import torch

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ..modules import *
from ..linear import *


#Assumption on memory
#It should contain [(x-\mu)/\sigma, x]

class LayerNormBackward(nn.Module):
    def __init__(self, config, din, use_softmax, retain_nablay=False, memory_index=-1):
        super(LayerNormBackward, self).__init__()

        assert use_softmax==False ,\
            "Currently I only use linear attention in this module"
        
        assert memory_index == -1 or memory_index >= din, \
            "memory crosses current signal"
    

        self.linear = LinearBackward(config, \
                                     din=din, \
                                     dout=din, \
                                     use_softmax=use_softmax, \
                                     retain_nablay=retain_nablay, \
                                     memory_index=memory_index, \
                                    )
        self.epsilon = config.epsilon
        self.memory_index = memory_index
        self.config = config
        
        head_dim  = config.hidden_size // config.num_attention_heads
        self.c_fc = Conv2D(config.num_attention_heads, head_dim, transpose=True, use_einsum=self.config.use_einsum)
        self.proj_fc = Conv2D(config.num_attention_heads, head_dim, transpose=True, use_einsum=self.config.use_einsum)
        
        self.config = config
        self.din = din
        
        
        c_fc_init = torch.zeros((head_dim, config.num_attention_heads, config.num_attention_heads))
        c_proj_init = torch.zeros((head_dim, config.num_attention_heads, config.num_attention_heads))
        
    

        
        assert din % head_dim == 0, \
            " 'din' should be a multiple of head_dim! "
        
        num_partitions = din // head_dim
        
        
        
        assert self.memory_index % head_dim == 0, \
            "Memory should start at a multiple of head_dim!"
        
        mem_head_start = self.memory_index // head_dim
        
        if retain_nablay:
            start_shift = num_partitions
        else:
            start_shift = 0
            
        c_fc_init[:, start_shift: start_shift + num_partitions, start_shift: start_shift + num_partitions] = 1. / config.scale_embeddings * torch.eye(num_partitions)
        #1. / config.scale_embeddings
        c_fc_init[:, start_shift: start_shift + num_partitions, mem_head_start + num_partitions: mem_head_start + 2*num_partitions] =  torch.eye(num_partitions)
        
        
        #Compute GeLU(x + 1/N \nabla y) - GeLU(x)

        c_proj_init[:, start_shift: start_shift + num_partitions, start_shift: start_shift + num_partitions] = config.scale_embeddings * torch.eye(num_partitions)
        c_proj_init[:, start_shift: start_shift + num_partitions, mem_head_start: mem_head_start + num_partitions] = -config.scale_embeddings  * torch.eye(num_partitions)
        
        
        with torch.no_grad():
            
            self.c_fc.weight.copy_(torch.swapaxes(c_fc_init, axis0=-1, axis1=-2))
            self.proj_fc.weight.copy_(torch.swapaxes(c_proj_init, axis0=-1, axis1=-2))
        
        #w acts like a gate to decide what portion of the embedding we apply layernorm on
        self.w   = torch.zeros (( 1, 1, config.hidden_size ))
        if retain_nablay:
            self.w [:, :, din : 2*din] += config.gate_scale
        else:
            self.w [:, :, : din] += config.gate_scale
        self.gate = torch.nn.Tanh()
        
        
        #mask out normalization on prefixes 
        self.normalization_gates = Gates (config)
        #Initialize Gates
        #Ignore the changes for the prefixes!
        #w, u, v, w_bias, u_bias, v_bias
        w = torch.zeros((1, 2*config.hidden_size))
        u = torch.zeros((1, 2*config.hidden_size))
        v = torch.zeros((1, 2*config.position_dim))
        w_bias = torch.zeros(2)
        u_bias = torch.zeros(2)
        v_bias = torch.zeros(2)

        #Input Gate is 1 on prefixes and 0 for non-prefixes
        v [0, config.seq_length: config.position_dim] = config.gate_scale * torch.ones(config.num_prefixes)
        

        #Change Gate is 0 on prefixes and 1 for non-prefixes
        v [0, config.position_dim+config.seq_length: 2*config.position_dim] = -config.gate_scale * torch.ones(config.num_prefixes)
        v_bias [1] += config.gate_scale

        self.normalization_gates.initialize_weights (w, u, v, w_bias, u_bias, v_bias)

        
        
    def forward(self, hidden_states, position_states, attention_mask=None, icl_mask=None):    
        
        weights = self.gate ( self.w ).to(hidden_states.device)
        
        back_gradient = self.linear.forward(hidden_states, position_states)
        
        #print (back_gradient[0, 12, :72], back_gradient[0, 12, -72:])
        #######################################################################
        #Next few lines compute the operation:
        # f(x) = (x - \mu(x)) / \nabla(x)
        # N (f(x + 1/N \nabla y) - f(x))
        #######################################################################
        first_layer = self.c_fc.forward ( back_gradient )
        first_layer = weights * first_layer + (1. - weights) * back_gradient
        
        #print (first_layer[0, 12, :72])
        
        mean = torch.sum(first_layer * weights, dim=-1, keepdim=True) / torch.sum(weights, dim=-1, keepdim=True)        
        var = ( self.epsilon + torch.sum( (weights * (first_layer - mean)) ** 2, dim=-1, keepdim=True) / torch.sum(weights, dim=-1, keepdim=True) ) ** 0.5
        
        #print (var)
        normalized_states = (first_layer - mean) / var
        #print (normalized_states[:, 192, :64])
        
        normalized_states = weights * normalized_states + (1. - weights) * first_layer
        
        second_layer = self.proj_fc.forward ( normalized_states )
        
        second_layer = weights * second_layer + (1. - weights) * normalized_states
        
        #######################################################################
            
        gated_output = self.normalization_gates.forward ( hidden_states, second_layer, position_states)
        
        return gated_output
        
        
        
class LayerNormDescent(nn.Module):
    def __init__ (self, config, din, use_softmax, memory_index=-1, debug_zero=False):
        super(LayerNormDescent, self).__init__()
        self.config=config
        self.linear = LinearDescent(config, din=din, dout=din, use_softmax=use_softmax, memory_index=memory_index, debug_zero=debug_zero, update_bias_only=self.config.ln_update_bias_only) 
        
    def forward(self, hidden_states, position_states, attention_mask, activation_memory=None, icl_mask=None):   
        return self.linear.forward(hidden_states, position_states, attention_mask)    

    
    
class LayerNormDescent_Backward(nn.Module):
    def __init__(self, config, din, use_softmax, debug_zero=False, retain_nablay=False, projection_matrix=None, memory_index=-1):
        super(LayerNormDescent_Backward, self).__init__()
        
        
        self.config = config
        self.backward = LayerNormBackward(config, \
                                          din=din, \
                                          use_softmax=use_softmax, \
                                          retain_nablay=retain_nablay, \
                                          memory_index=memory_index, \
                                         )
        
        self.descent = LayerNormDescent(config, \
                                        din=din, \
                                        use_softmax=use_softmax, \
                                        memory_index=memory_index,\
                                        debug_zero=debug_zero, \
                                       )
        
    def forward(self, hidden_states, position_embeddings, attention_mask, activation_memory=None, icl_mask=None):
        backward_out = self.backward(hidden_states, position_embeddings)
        descent_out  = self.descent (hidden_states, position_embeddings, attention_mask)
        
        return torch.cat( [ descent_out[:, :self.config.num_prefixes], backward_out[:, self.config.num_prefixes:] ], axis=1)
        