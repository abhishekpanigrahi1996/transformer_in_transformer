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



#------------------------------------------------------------------#
#embedding structure [ signal, memory, position ] 
#I will introduce k prefixes to separate different sub-sequences
#------------------------------------------------------------------#



#------------------------------------------------------------------#
#config contains the following important parameters: 
#config.signal_start : Start Index of current signal embeddings (0 always)
#config.signal_end : End Index of current signal
#config.memory_start : Start index of memorized embeddings (from a previous layer)
#config.memory_end : End Index of memorized embeddings (from a previous layer)
#config.position_start : Start index of one-hot position embeddings
#config.seq_length : Sequence length of the smaller model that we are trying to simulate
#config.blank_identifier : Index containing Identifiers for blank token
#config.num_prefixes : Number of prefixes to separate the sub-sequences
#config.num_attention_heads : Number of attention heads
#config.scale_embeddings : A scale to initialize different query, key matrices
#config.inner_lr : Inner learning rate to simulate sgd 
#config.epsilon: epsilon for the denominator of layernorm 
#config.gate_scale: Scale to use inside gates  
#------------------------------------------------------------------# 

class LayerNormForward(nn.Module):
    def __init__(self, config, din, use_softmax, memory_index=-1):
        super(LayerNormForward, self).__init__()
        assert use_softmax==False ,\
            "Currently I only use linear attention in this module"

        
        
        self.linear=LinearForward ( config, din=din, dout=din, use_softmax=use_softmax, memory_index=memory_index )
        self.din=din
        self.epsilon = config.epsilon
        self.config=config
        self.memory_index = memory_index
        
        #w acts like a gate to decide what portion of the embedding we apply layernorm on
        self.w   = torch.zeros (( 1, 1, config.hidden_size ))
        self.w [:, :, :din] += config.gate_scale
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
        
        
        #self.add_module('LayernormForward_weights', self.w)
        #self.add_module('LayernormForward_normgates', self.normalization_gates)
        #self.add_module('LayernormForward_Linearforward', self.linear)
        
        
    def forward(self, hidden_states, position_embeddings):
        
        #print ("------", torch.sum( torch.absolute(hidden_states[:, self.config.num_prefixes:, self.memory_index:])).item(), "------")
        #print (hidden_states[0, self.config.num_prefixes, self.memory_index:])
        weights = self.gate ( self.w ).to(hidden_states.device)
        mean = torch.sum(hidden_states * weights, dim=-1, keepdim=True) / torch.sum(weights, dim=-1, keepdim=True)
        
        var = ( self.epsilon + torch.sum( (weights * (hidden_states - mean)) ** 2, dim=-1, keepdim=True) / torch.sum(weights, dim=-1, keepdim=True) ) ** 0.5
        
        normalized_states = (hidden_states - mean) / var
        #print (normalized_states[0, self.config.num_prefixes, self.memory_index:])
        normalized_states = weights * normalized_states + (1. - weights) * hidden_states
        #print (normalized_states[0, self.config.num_prefixes, self.memory_index:])
        
        gated_output = self.normalization_gates.forward (hidden_states, normalized_states, position_embeddings)
        #print (gated_output[0, self.config.num_prefixes, self.memory_index:])
        
        output = self.linear.forward ( gated_output, position_embeddings )
        
        #print ("------", torch.sum( torch.absolute(output[:, self.config.num_prefixes:, self.memory_index:])).item(), "------")
        #print (output[0, self.config.num_prefixes, self.memory_index:])
        #store [(x-\mu)/\sigma, x] for memory in backward pass
        assert torch.sum( torch.absolute( output[:, self.config.num_prefixes:, self.memory_index+self.din:]) ).item() < 1e-10,\
               "Memory portion not empty!"
        output[:, self.config.num_prefixes:, self.memory_index+self.din: self.memory_index+2*self.din] += hidden_states[:, self.config.num_prefixes:, :self.din]
        #output[:, :, self.memory_index+self.din: self.memory_index+2*self.din] += hidden_states[:, :, :self.din]
        
        return output
    
