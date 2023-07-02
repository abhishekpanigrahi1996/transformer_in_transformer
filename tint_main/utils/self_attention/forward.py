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
#self.memory_index : Start index of memorized embeddings (from a previous layer)
#config.memory_end : End Index of memorized embeddings (from a previous layer)
#config.position_start : Start index of one-hot position embeddings
#config.seq_length : Sequence length of the smaller model that we are trying to simulate
#config.blank_identifier : Index containing Identifiers for blank token
#config.num_prefixes : Number of prefixes to separate the sub-sequences
#config.num_attention_heads : Number of attention heads
#config.scale_embeddings : A scale to initialize different query, key matrices
#config.inner_lr : Inner learning rate to simulate sgd  
#------------------------------------------------------------------# 



#------------------------------------------------------------------##------------------------------------------------------------------##------------------------------------------------------------------# 
 #Input: Number of attention heads in the smaller model, din denotes the embedding dimension through the attention module of small model whose forward pass we are trying to simulate
#Output: 3 attention layers
class AttentionForward (nn.Module):
    def __init__ (self, config, din, num_attnt_heads, use_softmax, projection_matrix=None, separate_QK=False, memory_index=0):
        super(AttentionForward, self).__init__()
        
        assert use_softmax==False ,\
            "Currently I only use linear attention in this module"
        
        assert num_attnt_heads <= config.num_attention_heads,\
            "Number of attention heads should be at least the number of attention heads necessary to simulate"
        
        self.separate_QK = separate_QK
        if projection_matrix is not None:
            dout = projection_matrix.shape[1]
        else:
            if separate_QK: dout = 2*din
            else: dout = din    

        self.linear = LinearForward(config, din=din, dout=dout, use_softmax=use_softmax, projection_matrix=projection_matrix, memory_index=-1)
        
        self.key_linear = self.linear
        #LinearForward(config, din=din, dout=dout, use_softmax=use_softmax, projection_matrix=projection_matrix, memory_index=-1)
        self.value_linear = self.linear
        #LinearForward(config, din=din, dout=dout, use_softmax=use_softmax, projection_matrix=projection_matrix, memory_index=-1)
        
        #if separate_QK:
        #self.value_linear = LinearForward(config, din=din, dout=din, use_softmax=use_softmax, shift_top=2*din, memory_index=memory_index+2*din) 
        #if not separate_QK:
        #    self.key_linear = LinearForward(config, din=din, dout=din, use_softmax=use_softmax, shift_top=din, memory_index=memory_index+din)


        
        self.gates = Gates (config)
        
        self.din = din
        self.num_attnt_heads = num_attnt_heads
        self.config = config
        self.memory_index = memory_index


        head_dim = config.hidden_size // config.num_attention_heads
        basemodel_head_dim = din // num_attnt_heads  
        
        self.attnt_module = Attention (config, normalize=True, proj_conv2d=True, proj_conv_dim=head_dim, proj_transpose=True)
        
        assert din % head_dim == 0, \
               "a bug! 'din' should be divisible by head dimensions"
        
        num_partitions = din // head_dim
        
        assert num_attnt_heads % num_partitions == 0, \
               "Num of attention heads should be divisible by num of partitions"
        
        num_attnt_heads_per_partition = num_attnt_heads // num_partitions
        
        #--------------------------------#--------------------------------#
        #For all Attention heads on the embeddings
        #Query uses the first set of din coordinates and splits them among the first 'num_attnt_heads' attention heads
        #Key uses the second set of din coordinates and splits them among the first 'num_attnt_heads' attention heads
        #Value uses the third set of din coordinates and splits them among the first 'num_attnt_heads' attention heads
        #Key and Query of embeddings ignore the position dependence.
        #--------------------------------#--------------------------------#
        
        q_attn_head = torch.zeros((head_dim, config.num_attention_heads, config.num_attention_heads))
        for i in range(num_partitions):
            for j in range(num_attnt_heads_per_partition):
                q_attn_head[ :, i * num_attnt_heads_per_partition + j, i ] = 1.
        
        
        
        q_attn = torch.zeros((config.num_attention_heads, head_dim, head_dim))
        for i in range(num_attnt_heads):
            partition = i % num_attnt_heads_per_partition
            q_attn[ i, :basemodel_head_dim, partition*basemodel_head_dim: (partition + 1)*basemodel_head_dim ] = torch.eye(basemodel_head_dim)
            
        
        k_attn_head = torch.zeros((head_dim, config.num_attention_heads, config.num_attention_heads))
        for i in range(num_partitions):
            for j in range(num_attnt_heads_per_partition):
                k_attn_head[ :, i * num_attnt_heads_per_partition + j, i + num_partitions] = 1.
        
         
        
        k_attn = torch.zeros((config.num_attention_heads, head_dim, head_dim))
        for i in range(num_attnt_heads):
            partition = i % num_attnt_heads_per_partition
            k_attn[ i, :basemodel_head_dim, partition*basemodel_head_dim: (partition + 1)*basemodel_head_dim ] = torch.eye(basemodel_head_dim)
            

        
        v_attn_head = torch.zeros((head_dim, config.num_attention_heads, config.num_attention_heads))
        for i in range(num_partitions):
            for j in range(num_attnt_heads_per_partition):
                v_attn_head[ :, i * num_attnt_heads_per_partition + j, i + 2 * num_partitions] = 1.
        
        
        v_attn = torch.zeros((config.num_attention_heads, head_dim, head_dim))
        for i in range(num_attnt_heads):
            partition = i % num_attnt_heads_per_partition
            v_attn[ i, partition*basemodel_head_dim: (partition + 1)*basemodel_head_dim, partition*basemodel_head_dim: (partition + 1)*basemodel_head_dim ] = torch.eye(basemodel_head_dim)
            

        #c_attn_init, c_attn_bias = torch.cat([query, key, value], axis=0), torch.zeros(5 * config.hidden_size)


        #--------------------------------#--------------------------------#
        #For all Attention heads on the positions
        #Query, Key are set such that we never attend to the blank tokens!
        #--------------------------------#--------------------------------#
        
        #--------------------------------#--------------------------------#
        #The projection matrix takes the output of the attention heads, which has the required signal only in its first basemodel_head_dim coordiantes
        #We merge them together and return them at the head of the embedding
        #--------------------------------#--------------------------------#
        c_proj_init = torch.zeros((head_dim, config.num_attention_heads, config.num_attention_heads))
        for i in  range(num_partitions):
            c_proj_init[:, i, i*num_attnt_heads_per_partition: (i+1)*num_attnt_heads_per_partition] = 1.
        
        
         
        self.attnt_module.initialize_weights(q_attn_init=q_attn,\
                                             q_attn_init_head=q_attn_head,\
                                             k_attn_init=k_attn,\
                                             k_attn_init_head=k_attn_head,\
                                             v_attn_init=v_attn,\
                                             v_attn_init_head=v_attn_head,\
                                             c_proj_init=c_proj_init )


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
        v [0, config.position_dim+config.seq_length: 2*config.position_dim] = - config.gate_scale * torch.ones(config.num_prefixes)
        v_bias [1] += config.gate_scale

        self.gates.initialize_weights (w, u, v, w_bias, u_bias, v_bias)

        #self.add_module('Attentionforward_Linearforward', self.linear)
        #self.add_module('Attentionforward_attention', self.attnt_module)
        #self.add_module('Attentionforward_gates', self.gates)
        
#Compute Q^{\top} K \sum_{j} a_{i, j} \nabla y_i^{\top} x_j x_j - Q^{\top} K  \nabla y_i^{\top} y_i y_i
# + K^{\top} Q \sum_{j} a_{j, i} \nabla y_j^{\top} x_i x_j - K^{\top} Q \sum_{j} a_{j, i} \nabla y_j^{\top} y_j x_j
# + \sum_{j} a_{j, i} \nabla y_j
       
    def forward(self, hidden_states, position_states, key_weights=None, value_weights=None, icl_mask=None):
        
        linear_output = self.linear.forward(hidden_states, position_states)
        #if self.separate_QK:
            #inp_hidden_states = hidden_states.clone()
            #this is an in-place operation, hence I need to do clone
            #inp_hidden_states[:, :self.config.num_prefixes] += ( value_weights - inp_hidden_states[:, :self.config.num_prefixes] )
        
        if not self.separate_QK:
            inp_hidden_states = torch.cat( [key_weights, hidden_states[:, self.config.num_prefixes:] ], axis=1)            
            #key_out = self.key_linear(inp_hidden_states, position_states)
            key_out = self.key_linear(inp_hidden_states, position_states)
            assert torch.sum(linear_output[:, self.config.num_prefixes:, self.din:]).item() < 1e-10,\
                   "Key portion not empty!"
            linear_output[:, self.config.num_prefixes:, self.din:] += key_out[:, self.config.num_prefixes:, :-self.din]

        
        
        inp_hidden_states = torch.cat( [value_weights, hidden_states[:, self.config.num_prefixes:] ], axis=1)            
        #value_out = self.value_linear(inp_hidden_states, position_states)
        value_out = self.value_linear(inp_hidden_states, position_states)
        
        assert torch.sum(linear_output[:, self.config.num_prefixes:, 2*self.din:]).item() < 1e-10,\
                "Value portion not empty!"
        linear_output[:, self.config.num_prefixes:, 2*self.din:] += value_out[:, self.config.num_prefixes:, :-2*self.din]

        #Send a mask such that the tokens don't attend to the blank tokens
        normalization_mask = torch.zeros( (1, 1, len(hidden_states[0]),  len(hidden_states[0]) ) )
        normalization_mask[:, :, :, :self.config.num_prefixes] = torch.finfo(self.attnt_module.p_attn.weight.dtype).min
        
        #icl_mask needs to be a 3D tensor of shape (batch_size, seqlen, seqlen)
        #icl_mask[i, j] = 1 if token i tends to token j
        
        if icl_mask is not None:
            
            bt = icl_mask.shape[0]
            for i in range( bt ):
                sq1 = icl_mask[i].shape[0]
                sq2 = icl_mask[i].shape[1]
                nb  = self.config.num_prefixes
            
                normalization_mask[i, :, nb: nb+sq1, nb: nb+sq2] = torch.tril( torch.round(torch.clip(1. - icl_mask[i], 0., 1.)) ) * torch.finfo(self.attnt_module.p_attn.weight.dtype).min
                
        #print ("------Attention-------")
        attnt_output  = self.attnt_module.forward(linear_output, position_states, normalization_mask=normalization_mask) [0]
        
        if self.memory_index != -1:
            #keep Qx, Kx in memory!
            #Keep also x separately afterwards!
            assert torch.sum(attnt_output[:, self.config.num_prefixes:, self.memory_index:]).item() < 1e-10,\
                   "Memory portion not empty!"
            
            attnt_output[:, self.config.num_prefixes:, self.memory_index: self.memory_index+2*self.din] += linear_output[:, self.config.num_prefixes:, :2*self.din]
            attnt_output[:, self.config.num_prefixes:, self.memory_index+2*self.din: self.memory_index+3*self.din] += hidden_states[:, self.config.num_prefixes:, :self.din]
        
        gate_output   = self.gates.forward(linear_output, attnt_output, position_states)
        
        return gate_output
