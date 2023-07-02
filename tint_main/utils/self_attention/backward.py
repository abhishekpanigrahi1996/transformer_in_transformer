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



#Implements the stop-attention gradient, where we don't compute the gradient w.r.t. the attention scores
class AttentionBackward(nn.Module):

    def __init__ (self, config, din, num_attnt_heads, use_softmax, retain_nablay=False, memory_index=-1):
        super(AttentionBackward, self).__init__()
        
        assert use_softmax==False ,\
            "Currently I only use linear attention in this module"
        
        
    
        self.attnt_gates = Gates (config)
        
        #self.backward = LinearBackward(config, din=din, dout=din, use_softmax=False, retain_nablay=retain_nablay) 
        self.retain_nablay = retain_nablay
        self.memory_index = memory_index
        self.config = config
        self.din = din
        
        ##### First attention module #######
        ########### Assumption #############
            #The memory part has the following format [Qx, Kx] which helps us to re-compute the attention scores
            #We compute \sum_j a_{j, i} \nabla y_j
        ########### Assumption #############
        

        head_dim = config.hidden_size // config.num_attention_heads
        basemodel_head_dim = din // num_attnt_heads     

        self.attnt_module = Attention (config, peak_into_future=True, normalize=True, attnt_back=True, proj_conv2d=True, proj_conv_dim=head_dim, proj_transpose=True)
        
        
        assert self.memory_index <= config.hidden_size - ( 3 * self.din ), \
            "Not enough memory to simulate backward pass"
        assert self.memory_index  == -1 or self.memory_index >= self.din, \
            "Memory is crossing current signal (and additional computation space)!"
        
        #assert config.seq_length <= head_dim ,\
        #    "Currently I assume the head dimension is atleast the sequence length of original model"
        
        #--------------------------------#--------------------------------#
        #For all Attention heads on the embeddings
        #Query uses the first set of din coordinates in memory and splits them among the first 'num_attnt_heads' attention heads
        #Key uses the second set of din coordinates in memory and splits them among the first 'num_attnt_heads' attention heads
        #Value uses the first set of din coordinates and splits them among the first 'num_attnt_heads' attention heads
        #Key and Query of embeddings ignore the position dependence.
        #--------------------------------#--------------------------------#
        
        num_partitions = din // head_dim
        
        assert num_attnt_heads % num_partitions == 0, \
               "Num of attention heads should be divisible by num of partitions"
        
        num_attnt_heads_per_partition = num_attnt_heads // num_partitions
        
        
        q_attn_head = torch.zeros((head_dim, config.num_attention_heads, config.num_attention_heads))
        
        assert memory_index % head_dim == 0,\
               "Memory index should be multiple of head_dim"
        mem_head_start = memory_index // head_dim
       
        
        for i in range(num_partitions):
            for j in range(num_attnt_heads_per_partition):
                q_attn_head[ :, i * num_attnt_heads_per_partition + j, i + mem_head_start ] = 1.
        #print (q_attn_head[0])
        #exit(0)
        
        #q_attn_head[:, :, 0] = 1.
        #q_attn_head = q_attn_head.view((head_dim, config.num_attention_heads, config.num_attention_heads))
        
        q_attn = torch.zeros((config.num_attention_heads, head_dim, head_dim))
        for i in range(num_attnt_heads):
            partition = i % num_attnt_heads_per_partition
            q_attn[ i, :basemodel_head_dim, partition*basemodel_head_dim: (partition + 1)*basemodel_head_dim ] = torch.eye(basemodel_head_dim)
            
            
        #query = torch.zeros((2*config.num_attention_heads, head_dim, config.hidden_size))
        #query [:num_attnt_heads, :basemodel_head_dim, :basemodel_head_dim] = torch.eye(basemodel_head_dim)
        #for i in range (num_attnt_heads):
        #    query[i, :basemodel_head_dim, self.memory_index + i*basemodel_head_dim: self.memory_index + (i+1)*basemodel_head_dim] = torch.eye(basemodel_head_dim)

        #query = query.view( (2*config.hidden_size, config.hidden_size) )
        
        #key = torch.zeros((2*config.num_attention_heads, head_dim, config.hidden_size))
        #key [:num_attnt_heads, :basemodel_head_dim, :basemodel_head_dim] = torch.eye(basemodel_head_dim)
        #for i in range (num_attnt_heads):
        #    key[i, :basemodel_head_dim, self.memory_index + din + i*basemodel_head_dim: self.memory_index + din + (i+1)*basemodel_head_dim] = torch.eye(basemodel_head_dim)
        #key = key.view( (2*config.hidden_size, config.hidden_size) )


        #value = torch.zeros((config.num_attention_heads, head_dim, config.hidden_size))
        #key [:num_attnt_heads, :basemodel_head_dim, :basemodel_head_dim] = torch.eye(basemodel_head_dim)
        #for i in range (num_attnt_heads):
        #    value[i, :basemodel_head_dim, i*basemodel_head_dim: (i+1)*basemodel_head_dim] = torch.eye(basemodel_head_dim)
        #value = value.view( (config.hidden_size, config.hidden_size) )
        
        #c_attn_init, c_attn_bias = torch.cat([query, key, value], axis=0), torch.zeros(5 * config.hidden_size)

        
        k_attn_head = torch.zeros((head_dim, config.num_attention_heads, config.num_attention_heads))
        for i in range(num_partitions):
            for j in range(num_attnt_heads_per_partition):
                k_attn_head[ :, i * num_attnt_heads_per_partition + j, i + num_partitions + mem_head_start] = 1.
        
         
        #q_attn_head[:, :, 0] = 1.
        #q_attn_head = q_attn_head.view((head_dim, config.num_attention_heads, config.num_attention_heads))
        
        k_attn = torch.zeros((config.num_attention_heads, head_dim, head_dim))
        for i in range(num_attnt_heads):
            partition = i % num_attnt_heads_per_partition
            k_attn[ i, :basemodel_head_dim, partition*basemodel_head_dim: (partition + 1)*basemodel_head_dim ] = torch.eye(basemodel_head_dim)
            
        
        value_head = 0
        #if not self.config.backprop_through_attention:
        #    value_head = (self.memory_index + 2*self.din) // head_dim
            
            
        v_attn_head = torch.zeros((head_dim, config.num_attention_heads, config.num_attention_heads))
        for i in range(num_partitions):
            for j in range(num_attnt_heads_per_partition):
                v_attn_head[ :, i * num_attnt_heads_per_partition + j, i + value_head ] = 1.
        
        #print (v_attn_head[:2])
        #exit(0) 
        #q_attn_head[:, :, 0] = 1.
        #q_attn_head = q_attn_head.view((head_dim, config.num_attention_heads, config.num_attention_heads))
        
        
        
        
        v_attn = torch.zeros((config.num_attention_heads, head_dim, head_dim))
        for i in range(num_attnt_heads):
            partition = i % num_attnt_heads_per_partition
            v_attn[ i, partition*basemodel_head_dim: (partition + 1)*basemodel_head_dim,  partition*basemodel_head_dim:  (partition + 1)*basemodel_head_dim ] = torch.eye(basemodel_head_dim)

        #--------------------------------#--------------------------------#
        #For all Attention heads on the positions
        #Query, Key are set such that we never attend to the blank tokens!
        #--------------------------------#--------------------------------#
        #position_dim_combined = config.position_dim * config.num_attention_heads

        #query = torch.zeros((config.num_attention_heads, head_dim, config.position_dim))
        #query[:num_attnt_heads, 0, :config.seq_length] = 1.
        #query = query.view( (config.hidden_size, config.position_dim) )
        
        
        #key = torch.zeros((config.num_attention_heads, head_dim, config.position_dim))        
        #key[:num_attnt_heads, 0, config.seq_length: config.position_dim] = -torch.finfo(self.attnt_module.c_attn.weight.dtype).max
        #key = key.view( (config.hidden_size, config.position_dim) )
        
        #value = torch.zeros((config.num_attention_heads * head_dim, config.position_dim))
        #p_attn_init = torch.cat([query, key, value], axis=0)
        

        #--------------------------------#--------------------------------------------#--------------------------------#
        #The projection matrix after the attention module places the output of the attention module after \nabla y at each position i
        #--------------------------------#--------------------------------------------#--------------------------------#
        #c_proj_init = torch.zeros((config.hidden_size, config.hidden_size))
        
        #if retain_nablay: start_index = din
        #else: start_index = 0
            
        #for i in range(din):
        #    c_proj_init [ i, (i // basemodel_head_dim) * head_dim + i % basemodel_head_dim] = 1.
        c_proj_init = torch.zeros((head_dim, config.num_attention_heads, config.num_attention_heads))
        #, torch.zeros(config.hidden_size)
        #c_proj_init[ :din, :din ] = torch.eye(din)
        #for i in range(din):
        #    c_proj_init[i, (i // basemodel_head_dim) * head_dim + i % basemodel_head_dim] = 1.
        for i in  range(num_partitions):
            c_proj_init[:, i + value_head, i*num_attnt_heads_per_partition: (i+1)*num_attnt_heads_per_partition] = 1.
        
        
        
        self.attnt_module.initialize_weights(q_attn_init=q_attn,\
                                             q_attn_init_head=q_attn_head,\
                                             k_attn_init=k_attn,\
                                             k_attn_init_head=k_attn_head,\
                                             v_attn_init=v_attn,\
                                             v_attn_init_head=v_attn_head,\
                                             c_proj_init=c_proj_init )
        
        #Initialize the first attention Gates
        #Ignore the changes for the prefixes!
        #w, u, v, w_bias, u_bias, v_bias
        w = torch.zeros((1, 2*config.hidden_size))
        u = torch.zeros((1, 2*config.hidden_size))
        v = torch.zeros((1, 2*config.position_dim))
        w_bias = torch.zeros(2)
        u_bias = torch.zeros(2)
        v_bias = torch.zeros(2)

        #if self.retain_nablay:
            #Input Gate is 1
        #    v_bias [0] += config.gate_scale
        #else:
        #Input Gate is 1 on prefixesß and 0 for non-prefixes
        v [0, config.seq_length:config.position_dim] = config.gate_scale * torch.ones(config.num_prefixes)
            
        #Change Gate is 0 on prefixes and 1 for non-prefixes
        v [0, config.position_dim+config.seq_length: 2*config.position_dim] = -config.gate_scale * torch.ones(config.num_prefixes)
        v_bias [1] += config.gate_scale

        self.attnt_gates.initialize_weights (w, u, v, w_bias, u_bias, v_bias)
        
        
        #self.add_module('Attentionback_attention', self.attnt_module)
        #self.add_module('Attentionback_gates', self.attnt_gates)
        #self.add_module('Attentionback_Linearback', self.backward)
        
    def forward(self, hidden_states, position_states, attention_mask, icl_mask=None):
        
        #print ("*******", hidden_states[0, :3])
        #add a mask to avoid attention on blank tokens!
        
        
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
        
        #print ("----Mask----", attention_mask)
        modified_attention_mask = attention_mask.detach().clone()
        modified_attention_mask[:, :, :, :self.config.num_prefixes] = 0.
        
        attnt_output  = self.attnt_module.forward(hidden_states, \
                                                  position_states, \
                                                  attention_mask=modified_attention_mask, \
                                                  normalization_mask=normalization_mask\
                                                 ) [0]
        
        #print (attnt_output[0, 2+self.config.num_prefixes])
        #if not self.config.backprop_through_attention:
        #    end_dim = self.memory_index + 2*self.din
        #    attnt_output[:, self.config.num_prefixes:, :self.din] += hidden_states[:, self.config.num_prefixes:, :self.din]
        #else:
        end_dim = self.memory_index + 3*self.din
            
        attnt_output[:, self.config.num_prefixes:, self.memory_index: end_dim] += hidden_states[:, self.config.num_prefixes:, self.memory_index: end_dim]
        
        
        gate_output   = self.attnt_gates.forward(hidden_states, \
                                                 attnt_output, \
                                                 position_states\
                                                ) 
        
        #print ("-------------------", gate_output[0, 5])
        #linear_output = self.backward.forward(gate_output, position_states)        
        return gate_output
    
    


    

    
#Implements descent w.r.t. the value matrix    
class AttentionDescent(nn.Module):
    def __init__ (self, config, din, num_attnt_heads, use_softmax, memory_index=-1, debug_zero=False, retain_nablay=False):
        super(AttentionDescent, self).__init__()
        self.linear = LinearDescent(config, din=din, dout=din, use_softmax=use_softmax, memory_index=memory_index+2*din, debug_zero=debug_zero) 
        #self.add_module('Attentiondescent_Lineardescent', self.linear)
        
        
    def forward(self, hidden_states, position_states, attention_mask, icl_mask=None):   
        return self.linear.forward(hidden_states, position_states, attention_mask)
    
    
    
    #self, config, din, dout, use_softmax, debug_zero=False, retain_nablay=False, projection_matrix=None, memory_index=-1
    #self, config, din, num_attnt_heads, use_softmax, retain_nablay=False, memory_index=-1
class AttentionBackward_Descent(nn.Module):
    def __init__ (self, config, din, num_attnt_heads, use_softmax, memory_index=-1, debug_zero=False, projection_matrix=None, retain_nablay=False):
        super(AttentionBackward_Descent, self).__init__()
        
        self.config = config
        self.memory_index = memory_index
        self.din = din
        #self, config, din, num_attnt_heads, use_softmax, retain_nablay=False, memory_index=-1
        self.attention_back = AttentionBackward(config, \
                                                 din=din, \
                                                 num_attnt_heads=num_attnt_heads, \
                                                 memory_index=memory_index, \
                                                 use_softmax=use_softmax, \
                                                 retain_nablay=retain_nablay,\
                                                )
        
        #self.add_module('Attention_backward', self.attention_back)
        
        self.linearback_descent = Linear_Descent_Backward(config, \
                                                          din=din, \
                                                          dout=din, \
                                                          use_softmax=use_softmax, \
                                                          memory_index=memory_index+2*din, \
                                                          debug_zero=debug_zero, \
                                                          projection_matrix=projection_matrix, \
                                                          retain_nablay=retain_nablay, \
                                                         ) 
        #self.add_module('Attention_Linearback_descent', self.linearback_descent)
        
        
    def forward(self, hidden_states, position_states, attention_mask, icl_mask=None):   
        if self.config.backprop_through_attention:
            attention_backout = self.attention_back(hidden_states, position_states, attention_mask, icl_mask=icl_mask)
        else:
            attention_backout = hidden_states
            
        #if activation_memory is not None:
        #    mem_replace_dim = activation_memory.shape[-1]
        #    mem_index = self.memory_index + self.din
        #    num_prefixes = self.config.num_prefixes
        #    attention_backout[:, num_prefixes:, mem_index: mem_index + mem_replace_dim] += (activation_memory - attention_backout[:, num_prefixes:, mem_index: mem_index + mem_replace_dim])    
        attention_descentout = self.linearback_descent(attention_backout, position_states, attention_mask)
        
        #if not self.config.backprop_through_attention:
        #    attention_descentout[:, self.config.num_prefixes:] = hidden_states[:, self.config.num_prefixes:]
        
        
        return attention_descentout    