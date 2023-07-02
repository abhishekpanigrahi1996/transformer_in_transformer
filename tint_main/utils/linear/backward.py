

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
import numpy as np



#LinearBackward module computes \nabla y_i W^\top at every position i
#To do so, we compute \sum_j (\nabla y )_j W_j at each position i.
#Here W_j, denotes j^th row of W.
#Important arguments: input dimension (din), output dimension (dout)
#output: Linear Backward layer, primarily containing a linear self_attention (see figure 4 in main paper)
#Currently, I use linear attention in this module. 

#Before calling this module, we have already used a residual connection
#to copy the relevant weights from the forward operation onto the prefix tokens


#If there is a projection involved, I always assume that nabla y 
#has already been projected using the projection matrix, before it enters this module!


#All arguments
#self, \
#config, \                   #TinT config file
#din, \                      #input dimension of auxiliary's linear layer
#dout, \                     #outut dimension of auxiliary's linear layer
#use_softmax=False, \        #linear self_attention used
#retain_nablay=False, \      #Unnecessary argument, to be cleared out in 2nd version
#projection_matrix=None, \   #Projection matrix used in forward operation
#memory_index=-1,\           #Start index where activations are stored in Linear Forward.

class LinearBackward(nn.Module):
    
    
    def __init__(self, \
                 config, \
                 din, \
                 dout, \
                 use_softmax=False, \
                 retain_nablay=False, \
                 projection_matrix=None, \
                 memory_index=-1,\
                ):
        super(LinearBackward, self).__init__()
        assert use_softmax==False ,\
            "Currently I only use linear attention in this module"

        
        
        num_attnt_heads_per_dim=dout // config.num_prefixes
        self.gates = Gates (config)
        self.retain_nablay = retain_nablay
        self.projection_matrix = projection_matrix
        self.memory_index = memory_index
        self.config = config
        
        #self.projection_place = projection_place
        #self.inner_gate = Gates (config)
        #self.outer_attnt_module = Attention (config)
        
        
        

        head_dim = config.num_prefixes
        
        assert config.hidden_size % head_dim == 0,\
            "Dimension should perfectly distribute over num_prefixes" 
        
        #head_dim = config.hidden_size // num_attention_heads
        num_attention_heads = config.hidden_size // head_dim
        attnt_head_per_wt= din // head_dim
        
        self.attnt_module = Attention (config, num_attention_heads=num_attention_heads, normalize=False, query_conv_dim=dout, proj_conv2d=True, proj_conv_dim=head_dim, proj_transpose=True)
        
        assert din % head_dim == 0,\
            "Currently this is a bug! I assume din is easily divisible across the heads we want to distribute to"
        
        assert memory_index == -1 or memory_index >= din + dout, \
            "Memory intersects with the current signal (and additional necessary space)"
        
        din_partition=din//attnt_head_per_wt
        
        
        
        
        assert num_attention_heads >= attnt_head_per_wt * num_attnt_heads_per_dim ,\
            "Currently I assume the number of attention heads is atleast the number of weights in each blank"
        
        #assert head_dim >= din, \
        #    "Attention head dimension must be greater than the dimension of linear input"
        
        
        #--------------------------------#---------------------------------#-------------------------------#
        #For all Attention heads on the embeddings
        #Key is all zeros.
        #Query takes \nabla y and reorders it so that the dimensions correspond to right indices in each attention head.
        #Query and Key only correspond to cross attention between position and signal (\nabla y)
        #
        #
        #Example Reordering for query: if the number of weight rows in each blank vector is 32 and the total
        #number of weight rows is 768, then 
        #the order will be 
        #[\nabla y_1, \nabla y_33, \nabla_65, ..., \nabla y_737, 0, ..., 0, \nabla y_2, \nabla y_34, \nabla_66, ..., \nabla y_738, 0, ..., 0, ....]. 
        #Here 0s separate the input to different attention heads!
        #
        #
        #Value is Identity matrix which simply gives the weight vectors stored in the blank tokens.
        #--------------------------------#--------------------------------#--------------------------------#
        #key = torch.zeros((2*config.num_attention_heads * head_dim, config.hidden_size))
        
        
        #query = torch.zeros((config.num_attention_heads, head_dim, config.hidden_size))
        query_conv_head = config.hidden_size // dout
        
        query_attn_head = torch.zeros((dout, query_conv_head, query_conv_head))
        for i in range(dout):
            query_attn_head[i, :, 0] = 1.  
        
        query_attn = torch.zeros((num_attention_heads, head_dim, dout))
        
        #num_repetitions = dout // din 
        num_useful_heads = dout // config.num_prefixes
        for i in range(dout):
            attnt_head = (i % num_useful_heads) * attnt_head_per_wt
            dim = i // num_useful_heads            
            for j in range(attnt_head_per_wt):
                query_attn[attnt_head + j, dim, i] = 1.
            #/config.scale_embeddings
        query_attn = query_attn.view((query_conv_head, dout, dout))
   
        key_attn = torch.zeros((num_attention_heads, head_dim, head_dim))
        
        
        #--------------------------------#--------------------------------#
        #For all Attention heads on the positions
        #Query is all zeros
        #Key copies the blank identifiers to the fore-front (for key position-query signal component).
        #Value is all zeros.
        #--------------------------------#--------------------------------#
        

        query = torch.zeros((head_dim, config.position_dim))
        
        
        key = torch.zeros((head_dim, config.position_dim))
             
         
        key[:config.num_prefixes, config.seq_length:config.position_dim] = torch.eye(config.num_prefixes)
        

        value = torch.zeros((head_dim, config.position_dim))
        

        p_attn_init = torch.cat([query, key, value], axis=0)
        
        expand_ = torch.zeros((3, 1, num_attention_heads))
        expand_[1, 0, :] = 1.
        
        #--------------------------------#--------------------------------#
        #After value projection, the embeddings contain [ \sum_{i \in H} ( \nabla y )_i w_i ]_H, 
        #where H denotes the set of weight row indices present in each blank.
        #The projection matrix after the attention module will compute \sum_i (\nabla y)_i w_i
        #if self.retain_nablay  is True, we store it after dout embeddings, since we still need \nabla y for descent!
        #--------------------------------#--------------------------------#
        c_proj_init = torch.zeros((config.hidden_size, config.hidden_size))
        num_useful_heads = dout // config.num_prefixes
        
        c_proj_init = torch.zeros((head_dim, num_attention_heads, num_attention_heads))
        start_head = 0
        if self.retain_nablay :
            start_head = dout // head_dim
            
        for i in range(num_useful_heads):
            c_proj_init[ :, start_head: start_head+attnt_head_per_wt, i*attnt_head_per_wt: (i+1)*attnt_head_per_wt ] = torch.eye(attnt_head_per_wt)
        
        
        self.projection_layer = None
        if projection_matrix is not None:
            if self.retain_nablay : start_index = dout
            else: start_index = 0
            
            if projection_matrix.shape[0] >= projection_matrix.shape[1]:
                self.projection_layer = up_projection (config, projection_matrix, signal_index=start_index, store_index=start_index)
            else:
                self.projection_layer = down_projection (config, projection_matrix, signal_index=start_index, store_index=start_index)
            
        
        self.attnt_module.initialize_weights(q_attn_init=query_attn, \
                                             q_attn_init_head=query_attn_head, \
                                             k_attn_init=key_attn, \
                                             p_attn_init=p_attn_init,\
                                             p_expand_init=expand_, \
                                             c_proj_init=c_proj_init, \
                                            )
        
        
        
        #Initialize Gates
        #Ignore the changes for the prefixes!
        #For non-prefixes, we have to add the change to the old value
        #w, u, v, w_bias, u_bias, v_bias
        w = torch.zeros((1, 2*config.hidden_size))
        u = torch.zeros((1, 2*config.hidden_size))
        v = torch.zeros((1, 2*config.position_dim))
        w_bias = torch.zeros(2)
        u_bias = torch.zeros(2)
        v_bias = torch.zeros(2)

        if self.retain_nablay :
            #gate on input is always 1
            v_bias [0] += config.gate_scale
        else:
            #gate on input is 1 (for prefixes) and 0 (for non-prefixes)
            v [ 0, config.seq_length: config.position_dim ] = config.gate_scale * torch.ones(config.num_prefixes)

            
        #gate on change is 0 (for prefixes) and 1 (for non-prefixes)
        v [ 0, config.position_dim + config.seq_length: 2 * config.position_dim ] = -config.gate_scale * torch.ones(config.num_prefixes)
        v_bias [1] += config.gate_scale
        
        
        self.gates.initialize_weights (w, u, v, w_bias, u_bias, v_bias)
        
    
        
    def forward(self, hidden_states, position_embeddings, attention_mask=None, icl_mask=None):
        output = self.attnt_module.forward(hidden_states=hidden_states, positions=position_embeddings, restrict_prefixes=self.config.restrict_prefixes)[0]
        if self.projection_layer is not None:
            output = self.projection_layer(output)
        
        #print (self.retain_nablay, self.memory_index, (not self.retain_nablay) and (self.memory_index != -1))
        if (not self.retain_nablay) and (self.memory_index != -1):
            output[:, self.config.num_prefixes:, self.memory_index:] += hidden_states[:, self.config.num_prefixes:, self.memory_index:]
            
        return self.gates.forward(hidden_states=hidden_states, \
                                  output_states=output, \
                                  position=position_embeddings\
                                 )
            
    
#LinearDescent module computes W - \sum_i \nabla y_i \nabla x_i^\top 
#Important arguments: output dimension (dout), input dimension (din)   
#output: Linear Descent layer, primarily containig a linear self_attention (see figure 5 in main paper)

#New arguments
#debug_zero=False, \             #unnecessary argument introduced for debugging purposes, to be cleaned up
#update_bias_only=False,\        #Introduced to check bitfit style training
class LinearDescent(nn.Module):    
    
    def __init__(self, \
                 config, \
                 din, \
                 dout, \
                 use_softmax=False, \
                 memory_index=-1, \
                 debug_zero=False, \
                 update_bias_only=False,\
                ):
        super(LinearDescent, self).__init__()
        
        assert use_softmax==False ,\
            "Currently I only use linear attention in this module"
        
        
        
        
        self.gates = Gates (config)
        self.memory_index = memory_index
        self.config=config
        #self.inner_gate = Gates (config)
        #self.outer_attnt_module = Attention (config)
        
        
        #head_dim = config.hidden_size // config.num_attention_heads
        head_dim = config.num_prefixes
        assert config.hidden_size % config.num_prefixes == 0,\
            "Dimension should perfectly distribute over all the attention heads" 

        num_attention_heads = config.hidden_size // head_dim
        
        self.head_dim = head_dim
        attnt_head_per_wt=int ( np.ceil(din / head_dim) )        
        extra_heads = int ( np.ceil (dout / head_dim) )
        copy_heads  = 0#int( np.ceil(din / head_dim) )
        num_attnt_heads_per_blank=dout // config.num_prefixes
        
        tot_wt_heads = attnt_head_per_wt * num_attnt_heads_per_blank
        tot_wt_bs_heads = tot_wt_heads + extra_heads
        
        self.attnt_module = Attention (config, num_attention_heads=num_attention_heads, normalize=use_softmax, peak_into_future=True, key_conv_dim=dout, total_position_embeddings=1, proj_conv2d=True, proj_conv_dim=head_dim, proj_transpose=True)
        
        
        if config.seq_length > head_dim:
            #we make head_dim chunk blocks to handle sequences that are longer than legnth head_dim
            #We will use attention masks to handle unnecessary dependencies
            position_projection_matrix = np.zeros((head_dim, config.seq_length))
            for i in range( int(np.ceil( config.seq_length // head_dim )) ):
                position_projection_matrix[ :, i * head_dim: min( (i+1) * head_dim,  config.seq_length) ] = np.eye( min(head_dim, config.seq_length - i * head_dim) )
                
            

            self.position_projection_tensor = torch.tensor(position_projection_matrix, dtype=self.attnt_module.p_attn.weight.dtype)
        else:
            self.position_projection_tensor = torch.eye(config.seq_length, dtype=self.attnt_module.p_attn.weight.dtype)
        
        assert din % attnt_head_per_wt == 0,\
            "Currently this is a bug! I assume that 'din' is easily divisible across the heads we want to distribute to"
        
        din_partition=din//attnt_head_per_wt
        
        assert num_attention_heads >= tot_wt_bs_heads + copy_heads,\
            "Currently I assume the number of attention heads is atleast the number of weights + biases in each blank"
        
        
        
        #assert head_dim >= din, \
        #    "Attention head dimension must be greater than the dimension of linear input"
        
        
        assert self.memory_index <= config.hidden_size - din, \
            "Not enough memory for backward pass"
        
        assert self.memory_index >= din + dout, \
            "Memory crosses current signal"
        
        #assert config.memory_start >= din * (dout // config.num_prefixes) + dout, \
        #    "I should have more space to store only the memory, where weights in prefixes don't intersect: this is a huge bug and need to change!"
        
        #assert config.num_attention_heads >= int (np.ceil(dout / head_dim)) + (dout // config.num_prefixes), \
        #    "Number of attention heads should be atleast the number of weights present in each blank"
        
        
        #--------------------------------#---------------------------------#-------------------------------#
        #For all Attention heads on the embeddings
        #Query is all zeros.
        #Key takes \nabla y and reorders it so that the dimensions correspond to right indices in each attention head.
        #Query and Key only correspond to cross attention between position and signal (\nabla y)
        #
        #
        #Example Reordering for key: if the number of weight rows in each blank vector is 32 and the total
        #number of weight rows is 768, then 
        #the order will be 
        #[\nabla y_1, \nabla y_33, \nabla_65, ..., \nabla y_737, 0, ..., 0, \nabla y_2, \nabla y_34, \nabla_66, ..., \nabla y_738, 0, ..., 0, ....]. 
        #Here 0s separate the input to different attention heads!
        #
        #
        #The memory contains input x to the linear layer.
        #Value caries the memory portion of the embeddings to the fore-front 
        #--------------------------------#--------------------------------#--------------------------------#
        
        #query = torch.zeros((config.num_attention_heads * head_dim, config.hidden_size))
        query_attn_head = torch.zeros((head_dim, num_attention_heads,  num_attention_heads))
        
        
        key_conv_head = config.hidden_size // dout
        key_attn_head = torch.zeros((dout, key_conv_head, key_conv_head))
        for i in range(dout):
            key_attn_head[i, :, 0] = 1.  
        
        key_attn = torch.zeros((num_attention_heads, head_dim, dout))
        
        #num_repetitions = dout // din 
        num_useful_heads = dout // config.num_prefixes
        for i in range(dout):
            attnt_head = (i % num_useful_heads) * attnt_head_per_wt
            dim = i // num_useful_heads            
            for j in range(attnt_head_per_wt):
                key_attn[attnt_head + j, dim, i] = 1.
            #/config.scale_embeddings
        key_attn = key_attn.view((key_conv_head, din, din))
        
        
        #value_conv_head = config.hidden_size // din
        value_attn_head = torch.zeros((head_dim, num_attention_heads, num_attention_heads))
        
        assert memory_index % head_dim == 0, \
            "Memory should start at multiple of head dimension"
        
        mem_head_start = memory_index // head_dim 
        mem_head_end   = mem_head_start + din // head_dim
        
        for i in range(num_useful_heads):
            if not debug_zero and not update_bias_only:
                value_attn_head[:, i * (mem_head_end - mem_head_start): (i+1) * (mem_head_end - mem_head_start), mem_head_start: mem_head_end] = - config.inner_lr * torch.eye (mem_head_end - mem_head_start)
            else:
                value_attn_head[:, i * (mem_head_end - mem_head_start): (i+1) * (mem_head_end - mem_head_start), mem_head_start: mem_head_end] = 0.
        #print (value_attn_head[0])
        #exit(0)
        
        value_attn_head = value_attn_head.view((head_dim,  num_attention_heads, num_attention_heads))
        
        
        
        #we keep few heads to compute the gradient for the bias
        for i in range(dout):
            desd_head = tot_wt_heads + i // head_dim
            desd_loc  = i % head_dim
            if not debug_zero:
                value_attn_head[desd_loc, desd_head, i // head_dim] = -config.inner_lr 
            else:
                value_attn_head[desd_loc, desd_head, i // head_dim] = 0.
                
        #we keep few heads to copy nabla x to the top
        #for i in range(din):
        #    desd_loc  = i % head_dim
        #    desd_head = tot_wt_bs_heads + i // head_dim
        #    attend_head = (i + dout) // head_dim
        #    value_attn_head[desd_loc, desd_head, attend_head] = 1.
        
        
        #--------------------------------#--------------------------------#
        #For all Attention heads on the positions (except one)
        #Key is all zeros
        #Query copies the blank identifiers to the fore-front (for key position-query signal component).
        #Value is all zeros.
        
        #We have one attention head that only keeps nabla x in the end at all non-blank positions
        #We have another attention head that simply uses \nabla y's to compute the gradient
        #for biases stored in the first blank!
        #--------------------------------#--------------------------------#
        

        key = torch.zeros((1, head_dim, config.position_dim))
        
        #key [0, :min(config.seq_length, head_dim), :config.seq_length] = self.position_projection_tensor @ torch.eye(config.seq_length)
        key [0, 0, :config.seq_length] = torch.ones(config.seq_length)
        key = key.view((head_dim, config.position_dim))
        
        
        
        query = torch.zeros((1, head_dim, config.position_dim))
        
        query[0, :config.num_prefixes, config.seq_length:config.position_dim] = torch.eye(config.num_prefixes)
        #query[1, :min(config.seq_length, head_dim), :config.seq_length] = self.position_projection_tensor @ torch.eye(config.seq_length)
        
        query = query.view((head_dim, config.position_dim))
        
        value = torch.zeros((head_dim, config.position_dim))
        

        p_attn_init = torch.cat([query, key, value], axis=0)
        expand_ = torch.zeros((3, 1, num_attention_heads))
        #First position pattern in query only appears in the first tot_wt_bs_heads positions
        expand_[0, 0, :tot_wt_bs_heads] = 1.
        #Second position pattern in query only appears in the copy_heads attention
        #expand_[1, 0, 0, tot_wt_bs_heads: tot_wt_bs_heads+copy_heads] = 1.                      
        #First position pattern in key only appears in the first copy_heads 
        #expand_[2, 0, 0, tot_wt_bs_heads:tot_wt_bs_heads+copy_heads] = 1.
        #Second position pattern in key only appears in the second set of attention heads
        expand_[1, 0, tot_wt_heads: tot_wt_bs_heads] = 1.                                    
        #p_attn_bias[config.position_dim:2*config.position_dim] -= np.log(config.scale_embeddings)

        #p_value_init, p_value_bias = value, torch.zeros(config.hidden_size)

        #--------------------------------#--------------------------------#
        #After value projection, the embeddings contain [ \sum_{i \in H} ( \nabla y )_i x_i ]_H, 
        #where H denotes the set of weight row indices present in each blank.
        #The projection matrix after the attention module will compute \sum_i (\nabla y)_i x_i
        #--------------------------------#--------------------------------#
        #c_proj_init, c_proj_bias = torch.zeros((config.hidden_size, config.hidden_size)), torch.zeros(config.hidden_size)
        #for head in range(num_useful_heads):
        #    c_proj_init[head * din: (head+1) * din, head_dim*head: head_dim*head + din] = torch.eye(din)
        c_proj_init = torch.zeros((head_dim, num_attention_heads, num_attention_heads))
        
        #for head in range(tot_wt_heads):
        c_proj_init[:, :tot_wt_bs_heads, :tot_wt_bs_heads] = torch.eye(tot_wt_bs_heads)
            
        
        #copy nabla x to the top    
        #c_proj_init[:din, tot_wt_bs_heads*head_dim: tot_wt_bs_heads*head_dim + din ] = torch.eye(din)
        #c_proj_init[:, :copy_heads, tot_wt_bs_heads: tot_wt_bs_heads+copy_heads] = torch.eye(copy_heads)
        
        #c_proj_init = c_proj_init.view((config.hidden_size, config.hidden_size))    
        
        self.attnt_module.initialize_weights(q_attn_init_head=query_attn_head, \
                                             k_attn_init=key_attn, \
                                             k_attn_init_head=key_attn_head, \
                                             v_attn_init_head=value_attn_head, \
                                             p_attn_init=p_attn_init, \
                                             p_expand_init=expand_, \
                                             c_proj_init=c_proj_init )

    
    
        #Initialize Gates
        #Ignore the changes for the non-prefixes!
        #For prefixes, we have to add the change to the old value
        #w, u, v, w_bias, u_bias, v_bias
        w = torch.zeros((1, 2*config.hidden_size))
        u = torch.zeros((1, 2*config.hidden_size))
        v = torch.zeros((1, 2*config.position_dim))
        w_bias = torch.zeros(2)
        u_bias = torch.zeros(2)
        v_bias = torch.zeros(2)

        #gate on input is 1 (for prefixes) and 0 (for non-prefixes)
        v [ 0,  config.seq_length: config.position_dim ] =  config.gate_scale * torch.ones(config.num_prefixes)
        
        #gate on change is always 1
        v_bias [ 1 ] += config.gate_scale

        self.gates.initialize_weights (w, u, v, w_bias, u_bias, v_bias)
        
        #self.add_module('Lineardescent_attention', self.attnt_module)
        #self.add_module('Lineardescent_gates', self.gates)
        
    def forward(self, hidden_states, position_embeddings, attention_mask, icl_mask=None):
        #modify attention mask slightly to not allow weights look at each other
        
        modified_attention_mask=attention_mask.detach().clone() 
        modified_attention_mask[:, :, :self.config.num_prefixes, :self.config.num_prefixes] = 0.
        #modified_attention_mask = modified_attention_mask[:, :, 0]
        
        
        output = self.attnt_module.forward(hidden_states=hidden_states, \
                                           positions=position_embeddings, \
                                           attention_mask=modified_attention_mask\
                                          )[0]

        gate_output = self.gates.forward(hidden_states=hidden_states, \
                                         output_states=output, \
                                         position=position_embeddings\
                                        )
        return gate_output
        
#This module combines Linear Descent layer and Linear backward layer,
#to parallelize passes through the two (see figure 1 on how we call
#both linear descent and linear backward in parallel)
class Linear_Descent_Backward(nn.Module): 
    
    def __init__(self, config, din, dout, use_softmax, debug_zero=False, retain_nablay=False, projection_matrix=None, memory_index=-1, update_bias_only=False):
        super(Linear_Descent_Backward, self).__init__()
        
        
        self.config = config
        self.backward = LinearBackward(config, \
                                       din=din, \
                                       dout=dout,\
                                       use_softmax=use_softmax,\
                                       projection_matrix=projection_matrix,\
                                       retain_nablay=retain_nablay,
                                       memory_index=memory_index,
                                      )
        #self.add_module('backward_layer', self.backward)
        
        self.descent = LinearDescent(config, \
                                     din=din, \
                                     dout=dout, \
                                     use_softmax=use_softmax, \
                                     memory_index=memory_index,\
                                     debug_zero=debug_zero, \
                                     update_bias_only=update_bias_only,
                                    )
        #self.add_module('descent_layer', self.descent)
        
    def forward(self, hidden_states, position_embeddings, attention_mask, activation_memory=None, icl_mask=None):
        backward_out = self.backward(hidden_states, position_embeddings, attention_mask=None)
        descent_out  = self.descent (hidden_states, position_embeddings, attention_mask)
        
        return torch.cat( [ descent_out[:, :self.config.num_prefixes], backward_out[:, self.config.num_prefixes:] ], axis=1)
        