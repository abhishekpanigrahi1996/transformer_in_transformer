

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



#------------------------------------------------------------------#
#LinearForward module computes Wx_i at every position i
#Important arguments: input dimension (din), output dimension (dout)   
#output: TinT LinearForward module, primarily containing a linear self_attention layer. (See figure 2 in the paper)
#We assume that the rows of W have been stacked onto the prefix tokens,
#before calling this module.

#We also allow projection matrix on the linear operations, however we donot 
#use them in the current version.
#------------------------------------------------------------------#


#All arguments
#self, \
#config, \                    #TinT config file
#din, \                       #input dimension
#dout, \                      #output dimension
#use_softmax=False, \         #use_softmax=False, implying we use linear attention head 
#projection_matrix=None, \    #projection_matrix to project the linear operation (not used in the current code)
#shift_top=0, \               #shifts the output to a shifted index if necessary
#memory_index=-1, \           #memory_index to store activations for the backward and descent pass!

class LinearForward(nn.Module):

    def __init__(self, \
                 config, \
                 din, \
                 dout, \
                 use_softmax=False, \
                 projection_matrix=None, \
                 shift_top=0, \
                 memory_index=-1, \
                ):
        super(LinearForward, self).__init__()
        
        self.attnt_module = None     #initialized later
        #We use gates to differentiate the operations on prefix embeddings and non-prefix embeddings.
        self.gates = Gates (config)
        self.din = din
        self.config = config
        self.projection_matrix = projection_matrix
        self.memory_index = memory_index
        
        #initialized later
        self.permutation_conv = None
        self.bias_add_conv = None
        self.projection_layer = None
        self.proj_conv2d = (config.hidden_size % dout == 0) 
        
        assert use_softmax == False, \
            "Currently only works without softmax!"
        
        head_dim = config.num_prefixes
        assert config.hidden_size % head_dim == 0,\
            "Dimension should perfectly distribute over the prefixes" 
        
        num_attention_heads = config.hidden_size // head_dim
        

        
        assert config.hidden_size >= din * (dout // config.num_prefixes), \
            "Total embedding size must be greater than the dimension necessary to store weights in the prefixes"
        
        assert dout % config.num_prefixes == 0, \
            "I assume uniform distribution of the weights over the prefixes"
        
        assert din % head_dim == 0,\
            "Currently this is a bug! I assume that the input dimension is easily divisible across the heads we want to distribute to"   
        assert dout % head_dim == 0,\
            "Currently this is a bug! I assume that the output dimension is easily divisible across the heads we want to distribute to"
        
        num_wts_per_blank = dout // config.num_prefixes
        #initialize attention module
        self.attnt_module = Attention (config, \
                                       num_attention_heads=num_attention_heads, \
                                       normalize=use_softmax, \
                                       proj_conv_dim=config.num_prefixes, \
                                       proj_transpose=True, \
                                       proj_conv2d=self.proj_conv2d\
                                      )            
        
        attnt_head_per_wt = din // head_dim 
        useful_attnt_heads=attnt_head_per_wt * (dout // config.num_prefixes)
        extra_heads=dout // head_dim
        
        #print (config.num_attention_heads,  extra_heads, useful_attnt_heads, attnt_head_per_wt)
        assert num_attention_heads >=  extra_heads + useful_attnt_heads, \
            "Number of attention heads should be atleast the number of weights + biases present in each blank"
        
        assert config.num_prefixes <= head_dim ,\
            "Currently I assume the head dimension is atleast the number of prefixes in the original model"
        
        #--------------------------------#--------------------------------#
        #For all Attention heads on the embeddings
        #Query repeats the first din dimensions dout times, so that we can split them among the different attention heads
        #Key is Identity
        #Value is all zeros
        #Key and Query of embeddings ignore the position dependence.
        
        #Final attention head simply copies the bias present in the first blank
        #--------------------------------#--------------------------------#
        key_attn = torch.zeros((num_attention_heads, head_dim, head_dim))
        
        din_partition = din // attnt_head_per_wt
        
        key_attn[:useful_attnt_heads] = torch.eye(head_dim)
        
        
        
        query_attn_head = torch.zeros((head_dim, num_attention_heads, num_attention_heads))        
        for i in range (dout // config.num_prefixes):
            query_attn_head[:, i*attnt_head_per_wt: (i+1)*attnt_head_per_wt, :attnt_head_per_wt] = torch.eye(attnt_head_per_wt)
        
        
        value_attn = torch.zeros((num_attention_heads, head_dim, head_dim))
        value_attn[useful_attnt_heads: useful_attnt_heads+extra_heads] = torch.eye(head_dim)
        
        
        #--------------------------------#--------------------------------#
        #For all Attention heads on the positions
        #Query is Identity (on the component corresponding to one-hot encodings 
        #of the input sequence to the smaller model)
        #Key is Identity (on the component corresponding to one-hot encodings 
        #of the input sequence to the smaller model) + all-ones on the the blank identifiers
        #Value moves the blank identifiers to the fore-front
        #Key and Query ignore dependence on the signal.
        
        #Final attention head simply copies the bias present in the first blank
        #--------------------------------#--------------------------------#
        
        
        query = torch.zeros((head_dim, config.position_dim))
        query[0, :config.seq_length] = 1.
        
        key = torch.zeros((head_dim, config.position_dim))
        key[:config.num_prefixes, config.seq_length: config.position_dim] = torch.eye(config.num_prefixes)
        
        value = torch.zeros((head_dim, config.position_dim))
        value[:config.num_prefixes, config.seq_length: config.position_dim] = torch.eye(config.num_prefixes)
        
        expand_ = torch.zeros((3, 1, num_attention_heads))
        #for query, we use position embedding only at heads useful_attnt_heads: useful_attnt_heads+extra_heads
        expand_[0, 0, useful_attnt_heads: useful_attnt_heads+extra_heads] = 1.
        #for key, we use position embedding only at heads useful_attnt_heads: useful_attnt_heads+extra_heads
        expand_[1, 0, useful_attnt_heads: useful_attnt_heads+extra_heads] = 1.
        #for value, we use position embedding only at heads :useful_attnt_heads
        expand_[2, 0, :useful_attnt_heads] = 1.
        

        p_attn_init = torch.cat([query, key, value], axis=0)

        #--------------------------------#--------------------------------#
        #The projection matrix after the attention module reorders such that 
        #<w_1, x>, <w_2, x>, ..., <w_dout, x> appear in a sequential order.
        #--------------------------------#--------------------------------#
        
        if not self.proj_conv2d:
            c_proj_init = torch.zeros(( config.hidden_size, config.hidden_size ))

            for i in range(dout):
                num_useful_heads = dout // config.num_prefixes
                desd_loc = head_dim * attnt_head_per_wt * (i % num_useful_heads) + i // num_useful_heads

                for sub_head in range(attnt_head_per_wt):
                    if use_softmax:
                        c_proj_init[shift_top+i, desd_loc + sub_head * head_dim] = config.scale_embeddings
                    else:
                        c_proj_init[shift_top+i, desd_loc + sub_head * head_dim] = 1.

    
            for i in range(dout):
                desd_loc = head_dim * num_useful_heads * attnt_head_per_wt + i
                c_proj_init[shift_top+i, desd_loc] = 1.

            if projection_matrix is not None:
                projection_tensor = torch.zeros((config.hidden_size, config.hidden_size))
                projection_tensor[shift_top:shift_top+projection_matrix.shape[0], :projection_matrix.shape[1]] = torch.tensor(projection_matrix, dtype=c_proj_init.dtype)
                c_proj_init = projection_tensor @ c_proj_init 
        
        else:
            assert head_dim % config.num_prefixes == 0, \
                "This is a bug! For simpler operation, I assume head_dim to be divisible by config.num_prefixes"
            
            num_partitions_head = head_dim // config.num_prefixes
            num_channels = config.hidden_size // config.num_prefixes
            num_wt_channels = dout // config.num_prefixes
            c_proj_init = torch.zeros(( config.num_prefixes, num_channels, num_channels ))
            for i in range(num_wt_channels):
                for j in range(attnt_head_per_wt * num_partitions_head):
                    c_proj_init[:, i, i * attnt_head_per_wt * num_partitions_head + j ] = 1.
            
                c_proj_init[:, num_wt_channels + i, num_wt_channels * attnt_head_per_wt * num_partitions_head + i] = 1.
            
            
            num_abs_heads = config.hidden_size // dout
            shift_top_head = shift_top // dout
            
            #permute the final computation
            self.permutation_conv = Conv2D(nf=num_abs_heads, nx=dout, transpose=False)
            permutation_wt = torch.zeros((num_abs_heads, dout, dout))
            for i in range(dout):
                desd_loc = ( i % num_wt_channels ) * config.num_prefixes + i // num_wt_channels
                permutation_wt[0, i, desd_loc] = 1.
            permutation_wt[1] = torch.eye(dout)
            with torch.no_grad():
                self.permutation_conv.weight.copy_(permutation_wt.transpose(-1, -2))


            #add bias to the <w,x>
            self.bias_add_conv = Conv2D(nf=num_abs_heads, nx=dout, transpose=True, use_einsum=self.config.use_einsum)
            bias_add_wt = torch.zeros((dout, num_abs_heads, num_abs_heads))
            bias_add_wt[:, shift_top_head, 0: 2 ] = 1.
            with torch.no_grad():
                self.bias_add_conv.weight.copy_(bias_add_wt.transpose(-1, -2))
        
            if projection_matrix is not None:
                start_index=shift_top
                if projection_matrix.shape[0] >= projection_matrix.shape[1]:
                    self.projection_layer = up_projection (config, projection_matrix, signal_index=start_index, store_index=start_index)
                else:
                    self.projection_layer = down_projection (config, projection_matrix, signal_index=start_index, store_index=start_index)


        self.attnt_module.initialize_weights(q_attn_init_head=query_attn_head, \
                                             k_attn_init=key_attn, \
                                             v_attn_init=value_attn,
                                             p_attn_init=p_attn_init, \
                                             p_expand_init=expand_,\
                                             c_proj_init=c_proj_init, \
                                            )

        
    
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

        self.gates.initialize_weights (w, u, v, w_bias, u_bias, v_bias)
        
        
    def forward(self, hidden_states, position_embeddings):
        output = self.attnt_module.forward(hidden_states=hidden_states, positions=position_embeddings, restrict_prefixes=self.config.restrict_prefixes)[0]
        
        if self.permutation_conv is not None:
            output =  self.permutation_conv(output) 
            output =  self.bias_add_conv( output )
        if self.projection_layer is not None:
            output = self.projection_layer(output)
        #store the input in memory for backward pass later on
        if self.memory_index != -1:
            assert torch.sum(output[:, self.config.num_prefixes:, self.memory_index: ]).item() < 1e-10,\
                   "Memory portion not empty!"
            
            output[:, self.config.num_prefixes:, self.memory_index: self.memory_index + self.din ] += hidden_states[:, self.config.num_prefixes:, :self.din]
        return self.gates.forward(hidden_states=hidden_states, \
                                  output_states=output, \
                                  position=position_embeddings\
                                 )
 