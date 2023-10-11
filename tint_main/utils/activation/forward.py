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
from ..activations import *




class ActivationForward (nn.Module):
    def __init__ (self, config, din, projection_matrix=None, memory_index=-1):
        super(ActivationForward, self).__init__()
        
        self.din=din
        self.config=config
        self.projection_matrix = projection_matrix
        self.memory_index = memory_index
        
        
        if projection_matrix is not None:
            self.dout = projection_matrix.shape[0]
        else:
            self.dout = din
        
            
        assert memory_index == -1 or memory_index >= self.dout,\
              "Memory interacts with final signal"
        
        assert memory_index == -1 or memory_index <= config.hidden_size - self.din, \
               "not enough space to store memory"
        
        if projection_matrix is not None:
            head_dim = self.dout
            num_channels = config.hidden_size // head_dim
        else:
            num_channels = config.num_attention_heads
            head_dim = config.hidden_size // num_channels
        
        self.mlp_module = MLP (config.hidden_size, \
                               config, \
                               conv2d=True, \
                               transpose_intermediate=True, \
                               transpose_proj=False, \
                               conv_proj_features=num_channels, \
                              )
        
        self.mlp_gates = Gates (config)
        self.projection_ = None
        
                            
        if projection_matrix is not None:
                
            assert projection_matrix.shape[1] == din,\
                   "Projection matrix must have 'din' in second coordinate"
            assert projection_matrix.shape[1] >= head_dim, \
                   "Currently, this projection only works when we project down to a lower dimension"
            assert projection_matrix.shape[1] % head_dim == 0, \
                   "Perfect division into channels"
            
            c_proj_init = torch.zeros((num_channels, head_dim, head_dim), dtype=self.mlp_module.c_proj.weight.dtype)
            num_useful_channels = projection_matrix.shape[1] // head_dim
            for i in range (num_useful_channels):
                c_proj_init[i] = torch.tensor(projection_matrix[:, i*head_dim: (i+1)*head_dim], dtype=self.mlp_module.c_proj.weight.dtype)
            self.mlp_module.initialize_weights(c_proj_init=c_proj_init)    
            
            self.projection_ = Conv2D( nf=num_channels, nx=head_dim, transpose=True, use_einsum=self.config.use_einsum )
            with torch.no_grad():    
                self.projection_.weight.copy_(torch.zeros(head_dim, num_channels, num_channels))
                self.projection_.weight[:, :num_useful_channels, 0] = 1.
            
        else:
            c_proj_init = torch.zeros((num_channels, head_dim, head_dim), dtype=self.mlp_module.c_proj.weight.dtype)
            
            if self.memory_index != -1:
                assert memory_index % head_dim == 0, \
                       "Memory should be divisible by the number of channels!"

                mem_head_start = memory_index // head_dim

                c_proj_init[:mem_head_start] = torch.eye(head_dim)
                self.mlp_module.initialize_weights(c_proj_init=c_proj_init)  
            else:
                c_proj_init[:] = torch.eye(head_dim)
                self.mlp_module.initialize_weights(c_proj_init=c_proj_init)  
                
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
        v [0, config.seq_length: config.position_dim] = config.gate_scale * torch.ones(config.position_dim-config.seq_length)


        #Change Gate is 0 on prefixes and 1 for non-prefixes
        v [0, config.position_dim+config.seq_length: 2*config.position_dim] = -config.gate_scale * torch.ones(config.position_dim-config.seq_length)
        v_bias [1] += config.gate_scale

        self.mlp_gates.initialize_weights (w, u, v, w_bias, u_bias, v_bias)

        
    def forward(self, hidden_states, position_embeddings):
        mlp_out = self.mlp_module.forward(hidden_states)
        if self.projection_ is not None:
            mlp_out = self.projection_(mlp_out)
        
        if self.memory_index != -1:
            assert torch.sum(torch.absolute(mlp_out[:, self.config.num_prefixes:, self.memory_index:])).item() < 1e-10,\
                   "Memory portion not empty!"

            mlp_out[:, self.config.num_prefixes:, self.memory_index: self.memory_index+self.din] += hidden_states[:, self.config.num_prefixes:, :self.din]

        gate_out = self.mlp_gates.forward(hidden_states, mlp_out, position_embeddings)
           
        return gate_out
    

    
