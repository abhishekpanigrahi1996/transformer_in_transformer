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
        
        
      
        
    def forward(self, hidden_states, position_embeddings):
        
        
        weights = self.gate ( self.w ).to(hidden_states.device)
        mean = torch.sum(hidden_states * weights, dim=-1, keepdim=True) / torch.sum(weights, dim=-1, keepdim=True)
        
        var = ( self.epsilon + torch.sum( (weights * (hidden_states - mean)) ** 2, dim=-1, keepdim=True) / torch.sum(weights, dim=-1, keepdim=True) ) ** 0.5
        
        normalized_states = (hidden_states - mean) / var
        normalized_states = weights * normalized_states + (1. - weights) * hidden_states
        
        gated_output = self.normalization_gates.forward (hidden_states, normalized_states, position_embeddings)
        
        output = self.linear.forward ( gated_output, position_embeddings )
        
        
        #store [(x-\mu)/\sigma, x] for memory in backward pass
        assert torch.sum( torch.absolute( output[:, self.config.num_prefixes:, self.memory_index+self.din:]) ).item() < 1e-10,\
               "Memory portion not empty!"
        output[:, self.config.num_prefixes:, self.memory_index+self.din: self.memory_index+2*self.din] += hidden_states[:, self.config.num_prefixes:, :self.din]
        
        return output
    

