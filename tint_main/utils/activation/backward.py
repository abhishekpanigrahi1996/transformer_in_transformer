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



class ActivationBackward (nn.Module):
    def __init__ (self, config, din, input_projection=None, projection_matrix=None, memory_index=-1, retain_og_act=False):
        super(ActivationBackward, self).__init__()

        
        
        assert memory_index == -1 or memory_index >= din, \
            "memory crosses current signal"
    

        
        self.epsilon = config.epsilon
        self.memory_index = memory_index
        self.config = config
        
        head_dim  = config.hidden_size // config.num_attention_heads
        self.c_fc = Conv2D(config.num_attention_heads, head_dim, transpose=True, use_einsum=self.config.use_einsum)
        self.proj_fc = Conv2D(config.num_attention_heads, head_dim, transpose=True, use_einsum=self.config.use_einsum)
        
        self.config = config
        self.din = din
        self.act = ACT2FN[config.activation_function]
        
        
        c_fc_init = torch.zeros((head_dim, config.num_attention_heads, config.num_attention_heads))
        c_proj_init = torch.zeros((head_dim, config.num_attention_heads, config.num_attention_heads))
        
    
        
        assert din % head_dim == 0, \
            " 'din' should be a multiple of head_dim! "
        
        num_partitions = din // head_dim
        
        
        
        assert self.memory_index % head_dim == 0, \
            "Memory should start at a multiple of head_dim!"
        
        mem_head_start = self.memory_index // head_dim
        
        
        start_shift = 0
        c_fc_init[:, start_shift: start_shift + num_partitions, start_shift: start_shift + num_partitions] = 1. / config.scale_embeddings * torch.eye(num_partitions)
        c_fc_init[:, start_shift: start_shift + num_partitions, mem_head_start: mem_head_start + num_partitions] =  torch.eye(num_partitions)
        
        #pass x as well
        c_fc_init[:, mem_head_start: mem_head_start + num_partitions, mem_head_start: mem_head_start + num_partitions] =  torch.eye(num_partitions)
        
        
        #Compute GeLU(x + 1/N \nabla y) - GeLU(x)

        c_proj_init[:, start_shift: start_shift + num_partitions, start_shift: start_shift + num_partitions] = config.scale_embeddings * torch.eye(num_partitions)
        c_proj_init[:, start_shift: start_shift + num_partitions, mem_head_start: mem_head_start + num_partitions] = -config.scale_embeddings  * torch.eye(num_partitions)
        
        #retain Act (x) for future purposes?
        if retain_og_act:
            c_proj_init[:, mem_head_start: mem_head_start + num_partitions, mem_head_start: mem_head_start + num_partitions] = torch.eye(num_partitions)
        
        
        with torch.no_grad():
            self.c_fc.weight.copy_(torch.swapaxes(c_fc_init, axis0=-1, axis1=-2))
            self.proj_fc.weight.copy_(torch.swapaxes(c_proj_init, axis0=-1, axis1=-2))

        
        
    def forward(self, hidden_states, position_embeddings, attention_mask=None, activation_memory=None, icl_mask=None):
        output = self.proj_fc ( self.act( self.c_fc(hidden_states) ) )
        return output     
    
