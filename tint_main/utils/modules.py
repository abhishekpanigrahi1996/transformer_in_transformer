import torch
from torch import nn
from typing import Optional, Tuple, Union
from .activations import *


class OPTLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, attention_mask: torch.LongTensor, position_ids: None, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        attention_mask = attention_mask.long()

        # create positions depending on attention_mask
        if position_ids is None:
            positions = (torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask).long() - 1
        else:
            positions = position_ids
        # cut positions if `past_key_values_length` is > 0
        positions = positions[:, past_key_values_length:]

        return super().forward(positions + self.offset)

    
    
class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).
    Basically works like a linear layer but the weights are transposed.
    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x


class position_conv(nn.Module):
    """
    Expand positions across heads, 
    with a single linear weights
    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
        n_var (int) : Representing Query, Key, Value
    """
    def __init__(self, nf, nx, n_var=3):
        super().__init__()
        self.nx = nx
        self.nf = nf
        
        assert n_var % 3 == 0, \
               "'n_var' must be divisible by 3"
        
        self.n_var = n_var
        
        w = torch.zeros(n_var, 1, nf)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))
    
    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf * self.nx * 3,)
        x = x.view(-1, self.n_var, self.nx, 1) @ self.weight + self.bias        
        x = x.transpose(-1, -2)
        
        if self.n_var > 3:
            q, k, v = torch.split(x, split_size_or_sections=2, dim=-3)
            x = torch.cat([torch.sum(q, axis=-3), torch.sum(k, axis=-3), torch.sum(v, axis=-3)], dim=-2) 
                   
        x = x.contiguous().view(size_out)
          
        return x

class Conv2D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).
    Basically works like a linear layer but the weights are transposed.
    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx, transpose=False, use_einsum=False):
        super().__init__()
        self.nf = nf
        self.nx = nx
        self.use_einsum = use_einsum
        if not transpose: w = torch.stack( [torch.eye(nx) for _ in range(nf)] )
        else: w = torch.stack( [torch.eye(nf) for _ in range(nx)] )
            
        self.weight = nn.Parameter(w)
        
        if not transpose: self.bias = nn.Parameter(torch.zeros(nf, nx))
        else: self.bias = nn.Parameter(torch.zeros(nx, nf))
        
        self.transpose = transpose
        
    def forward(self, x):
        
        size_out = x.size()[:-1] + (self.nf * self.nx,)
        if not self.transpose:
            x = x.view(-1, self.nf, self.nx) 
            x = torch.stack([x [:, i] @ self.weight [i] for i in range(self.nf)], dim=1) + self.bias
            x = x.view(size_out)
        else:
            
            if self.use_einsum:
                x = torch.einsum( 'ijk,ljk->ilk', x.view(-1, self.nf, self.nx), self.weight.transpose(-3, -1)  ) + self.bias.transpose(-1, -2)
                x = x.contiguous().view(size_out) 
            else:
                x = (x.view(-1, self.nf, 1, self.nx).transpose(-3, -1) @ self.weight).squeeze(dim=-2) + self.bias
                x = x.transpose(-1, -2)
                x = x.contiguous().view(size_out)
            
            
        return x

    
class up_projection(nn.Module):
    def __init__(self, config, projection_matrix, signal_index=0, store_index=0):
        super().__init__()
        
        assert projection_matrix.shape[0] >= projection_matrix.shape[1], \
               "Call this module only for upward projection"
        
        channel_dim = projection_matrix.shape[1]
        num_channels = config.hidden_size // channel_dim
        
        self.channel_projection = Conv2D(nf=num_channels, nx=channel_dim, transpose=True)
        self.projection = Conv2D(nf=num_channels, nx=channel_dim, transpose=False)
        
        assert projection_matrix.shape[0] % channel_dim == 0, \
               "Larger dimension must be multiple of the smaller dimension"
                
        assert signal_index % channel_dim == 0, \
               "Signal index must be multiple of smaller dimension"
        
        assert store_index % channel_dim == 0, \
               "Store index must be multiple of smaller dimension"
        
        signal_head_start = signal_index // channel_dim
        store_head_start  = store_index // channel_dim
        num_useful_channels = projection_matrix.shape[0] // channel_dim
        
        
        assert num_useful_channels + store_head_start <= num_channels, \
               "Not sufficient to store the final result"
        
        
        c_proj_init = torch.zeros((num_channels, channel_dim, channel_dim), dtype=self.channel_projection.weight.dtype)
        for i in range (num_useful_channels):
            c_proj_init[store_head_start+i] = torch.tensor(projection_matrix[i*channel_dim: (i+1)*channel_dim, :], dtype=self.channel_projection.weight.dtype)

        with torch.no_grad():    
            
            self.channel_projection.weight.copy_(torch.zeros(channel_dim, num_channels, num_channels))
            self.channel_projection.weight[:, signal_head_start, store_head_start:store_head_start+num_useful_channels] = 1.
            
            self.projection.weight.copy_(c_proj_init.transpose(-1, -2))
    
    def forward(self, hidden_states):
        return self.projection( self.channel_projection(hidden_states) )
    
    
class down_projection(nn.Module):
    def __init__(self, config, projection_matrix, signal_index=0, store_index=0):
        super().__init__()
        
        assert projection_matrix.shape[1] >= projection_matrix.shape[0], \
               "Call this module only for downward projection"
        
        channel_dim = projection_matrix.shape[0]
        num_channels = config.hidden_size // channel_dim
        self.channel_projection = Conv2D(nf=num_channels, nx=channel_dim, transpose=True)
        self.projection = Conv2D(nf=num_channels, nx=channel_dim, transpose=False)
        
        assert projection_matrix.shape[1] % channel_dim == 0, \
               "Larger dimension must be multiple of the smaller dimension"
                
        assert signal_index % channel_dim == 0, \
               "Signal index must be multiple of smaller dimension"
        
        assert store_index % channel_dim == 0, \
               "Store index must be multiple of smaller dimension"
        
        signal_head_start = signal_index // channel_dim
        store_head_start  = store_index // channel_dim
        num_useful_channels = projection_matrix.shape[1] // channel_dim
        
        assert num_useful_channels + signal_head_start <= num_channels, \
               "Not sufficient to have whole signal"
        
        c_proj_init = torch.zeros((num_channels, channel_dim, channel_dim), dtype=self.channel_projection.weight.dtype)
        for i in range (num_useful_channels):
            c_proj_init[signal_head_start+i] = torch.tensor(projection_matrix[:, i*channel_dim: (i+1)*channel_dim], dtype=self.channel_projection.weight.dtype)

        with torch.no_grad():    
            self.projection.weight.copy_(c_proj_init.transpose(-1, -2))
            
            
            self.channel_projection.weight.copy_(torch.zeros(channel_dim, num_channels, num_channels))
            self.channel_projection.weight[:, signal_head_start: signal_head_start+num_useful_channels, store_head_start] = 1.
            
    def forward(self, hidden_states):
        return self.channel_projection( self.projection(hidden_states) )
        
        
class Gates (nn.Module):
    def __init__(self, config):
        super().__init__()
        self.position_dim = config.position_dim
        
        self.old_v = Conv1D(1, self.position_dim)
        

        self.new_v = Conv1D(1, self.position_dim)
        
        self.act = torch.nn.Tanh()
        
    def initialize_weights(self, w, u, v, w_bias, u_bias, v_bias):
        with torch.no_grad():
            self.old_v.weight.copy_(v[:, :self.position_dim].T)
            self.old_v.bias.copy_(v_bias[:1])


            self.new_v.weight.copy_(v[:, self.position_dim:].T)
            self.new_v.bias.copy_(v_bias[1:])


    
    def forward(self, hidden_states, output_states, position) -> torch.FloatTensor:
        
        old_output_gate = self.act( self.old_v(position) )
        new_output_gate = self.act( self.new_v(position) )
        
        combined_hidden_states = old_output_gate * hidden_states + new_output_gate * output_states
        return combined_hidden_states
    
class MLP(nn.Module):
    def __init__(self, \
                 intermediate_size, \
                 config, \
                 conv2d=False, \
                 conv_intermediate_features=None, \
                 conv_proj_features=None, \
                 transpose_intermediate=False,
                 transpose_proj=False,
                ):
        super().__init__()
        embed_dim = config.hidden_size
        self.conv2d = conv2d
        self.config = config
        if not conv2d:
            self.c_fc = Conv1D(intermediate_size, embed_dim)
            self.c_proj = Conv1D(embed_dim, intermediate_size)
        else:
            if conv_intermediate_features is None:
                conv_intermediate_features = config.num_attention_heads
                
            channel_dim = embed_dim // conv_intermediate_features
            self.c_fc = Conv2D(conv_intermediate_features, channel_dim, transpose=transpose_intermediate, use_einsum=self.config.use_einsum)
            
            
            if conv_proj_features is None:
                conv_proj_features = config.num_attention_heads
                
            proj_channel_dim = (conv_intermediate_features * channel_dim) // conv_proj_features
            self.c_proj = Conv2D(conv_proj_features, proj_channel_dim, transpose=transpose_proj, use_einsum=self.config.use_einsum)
            
            
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def initialize_weights(self, \
                           c_fc_init=None, \
                           c_fc_bias=None, \
                           c_proj_init=None, \
                           c_proj_bias=None, \
                          ):
        with torch.no_grad():
            if self.conv2d:
                if c_proj_init is not None:
                    self.c_proj.weight.copy_(c_proj_init.transpose(-1, -2))
                if c_proj_bias is not None:
                    self.c_proj.bias.copy_(c_proj_bias)
                
                if c_fc_init is not None:
                    self.c_fc.weight.copy_(c_fc_init.transpose(-1, -2))
                if c_fc_bias is not None:
                    self.c_fc.bias.copy_(c_fc_bias)
            else:    
                if c_proj_init is not None:
                    self.c_proj.weight.copy_(c_proj_init.T)
                if c_proj_bias is not None:
                    self.c_proj.bias.copy_(c_proj_bias)
                
                if c_fc_init is not None:
                    self.c_fc.weight.copy_(c_fc_init.T)
                if c_fc_bias is not None:
                    self.c_fc.bias.copy_(c_fc_bias)
        
        
    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        inter_hidden_states = self.c_fc(hidden_states)
        inter_hidden_states = self.act(inter_hidden_states)
        inter_hidden_states = self.c_proj(inter_hidden_states)
        inter_hidden_states = self.dropout(inter_hidden_states)
        
        return inter_hidden_states

    
#------------------------------------------------------------------------------------------------------#
#Modification: Attention is given by (Q(x_i + P_i))^\top K(x_j + P_j)
#where p_i and p_j are explicit position embeddings, P_k and P_q are additional Query and Key matrices.
#Value matrix is given by V x_j + P_v p_j
#P_v is an additional Value matrix
#------------------------------------------------------------------------------------------------------#
class Attention(nn.Module):
    def __init__(self, \
                 config,\
                 num_attention_heads=-1,\
                 is_cross_attention=False,\
                 layer_idx=None,\
                 peak_into_future=False,\
                 normalize=False,\
                 reorder_and_upcast_attn=False,\
                 scale_attn_by_inverse_layer_idx=False,\
                 attnt_back=False, \
                 query_conv_dim=None, \
                 key_conv_dim=None, \
                 value_conv_dim=None, \
                 proj_conv2d=False, \
                 proj_conv_dim=None, \
                 proj_transpose=False, \
                 total_position_embeddings=1, \
                ):
        super().__init__()

        max_positions = config.max_position_embeddings
        if not peak_into_future or attnt_back :
            self.register_buffer(
                "bias",
                torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
                    1, 1, max_positions, max_positions
                ),
            )
        else:
            self.register_buffer(
                "bias",
                torch.ones((max_positions, max_positions), dtype=torch.uint8).view(
                    1, 1, max_positions, max_positions
                ),
            )
            
        self.register_buffer("masked_bias", torch.tensor(-1e4))
        
        self.config = config

        self.embed_dim = config.hidden_size
        self.position_dim = config.position_dim
        
        if num_attention_heads == -1:
            self.num_heads = config.num_attention_heads
        else:
            self.num_heads = num_attention_heads
        #config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        #self.position_dim_combined = self.position_dim * self.num_heads 
        self.normalize = normalize
        self.peak_into_future = peak_into_future
        
        self.split_size = self.embed_dim
        #self.normalize = normalize
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        
        
        self.scale_attn_weights = config.scale_attn_weights and self.normalize
        self.initial_scale = config.initial_scale
        
        
        
        self.is_cross_attention = is_cross_attention
        self.attnt_back = attnt_back
        
        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = reorder_and_upcast_attn

        #--------------------------------------------------------------------------------------#
        #Total parameters involved: (embed_dim ** 2) * 5 + (embed_dim * position_dim) * 5
        #Typically to simulate GPT-2: 
        #Number of attention of heads = 32
        #GPT-2 dimension = 768
        #embed_dim = 768 * 32, position_embed = 512
        #Total parameters involved ~ 6B
        #---------------------------------------------------------------------------------------#
        
        if self.is_cross_attention:
            self.v_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
        else:
            if query_conv_dim is not None:
                assert self.embed_dim % query_conv_dim == 0,\
                       "Should get an integer number of channels"
                num_channels = self.embed_dim // query_conv_dim
                self.q_attn  = Conv2D(num_channels, query_conv_dim)
                self.q_attn_head = Conv2D(num_channels, query_conv_dim, transpose=True, use_einsum=self.config.use_einsum)
            else:
                self.q_attn = Conv2D(self.num_heads, self.head_dim)
                self.q_attn_head = Conv2D(self.num_heads, self.head_dim, transpose=True, use_einsum=self.config.use_einsum)
            
            self.q_attn_update = False
            self.q_attn_head_update = False 
            
            if key_conv_dim is not None:
                assert self.embed_dim % key_conv_dim == 0,\
                       "Should get an integer number of channels"
                num_channels = self.embed_dim // key_conv_dim
                self.k_attn  = Conv2D(num_channels, key_conv_dim)
                self.k_attn_head = Conv2D(num_channels, key_conv_dim, transpose=True, use_einsum=self.config.use_einsum)
            else:
                self.k_attn = Conv2D(self.num_heads, self.head_dim)
                self.k_attn_head = Conv2D(self.num_heads, self.head_dim, transpose=True, use_einsum=self.config.use_einsum)
            self.k_attn_update = False
            self.k_attn_head_update = False 
            
            if value_conv_dim is not None:
                assert self.embed_dim % value_conv_dim == 0,\
                       "Should get an integer number of channels"
                num_channels = self.embed_dim // value_conv_dim
                self.v_attn  = Conv2D(num_channels, value_conv_dim)
                self.v_attn_head = Conv2D(num_channels, value_conv_dim, transpose=True, use_einsum=self.config.use_einsum)
            else:
                self.v_attn = Conv2D(self.num_heads, self.head_dim)
                self.v_attn_head = Conv2D(self.num_heads, self.head_dim, transpose=True, use_einsum=self.config.use_einsum)
            self.v_attn_update = False
            self.v_attn_head_update = False 
            
        self.p_attn = Conv1D(total_position_embeddings * 3 * self.head_dim, self.position_dim) 
        self.p_expand = position_conv(self.num_heads, self.head_dim, n_var=3*total_position_embeddings) 
        self.total_position_embeddings = total_position_embeddings
        
        
        self.proj_conv2d = proj_conv2d
        if proj_conv2d:
            if proj_conv_dim is not None:
                num_channels = config.hidden_size // proj_conv_dim
                self.c_proj = Conv2D(nf=num_channels, nx=proj_conv_dim, transpose=proj_transpose, use_einsum=self.config.use_einsum)
            else:
                num_channels = config.num_attention_heads
                proj_conv_dim = config.hidden_size // num_channels
                self.c_proj = Conv2D(nf=num_channels, nx=proj_conv_dim, transpose=proj_transpose, use_einsum=self.config.use_einsum)
              
        else:
            self.c_proj = Conv1D(self.embed_dim, self.embed_dim)
            
        
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.pruned_heads = set()

    def initialize_weights(self, \
                           q_attn_init=None, \
                           k_attn_init=None, \
                           v_attn_init=None, \
                           q_attn_init_head=None, \
                           k_attn_init_head=None, \
                           v_attn_init_head=None, \
                           p_attn_init=None, \
                           c_proj_init=None, \
                           q_attn_bias=None, \
                           k_attn_bias=None, \
                           v_attn_bias=None, \
                           q_attn_bias_head=None, \
                           k_attn_bias_head=None, \
                           v_attn_bias_head=None, \
                           p_expand_init=None,\
                           p_expand_bias=None,\
                           p_attn_bias=None, \
                           c_proj_bias=None, \
                          ):
        with torch.no_grad():
            #if not self.proj_conv2d:
            if c_proj_init is not None:
                self.c_proj.weight.copy_(c_proj_init.transpose(-1, -2))
            if c_proj_bias is not None:
                self.c_proj.bias.copy_(c_proj_bias)
                
            if p_attn_init is not None:
                self.p_attn.weight.copy_(p_attn_init.transpose(-1, -2))
            if p_attn_bias is not None:
                self.p_attn.bias.copy_(p_attn_bias)
            
            if p_expand_init is not None:
                self.p_expand.weight.copy_(p_expand_init)
            if p_expand_bias is not None:
                self.p_expand.bias.copy_(p_expand_bias)
                
            
            if self.is_cross_attention:
                self.c_attn.weight.copy_(c_attn_init[:3 * self.embed_dim].T)
                self.c_attn.bias.copy_(c_attn_bias[:3 * self.embed_dim])
                self.q_attn.weight.copy_(c_attn_init[3 * self.embed_dim:].T)
                self.q_attn.bias.copy_(c_attn_bias[3 * self.embed_dim:])
            else:
                if q_attn_init is not None:
                    self.q_attn.weight.copy_(q_attn_init.transpose(-1, -2))
                    self.q_attn_update = True
                if k_attn_init is not None:
                    self.k_attn.weight.copy_(k_attn_init.transpose(-1, -2))
                    self.k_attn_update = True
                if v_attn_init is not None:
                    self.v_attn.weight.copy_(v_attn_init.transpose(-1, -2))
                    self.v_attn_update = True
                    
                if q_attn_bias is not None:
                    self.q_attn.bias.copy_(q_attn_bias)
                    self.q_attn_update = True
                if k_attn_bias is not None:
                    self.k_attn.bias.copy_(k_attn_bias)
                    self.k_attn_update = True
                if v_attn_bias is not None:
                    self.v_attn.bias.copy_(v_attn_bias)
                    self.v_attn_update = True
    
                if q_attn_init_head is not None:
                    self.q_attn_head.weight.copy_(q_attn_init_head.transpose(-1, -2))
                    self.q_attn_head_update = True
                if k_attn_init_head is not None:
                    self.k_attn_head.weight.copy_(k_attn_init_head.transpose(-1, -2))
                    self.k_attn_head_update = True
                if v_attn_init_head is not None:
                    self.v_attn_head.weight.copy_(v_attn_init_head.transpose(-1, -2))
                    self.v_attn_head_update = True
                    
                if q_attn_bias_head is not None:
                    self.q_attn_head.bias.copy_(q_attn_bias_head)
                    self.q_attn_head_update = True
                if k_attn_bias_head is not None:
                    self.k_attn_head.bias.copy_(k_attn_bias_head)
                    self.k_attn_head_update = True
                if v_attn_bias_head is not None:
                    self.v_attn_head.bias.copy_(v_attn_bias_head)
                    self.v_attn_head_update = True
    
    #In linear attention, I assume attention mask is of shape (b, 1, n)!
    def linear_attn(self, query, key, value, attention_mask=None, restrict_prefixes=False):
        
        attn_weights = torch.matmul(query[:, :, self.config.num_prefixes:], key[:, :, :self.config.num_prefixes].transpose(-1, -2))
        qvk = torch.matmul(attn_weights, value[:, :, :self.config.num_prefixes])
        
        qvk = torch.cat([ torch.zeros( (qvk.shape[0], qvk.shape[1], self.config.num_prefixes, qvk.shape[3]), device=qvk.device ),  qvk ], dim=2)
        
        return qvk, None
        

    def _attn(self, query, key, value, attention_mask=None, normalization_mask=None, head_mask=None, restrict_prefixes=False):
        
        if restrict_prefixes and not self.normalize:
            attn_output, attn_weights = self.linear_attn(query, key, value, attention_mask=attention_mask)
            return (attn_output, attn_weights)
        
        #Modify the attention weights to involve a component from the position embeddings
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
         
        
        if self.scale_attn_weights:
            attn_weights = self.initial_scale * attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].to(torch.bool)
            if self.normalize:
                mask_value = torch.finfo(attn_weights.dtype).min
            else:
                mask_value = 0.
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

            
        if normalization_mask is not None:
            if self.normalize: attn_weights = attn_weights + normalization_mask.to(attn_weights.device)
            
        
        
        if self.normalize:
               
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        
        if self.attnt_back and self.peak_into_future:
            attn_weights = torch.swapaxes(attn_weights, axis0=-1, axis1=-2)
        
        if attention_mask is not None:
            if not(self.normalize and not self.attnt_back): attn_weights = attn_weights * attention_mask
            
        
        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        
        attn_output = torch.matmul(attn_weights, value)
        
       
        return attn_output, attn_weights

    def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None, head_mask=None):
        # Use `torch.baddbmm` (a bit more efficient w/ alpha param for scaling -- from Megatron-LM)
        bsz, num_heads, q_seq_len, dk = query.size()
        _, _, k_seq_len, _ = key.size()

        # Preallocate attn_weights for `baddbmm`
        attn_weights = torch.empty(bsz * num_heads, q_seq_len, k_seq_len, dtype=torch.float32, device=query.device)

        # Compute Scale Factor
        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value.size(-1)) ** 0.5

        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= float(self.layer_idx + 1)

        # Upcast (turn off autocast) and reorder (Scale K by 1 / root(dk))
        with autocast(enabled=False):
            q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
            attn_weights = torch.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
            attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        if self.normalize:    
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op if otherwise
        if attn_weights.dtype != torch.float32:
            raise RuntimeError("Error with upcasting, attn_weights does not have dtype torch.float32")
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        positions: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        normalization_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        restrict_prefixes=False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            if self.q_attn_head_update and self.q_attn_update:
                query = self.q_attn( self.q_attn_head(hidden_states) )
            elif self.q_attn_update and not self.q_attn_head_update: 
                query = self.q_attn( hidden_states) 
            elif not self.q_attn_update and self.q_attn_head_update: 
                query = self.q_attn_head(hidden_states)
            else:
                query = hidden_states
                
            if self.k_attn_head_update and self.k_attn_update:
                key = self.k_attn( self.k_attn_head(hidden_states) )
            elif self.k_attn_update and not self.k_attn_head_update: 
                key = self.k_attn( hidden_states) 
            elif not self.k_attn_update and self.k_attn_head_update: 
                key = self.k_attn_head(hidden_states)
            else:
                key = hidden_states
            
            if self.v_attn_head_update and self.v_attn_update:
                value = self.v_attn( self.v_attn_head(hidden_states) )
            elif self.v_attn_update and not self.v_attn_head_update: 
                value = self.v_attn( hidden_states) 
            elif not self.v_attn_update and self.v_attn_head_update: 
                value = self.v_attn_head(hidden_states)
            else:
                value = hidden_states
            
                       
        
        p_attn_out = self.p_attn(positions)
        query_position, key_position, value_position = self.p_expand(p_attn_out).split(self.split_size, dim=2) 
        
        query = self._split_heads(query, self.num_heads, self.head_dim) \
                + self._split_heads(query_position, self.num_heads, self.head_dim)
             
        key = self._split_heads(key, self.num_heads, self.head_dim) \
                + self._split_heads(key_position, self.num_heads, self.head_dim)
               
        value = self._split_heads(value, self.num_heads, self.head_dim) \
                + self._split_heads(value_position, self.num_heads, self.head_dim)
        
        
        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None
                
        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask, restrict_prefixes)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, normalization_mask, head_mask, restrict_prefixes)
        
        
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        
        attn_output = self.c_proj(attn_output)
        
        
        attn_output = self.resid_dropout(attn_output)
    
        
        outputs = (attn_output,)
        if use_cache is True:
            outputs += (present,)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs 