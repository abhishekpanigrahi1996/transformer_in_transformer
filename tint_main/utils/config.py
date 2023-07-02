import numpy as np


class Construct_config:
    def __init__(self, model_config, construction_args):
        self.seq_length = construction_args.seq_length
        self.position_dim = construction_args.seq_length + construction_args.num_prefixes
        self.num_prefixes = construction_args.num_prefixes
        self.num_attention_heads = construction_args.num_attention_heads
        self.scale_embeddings = construction_args.scale_embeddings
        self.inner_lr = construction_args.inner_lr
        self.gate_scale = construction_args.gate_scale
        self.hidden_size = construction_args.hidden_size 
        self.max_position_embeddings = construction_args.max_position_embeddings
        #self.scale_attn_weights=scale_attn_weights
        #self.scale_attn_by_inverse_layer_idx=False
        #self.reorder_and_upcast_attn=False
        self.embd_pdrop=construction_args.embd_pdrop
        self.attn_pdrop=construction_args.attn_pdrop
        self.resid_pdrop=construction_args.resid_pdrop
        self.activation_function=construction_args.activation_function
        self.epsilon=construction_args.epsilon
        self.scale_attn_weights=construction_args.scale_attn_weights
        
        self.initial_scale=np.sqrt( (construction_args.hidden_size / model_config.hidden_size) * (model_config.num_attention_heads / construction_args.num_attention_heads) )
        
        
        self.n_simulation_layers=construction_args.n_simulation_layers
        self.n_forward_backward=construction_args.n_forward_backward
        self.n_debug_layers=construction_args.n_debug_layers
        self.projection_paths=construction_args.projection_paths
        self.backprop_through_attention=construction_args.backprop_through_attention
        self.restrict_prefixes=construction_args.restrict_prefixes
        self.use_einsum=construction_args.use_einsum
        self.use_prediction_loss=construction_args.use_prediction_loss
        self.use_quad=construction_args.use_quad
        #self.n_gpus = construction_args.n_gpus
        self.n_layers_pergpu = construction_args.n_layers_pergpu
        self.device = construction_args.device

        
        self.reuse_forward_blocks = construction_args.reuse_forward_blocks
        self.reuse_backward_blocks = construction_args.reuse_backward_blocks
        
        self.ln_update_bias_only = construction_args.ln_update_bias_only
    