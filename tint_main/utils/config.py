import numpy as np


class Construct_config:
    def __init__(self, model_config, construction_args):
        #Sequence length of auxiliary model
        self.seq_length = construction_args.seq_length
        #Sequence length of auxiliary model + Number of prefix tokens
        self.position_dim = construction_args.seq_length + construction_args.num_prefixes
        #Number of prefix tokens
        self.num_prefixes = construction_args.num_prefixes
        #Number of attention heads in TinT
        self.num_attention_heads = construction_args.num_attention_heads
        #Scale_embeddings (for softmax attention, unnecessary argument for now)
        self.scale_embeddings = construction_args.scale_embeddings
        #inner_lr for dynamic evaluation
        self.inner_lr = construction_args.inner_lr
        #Scaling for sigmoid gates to behave as hard gates
        self.gate_scale = construction_args.gate_scale
        #Embedding dimension of TinT
        self.hidden_size = construction_args.hidden_size 
        #A max bound on the final sequence length, for initialization purposes
        self.max_position_embeddings = construction_args.max_position_embeddings
        
        #Following three arguments will be useful only when we pre-train
        #Dropout rate of embedding
        self.embd_pdrop=construction_args.embd_pdrop
        #Dropout rate of attention
        self.attn_pdrop=construction_args.attn_pdrop
        #Dropout rate of residual connection
        self.resid_pdrop=construction_args.resid_pdrop
        
        #Auxiliary's activation function
        self.activation_function=construction_args.activation_function
        #Auxiliary's error term for layernorm
        self.epsilon=construction_args.epsilon
        #Whether Attention score scales before softmax, as determined by Auxiliary model
        self.scale_attn_weights=construction_args.scale_attn_weights
        #Attention score appropriate scaling, as determined by Auxiliary model
        self.initial_scale=np.sqrt( (construction_args.hidden_size / model_config.hidden_size) * (model_config.num_attention_heads / construction_args.num_attention_heads) )
        
        #Number of layers to involve in SGD
        self.n_simulation_layers=construction_args.n_simulation_layers
        #Number of SGD steps
        self.n_forward_backward=construction_args.n_forward_backward
        #Unnecessary argument, was used for debugging
        self.n_debug_layers=construction_args.n_debug_layers
        #Unnecessary argument, was used for projection
        self.projection_paths=construction_args.projection_paths
        #We never backprop through attention, hence unnecessary argument for now
        self.backprop_through_attention=construction_args.backprop_through_attention
        #Whether to restrict attention between prefix and non-prefix tokens for linear operations
        self.restrict_prefixes=construction_args.restrict_prefixes
        #Whether to use einsum, which speeds up inference
        self.use_einsum=construction_args.use_einsum
        #Whether to use classification loss with softmax, for computing gradients
        self.use_prediction_loss=construction_args.use_prediction_loss
        #Whether to use quad loss from Saunshi et al., for computing gradients
        self.use_quad=construction_args.use_quad
        #self.n_gpus = construction_args.n_gpus
        #For multiple gpus, we can further partition the model across multiple gpus
        self.n_layers_pergpu = construction_args.n_layers_pergpu
        #'cuda'/'cpu'
        self.device = construction_args.device

        #Whether to reuse forward blocks, when we do multiple forward passes
        self.reuse_forward_blocks = construction_args.reuse_forward_blocks
        #Whether to reuse backward blocks, when we do multiple forward passes
        self.reuse_backward_blocks = construction_args.reuse_backward_blocks
        
        #Whether to only update biases in layernorm.
        self.ln_update_bias_only = construction_args.ln_update_bias_only
    