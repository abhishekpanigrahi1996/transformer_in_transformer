from transformers import HfArgumentParser
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Union, List

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    
        
    construct_save_model:  Optional[bool] = field(
        default=False, metadata={"help": "Whether to save constructed model"}
    ) 
    
    
    construct_load_model:  Optional[bool] = field(
        default=False, metadata={"help": "Whether to load constructed model from a specific path"}
    )
        
    construct_model_path:   Optional[str] = field(
        default="", metadata={"help": "Path to save the model"}
    )
        
      
        
      
          
  
@dataclass    
class ConstructionArguments:
    """
    Arguments pertaining
    """
    
    device: Optional[str] = field(
        default="cuda", metadata={"help": "cuda/cpu"}
    )
        
    seq_length: Optional[int] = field(
        default=1024, metadata={"help": "Sequence length for the smaller model"}
    )
        
    #position_dim: Optional[int] = field(
    #    default=1024 + 256, metadata={"help": ""}
    #)
        
    num_prefixes: Optional[int] = field(
        default=256, metadata={"help": "Number of prefixes to encode at the start of the sequence"}
    )
        
    num_attention_heads: Optional[int] = field(
        default=12, metadata={"help": "Number of attention heads"}
    )
    
    scale_embeddings: Optional[float] = field(
        default=1000., metadata={"help": "Scale factor to minimize error introduced by multiplications with GeLU"}
    )
        
    inner_lr: Optional[float] = field(
        default=0.000001, metadata={"help": "Learning rate to simulate SGD inside the model"}
    )
        
    gate_scale: Optional[float] = field(
        default=10., metadata={"help": "Initial scale inside gate weights to simulate 0/1 switch"}
    )
    hidden_size: Optional[int] = field(
        default=4, metadata={"help": "Multiple of Embedding size (of the smaller model) for the construction"}
    )
        
    max_position_embeddings: Optional[int] = field(
        default=2048, metadata={"help": "Max sequence length"}
    )
        
    embd_pdrop: Optional[float] = field(
        default=0.1, metadata={"help": "Dropout on attention weights"}
    )
    
    attn_pdrop: Optional[float] = field(
        default=0.1, metadata={"help": "Dropout on attention weights"}
    )
    resid_pdrop: Optional[float] = field(
        default=0.1, metadata={"help": "Dropout on residual connections"}
    )
        
    activation_function: Optional[str] = field(
        default="gelu", metadata={"help": "Activation: gelu/relu"}
    )
    epsilon: Optional[float] = field(
        default=1e-05, metadata={"help": "Epsilon for layernorm computation"}
    )
        
    scale_attn_weights: Optional[bool] = field(
        default=True, metadata={"help": "Whether to scale attention weights"}
    ) 
    
    n_simulation_layers: Optional[int] = field(
        default=-1, metadata={"help": "Number of layers to simulate forward-backward passes"}
    )
     
    n_forward_backward: Optional[int] = field(
        default=1, metadata={"help": "Number of forward-backward passes"}
    )
        
    n_debug_layers: Optional[int] = field(
        default=-1, metadata={"help": "Number of layers of smaller model that we use (for debugging purposes)"}
    )  
    
    projection_paths: Optional[str] = field(
        default="./projections", metadata={"help": "Path to all the projection matrices"}
    )  
    
    backprop_through_attention:  Optional[bool] = field(
        default=True, metadata={"help": "Whether to look ahead during backprop through attention"}
    ) 
    
    restrict_prefixes: Optional[bool] = field(
        default=False, metadata={"help": "Whether to restrict attention to blank tokens only in linear forward/backward"}
    ) 
    
    use_einsum: Optional[bool] = field(
        default=True, metadata={"help": "Whether to use einsum in dimension wise linear convolutions"}
    ) 
    
    use_prediction_loss: Optional[bool] = field(
        default=True, metadata={"help": "If true, gradient w.r.t. loss is E(p-q) with p being the softmax prediction, if false (and use_quad is false), gradient w.r.t. loss is E(1 - q)"}
    )
        
    use_quad: Optional[bool] = field(
        default=False, metadata={"help": "If true, we use the quad loss to compute the gradient from Saunshi & Malladi et al. (2020)"}
    )
        
    
    n_layers_pergpu: Optional[int] = field(
        default=100000, metadata={"help": "Number of layers simulated per gpu"}
    )
        
    reuse_forward_blocks: Optional[bool] = field(
        default=False, metadata={"help": "Whether to re-use forward blocks"}
    )
        
    reuse_backward_blocks: Optional[bool] = field(
        default=False, metadata={"help": "Whether to re-use backward blocks"}
    )
    
    ln_update_bias_only: Optional[bool] = field(
        default=True, metadata={"help": "Whether to update only biases in layernorm modules; current weight update of layernorm modules is very noisy, hence it's best to avoid!"}
    )
    
    
@dataclass
class DynamicArguments:
    
    data_cache_dir: Optional[str] = field(
        default='/scratch/gpfs/ap34/icl-as-ft/Dynamic_initialization/data',
        metadata={
            'help': 'where to store downloaded model, datasets, and tokenizer'
        }
    )
    
    log_dir: Optional[str] = field(
        default='/scratch/gpfs/smalladi/icl_as_ft/logs'
    )

    dataset: Optional[str] = field(
        default='wikitext-103',
        metadata={
            'help': 'dataset to use, should be in HF datasets repo'
        }
    )
    
    incontext_dataset: Optional[str] = field(
        default="", metadata={"help": "Dataset for in-context experiments!"}
    ) 
    
    incontext_stem: Optional[str] = field(
        default="", metadata={"help": "Directory containing the files!"} 
    ) 
        
    incontext_n_shot: Optional[int] = field(
        default=4, metadata={"help": "Number of demonstations in the in-context prompt!"} 
    )     
        
    prompt_variant:  Optional[int] = field(
        default=0, metadata={"help": "Variants to try for sst-2!"} 
    )        

        
    use_eval_set: bool = field(
        default=True,
        metadata={'help': 'when on, uses the eval dataset only for ppl measurement/in-context expts.'}
    )
        
    use_test_set: bool = field(
        default=False,
        metadata={'help': 'when on, uses the test dataset only for ppl measurement/in-context expts.'}
    )    

    ### finetuning strategies ###
    
    chunk_data: bool = field(
        default=True,
        metadata={'help': 'when on, chunks the data into block_size chunks'}
    )
    
    baseline: bool = field(
        default=False
    )
    
    num_subsets: int = field(
        default=16
    )
    data_chunk_index: int = field(
        default=0
    )

    block_size: Optional[int] = field(
        default=1024,
        metadata={'help': 'size of chunks to break corpus into'}
    )
    
    num_workers: Optional[int] = field(
        default=4,
        metadata={'help': 'number of workers'}
    )
    
    data_subset: Optional[int] = field(
        default=-1, metadata={"help": "Number of training examples to use in the subset!"}
    )

    train_fraction: Optional[float] = field(
        default=0.5, metadata={"help": "Fraction of sentence for training!"}
    )
        
        
    batch_size: Optional[int] = field(
        default=1, metadata={"help": "Batch size for training!"}
    )
        
    