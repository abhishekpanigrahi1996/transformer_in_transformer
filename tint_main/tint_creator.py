import torch


from .tint_gpt import *
from .tint_opt import *
from transformers import GPT2LMHeadModel, OPTForCausalLM
from transformers import HfArgumentParser
from transformers import AutoConfig
from .utils.config import Construct_config


#Creates the tree view of the model layers
def nested_children(m: torch.nn.Module):
    children = dict(m.named_children())
    output = {}
    if children == {}:
        # if module has no children; m is last child! :O
        return m
    else:
        # look for children from children... to the last child!
        for name, child in children.items():
            try:
                output[name] = nested_children(child)
            except TypeError:
                output[name] = nested_children(child)
    return output


#Creates TinT on a given auxiliary models
#model_args contains the necessary config for auxiliary model
#construction_args contains the necessary config for TinT
#The module calls TinT_gpt for gpt models and TinT_opt for opt models
def TinT_Creator(model_args, construction_args):
    model_config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir
    )
    
    
    construction_args.activation_function = model_config.activation_function
    construction_args.seq_length = model_config.max_position_embeddings
    
    config = Construct_config(model_config, construction_args)
    
    if 'gpt' in model_args.model_name_or_path:
        model_fn = GPT2LMHeadModel
    elif 'opt' in model_args.model_name_or_path:
        model_fn = OPTForCausalLM
    else:
        raise NotImplmentedError
    
    model = model_fn.from_pretrained(
        model_args.model_name_or_path,
        config=model_config,
        cache_dir=model_args.cache_dir,
    )
    
    
    print ("....Constructing the model....") 
    
    if  model_args.construct_load_model:  
        print ("....Load constructed model....")
        checkpoint = torch.load(model_args.construct_model_path)
        model_config = checkpoint['model_config']
        config = checkpoint['construction_config']
    
    print (nested_children(model))
    if 'gpt' in model_args.model_name_or_path:
        constructed_model = TinT_gpt(config, model_config, nested_children(model))
    elif 'opt' in model_args.model_name_or_path:
        constructed_model = TinT_opt(config, model_config, nested_children(model))
    else:
        raise NotImplmentedError
    
    if  model_args.construct_load_model:  
        constructed_model.load_state_dict(checkpoint['model_state_dict'])
    
    
    if model_args.construct_save_model:
        print ("....Store constructed model....")
        torch.save({'model_state_dict': constructed_model.state_dict(),\
                    'model_config': model_config,\
                    'construction_config': config},\
                   model_args.construct_model_path,\
                  )


    tot = 0
    for parameter in constructed_model.parameters():
        tot += parameter.numel()
    print ("Total trainable parameters in constructed model", tot)
    
    return (constructed_model, model, model_config, config)
    


#Creates TinT from a config checkpoint
#model_path: auxiliary model name gpt2 or facebook/opt-125m
#cache_dir is where the auxiliary model has been saved from huggingface
#construction_args contains the necessary config for TinT
#The module calls TinT_gpt for gpt models and TinT_opt for opt models
def load_construction(model_path, \
                      cache_dir, \
                      construction_path, \
                     ):
    
    model_config = AutoConfig.from_pretrained(
        model_path,
        cache_dir=cache_dir
    )
    

    if 'gpt' in model_path:
        model_fn = GPT2LMHeadModel
    elif 'opt' in model_path:
        model_fn = OPTForCausalLM
    else:
        raise NotImplmentedError
    
    model = model_fn.from_pretrained(
        model_path,
        config=model_config,
        cache_dir=cache_dir,
    )
    
    import time
    start = time.time() 
    print ("....Load constructed model....")
    checkpoint = torch.load(construction_path)
    model_config = checkpoint['model_config']
    config = checkpoint['construction_config']
    
    if 'gpt' in model_path:
        constructed_model = TinT_gpt(config, model_config, nested_children(model))
    elif 'opt' in model_path:
        constructed_model = TinT_opt(config, model_config, nested_children(model))
    else:
        raise NotImplmentedError
    
    constructed_model.load_state_dict(checkpoint['model_state_dict'])
    end = time.time()
    print("Time for loading constructed model: ", round(end - start, 2))
    return constructed_model
    

if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments,))
    model_args, = parser.parse_args_into_dataclasses()    

    constructed_model = load_construction(model_args.model_name_or_path, \
                                           model_args.cache_dir, \
                                           model_args.construct_model_path, \
                                          )
        

    
    
