import torch
from tqdm import tqdm
from Conversion import *
from transformers import AutoTokenizer, OPTForCausalLM
from datasets import load_dataset
import copy
from transformers import HfArgumentParser
from filelock import Timeout, FileLock
from data_utils import *
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
    
    device: Optional[str] = field(
        default="cuda", metadata={"help": "cuda/cpu"}
    )
    
    
    final_layers_to_train: Optional[int] = field(
        default=-1, metadata={"help": "Number of layers to train"}
    )
    
    gradient_steps : Optional[int] = field(
        default=1, metadata={"help": "Number of gradient steps"}
    )
    
    learning_rate: Optional[float] = field(
        default=1e-04, metadata={"help": "Learning rate for projection!"}
    )    
        
    batch_size: Optional[int] = field(
        default=1, metadata={"help": "Batch size for training!"}
    )
        
    train_fraction: Optional[float] = field(
        default=0.5, metadata={"help": "Fraction of sentence for training!"}
    )    
     
    dynamic_chunks: Optional[int] = field(
        default=1, metadata={"help": "Number of dynamic chunks (before the final test per sequence)!"}
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
        
    use_eval_set: bool = field(
        default=True,
        metadata={'help': 'when on, uses the eval dataset only for ppl measurement.'}
    )
        
    use_test_set: bool = field(
        default=False,
        metadata={'help': 'when on, uses the test dataset only for ppl measurement.'}
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

    
      
        
parser = HfArgumentParser((ModelArguments, DynamicArguments))
model_args, data_args = parser.parse_args_into_dataclasses()
  
learning_rate = model_args.learning_rate
gradient_steps = model_args.gradient_steps
final_layers_to_train = model_args.final_layers_to_train
train_fraction = model_args.train_fraction
dynamic_chunks = model_args.dynamic_chunks

device = model_args.device
model_config = AutoConfig.from_pretrained(
    model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    cache_dir=model_args.cache_dir
)


if 'gpt' in model_args.model_name_or_path:
    model_fn = GPT2LMHeadModel
elif 'opt' in model_args.model_name_or_path:
    model_fn = OPTForCausalLM
else:
    raise NotImplmentedError

simulated_gpt2 = model_fn.from_pretrained(
    model_args.model_name_or_path,
    config=model_config,
    cache_dir=model_args.cache_dir,
)

simulated_gpt2.to(device)
simulated_gpt2.eval()


#dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir='data', download_mode='reuse_cache_if_exists')
#test_data = dataset['test']
#test_data = dataset['test']
#valid_data = dataset['validation']

batch_size=model_args.batch_size

assert batch_size == 1, \
        "Assume batch size as 1 for proper perplexity computation (lazy to do for multi input batch)"

# Download vocabulary from huggingface.co and cache.
tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir="../..")
if 'gpt' in model_args.model_name_or_path:
    tokenizer.pad_token = tokenizer.eos_token
    pad_token_id=tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
           
elif 'opt' in model_args.model_name_or_path:
    tokenizer.bos_token_id = 0
    
dataset = preprocess_dataset(data_args, tokenizer)
#_, simulated_gpt2, model_config, config = Construct_NASgpt()


simulated_gpt2.eval()
device=next(simulated_gpt2.parameters()).device



num_valid_batches = len(dataset) // batch_size

if data_args.data_subset != -1:
    num_valid_batches = min(data_args.data_subset, num_valid_batches)
    
avg_model_perplexity = 0.
avg_eval_perplexity = 0.
total_words = 0.


avg_model_test_perplexity = 0.
avg_eval_test_perplexity = 0.
total_test_words = 0.



for batch_id in tqdm( range( num_valid_batches ) ):
    if data_args.dataset == 'c4' and batch_id == 100: break
    model_copy = copy.deepcopy(simulated_gpt2)
    model_copy.eval()
    
    trainable_parameters = []
    trainable_name = []
    if final_layers_to_train == -1:
        final_layers_to_train = 12
        
    #for i in range( 12 - final_layers_to_train, 12 ):
    for n, p in model_copy.named_parameters():
        if 'ln_' not in n and '.h.' in n:
            layer_num = int(n.split('.h.')[-1].split('.')[0])
            if layer_num >= 12 - final_layers_to_train:
                trainable_parameters += [ p ] 
                trainable_name += [n]
        
    
    optimizer = torch.optim.SGD(model_copy.parameters(), lr=learning_rate)
    model_copy.zero_grad()
    
    
    
    data = dataset [ batch_id * batch_size : (batch_id + 1) * batch_size ]
    batch_sentences = torch.tensor(  data ['input_ids'] )
    attention_mask  = torch.tensor(  data ['attention_mask'] )
    labels = torch.tensor(  data ['labels'] )
    
    
    
    
    
    
    if len(batch_sentences.shape) == 1:
        batch_sentences = batch_sentences.view((1, -1))
        attention_mask = attention_mask.view((1, -1))
        labels = labels.view((1, -1))
    
    train_subchunk_fraction = train_fraction / dynamic_chunks 
    
    #initialize the dynamic loss
    final_loss = 0.
    og_loss = 0.
    total_terms = 0.
    loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
    
    
    og_test_loss = 0. 
    final_test_loss = 0.
    #original model's loss
    #with torch.no_grad():
    #    train_seq_loss = model_copy(batch_sentences.cuda(), \
    #                                output_hidden_states=False, \
    #                                labels=labels.long().to(device))[0].item()
    
    #dynamic training
    for dynamic_chunk_id in range(dynamic_chunks):
       
        batch_seq_lengths = torch.sum(attention_mask.int(), dim=-1)
        mask = torch.zeros_like(attention_mask)
        
        for i in range(len(batch_seq_lengths)):
            len_chunk = int(batch_seq_lengths[i] * train_subchunk_fraction)
            actual_fraction = len_chunk / batch_seq_lengths[i] 
            #print (actual_fraction, train_subchunk_fraction)
            mask[i, dynamic_chunk_id * len_chunk: dynamic_chunk_id * len_chunk + len_chunk] = 1.
        bidirection_mask = mask.float()
   
        target = labels.detach().clone()
        target[ torch.where(bidirection_mask == 0.)  ] = -100

    
        #print (target)
        #with torch.no_grad():
        input_ids = batch_sentences.to(device)
        target = target.to(device)
        
        #first a simple evaluation on the current model
        with torch.no_grad():
            output = model_copy(input_ids, \
                                attention_mask=attention_mask.to(device), \
                                output_hidden_states=False, 
                               )
            
            print (output[0].shape, input_ids.shape)
            final_loss +=  loss_fct( output[0][:, :-1].view((-1, model_config.vocab_size)), \
                                     target[:, 1:].long().view((-1,)) \
                                   ).item()
           
            gpt_output = simulated_gpt2(input_ids, \
                                        attention_mask=attention_mask.to(device), \
                                        output_hidden_states=False, \
                                       )
            og_loss +=  loss_fct( gpt_output[0][:, :-1].view((-1, model_config.vocab_size)), \
                                  target[:, 1:].long().view((-1,)) \
                                ).item()
            total_terms += bidirection_mask.sum()
            
            
        for _ in range(gradient_steps):
            simulated_output = model_copy(input_ids, \
                                          attention_mask=attention_mask.to(device), \
                                          output_hidden_states=True, \
                                          labels=target.long()\
                                         )
            small_model_loss = simulated_output[0]
            #print (small_model_loss.item())
            small_model_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        for n, p in model_copy.named_parameters():
            for n_, p_ in simulated_gpt2.named_parameters():
                if n == n_ and n in trainable_name:
                    print ( n, torch.max (torch.absolute(p - p_)) )

    
    
    
    #print ([p for n, p in model_copy.named_parameters() if  'ln_f' in n])
    #test on the remaining chunk
    
    batch_seq_lengths = torch.sum(attention_mask.int(), dim=-1)
    mask = torch.zeros_like(attention_mask)
    

    for i in range(len(batch_seq_lengths)):
        len_chunk = dynamic_chunks * int(batch_seq_lengths[i] * train_subchunk_fraction)
        test_fraction = 1. - len_chunk / (1. * batch_seq_lengths[i])
        mask[i, len_chunk:] = 1.
    bidirection_mask = mask.float()

        
    with torch.no_grad():
        target = labels.detach().clone()
        target[ torch.where(bidirection_mask == 0.)  ] = -100
        target = target.to(device)
        
        
        
        simulated_output = model_copy(input_ids, \
                                      attention_mask=attention_mask.to(device), \
                                      output_hidden_states=False, 
                                     )
        loss = loss_fct( simulated_output[0][:, :-1].view((-1, model_config.vocab_size)), \
                         target[:, 1:].long().view((-1,)) \
                       ).item()
        final_loss += loss
        avg_eval_test_perplexity += loss
        
        gpt_output = simulated_gpt2(input_ids, \
                                    attention_mask=attention_mask.to(device), \
                                    output_hidden_states=False, 
                                   )
        
        loss = loss_fct( gpt_output[0][:, :-1].view((-1, model_config.vocab_size)), \
                         target[:, 1:].long().view((-1,)) \
                       ).item()
        og_loss +=  loss
        avg_model_test_perplexity += loss
        
        total_terms += bidirection_mask.sum()
        total_test_words += bidirection_mask.sum()
            
    del (model_copy)
    avg_model_perplexity += og_loss
    avg_eval_perplexity  += final_loss 
    total_words += total_terms

    
final_result = {}
final_result[ 'Validation Dynamic eval acc' ] = np.exp(avg_eval_perplexity / total_words)
final_result[ 'Validation Model acc' ] = np.exp(avg_model_perplexity / total_words)

final_result[ 'Validation Dynamic eval acc (on test)' ] = np.exp(avg_eval_test_perplexity  / total_test_words)
final_result[ 'Validation Model acc (on test)' ] = np.exp(avg_model_test_perplexity / total_test_words)


with FileLock('log_exp.lock'):
    with open('log_exp', 'a') as f:
        final_result.update(vars(model_args))
        final_result.update(vars(data_args))
        f.write(str(final_result) + '\n')