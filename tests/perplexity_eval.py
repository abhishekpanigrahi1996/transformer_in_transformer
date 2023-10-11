import torch
from tqdm import tqdm
import os

from tint_main.tint_creator import *
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import HfArgumentParser
from filelock import Timeout, FileLock
from .data_utils.data_utils import *
from tint_main.utils.all_arguments import *
      
parser = HfArgumentParser((ModelArguments, ConstructionArguments, DynamicArguments,))
model_args, construction_args, data_args, = parser.parse_args_into_dataclasses()
        

tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
if 'gpt' in model_args.model_name_or_path:
    tokenizer.pad_token = tokenizer.eos_token
    pad_token_id=tokenizer.convert_tokens_to_ids(tokenizer.pad_token)    
elif 'opt' in model_args.model_name_or_path:
    tokenizer.bos_token_id = 0
    
dataset = preprocess_dataset(data_args, tokenizer)           
batch_size=data_args.batch_size
constructed_model = load_construction(model_args.model_name_or_path, model_args.cache_dir, model_args.construct_model_path)



constructed_model.eval()
if construction_args.device == 'cuda': device='cuda:0'
else: device='cpu'   
    
num_valid_batches = len(dataset) // batch_size
train_fraction = data_args.train_fraction

if data_args.data_subset == 0:
    exit(0)

if data_args.data_subset != -1:
    num_valid_batches = min(data_args.data_subset // batch_size, num_valid_batches)
    

avg_model_test_perplexity = 0.
avg_eval_test_perplexity = 0.
total_test_words = 0.



for batch_id in tqdm( range( num_valid_batches ), desc='Inference' ):
    
   
    
    data = dataset [ batch_id * batch_size : (batch_id + 1) * batch_size ]
    batch_sentences = torch.tensor(  data ['input_ids'] )
    attention_mask  = torch.tensor(  data ['attention_mask'] )
    labels = torch.tensor(  data ['labels'] )
    
    if len(batch_sentences.shape) == 1:
        batch_sentences = batch_sentences.view((1, -1))
        attention_mask = attention_mask.view((1, -1))
        labels = labels.view((1, -1))
    
    
    
    mask = torch.zeros_like(attention_mask)
    target = batch_sentences.detach().clone()
    target [ torch.where(attention_mask == 0.) ] = -100
    
    batch_seq_lengths = torch.sum(attention_mask, dim=-1)
    for i in range(len(batch_seq_lengths)):
        mask[i, :int(batch_seq_lengths[i] * train_fraction)] = 1. 
        target[ i, :int(batch_seq_lengths[i] * train_fraction) ] = -100
    
    bidirection_mask = mask.float()
    gradient_mask = None
    
    with torch.no_grad():
        results = constructed_model.forward(batch_sentences.to(device), bidirection_mask.to(device), gradient_mask=gradient_mask, test_backward_pass=True, continue_from_first_forward_pass=False, labels=target.to(device))
        original_loss, final_loss = results.original_loss, results.final_loss
        
    
    total_terms = torch.ne(target, -100).float().sum()
    
    avg_model_test_perplexity += original_loss.item() * total_terms
    avg_eval_test_perplexity  += final_loss.item() *  total_terms
    total_test_words += total_terms

    
final_result = {}
final_result[ 'Validation Dynamic eval acc (on test)' ] = np.exp(avg_eval_test_perplexity / total_test_words)
final_result[ 'Validation Model acc (on test)' ] = np.exp(avg_model_test_perplexity / total_test_words)



with FileLock('log_exp_construct.lock'):
    with open('log_exp_construct', 'a') as f:
        final_result.update(vars(model_args))
        final_result.update(vars(data_args))
        final_result.update(vars(construction_args))
        f.write(str(final_result) + '\n')
        
import torch.utils.benchmark as benchmark
def benchmark_forward(fn, *inputs, repeats=10, desc='', verbose=True, amp=False,
                      amp_dtype=torch.float16, **kwinputs):
    """ Use Pytorch Benchmark on the forward pass of an arbitrary function. """
    if verbose:
        print(desc, '- Forward pass')
    def fn_amp(*inputs, **kwinputs):
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp):
            fn(*inputs, **kwinputs)
    for _ in range(repeats):  # warmup
        fn_amp(*inputs, **kwinputs)
    t = benchmark.Timer(
            stmt='fn_amp(*inputs, **kwinputs)',
            globals={'fn_amp': fn_amp, 'inputs': inputs, 'kwinputs': kwinputs},
            num_threads=torch.get_num_threads(),
            )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m        