import logging

import torch
import copy
from filelock import FileLock
from itertools import chain

from accelerate import Accelerator
from datasets import load_dataset, concatenate_datasets
logger = logging.getLogger(__name__)

def get_raw_data(args):
    if args.dataset == 'wikitext-103':
        dataset = load_dataset('wikitext', 'wikitext-103-v1', cache_dir=args.data_cache_dir)
        if args.use_eval_set:
            dataset = dataset['validation']
        elif args.use_test_set:
            dataset = dataset['test']    
        else:
            dataset = dataset['train']
        return dataset
    elif args.dataset == 'wikitext-2':
        print (args.data_cache_dir)
        dataset = load_dataset('wikitext', 'wikitext-2-v1', cache_dir=args.data_cache_dir)
        if args.use_eval_set:
            dataset = dataset['validation']
        elif args.use_test_set:
            dataset = dataset['test']        
        else:
            dataset = dataset['train']
        return dataset
    elif args.dataset == 'c4':
        print (args.data_cache_dir)
        dataset = load_dataset('c4', 'realnewslike', cache_dir=args.data_cache_dir)
        if args.use_eval_set:
            dataset = dataset['validation']
        elif args.use_test_set:
            dataset = dataset['validation']      
        else:
            dataset = dataset['train']
        return dataset
    raise NotImplementedError

def preprocess_dataset(args, tokenizer):
    raw_datasets = get_raw_data(args)
    
    text_column_name = "text"
    column_names = raw_datasets.column_names
    
        
    accelerator = Accelerator()

    def tokenize_function(examples):
        return tokenizer([i for i in examples[text_column_name] if len(i)>0])

    with accelerator.main_process_first():
        datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.num_workers,
            remove_columns=column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )

    block_size = args.block_size 
    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def concat_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        concatenated_examples['labels'] = concatenated_examples['input_ids'].copy()
        return concatenated_examples

    if args.chunk_data:
        with accelerator.main_process_first():
            datasets = datasets.map(
                group_texts,
                batched=True,
                load_from_cache_file=True,
                num_proc=args.num_workers,
                desc=f"Grouping texts in chunks of {block_size}",
            )
    else:
        with accelerator.main_process_first():
            datasets = datasets.map(
                concat_texts,
                batched=True,
                load_from_cache_file=True,
                num_proc=args.num_workers,
                desc=f"Concatenating the texts",
            )

    return datasets


def prepare_inputs(batch, padding=None):
    for k in batch:
        if len(batch[k].shape) == 1:
            batch[k] = batch[k].reshape(1, -1)
        if padding is not None:
            for k in batch:
                batch[k] = torch.cat((batch[k], padding))
        batch[k] = batch[k].cuda()

    return batch

def init_model_and_optimizer(model_init, args):
    model = copy.deepcopy(model_init)
    if args.freeze_embs:
        if 'opt' in args.model_name:
            model.model.decoder.embed_tokens.requires_grad = False
            model.model.decoder.embed_positions.requires_grad = False
        else:
            model.transformer.wte.requires_grad = False
            model.transformer.wpe.requires_grad = False
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    return model, optimizer