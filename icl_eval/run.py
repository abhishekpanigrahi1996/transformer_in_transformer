import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import argparse
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, Trainer, HfArgumentParser, TrainingArguments, DataCollatorWithPadding, DataCollatorForTokenClassification
import torch
from torch.nn.parameter import Parameter
import numpy as np
from dataclasses import dataclass, is_dataclass, asdict
from tqdm import tqdm
from tasks import get_task
import json
import torch.nn.functional as F
from metrics import calculate_metric
from utils import *
from dataclasses import dataclass, field, fields
import sys

# TODO: fix
sys.path.append("/scratch/gpfs/mengzhou/space9/icl-as-ft/Dynamic_initialization")
sys.path.append("/n/fs/nlp-mengzhou/space9/icl-as-ft/Dynamic_initialization")
from load_models import load_construction

CACHE_DIR=os.environ["n"] + "/.cache"

@dataclass
class OurArguments(TrainingArguments):
    # dataset and sampling strategy
    task_name: List[str] = field(default_factory=list, metadata={"help": "task nam e should match the string before Dataset in the Dataset class name."})
    num_train: int = 0 # number of ICL samples to encode
    num_eval: int = None # sample evaluation samples
    train_set_seed: int = None # designated seed to sample training samples/demos
    test_set_seed: int = None # designated seed to sample evaluation samples
    eval_set_seed: int = None # designated seed to sample test samples
    result_file: str = None # output file name; if None, then use the task name, model name, and config
    pred_file: str = None # prediction file name; if None, then use the task name, model name, and config

    # model
    model_path: str = "opt-125m" # model path 
    model_name: str = "facebook/opt-125m" # model name
    load_float16: bool = False # load model as float16
    max_length: int = 2048 # max length the model can take
    pruned: bool = False # whether to use pruned model

    # calibration
    sfc: bool = False # whether to use SFC calibration
    icl_sfc: bool = False # whether to use SFC calibration for ICL samples
    loss_label_only: bool = False #Only for construction; whether to compute loss only on the labels
    unbiased_sampling: bool = True #Whether to have unbiased sampling in demonstrations    

    # generation
    sampling: bool = False # whether to use sampling
    temperature: float = 1.0 # temperature for generation
    num_beams: int = 1 # number of beams for generation
    top_k: int = None # top-k for generation
    top_p: float = 0.95 # top-p for generation
    max_new_tokens: int = 50 # max number of new tokens to generate
    eos_token: str = "\n" # end of sentence token

    # saving
    save_model: bool = False # whether to save the model
    tag: str = "" # saving tag
    
    # deprecated 
    make_first_token_one: bool = False # whether to make the first token one
    make_first_token_zero: bool = False # whether to make the first token zero
    
    # dynamic eval 
    dynamic_eval_icl: bool = False # if True, then use dynamic eval; if False, then use static eval
    dynamic_eval_lr: float = 1e-3 # learning rate for dynamic eval
    loss_type: str = "label_only" # loss_only; 
    exclusive_training: bool = False # multi-FT, exclude other examples 
    single_FT: bool = False # whether to use single FT for dynamic eval
    num_train_layers: int = 12
    num_epochs: int = 1

    # test_mode
    test_mode: bool = False # if True, then only run a few examples for debugging 
    
    #additional construction arguments
    restrict_attention_demonstration: bool = False # whether to use restrict the attention of each demonstration to itself in icl experiments
    position_modify_demonstration: bool = False # whether to change the position ids of each demonstration to top; only relevant if restrict_attention_demonstration is true
    mask_demonstration_eval: bool = False # whether to mask demonstrations when evaluating on eval example
    position_modify_eval: bool = False    # whether to change the position ids of eval example; only relevant if mask_demonstration_eval is true

    # synthatic labels
    template_id: int = 0 # whether to use synthetic labels; 0 is normal template and 1 is synthetic label template

def parse_args():
    parser = argparse.ArgumentParser()
    parser = HfArgumentParser(OurArguments)
    args = parser.parse_args_into_dataclasses()[0]
    return args

class Framework:
    def __init__(self, args):
        self.args = args
        self.local_paths = {f"llama-{size}": f"/projects/DANQIC/mengzhou/LLaMA/hf-{size.upper()}" for size in ["7b", "13b", "30b", "66b"]}
        self.local_paths["alpaca"] = "/scratch/gpfs/tianyug/stanford_alpaca/ckpt/alpaca-7B"

        self.config = self.load_config()
        self.model, self.tokenizer = self.load_model()
    
    def freeze_layers(self, model):
        model.model.decoder.embed_tokens.weight.requires_grad = False
        model.model.decoder.embed_positions.weight.requires_grad = False
        num_layers = model.config.num_hidden_layers
        layers = model.model.decoder.layers[:num_layers-self.args.num_train_layers]
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False
        logger.info(f"Freezing layers: {num_layers-self.args.num_train_layers}")
        logger.info(f"Training layers: {self.args.num_train_layers}")
        
    def load_config(self):
        if self.is_llama:
            config_name = "-".join(self.args.model_name.split("-")[:2])
            if config_name in self.local_paths:
                config_path = self.local_paths[config_name]
            else:
                config_path = self.args.model_path
            config = LlamaConfig.from_pretrained(config_path)
        elif self.is_constructed:
            if "opt" in self.args.model_name: 
                config_path = "facebook/opt-125m"
                config = AutoConfig.from_pretrained(config_path)
            elif "gpt2" in self.args.model.name: 
                config_path = "gpt2"
                config = AutoConfig.from_pretrained(config_path)
        else:
            config_path = self.args.model_path
            config = AutoConfig.from_pretrained(config_path)
        self.config_path = config_path
        return config

    def get_tokenizer_path(self):
        if self.is_llama: 
            tokenizer_path ="/projects/DANQIC/mengzhou/LLaMA/hf-7B"
        elif self.is_constructed:
            if "opt" in self.args.model_name: tokenizer_path = "facebook/opt-125m"
            elif "gpt2" in self.args.model_name: tokenizer_path = "gpt2"
        else:
            tokenizer_path = self.args.model_path 
        return tokenizer_path
    
    @property
    def is_llama(self):
        return "llama" in self.args.model_name
    
    @property
    def is_constructed(self):
        return "constructed" in self.args.model_name
    
    @property
    def is_opt(self):
        return "opt" in self.args.model_name
    
    def load_model(self):
        # load model using HF's accelerate
        with count_time("Loading model with FP%d" % (16 if self.args.load_float16 else 32)):
            free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
            if self.is_constructed:
                if "opt" in self.args.model_name: mm = "facebook/opt-125m"
                elif "gpt2" in self.args.model_name: mm = "gpt2"
                model = load_construction(mm, CACHE_DIR, self.args.model_path)
                model.cuda()
                model.device = torch.device("cuda")
            elif self.is_opt:
                from models.modeling_opt import LocalOPTForCausalLM
                model = LocalOPTForCausalLM.from_pretrained(
                    self.args.model_path,
                    config=self.config,
                    device_map="auto",
                    torch_dtype=torch.float16 if self.args.load_float16 else torch.float32,
                    max_memory={i: f'{free_in_GB-5}GB' for i in range(torch.cuda.device_count())}, 
                )
                self.freeze_layers(model)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    self.args.model_path,
                    config=self.config,
                    device_map='auto',
                    torch_dtype=torch.float16 if self.args.load_float16 else torch.float32,
                    max_memory={i: f'{free_in_GB-5}GB' for i in range(torch.cuda.device_count())},
                )
            model.eval()
            
        # load tokenizer
        tokenizer_path = self.get_tokenizer_path()
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
        
        if tokenizer.pad_token is None: 
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # HF bug fix
        if "opt" in self.args.model_name:
            tokenizer.bos_token_id = 0

        return model, tokenizer
    
    def constructed_model_forward(self, input_ids, option_len=None, **kwargs):
        input_ids = torch.tensor([input_ids]).to(self.model.device)
        mask = torch.zeros_like(input_ids)
        mask[:, :-option_len] = 1.
        gradient_mask, icl_mask, position_ids = kwargs['gradient_mask'], kwargs['icl_mask'], kwargs['position_ids']
        label_ids = kwargs["label_ids"]
        
        if gradient_mask is not None:
            gradient_mask = torch.tensor(gradient_mask).unsqueeze(0).to(self.model.device)
        if icl_mask is not None:
            icl_mask = torch.tensor(icl_mask).unsqueeze(0).to(self.model.device)
        if position_ids is not None:
            position_ids = torch.tensor(position_ids).unsqueeze(0).to(self.model.device)
        
   
        with torch.inference_mode():
            self.model.eval()
            outputs = self.model(input_ids=input_ids, bidirection_mask=mask, labels=input_ids,  continue_from_first_forward_pass=False, gradient_mask=gradient_mask, icl_mask=icl_mask, position_ids=position_ids)
            logits = outputs.logits
            # loss = outputs.loss
            # labels = input_ids[0, 1:]
            logits = logits[0, :-1][-1] 
            log_probs = F.log_softmax(logits, dim=-1)
            selected_log_probs = log_probs[label_ids].cpu().detach()
            # selected_log_probs = log_probs[torch.arange(len(labels)).to(labels.device), labels]
            # selected_log_probs = selected_log_probs.cpu().detach()
            return selected_log_probs
    
    
    def train(self, input_ids, labels, attention_mask, twod_attention_mask=None):
        lr = self.args.dynamic_eval_lr
        train_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(train_params, lr=lr)
        print("Trainable parameters: ", sum([tp.nelement() for tp in train_params]))
        
        self.model.train()
        for j in range(self.args.num_epochs):
            # input_ids_ba = input_ids[i:i+4]; labels_ba = labels[i:i+4]; attention_mask_ba = attention_mask[i:i+4]
            input_ids_ba = input_ids; labels_ba = labels; attention_mask_ba = attention_mask
            loss = self.model(input_ids=input_ids_ba, attention_mask=attention_mask_ba, labels=labels_ba, twod_attention_mask=twod_attention_mask).loss
            loss.backward()
            logger.info(f"train loss: {loss.item()}")
            optimizer.step()
            optimizer.zero_grad()
            self.model.zero_grad()
    
    def dynamic_eval(self, task, train_samples, eval_samples, test_samples=None, one_train_set_per_eval_sample=False, FT=True):
        all_metrics = []
        template = task.get_template(self.args.template_id)
        values = list(template.verbalizer.values())
        label_ids = [self.tokenizer.encode(f" {value}", add_special_tokens=False)[0] for value in values]
        logger.info(f"label words {values}")
        logger.info(f"label ids {label_ids}" )
        
        train_exs = []
        for train_sample in train_samples:
            train_candidates, train_option_lens, _ = encode_prompt(task, task.get_template(self.args.template_id), [], train_sample, self.tokenizer, max_length=self.args.max_length, generation=task.generation)
            train_ex = train_candidates[train_sample.correct_candidate]
            train_exs.append(train_ex)
            
        input_ids = [torch.tensor(train_ex) for train_ex in train_exs] 
        attention_mask = [torch.ones_like(input_id) for input_id in input_ids]
        for mask in attention_mask: mask[-1] = 0 # not necessary
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.eos_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = input_ids.clone()
        
        if self.args.loss_type == "label_only":
            labels[...] = -100
            label_word_index = (input_ids == label_ids[0])
            logger.info(f"label 1: {label_word_index.sum().item()}")
            for i in range(1, len(label_ids)):
                new_word_index = (input_ids == label_ids[i])
                label_word_index = label_word_index | new_word_index
                logger.info(f"label {i+1}: {new_word_index.sum().item()}")
            labels[label_word_index] = input_ids[label_word_index]
        
        input_ids = input_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)
        labels = labels.to(self.model.device)
        logger.info(f"pos: {(labels == label_ids[0]).sum().item()}")
        logger.info(f"neg: {(labels == label_ids[1]).sum().item()}")
        self.train(input_ids, labels, attention_mask)
        
        all_metrics = self.eval_multiple_settings(task, train_samples, eval_samples, test_samples, self.args.test_mode) 
        return None, all_metrics
         
    def dynamic_eval_icl(self, task, train_samples, eval_samples, test_samples=None, one_train_set_per_eval_sample=False, FT=False):
        if FT:
            template = task.get_template(self.args.template_id)
            values = list(template.verbalizer.values())
            label_ids = [self.tokenizer.encode(f" {value}", add_special_tokens=False)[0] for value in values]
            logger.info(f"label words, {values}")
            logger.info(f"label ids, {label_ids}")
            
            train_candidates, train_option_lens, _ = encode_prompt(task, template, train_samples[:-1], train_samples[-1], self.tokenizer, max_length=self.args.max_length, generation=task.generation) 
            train_ex = torch.tensor(train_candidates[train_samples[-1].correct_candidate])
            
            twod_attention_mask = None 
            if self.args.exclusive_training: # during training, each example only gets to see its own 
                train_sep_id = self.tokenizer.encode("I" + task.train_sep + "I", add_special_tokens=False)[1:-1]
                train_sep_id_string = " ".join([str(v) for v in train_sep_id])
                logger.info(f"Train sep: {task.train_sep}")
                logger.info(f"Train sep id: {train_sep_id}")
                exs = [ex.strip() for ex in " ".join([str(v.item()) for v in train_ex]).split(train_sep_id_string)]
                exs = [[int(v) for v in ex.split()] for ex in exs]
                lens = [len(ex) for ex in exs]
                logger.info("Training example lens")
                logger.info(lens)
                cumsum = np.cumsum(lens)
                twod_attention_mask = torch.zeros((1, 1, cumsum[-1], cumsum[-1]))
                twod_attention_mask.fill_(torch.finfo(next(self.model.parameters()).dtype).min)
                
                for i in range(len(cumsum)):
                    if i == 0: start_index = 0
                    else: start_index = cumsum[i-1]
                    twod_attention_mask[0][0][start_index: cumsum[i], start_index: cumsum[i]] = 0 
                twod_attention_mask = twod_attention_mask.to(self.model.device)
                train_ex = torch.cat([torch.tensor(ex) for ex in exs])

            input_ids = train_ex.unsqueeze(0).to(self.model.device)
            attention_mask = torch.ones_like(input_ids)
            labels = input_ids.clone()
            labels = torch.ones_like(input_ids)
            if self.args.loss_type == "label_only":
                labels[...] = -100
                label_word_index = (input_ids == label_ids[0])
                logger.info("label 1: {label_word_index.sum().item()}")
                for i in range(1, len(label_ids)):
                    new_word_index = (input_ids == label_ids[i])
                    label_word_index = label_word_index | new_word_index
                    logger.info(f"label {i+1}: {new_word_index.sum().item()}")
                labels[label_word_index] = input_ids[label_word_index]
            self.train(input_ids, labels, attention_mask, twod_attention_mask=twod_attention_mask)
        
        all_metrics = self.eval_multiple_settings(task, train_samples, eval_samples, test_samples, verbose=self.args.test_mode)
        return None, all_metrics
    
    def eval_multiple_settings(self, task, train_samples, eval_samples, test_samples=None, verbose=False):
        is_test = False
        if test_samples is not None:
            eval_samples = test_samples; is_test = True
            
        all_metrics = []

        self.model.eval()

        sfc = self.args.sfc; icl_sfc = self.args.icl_sfc
        if self.args.single_FT:
            sfc = self.args.sfc; icl_sfc = self.args.icl_sfc
            preds, metrics = self.evaluate(task, [], eval_samples, one_train_set_per_eval_sample=False, verbose=verbose, twod_attention_mask=False) 
            metrics["sfc"] = sfc; metrics["icl_sfc"] = icl_sfc; metrics["demonstrations"] = False;  metrics["mask_demonstrations"] = False; metrics["is_test"] = is_test 
            logger.info(metrics)
            all_metrics.append(metrics)
            write_predictions_to_file(preds, os.path.join(self.args.output_dir, "no-demonstrations.jsonl"))
        else:
            preds, metrics = self.evaluate(task, train_samples, eval_samples, one_train_set_per_eval_sample=False, verbose=verbose, twod_attention_mask=False) 
            metrics["sfc"] = sfc; metrics["icl_sfc"] = icl_sfc; metrics["demonstrations"] = True;  metrics["mask_demonstrations"] = False; metrics["is_test"] = is_test
            logger.info(metrics)
            all_metrics.append(metrics)
            write_predictions_to_file(preds, os.path.join(self.args.output_dir, "with-demonstrations.jsonl"))
        
        # preds, metrics = self.evaluate(task, train_samples, eval_samples, one_train_set_per_eval_sample=False, verbose=verbose, twod_attention_mask=True) 
        # metrics["sfc"] = sfc; metrics["icl_sfc"] = icl_sfc; metrics["demonstrations"] = True;  metrics["mask_demonstrations"] = True; metrics["is_test"] = is_test
        # logger.info(metrics)
        # all_metrics.append(metrics)
        # write_predictions_to_file(preds, os.path.join(self.args.output_dir, "mask-demonstrations.jsonl"))
        
        return all_metrics
    
    def eval_multiple_settings_constructed_model(self, task, train_samples, eval_samples, test_samples=None, verbose=False):
        is_test = False
        if test_samples is not None:
            eval_samples = test_samples; is_test = True
            
        all_metrics = []

        self.model.eval()

        sfc = self.args.sfc; icl_sfc = self.args.icl_sfc
        if self.args.single_FT:
            self.args.mask_demonstration_eval = True; self.args.position_modify_eval = True
            preds, metrics = self.evaluate(task, train_samples, eval_samples, one_train_set_per_eval_sample=False, verbose=verbose, twod_attention_mask=False) 
            metrics["sfc"] = sfc; metrics["icl_sfc"] = icl_sfc; metrics["demonstrations"] = False;  metrics["mask_demonstrations"] = False; metrics["is_test"] = is_test 
            logger.info(metrics)
            all_metrics.append(metrics)
            write_predictions_to_file(preds, os.path.join(self.args.output_dir, "no-demonstrations.jsonl"))
        
        else:
            self.args.mask_demonstration_eval = False; self.args.position_modify_eval = False
            preds, metrics = self.evaluate(task, train_samples, eval_samples, one_train_set_per_eval_sample=False, verbose=verbose, twod_attention_mask=False) 
            metrics["sfc"] = sfc; metrics["icl_sfc"] = icl_sfc; metrics["demonstrations"] = True;  metrics["mask_demonstrations"] = False; metrics["is_test"] = is_test
            logger.info(metrics)
            all_metrics.append(metrics)
            write_predictions_to_file(preds, os.path.join(self.args.output_dir, "mask-demonstrations.jsonl"))
        
        return None, all_metrics
    
    def split_sequence(self, task, input_ids):
        seq_len = len(input_ids)
        train_sep_id = self.tokenizer.encode("I" + task.train_sep + "I", add_special_tokens=False)[1:-1]
        train_sep_id_string = " ".join([str(v) for v in train_sep_id])
        exs = [ex.strip() for ex in " ".join([str(v) for v in input_ids]).split(train_sep_id_string)]
        exs = [[int(v) for v in ex.split()] for ex in exs]
        train_lens = sum(len(ex) for ex in exs[:-1]) + len(exs[:-1]) * len(train_sep_id)
        attention_mask = torch.ones((1, seq_len))
        attention_mask[0, :train_lens] = 0
        return attention_mask
        
    def forward(self, input_ids, option_len=None, generation=False, gradient_mask=None, attention_mask=None):
        """
        Given input_ids and the length of the option, return the log-probability of the option (each token)
        """
        input_ids = torch.tensor([input_ids]).to(self.model.device)

        if generation:
            args = self.args
            outputs = self.model.generate(
                input_ids, do_sample=args.sampling, temperature=args.temperature, 
                num_beams=args.num_beams, top_p=args.top_p, top_k=args.top_k, max_new_tokens=min(args.max_new_tokens, args.max_length - input_ids.size(1)), 
                num_return_sequences=1, eos_token_id=[self.tokenizer.encode(args.eos_token, add_special_tokens=False)[0], self.tokenizer.eos_token_id],
            )
            output_text = self.tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True).strip()
            return output_text
        else:
            with torch.inference_mode():
                self.model.eval()
                # this is all for opt 
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            labels = input_ids[0, 1:]
            logits = logits[0, :-1] 
            log_probs = F.log_softmax(logits, dim=-1)

            selected_log_probs = log_probs[torch.arange(len(labels)).to(labels.device), labels]
            selected_log_probs = selected_log_probs.cpu().detach()
            return selected_log_probs[-option_len:]

    def one_step_pred(self, task, train_samples, eval_sample, verbose=False, add_twod_attention_mask=False):
        """
        Given the training samples and the evaluation sample, return the prediction
        """
        if verbose:
            logger.info("========= Example =========")
            logger.info(f"Candidate: {eval_sample[0].candidates}")
        
        ######################### Construction code introduced by Abhishek ###################    
        if self.is_constructed:
            kwargs={'restrict_attention_demonstration': self.args.restrict_attention_demonstration,\
                    'position_modify_demonstration': self.args.position_modify_demonstration,\
                    'mask_demonstration_eval': self.args.mask_demonstration_eval,\
                    'position_modify_eval': self.args.position_modify_eval,}
            encode_fn = encode_prompt_with_construction
            # encode (add prompt and tokenize) the sample; if multiple-choice/classification, encode all candidates (options)
            encoded_candidates, \
            option_lens, \
            label_mask, \
            icl_mask, \
            position_ids = encode_fn(task, task.get_template(self.args.template_id), train_samples, eval_sample, self.tokenizer, max_length=self.args.max_length, generation=task.generation, **kwargs)

            if self.args.sfc or self.args.icl_sfc:
                sfc_encoded_candidates, \
                sfc_option_lens, \
                sfc_label_mask, \
                sfc_icl_mask, \
                sfc_position_ids = encode_fn(task, task.get_template(self.args.template_id), train_samples, eval_sample, self.tokenizer, max_length=self.args.max_length, sfc=self.args.sfc, icl_sfc=self.args.icl_sfc, generation=task.generation, **kwargs)
            
  
        else:
        ######################### Construction code introduced by Abhishek ###################
            encoded_candidates, option_lens, label_mask = encode_prompt(task, task.get_template(self.args.template_id), train_samples, eval_sample, self.tokenizer, max_length=self.args.max_length, generation=task.generation)
            if self.args.sfc or self.args.icl_sfc:
                sfc_encoded_candidates, sfc_option_lens, sfc_label_mask = encode_prompt(task, task.get_template(self.args.template_id), train_samples, eval_sample, self.tokenizer, max_length=self.args.max_length,
                    sfc=self.args.sfc, icl_sfc=self.args.icl_sfc, generation=task.generation
                )
        # calculate the probabilities of all candidates
        outputs = []
        if task.generation:
            output_text = self.forward(encoded_candidates[0], generation=True)
            if verbose:
                logger.info("=== Prompt ===")
                logger.info(self.tokenizer.decode(encoded_candidates[0]))
                logger.info(f"Output: {output_text}") 
            return Prediction(correct_candidate=eval_sample.correct_candidate, predicted_candidate=output_text)
        else:
            label_ids = torch.tensor([encoded_candidate[-1] for encoded_candidate in encoded_candidates])
            # print("label_ids", label_ids)

            for candidate_id, encoded_candidate in enumerate(encoded_candidates):
                forward_func = self.forward if not self.is_constructed else self.constructed_model_forward
                ########### Abhishek: Separated the calls for ease! ###########
                if not self.is_constructed:
                    gradient_mask = None
                    if add_twod_attention_mask:
                        attention_mask = self.split_sequence(task, encoded_candidate)
                    else:
                        attention_mask = None
                    selected_log_probs = forward_func(encoded_candidate, option_len=option_lens[candidate_id], gradient_mask=gradient_mask, attention_mask=attention_mask)

                else:
                    ######################### Construction code introduced by Abhishek ###################
                    gradient_mask = label_mask[candidate_id] if self.args.loss_label_only else None
                    icl_mask_cand      = icl_mask[candidate_id] 
                    position_ids_cand  = position_ids[candidate_id] 

                    kwargs = {'gradient_mask': gradient_mask, \
                              'icl_mask': icl_mask_cand, \
                              'position_ids': position_ids_cand, \
                              'label_ids': label_ids
                             }

                    selected_log_probs = forward_func(encoded_candidate, option_len=option_lens[candidate_id], **kwargs)
                    ######################### Construction code introduced by Abhishek ###################

                if verbose:
                    if candidate_id == 0:
                        logger.info("=== Candidate %d ===" % candidate_id)
                        logger.info(self.tokenizer.decode(encoded_candidate))
                    else:
                        logger.info("=== Candidate %d (without context) ===" % candidate_id)
                        logger.info(self.tokenizer.decode(encoded_candidate).split(task.train_sep)[-1])
                    logger.info(f"Log probabilities of the option tokens: {selected_log_probs}")

                if self.args.sfc or self.args.icl_sfc:
                    if not self.is_constructed:
                        gradient_mask = None
                        sfc_selected_log_probs = forward_func(sfc_encoded_candidates[candidate_id], option_len=sfc_option_lens[candidate_id], gradient_mask=gradient_mask)
                    else:
                        ######################### Construction code introduced by Abhishek ###################
                        gradient_mask = sfc_label_mask[candidate_id] if self.args.loss_label_only else None
                        icl_mask_cand      = sfc_icl_mask[candidate_id] if self.is_constructed else None
                        position_ids_cand  = sfc_position_ids[candidate_id] if self.is_constructed else None
                    
                        kwargs = {'gradient_mask': gradient_mask, \
                                  'icl_mask': icl_mask_cand, \
                                  'position_ids': position_ids_cand, \
                                  'label_ids': label_ids
                                 }

                        sfc_selected_log_probs = forward_func(sfc_encoded_candidates[candidate_id], option_len=sfc_option_lens[candidate_id], **kwargs)
                        ######################### Construction code introduced by Abhishek ###################
                        
                    if verbose:
                        logger.info("=== Candidate %d (without context) SFC ===" % candidate_id)
                        logger.info(self.tokenizer.decode(sfc_encoded_candidates[candidate_id]).split(task.train_sep)[-1])
                        logger.info(f"Log probabilities of the option tokens: {sfc_selected_log_probs}")

                if self.is_constructed:
                    for i in range(len(selected_log_probs)):
                        log_probs = selected_log_probs[i]
                        sfc_log_probs = sfc_selected_log_probs[i] if self.args.sfc or self.args.icl_sfc else None
                        outputs.append({"log_probs": log_probs, "sfc_log_probs": sfc_log_probs if self.args.sfc or self.args.icl_sfc else None})
                    break
                else:
                    outputs.append({"log_probs": selected_log_probs, "sfc_log_probs": sfc_selected_log_probs if self.args.sfc or self.args.icl_sfc else None})
                    

            if self.args.sfc or self.args.icl_sfc:
                scores = [x['log_probs'].sum().item() - x['sfc_log_probs'].sum().item() for x in outputs]
            else:
                scores = [x['log_probs'].mean().item() for x in outputs]

            # if verbose:
            logger.info(f"Prediction scores: {scores}")

            if isinstance(eval_sample.correct_candidate, list):
                correct_candidate_id = [eval_sample.candidates.index(c) for c in eval_sample.correct_candidate]
            else:
                correct_candidate_id = eval_sample.candidates.index(eval_sample.correct_candidate)
            return Prediction(correct_candidate=correct_candidate_id, predicted_candidate=int(np.argmax(scores)), scores=scores)

    
    def evaluate(self, task, train_samples, eval_samples, one_train_set_per_eval_sample=False, verbose=True, twod_attention_mask=False):
        if one_train_set_per_eval_sample:
            logger.info(f"There are {len(eval_samples)} validation samples and one train set per eval sample")
        else:
            logger.info(f"There are {len(train_samples)} training samples and {len(eval_samples)} validation samples")

        # prediction loop
        predictions = []  
        for eval_id, eval_sample in enumerate(tqdm(eval_samples)):
            predictions.append(self.one_step_pred(task, train_samples, eval_sample, verbose=(eval_id < 3) & verbose, add_twod_attention_mask=twod_attention_mask))

        # calculate metrics (only support acc for now)
        metric_name = getattr(task, "metric_name", "accuracy")
        metrics = {}
        if isinstance(metric_name, list):
            for metric in metric_name:
                metrics[metric] = calculate_metric(predictions, metric)
        else:
            metrics = {metric_name: calculate_metric(predictions, metric_name)}
        metrics["eval_info"] = {"model_name": self.args.model_name, "model_path": self.args.model_path, "task": task.get_task_name(), "num_train": self.args.num_train, "num_eval": len(eval_samples)}
        return predictions, metrics
    
def result_file_tag(args, task_name):
    save_model_name = args.model_name
    sfc_tag = "-sfc" if args.sfc else ""
    icl_sfc_tag = "-icl_sfc" if args.icl_sfc else ""
    loss_label_only = "-loss_label_only" if args.loss_label_only else ""
    unbiased_sampling = "-unbiased_sampling" if args.loss_label_only else ""
    sample_eval_tag = "-sampleeval%d" % args.num_eval if args.num_eval is not None else ""
    sample_train_tag = "-ntrain%d" % args.num_train if args.num_train > 0 else ""
    customized_tag = f"-{args.tag}" if len(args.tag) > 0 else ""
    
    ######################### Additional flags introduced by Abhishek ###################
    restrict_attention_traintag = "-restrict_attention_train" if args.restrict_attention_demonstration else ""
    position_modify_traintag = "-position_modify_train" if args.position_modify_demonstration else "" 
    mask_evaltag = "-masktrain_duringeval" if args.mask_demonstration_eval else ""
    position_modify_evaltag = "-position_modify_eval" if args.position_modify_eval else ""
    ######################### Additional flags introduced by Abhishek ###################
    
    return f"{task_name}-{save_model_name}" + sfc_tag + icl_sfc_tag + sample_eval_tag + sample_train_tag + customized_tag + loss_label_only + unbiased_sampling + restrict_attention_traintag + position_modify_traintag + mask_evaltag + position_modify_evaltag

def main():
    args = parse_args()
    # initialize trainer and load model
    
    framework = Framework(args)

    if args.test_mode:
        verbose = True
        args.num_eval = 4 
        
    for task_name in args.task_name: 
        task = get_task(task_name)
        
        test_index = None; test_samples = None 
        if args.test_set_seed is not None:
            test_index, test_samples = task.sample_subset(data_split="valid", seed=args.test_set_seed, num=args.num_eval, include_index=True, unbiased=args.unbiased_sampling)

        train_sets = task.sample_train_sets(num_train=args.num_train, num_eval=args.num_eval, seed=args.train_set_seed, unbiased=args.unbiased_sampling)
        assert args.train_set_seed is not None
        
        # eval samples share one (or multiple) training set(s)
        for train_set_id, train_samples in enumerate(train_sets):
            # get eval samples
            if args.num_eval is not None:
                eval_index, eval_samples = task.sample_subset(data_split="valid", seed=args.eval_set_seed, num=args.num_eval, include_index=True, unbiased=args.unbiased_sampling)
            else:
                eval_index, eval_samples = task.valid_samples
            
            # get test samples    
            test_index = None; test_samples = None 
            if args.test_set_seed is not None:
                test_index, test_samples = task.sample_subset(data_split="valid", seed=args.test_set_seed, num=args.num_eval, exclude=eval_index, include_index=True, unbiased=args.unbiased_sampling)

            add_args = {}       
            
            # whether finetune on the training examples during inference     
            add_args["FT"] = args.dynamic_eval_icl
            add_args["test_samples"] = test_samples
            
            # constructed model
            if framework.is_constructed: 
                func = framework.eval_multiple_settings_constructed_model
                add_args.pop("FT")
            # original model
            else:
                if args.single_FT:
                    logger.info("Use single FT")
                    func = framework.dynamic_eval
                else:
                    logger.info("Use multiple FT")
                    func = framework.dynamic_eval_icl
                
            
            if test_index is not None: 
                assert set(eval_index).isdisjoint(set(test_index)) 
            logger.info(f"Eval index: {eval_index[:10]}")
            if test_index is not None:
                logger.info(f"Test index: {test_index[:10]}")
            predictions, metrics = func(task, train_samples, eval_samples, **add_args)
            
            # report and save results
            logger.info("===== Train set %d =====" % train_set_id)
            logger.info(metrics)
            metric_file = os.path.join(args.output_dir,  result_file_tag(args, task_name) + f"-trainset{train_set_id}.json" if args.result_file is None else args.result_file)
            pred_file = os.path.join(args.output_dir, result_file_tag(args, task_name) + f"-trainset{train_set_id}.pred" if args.pred_file is None else args.pred_file)
            write_metrics_to_file(metrics, metric_file)
            if predictions is not None:
                write_predictions_to_file(predictions, pred_file)

if __name__ == "__main__": 
    main()

