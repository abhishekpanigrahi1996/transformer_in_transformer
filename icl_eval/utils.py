import json
import os
import contextlib
from typing import Optional, Union
import numpy as np
from dataclasses import dataclass, is_dataclass, asdict
import logging
import time
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch
from transformers.utils import PaddingStrategy
from transformers import PreTrainedTokenizerBase
from typing import Optional, Union, List, Dict, Any

logger = logging.getLogger(__name__)

@dataclass
class Prediction:
    correct_candidate: Union[int, str]
    predicted_candidate: Union[int, str]
    scores: List[float] = None

@contextlib.contextmanager
def count_time(name):
    logger.info("%s..." % name)
    start_time = time.time()
    try:
        yield
    finally:
        logger.info("Done with %.2fs" % (time.time() - start_time))

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

def forward_wrap_with_option_len(self, input_ids=None, labels=None, option_len=None, num_options=None, return_dict=None, **kwargs):
    outputs = self.original_forward(input_ids=input_ids, **kwargs)
    if labels is None:
        return outputs
    logits = outputs.logits

    loss = None
    # shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    # here we use input_ids (which should always = labels) bc sometimes labels are correct candidate IDs
    shift_labels = torch.clone(input_ids)[..., 1:].contiguous()
    shift_labels[shift_labels == self.config.pad_token_id] = -100

    # apply option len
    for _i, _len in enumerate(option_len):
        shift_labels[_i, :-_len] = -100

    # calculate the loss
    loss_fct = CrossEntropyLoss(ignore_index=-100)
    if num_options is not None: 
        # Train as classificaiton tasks
        log_probs = F.log_softmax(shift_logits, dim=-1)
        mask = shift_labels != -100 # option part
        shift_labels[~mask] = 0 # so that it doesn't mess up with indexing

        selected_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1) # (bsz x num_options, len)
        selected_log_probs = (selected_log_probs * mask).sum(-1) # (bsz x num_options)

        num_options = num_options[0]
        selected_log_probs = selected_log_probs.view(-1, num_options) # (bsz, num_options)
        labels = labels.view(-1, num_options)[:, 0] # labels repeat so we only take the first one
        loss = loss_fct(selected_log_probs, labels)
    else:
        loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

@dataclass
class DataCollatorWithPaddingAndNesting:

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features = [ff for f in features for ff in f]
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch

    
def encode_prompt(task, template, train_samples, eval_sample, tokenizer, max_length, sfc=False, icl_sfc=False, generation=False, generation_with_gold=False, **kwargs):
    """
    sfc: calibration (surface form competition)
    icl_sfc: calibration (surface form competition) with in-context demonstrations
    """
    train_prompts = [template.verbalize(sample, sample.correct_candidate).strip() for sample in train_samples]
    
    
    ############### New code to include a label mask ################## 
    train_label_positions = []
    correct_label_lengths = []
    prompt = ''
    
    for (sample, sample_cand) in zip( train_samples, train_prompts ):

        example_prompt = (prompt + sample_cand).strip()
        example_sent   = (prompt + template.encode(sample)).strip()
        
        label_len = len(tokenizer.encode(example_prompt)) - len(tokenizer.encode(example_sent))
        train_label_positions += [ len(tokenizer.encode(example_sent)) ]
        correct_label_lengths += [ label_len ]
        prompt = example_prompt + task.train_sep   
    label_mask = None
    ############### New code to include a label mask ##################

    train_prompts = task.train_sep.join(train_prompts).strip()
    
    if sfc or icl_sfc:
        encode_fn = template.encode_sfc; verbalize_fn = template.verbalize_sfc
    else: 
        encode_fn = template.encode; verbalize_fn = template.verbalize 
            
    unverbalized_eval_prompt = encode_fn(eval_sample).strip(' ')
    if not generation:
        verbalized_eval_prompts = [verbalize_fn(eval_sample, cand).strip(' ') for cand in eval_sample.candidates]
        unverbalized_eval_prompt_length = len(tokenizer.encode(unverbalized_eval_prompt))
        option_lens = [(len(tokenizer.encode(verbalized_eval_prompt)) - unverbalized_eval_prompt_length) for verbalized_eval_prompt in verbalized_eval_prompts]

        if sfc or train_prompts == '':
            # without demonstrations
            final_prompts = verbalized_eval_prompts 
        else:
            # with demonstrations
            final_prompts = [(train_prompts + task.train_sep + eval_prompt).lstrip().strip(' ') for eval_prompt in verbalized_eval_prompts]
            
    else:
        assert not sfc and not icl_sfc, "Generation tasks do not support SFC"
        if generation_with_gold:
            verbalized_eval_prompts = [verbalize_fn(eval_sample, eval_sample.correct_candidate)]
            unverbalized_eval_prompt_length = len(tokenizer.encode(unverbalized_eval_prompt))
            option_lens = [(len(tokenizer.encode(verbalized_eval_prompt)) - unverbalized_eval_prompt_length) for verbalized_eval_prompt in verbalized_eval_prompts]
            final_prompts = [(train_prompts + task.train_sep + eval_prompt).lstrip().strip(' ') for eval_prompt in verbalized_eval_prompts] 
        else:
            option_lens = [0]
            final_prompts = [(train_prompts + task.train_sep + unverbalized_eval_prompt).lstrip().strip(' ')]

    # tokenize 
    encodings = [tokenizer.encode(final_prompt) for final_prompt in final_prompts]
    
    ############### New code to include a label mask ##################
    if not generation:
        if sfc:
            label_mask = [np.zeros( len(encoding) ) for encoding in encodings]
        else:    
            label_mask = []
            
            for enc_order in range(len(encodings)):
                mask = np.zeros( len(encodings[enc_order]) )
                for (pos, length) in zip( train_label_positions, correct_label_lengths ):
                    mask [pos: pos+length] = 1.
                label_mask += [mask]
        # label_mask = np.stack(label_mask)
    ############### New code to include a label mask ##################    
    # print (label_mask)    
    # truncate
    if any([len(encoding) > max_length for encoding in encodings]):
        logger.warn("Exceed max length")
    if tokenizer.add_bos_token:
        encodings = [encoding[0:1] + encoding[1:][-(max_length-1):] for encoding in encodings]  
        # label_mask = [lmask[0:1] + lmask[1:][-(max_length-1):] for lmask in label_mask]  
    else:
        encodings = [encoding[-max_length:] for encoding in encodings]  
        # label_mask = [lmask[-max_length:] for lmask in label_mask]  

    return encodings, option_lens, label_mask 


def encode_prompt_with_construction(task, template, train_samples, eval_sample, tokenizer, max_length, sfc=False, icl_sfc=False, generation=False, generation_with_gold=False, **kwargs):
    
    """
    sfc: calibration (surface form competition)
    icl_sfc: calibration (surface form competition) with in-context demonstrations
    """
    
    restrict_attention_demonstration=kwargs['restrict_attention_demonstration']
    position_modify_demonstration=kwargs['position_modify_demonstration']
    mask_demonstration_eval=kwargs['mask_demonstration_eval']
    position_modify_eval=kwargs['position_modify_eval'] 
    
    
    assert restrict_attention_demonstration or not position_modify_demonstration, \
           "Can't change position if we are not restricting attention of each demonstration to itself"
    
    
    assert mask_demonstration_eval or not position_modify_eval, \
           "Can't change position if we are not masking demonstrations"
    
    
    
    train_prompts = [template.verbalize(sample, sample.correct_candidate).strip() for sample in train_samples]
    
    ############### New code to include a label mask, icl mask, and modify positions ##################  
    train_example_start = []
    train_example_end   = []
    
    train_label_positions = []
    correct_label_lengths = []

    prompt = ''
    if tokenizer.add_bos_token: shift_right = 1
    else: shift_right = 0   
    for (sample, sample_cand) in zip( train_samples, train_prompts ):

        example_prompt = (prompt + sample_cand).lstrip()
        example_sent   = (prompt + template.encode(sample)).lstrip()
        
        example_prompt_enc = tokenizer.encode(example_prompt)
        example_sent_enc   = tokenizer.encode(example_sent)
        example_sample_enc = tokenizer.encode(sample_cand.strip())
        
        label_len = len(example_prompt_enc) - len(example_sent_enc)
   
        train_example_start += [len(example_prompt_enc) - len(example_sample_enc) + shift_right]
        train_example_end += [len(example_prompt_enc)]
       
        train_label_positions += [ len(example_sent_enc) ]
        correct_label_lengths += [ label_len ]

        prompt = example_prompt + task.train_sep   
    label_mask = None
    test_example_start = len(tokenizer.encode(prompt + 'end')) - len(tokenizer.encode('end')) + shift_right
    
    
    ############### New code to include a label mask ##################
    
    train_prompts = task.train_sep.join(train_prompts).strip()
    
    if sfc or icl_sfc:
        encode_fn = template.encode_sfc; verbalize_fn = template.verbalize_sfc
    else: 
        encode_fn = template.encode; verbalize_fn = template.verbalize 
            
    unverbalized_eval_prompt = encode_fn(eval_sample).strip(' ')
    if not generation:
        verbalized_eval_prompts = [verbalize_fn(eval_sample, cand).strip(' ') for cand in eval_sample.candidates]
        unverbalized_eval_prompt_length = len(tokenizer.encode(unverbalized_eval_prompt))
        option_lens = [(len(tokenizer.encode(verbalized_eval_prompt)) - unverbalized_eval_prompt_length) for verbalized_eval_prompt in verbalized_eval_prompts]

        if sfc or train_prompts == '':
            # without demonstrations
            final_prompts = verbalized_eval_prompts 
        else:
            # with demonstrations
            final_prompts = [(train_prompts + task.train_sep + eval_prompt).lstrip().strip(' ') for eval_prompt in verbalized_eval_prompts]
            
    else:
        assert not sfc and not icl_sfc, "Generation tasks do not support SFC"
        if generation_with_gold:
            verbalized_eval_prompts = [verbalize_fn(eval_sample, eval_sample.correct_candidate)]
            unverbalized_eval_prompt_length = len(tokenizer.encode(unverbalized_eval_prompt))
            option_lens = [(len(tokenizer.encode(verbalized_eval_prompt)) - unverbalized_eval_prompt_length) for verbalized_eval_prompt in verbalized_eval_prompts]
            final_prompts = [(train_prompts + task.train_sep + eval_prompt).lstrip().strip(' ') for eval_prompt in verbalized_eval_prompts] 
        else:
            option_lens = [0]
            final_prompts = [(train_prompts + task.train_sep + unverbalized_eval_prompt).lstrip().strip(' ')]

    # tokenize 
    encodings = [tokenizer.encode(final_prompt) for final_prompt in final_prompts]
    
    ############### New code to include a label mask ##################
    if not generation:
        if sfc:
            label_mask = [np.zeros( len(encoding) ) for encoding in encodings]
        else:    
            label_mask = []
            
            for enc_order in range(len(encodings)):
                mask = np.zeros( len(encodings[enc_order]) )
                for (pos, length) in zip( train_label_positions, correct_label_lengths ):
                    mask [pos: pos+length] = 1.
                label_mask += [mask]
        label_mask = np.stack(label_mask)
    ############### New code to include a label mask ##################    
    
    ############### New code to include a icl mask and position ids##################
    icl_mask = [ np.ones( (len(encoding), len(encoding)) ) for encoding in encodings ]
    if restrict_attention_demonstration:
        for (start, end) in zip(train_example_start, train_example_end):
            for i in range(len(encodings)):
                icl_mask[i][start:end, start:end] = 1.
                icl_mask[i][start:end, :start]    = 0.
                icl_mask[i][start:end, end:]    = 0.
  
    if mask_demonstration_eval:
        for i in range(len(encodings)):
            icl_mask[i][test_example_start:, test_example_start:] = 1.
            icl_mask[i][test_example_start:, :test_example_start]   = 0.
        
    if  tokenizer.add_bos_token: 
        for i in range(len(encodings)): icl_mask[i][:, 0] = 1.
        
    position_ids = [ np.arange( len(encoding) ) for encoding in encodings   ]    
    if position_modify_demonstration:
        if  tokenizer.add_bos_token: start_id = 1
        else: start_id = 0
            
        for (start, end) in zip(train_example_start, train_example_end):
            example_length = end - start
            for i in range(len(encodings)):
                position_ids [i][start:end] = np.arange( start_id, start_id + example_length )
            
    if position_modify_eval:  
        if  tokenizer.add_bos_token: start_id = 1
        else: start_id = 0
        
        for i in range( len(encodings) ):
            start = test_example_start
            end   = len(encodings[i])
            
            example_length = end - start
            position_ids [i][start:end] = np.arange( start_id, start_id + example_length )

    ############### New code to include a icl mask and position ids ##################
    
    
    # truncate
    if any([len(encoding) > max_length for encoding in encodings]):
        logger.warn("Exceed max length")
    if tokenizer.add_bos_token:
        
        new_position_ids = []
        new_label_mask = []
        new_icl_mask = []
        for i in range(  len(encodings) ):
            max_len = min(max_length, len(encodings[i]))
            
            nicm = np.zeros((max_len, max_len))
            nicm [ -(max_len-1):, -(max_len-1): ] = icl_mask[i] [-(max_len-1):, -(max_len-1):]
            nicm [ :, 0 ] = 1.
            
            new_icl_mask += [nicm]
            new_position_ids += [ np.concatenate( [np.asarray([0]), position_ids[i][ -(max_len-1): ] ], axis=0 ) ]
            new_label_mask   += [ np.concatenate( [np.asarray([label_mask[i][0]]), label_mask[i][ -(max_len-1): ] ], axis=0 ) ]
        icl_mask = new_icl_mask
        position_ids = new_position_ids
        label_mask = new_label_mask
       
        encodings = [encoding[0:1] + encoding[1:][-(max_length-1):] for encoding in encodings]  
    else:
        encodings = [encoding[-max_length:] for encoding in encodings]  
        label_mask = [lmask[-max_length:] for lmask in label_mask]  
        position_ids = [ position_id[ -max_length: ] for position_id in position_ids]
        icl_mask = [ icm[-max_length:, -max_length:] for icm in icl_mask ]
    
    return encodings, option_lens, label_mask, icl_mask, position_ids

 
def load_generation():
    out_dir = "/scratch/gpfs/mengzhou/space6/out/the_pile_corrected"
    json_files = [f"{out_dir}/ft/ft_opt-125m-lr1e-4/generation_downstream/ft_opt-125m-lr1e-4-hf-sample-0.90-len20-num5-copa.json"]
    for model in ["350m", "1.3b", "2.7b"]:
        json_files.append(f"{out_dir}/kd_pretrained_ce1_layer0_bs512/kd_pretrained_temp1_tmodelft{model}_lr1e-4/generation_downstream/kd_pretrained_temp1_tmodelft{model}_lr1e-4-hf-sample-0.90-len20-num5-copa.json")
    
    i = 0
    for json_file in json_files[3:4]:
        name = os.path.basename(os.path.dirname(os.path.dirname(json_file)))
        generations = json.load(open(json_file))
        gen = generations[i]
        print(name)
        print("\033[1mprefix\033[0m:", gen["prefix"])
        print("\033[1mcorrect_options\033[0m:", gen["correct_options"])
        for i in range(len(gen["incorrect_options"])):
            print("\033[1mincorrect_options\033[0m" + f" {i}:", end=" ")
            print(gen["incorrect_options"][i])
        for i in range(len(gen["generated"])):
            print("\033[1mgenerated_option\033[0m" + f" {i}:", end=" ") 
            print(f"[{round(gen['scores'][i], 2)}]:", end=" ")
            print(gen["generated"][i])

def read_jsonl(file):
    ds = []
    try:
        with open(file) as f:
            for i, line in enumerate(f):
                d = json.loads(line.strip())
                ds.append(d)
    except:
        import pdb
        pdb.set_trace()
    return ds


from collections.abc import Mapping
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
InputDataClass = NewType("InputDataClass", Any)
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import torch
@dataclass
class ICLCollator:
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(features[0], Mapping):
            features = [vars(f) for f in features]
        first = features[0]
        batch = {}
        
        pad_id = self.tokenizer.pad_token_id

        pad_ids = {"input_ids": pad_id, "attention_mask": 0, "sfc_input_ids": pad_id, "sfc_attention_mask": 0, "labels": pad_id}
        for key in first:
            pp = pad_ids[key]
            lens = [len(f[key]) for f in features]
            max_len = max(lens)
            feature = np.stack([np.pad(f[key], (0, max_len - lens[i]), "constant", constant_values=(0, pp)) for i, f in enumerate(features)])
            padded_feature = torch.from_numpy(feature).long()
            batch[key] = padded_feature
            
        return batch


import json
class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if is_dataclass(o):
            return asdict(o)
        return super().default(o)

def write_predictions_to_file(final_preds, output):
    with open(output, "w") as f:
        for pred in final_preds:
            f.write(json.dumps(pred, cls=EnhancedJSONEncoder) + "\n")

def write_metrics_to_file(metrics, output):
    with open(output, "w") as f:
        if type(metrics) == list:
            for metric in metrics:
                json.dump(metric, f, cls=EnhancedJSONEncoder)
                f.write("\n")
        else:
            json.dump(metrics, f, cls=EnhancedJSONEncoder, indent=4)
        
if __name__ == "__main__":
    load_generation()     
    