from templates import *
from utils import read_jsonl, temp_seed
import json
import os
from datasets import load_dataset, load_from_disk
from dataclasses import dataclass
from typing import List, Union
import string
import random
import datasets
import sys
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DATA_DIR = "/scratch/gpfs/mengzhou/space6/data"
META_DATA_DIR = "/scratch/gpfs/mengzhou/meta-internship-2022/data"
CACHE_DIR = "/scratch/gpfs/mengzhou/.cache/datasets"


# def load_dataset(*args):
#     return load_dataset(*args, cache_dir=CACHE_DIR)

def get_task(task_name):
    aa = task_name.split("__")
    if len(aa) == 2:
        task_group, subtask = aa
    else:
        task_group = aa[0]
        subtask = None
    class_ = getattr(sys.modules[__name__], f"{task_group}Dataset")
    instance = class_(subtask)
    return instance

@dataclass
class Sample:
    id: int = None
    data: dict = None
    correct_candidate: Union[str, List[str]] = None
    candidates: List[str] = None
    
class Dataset:
    mixed_set = False
    train_sep = "\n\n"
    generation = False # whether this is a generation task

    def __init__(self, subtask=None, **kwargs) -> None:
        self.subtask = subtask
    
    def get_task_name(self):
        task_name = self.__class__.__name__[:-7]
        if self.subtask is not None:
            task_name += f"__{self.subtask}"
        return task_name
        
    def load_dataset():
        raise NotImplementedError
    
    def get_template(self, template_version=0):
       templates = {0: Template}
       return templates[template_version]
   
    def build_sample(self, example):
        return 
     
    def sample_train_sets(self, num_train=32, num_dev=None, num_eval=None, num_train_sets=None, seed=None, unbiased=False):
        if seed is not None:
            # one train/demo set using the designated seed
            seeds = [seed]
        elif num_train_sets is not None:
            # num_train_sets train/demo sets
            seeds = list(range(num_train_sets))
        else: 
            # one train/demo set per evaluation sample
            assert num_dev is None # not supported
            len_valid_samples = len(self.samples["valid"]) if num_eval is None else num_eval
            with temp_seed(0):
                seeds = np.random.randint(0, 10000, len_valid_samples)

        train_samples = [] 
        for i, set_seed in enumerate(seeds):
            if self.mixed_set:
                # raise NotImplementedError
                # TODO: fix bug for mixed set sampling
                train_samples.append(self.sample_subset(data_split="valid", seed=set_seed, num=num_train, exclude=[i], unbiased=unbiased))
            else:
                if num_dev is not None:
                    train_samples.append(self.sample_subset(data_split="train", seed=set_seed, num=num_train+num_dev, unbiased=unbiased)) # dev set is included at the end of train set
                    if num_train + num_dev > len(self.samples["train"]):
                        logger.warn("num_train + num_dev > available training examples")
                else:
                    train_samples.append(self.sample_subset(data_split="train", seed=set_seed, num=num_train, unbiased=unbiased))
                if num_dev is not None:
                    logger.info(f"Sample train set {len(train_samples[-1])}/{len(self.samples['train'])}")
                    logger.info(f"... including dev set {num_dev} samples")
        return train_samples

    def sample_subset(self, data_split="train", seed=0, num=100, exclude=None, unbiased=False, include_index=False):
        with temp_seed(seed):
            samples = self.samples[data_split] 
            lens = len(samples)
            index = np.random.permutation(lens).tolist()
            #[:num if exclude is None else num+1]
            if unbiased:
                new_index = []
                unique_labels = set( [ sample.correct_candidate for sample in samples ])
                
                 
                assert num % len(unique_labels) == 0, \
                    f"Number of training examples ({num}) should be divisible by number of classes to be fair (otherwise don't bother)!" 
                
                examples_per_label = num // len(unique_labels)
                count_labels = {}
                
                for label in unique_labels:
                    count_labels[label] = examples_per_label
                
                for i in index:
                    if exclude is not None and i in exclude: continue
                    label = samples[i].correct_candidate
                    if count_labels[label] > 0:
                        new_index += [i]
                        count_labels[label] -= 1
                        if len(new_index) == num: break    
                index = new_index            
            else:
                index = index[:num if exclude is None else num+1]
            
            if exclude is not None:
                index = list(set(index) - set(exclude))
            else:
                index = index[:num]
            if include_index:
                return index, [samples[i] for i in index]
            else:
                return [samples[i] for i in index]
    
    @property
    def valid_samples(self):
        return self.samples["valid"]
    
class SST2Dataset(Dataset):
    train_sep = "\n\n"
    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)
        
    def load_dataset(self, path, **kwargs):
        d = load_dataset('glue', 'sst2')
        train_d = d["train"]
        validation_d = d["validation"]
        
        train_samples = [self.build_sample(example) for example in train_d]
        valid_samples = [self.build_sample(example) for example in validation_d]
        
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    # for generative tasks, candidates are []
    def build_sample(self, example):
        label = int(example["label"])
        return Sample(id=example["idx"], data=example, correct_candidate=label, candidates=[0, 1])
        
    def get_template(self, template_version=0):
        return {0: SST2v2Template, 1: SST2SyntheticTemplate, 2: SST2Template}[template_version]()

class AmazonPolarityDataset(Dataset):
    train_sep = "\n\n"
    
    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        # d = load_dataset("amazon_polarity") 
        d = load_from_disk("/scratch/gpfs/mengzhou/space6/data/amazon_polarity_short.pt")
        train_set = d["train"]
        valid_set = d["test"]
        
        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        example["sentence"] = example["content"]
        label = int(example["label"])
        return Sample(data=example, correct_candidate=label, candidates=[0, 1])
        
    def get_template(self, template_version=0):
        return {0: SST2v2Template, 1: SST2SyntheticTemplate, 2: SST2Template}[template_version]()

    
    
class YelpPolarityDataset(Dataset):
    train_sep = "\n\n"
    
    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        # d = load_dataset("yelp_polarity") 
        d = load_from_disk("/scratch/gpfs/mengzhou/space6/data/yelp_polarity_short.pt")
        train_set = d["train"]
        valid_set = d["test"]
        
        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        example["sentence"] = example["text"]
        label = int(example["label"])
        return Sample(data=example, correct_candidate=label, candidates=[0, 1])
        
    def get_template(self, template_version=0):
        return {0: SST2v2Template, 1: SST2SyntheticTemplate, 2: SST2Template}[template_version]()

      
class CRDataset(Dataset):
    train_sep = "\n\n"
    
    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        train_set = []
        for line in open("/scratch/gpfs/ap34/icl-as-ft/llm_eval/data/cr/train.jsonl", 'r'):
            train_set += [ json.loads(line.strip()) ]
        
        valid_set = []
        for line in open("/scratch/gpfs/ap34/icl-as-ft/llm_eval/data/cr/test.jsonl", 'r'):
            valid_set += [ json.loads(line.strip()) ]
        
        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        example["sentence"] = example["text"]
        label = int(example["label"])
        return Sample(data=example, correct_candidate=label, candidates=[0, 1])
        
    def get_template(self, template_version=0):
        return {0: SST2v2Template, 1: SST2SyntheticTemplate, 2: SST2Template}[template_version]()
    
class MRDataset(Dataset):
    train_sep = "\n\n"
    
    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        import pandas as pd
        train_set = []
        for line in pd.read_csv("/scratch/gpfs/ap34/icl-as-ft/llm_eval/data/mr/train.csv", header=None).values.tolist():
            label, text = line
            exmaple = {}
            exmaple['label'] = int(label)
            exmaple['text'] = text
            train_set += [ exmaple ]
        
        valid_set = []
        for line in pd.read_csv("/scratch/gpfs/ap34/icl-as-ft/llm_eval/data/mr/test.csv", header=None).values.tolist():
            label, text = line
            exmaple = {}
            exmaple['label'] = int(label)
            exmaple['text'] = text
            valid_set += [ exmaple ]
            
        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        example["sentence"] = example["text"]
        label = int(example["label"])
        return Sample(data=example, correct_candidate=label, candidates=[0, 1])
        
    def get_template(self, template_version=0):
        return {0: SST2v2Template, 1: SST2SyntheticTemplate, 2: SST2Template}[template_version]()
    
    
class MPQADataset(Dataset):
    train_sep = "\n\n"
    
    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        import pandas as pd
        train_set = []
        for line in pd.read_csv("/scratch/gpfs/ap34/icl-as-ft/llm_eval/data/mpqa/train.csv", header=None).values.tolist():
            label, text = line
            exmaple = {}
            exmaple['label'] = int(label)
            exmaple['text'] = text
            train_set += [ exmaple ]
        
        valid_set = []
        for line in pd.read_csv("/scratch/gpfs/ap34/icl-as-ft/llm_eval/data/mpqa/test.csv", header=None).values.tolist():
            label, text = line
            exmaple = {}
            exmaple['label'] = int(label)
            exmaple['text'] = text
            valid_set += [ exmaple ]
        
        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        example["sentence"] = example["text"]
        label = int(example["label"])
        return Sample(data=example, correct_candidate=label, candidates=[0, 1])
        
    def get_template(self, template_version=0):
        return {0: SST2v2Template, 1: SST2SyntheticTemplate, 2: SST2Template}[template_version]()
        
class imdbDataset(Dataset):
    train_sep = "\n\n"
    
    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        d = load_dataset("imdb") 
        train_set = d["train"]
        valid_set = d["test"]
        
        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        example["sentence"] = example["text"]
        label = int(example["label"])
        return Sample(data=example, correct_candidate=label, candidates=[0, 1])
        
    def get_template(self, template_version=0):
        return {0: SST2v2Template, 1: SST2SyntheticTemplate, 2: SST2Template}[template_version]()
    
          
class StoryclozeDataset(Dataset):
    train_sep = "\n\n"
    mixed_set = True

    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)
        
    def load_dataset(self, path, **kwargs):
        import pandas as pd
        d = pd.read_csv(f"/scratch/gpfs/mengzhou/space6/data/storycloze/storycloze_winter2018_val.csv") 
    
        train_samples = [self.build_sample(d.iloc[i].to_dict()) for i in range(len(d))]
        valid_samples = [self.build_sample(d.iloc[i].to_dict()) for i in range(len(d))]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    # for generative tasks, candidates are []
    def build_sample(self, example):
        sample = \
            Sample(
                id=example["InputStoryid"],
                data=example,
                candidates=[example["RandomFifthSentenceQuiz1"], example["RandomFifthSentenceQuiz2"]],
                correct_candidate=example[f"RandomFifthSentenceQuiz{example['AnswerRightEnding']}"],
            )
        
        return sample
        
    def get_template(self, template_version=0):
        return {0: StoryclozeTemplate}[template_version]()
    
class CopaDataset(Dataset):
    train_sep = "\n\n"
    mixed_set = False

    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)
        
    def load_dataset(self, path, **kwargs):
        train_examples = load_dataset('super_glue', "copa")["train"]
        valid_examples = load_dataset('super_glue', "copa")["validation"]
    
        train_samples = [self.build_sample(example) for example in train_examples]
        valid_samples = [self.build_sample(example) for example in valid_examples]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    # for generative tasks, candidates are []
    def build_sample(self, example):
        sample = \
            Sample(
                id=example["idx"],
                data=example,
                candidates=[example["choice1"], example["choice2"]],
                correct_candidate=example[f"choice{example['label'] + 1}"],
            )
        
        return sample
        
    def get_template(self, template_version=0):
        return {0: CopaTemplate}[template_version]()
    
class HellaswagDataset(Dataset):
    train_sep = "\n\n"
    mixed_set = False

    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)
        
    def load_dataset(self, path, **kwargs):
        train_examples = load_dataset('hellaswag')["train"]
        valid_examples = load_dataset('hellaswag')["validation"]
    
        train_samples = [self.build_sample(example) for example in train_examples]
        valid_samples = [self.build_sample(example) for example in valid_examples]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        return Sample(
                id=example["ind"],
                data=example,
                candidates=example["endings"],
                correct_candidate=example["endings"][int(example["label"])],
            )
        
    def get_template(self, template_version=0):
        return {0: HellaswagTemplate}[template_version]()    
    
class BigbenchDataset(Dataset):
    train_sep = "\n\n"
    mixed_set = True
    subtasks =  ['abstract_narrative_understanding', 'anachronisms', 'analogical_similarity', 'analytic_entailment', 'arithmetic', 'ascii_word_recognition', 'authorship_verification', 'auto_categorization', 'auto_debugging', 'bbq_lite_json', 'bridging_anaphora_resolution_barqa', 'causal_judgment', 'cause_and_effect', 'checkmate_in_one', 'chess_state_tracking', 'chinese_remainder_theorem', 'cifar10_classification', 'code_line_description', 'codenames', 'color', 'common_morpheme', 'conceptual_combinations', 'conlang_translation', 'contextual_parametric_knowledge_conflicts', 'crash_blossom', 'crass_ai', 'cryobiology_spanish', 'cryptonite', 'cs_algorithms', 'dark_humor_detection', 'date_understanding', 'disambiguation_qa', 'discourse_marker_prediction', 'disfl_qa', 'dyck_languages', 'elementary_math_qa', 'emoji_movie', 'emojis_emotion_prediction', 'empirical_judgments', 'english_proverbs', 'english_russian_proverbs', 'entailed_polarity', 'entailed_polarity_hindi', 'epistemic_reasoning', 'evaluating_information_essentiality', 'fact_checker', 'fantasy_reasoning', 'few_shot_nlg', 'figure_of_speech_detection', 'formal_fallacies_syllogisms_negation', 'gem', 'gender_inclusive_sentences_german', 'general_knowledge', 'geometric_shapes', 'goal_step_wikihow', 'gre_reading_comprehension', 'hhh_alignment', 'hindi_question_answering', 'hindu_knowledge', 'hinglish_toxicity', 'human_organs_senses', 'hyperbaton', 'identify_math_theorems', 'identify_odd_metaphor', 'implicatures', 'implicit_relations', 'intent_recognition', 'international_phonetic_alphabet_nli', 'international_phonetic_alphabet_transliterate', 'intersect_geometry', 'irony_identification', 'kanji_ascii', 'kannada', 'key_value_maps', 'known_unknowns', 'language_games', 'language_identification', 'linguistic_mappings', 'linguistics_puzzles', 'list_functions', 'logic_grid_puzzle', 'logical_args', 'logical_deduction', 'logical_fallacy_detection', 'logical_sequence', 'mathematical_induction', 'matrixshapes', 'metaphor_boolean', 'metaphor_understanding', 'minute_mysteries_qa', 'misconceptions', 'misconceptions_russian', 'mnist_ascii', 'modified_arithmetic', 'moral_permissibility', 'movie_dialog_same_or_different', 'movie_recommendation', 'mult_data_wrangling', 'multiemo', 'natural_instructions', 'navigate', 'nonsense_words_grammar', 'novel_concepts', 'object_counting', 'odd_one_out', 'operators', 'paragraph_segmentation', 'parsinlu_qa', 'parsinlu_reading_comprehension', 'penguins_in_a_table', 'periodic_elements', 'persian_idioms', 'phrase_relatedness', 'physical_intuition', 'physics', 'physics_questions', 'play_dialog_same_or_different', 'polish_sequence_labeling', 'presuppositions_as_nli', 'qa_wikidata', 'question_selection', 'real_or_fake_text', 'reasoning_about_colored_objects', 'repeat_copy_logic', 'rephrase', 'riddle_sense', 'ruin_names', 'salient_translation_error_detection', 'scientific_press_release', 'semantic_parsing_in_context_sparc', 'semantic_parsing_spider', 'sentence_ambiguity', 'similarities_abstraction', 'simp_turing_concept', 'simple_arithmetic_json', 'simple_arithmetic_json_multiple_choice', 'simple_arithmetic_json_subtasks', 'simple_arithmetic_multiple_targets_json', 'simple_ethical_questions', 'simple_text_editing', 'snarks', 'social_iqa', 'social_support', 'sports_understanding', 'strange_stories', 'strategyqa', 'sufficient_information', 'suicide_risk', 'swahili_english_proverbs', 'swedish_to_german_proverbs', 'symbol_interpretation', 'temporal_sequences', 'tense', 'timedial', 'topical_chat', 'tracking_shuffled_objects', 'understanding_fables', 'undo_permutation', 'unit_conversion', 'unit_interpretation', 'unnatural_in_context_learning', 'vitaminc_fact_verification', 'what_is_the_tao', 'which_wiki_edit', 'winowhy', 'word_sorting', 'word_unscrambling']
    
    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)
        
    def load_dataset(self, path, **kwargs):
        d = load_dataset('bigbench', path)["default"]
        d = d.filter(lambda x: len(x["multiple_choice_targets"]) > 0)
    
        train_samples = [self.build_sample(example) for example in d]
        valid_samples = [self.build_sample(example) for example in d]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    # for generative tasks, candidates are []
    def build_sample(self, example):
        index = example["multiple_choice_scores"].index(1)
        sample = \
            Sample(
                id=example["idx"],
                data=example,
                candidates=[a for a in example["multiple_choice_targets"]],
                correct_candidate=example["multiple_choice_targets"][index],
            )
        
        return sample
        
    def get_template(self, template_version=0):
        return {0: BigbenchTemplate}[template_version]()
    
class ValidationDataset(Dataset):
    data_dir = f"{META_DATA_DIR}/valid"
    train_sep = " "
    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)
        
    def load_dataset(self, path, **kwargs):
        file_name = os.path.join(self.data_dir, path, "00", f"{path}.jsonl")
        d = read_jsonl(file_name)
        
        for id, example in enumerate(d):
            example["id"] = id
            
        train_samples = [self.build_sample(example) for example in d]
        valid_samples = [self.build_sample(example) for example in d]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        text = " ".join(example["text"].split(" "))
        sample = \
            Sample(
                id=example["id"],
                data=example,
                candidates=[text],
                correct_candidate=text,
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: NoTemplate}[template_version]()
    
class StrategyQADataset(Dataset):
    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)
        
    def load_dataset(self, path, **kwargs):
        file_name = os.path.join("/scratch/gpfs/mengzhou/space7/davinci-memorization/dataset_Arithmetic/StrategyQA/task.json")
        d = json.load(open(file_name))["examples"]
        
        for id, example in enumerate(d):
            example["id"] = id
            
        train_samples = [self.build_sample(example) for example in d]
        valid_samples = [self.build_sample(example) for example in d]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        sample = \
            Sample(
                id=example["id"],
                data=example,
                candidates=["Yes", "No"],
                correct_candidate="Yes" if example["target_scores"]["Yes"] == 1 else "No",
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: StrategyQATemplate}[template_version]()

class CoinflipDataset(Dataset) :
    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        file_name = os.path.join("/scratch/gpfs/mengzhou/space7/davinci-memorization/dataset_Arithmetic/coin_flip/coin_flip.json")
        d = json.load(open(file_name))["examples"]
        
        for id, example in enumerate(d):
            example["id"] = id
            
        train_samples = [self.build_sample(example) for example in d]
        valid_samples = [self.build_sample(example) for example in d]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        example["input"] = example["question"]
        sample = \
            Sample(
                id=example["id"],
                data=example,
                candidates=["Yes", "No"],
                correct_candidate="Yes" if example["answer"] == "yes" else "No",
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: StrategyQATemplate}[template_version]()

class BoolQDataset(Dataset):
    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        d = load_dataset("boolq")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=["Yes", "No"],
                correct_candidate="Yes" if example["answer"] else "No",
            )
        
        return sample
    
    def get_template(self, template_version=2):
        return {0: BoolQTemplate, 1: BoolQTemplateV2, 2: BoolQTemplateV3}[template_version]()

class RhymingDataset(Dataset):
    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        d = json.load(open("/scratch/gpfs/mengzhou/space7/davinci-memorization/rhyming.json"))
        
        train_samples = [self.build_sample(example) for example in d["examples"]]
        valid_samples = [self.build_sample(example) for example in d["examples"]]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        for key in example["target_scores"]:
            if example["target_scores"][key] == 1:
                correct_candidate = key
                break
        sample = \
            Sample(
                data=example,
                candidates=list(example["target_scores"].keys()),
                correct_candidate=correct_candidate,
            ) 
        return sample
    
    def get_template(self, template_version=0):
        return {0: RhymingTemplate}[template_version]()

class MultiArithDataset(Dataset) :
    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        file_name = os.path.join("/scratch/gpfs/mengzhou/space7/davinci-memorization/dataset_Arithmetic/MultiArith/MultiArith.json")
        d = json.load(open(file_name))
        
            
        train_samples = [self.build_sample(example) for example in d]
        valid_samples = [self.build_sample(example) for example in d]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        example["input"] = example["sQuestion"]
        correct_answer = str(example["lSolutions"][0])
        incorrect_answers = [str(example["lSolutions"][0] - 1.0), str(example["lSolutions"][0] + 1.0)]
        sample = \
            Sample(
                id=example["iIndex"],
                data=example,
                candidates=[correct_answer] + incorrect_answers,
                correct_candidate=correct_answer,
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: MultiArithTemplate}[template_version]() 

class CommonsenseQADataset(Dataset) :
    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        file_name = os.path.join("/scratch/gpfs/mengzhou/space7/davinci-memorization/dataset_Arithmetic/CommonsenseQA/dev_rand_split.jsonl")
        d = read_jsonl(file_name)
            
        train_samples = [self.build_sample(example) for example in d]
        valid_samples = [self.build_sample(example) for example in d]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        candidates = [c["text"] for c in example["question"]["choices"]]
        answer = example["answerKey"]
        index = string.ascii_uppercase.index(answer)
        correct_candidate = candidates[index]
        sample = \
            Sample(
                id=example["id"],
                data=example,
                candidates=candidates,
                correct_candidate=correct_candidate,
            )
        return sample
    
    def get_template(self, template_version=0):
        return {0: CommonsenseQATemplate}[template_version]() 

class AddSubDataset(Dataset):
    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        file_name = os.path.join("/scratch/gpfs/mengzhou/space7/davinci-memorization/dataset_Arithmetic/AddSub/AddSub.json")
        d = json.load(open(file_name))
            
        train_samples = [self.build_sample(example) for example in d]
        valid_samples = [self.build_sample(example) for example in d]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        example["input"] = example["sQuestion"]
        correct_answer = str(example["lSolutions"][0])
        incorrect_answers = [str(int(example["lSolutions"][0]) - 1), str(int(example["lSolutions"][0]) + 1)]
        sample = \
            Sample(
                id=example["iIndex"],
                data=example,
                candidates=[correct_answer] + incorrect_answers,
                correct_candidate=correct_answer,
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: MultiArithTemplate}[template_version]() 
    
class LastLettersDataset(Dataset):
    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        file_name = os.path.join("/scratch/gpfs/mengzhou/space7/davinci-memorization/dataset_Arithmetic/last_letters/last_letters.json")
        d = json.load(open(file_name))["examples"]
        
        for i, example in enumerate(d):
            example["id"] = i
            
        train_samples = [self.build_sample(example) for example in d]
        valid_samples = [self.build_sample(example) for example in d]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def shuffle_string(self, s):
        l = list(s)
        random.shuffle(l)
        result = ''.join(l)
        return result
        
    def build_sample(self, example):
        correct_answer = example["answer"]
        incorrect_answers = []
        if len(set(correct_answer)) == 1:
            while len(incorrect_answers) < 2:
                a = random.choice(string.ascii_lowercase) * 4
                if a != correct_answer and a not in incorrect_answers:
                    incorrect_answers.append(a)
        
        while len(incorrect_answers) < 2:
            a = self.shuffle_string(correct_answer)
            print(a)
            if a != correct_answer and a not in incorrect_answers:
                incorrect_answers.append(a)
                
        sample = \
            Sample(
                id=example["id"],
                data=example,
                candidates=[correct_answer] + incorrect_answers,
                correct_candidate=correct_answer,
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: LastLettersTemplate}[template_version]() 

class LMDataset(Dataset):
    lm_datasets = {"the_pile_valid": f"{DATA_DIR}/the_pile/val_sample_texts.pt", 
                   "Wikipedia": f"{DATA_DIR}/the_pile/category/val/Wikipedia-valid-1.536M_sample_texts.pt"}
     
    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)
        
    # default is to load local file
    def load_dataset(self, subtask, **kwargs):
        d = datasets.load_from_disk(self.lm_datasets[subtask])
            
        d = d.add_column("id", [i for i in range(len(d))])
        valid_samples = [self.build_sample(example) for example in d]
        self.samples = {"valid": valid_samples}
    
    def build_sample(self, example):
        sample = \
            Sample(
                id=example["id"],
                data=example,
                candidates=[example["text"]],
                correct_candidate=example["text"],
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: LMTemplate}[template_version]() 

class NQDataset(Dataset):
    train_sep = "\n\n"
    mixed_set = False
    metric_name = ["em", "substring_em"]
    generation = True

    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset()
        
    def load_dataset(self):
        # dataset = load_dataset("gaotianyu1350/natural_question_dpr_version")
        dataset = load_dataset("csv", data_files={"train": "/home/tianyug/p-zero-order/llm_eval/data/nq-dpr-train.csv", "validation": "/home/tianyug/p-zero-order/llm_eval/data/nq-dpr-dev.csv"})
        train_examples = dataset["train"]
        valid_examples = dataset["validation"]
    
        train_samples = [self.build_sample(example, idx) for idx, example in enumerate(train_examples)]
        valid_samples = [self.build_sample(example, idx) for idx, example in enumerate(valid_examples)]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    # for generative tasks, candidates are []
    def build_sample(self, example, idx):
        sample = \
            Sample(
                id=idx, 
                data=example,
                candidates=None,
                correct_candidate=eval(example['answers']),
            )
        
        return sample
        
    def get_template(self, template_version=0):
        return {0: NQTemplate}[template_version]()

class ARCEDataset(Dataset):
    train_sep = "\n\n"
    mixed_set = False

    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)
        
    def load_dataset(self, path, **kwargs):

        train_examples = load_dataset('ai2_arc', "ARC-Easy")["train"]
        valid_examples = load_dataset('ai2_arc', "ARC-Easy")["validation"]
    
        train_samples = [self.build_sample(example) for example in train_examples]
        valid_samples = [self.build_sample(example) for example in valid_examples]
        self.samples = {"train": train_samples, "valid": valid_samples}
   
    # for generative tasks, candidates are []
    def build_sample(self, example):
        answer_id = example['choices']['label'].index(example['answerKey'])
        sample = \
            Sample(
                id=example["id"],
                data=example,
                candidates=example['choices']['text'],
                correct_candidate=example['choices']['text'][answer_id],
            )
        
        return sample
        
    def get_template(self, template_version=0):
        return {0: ARCTemplate}[template_version]() 

class ARCCDataset(Dataset):
    train_sep = "\n\n"
    mixed_set = False

    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)
        
    def load_dataset(self, path, **kwargs):

        train_examples = load_dataset('ai2_arc', "ARC-Challenge")["train"]
        valid_examples = load_dataset('ai2_arc', "ARC-Challenge")["validation"]
    
        train_samples = [self.build_sample(example) for example in train_examples]
        valid_samples = [self.build_sample(example) for example in valid_examples]
        self.samples = {"train": train_samples, "valid": valid_samples}
   
    # for generative tasks, candidates are []
    def build_sample(self, example):
        answer_id = example['choices']['label'].index(example['answerKey'])
        sample = \
            Sample(
                id=example["id"],
                data=example,
                candidates=example['choices']['text'],
                correct_candidate=example['choices']['text'][answer_id],
            )
        
        return sample
        
    def get_template(self, template_version=0):
        return {0: ARCTemplate}[template_version]() 


class OBQADataset(Dataset):
    train_sep = "\n\n"
    mixed_set = False

    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)
        
    def load_dataset(self, path, **kwargs):

        train_examples = load_dataset('openbookqa', "main")["train"]
        valid_examples = load_dataset('openbookqa', "main")["validation"]
    
        train_samples = [self.build_sample(example) for example in train_examples]
        valid_samples = [self.build_sample(example) for example in valid_examples]
        self.samples = {"train": train_samples, "valid": valid_samples}
   
    # for generative tasks, candidates are []
    def build_sample(self, example):
        answer_id = example['choices']['label'].index(example['answerKey'])
        sample = \
            Sample(
                id=example["id"],
                data=example,
                candidates=example['choices']['text'],
                correct_candidate=example['choices']['text'][answer_id],
            )
        
        return sample
        
    def get_template(self, template_version=0):
        return {0: OBQATemplate}[template_version]() 


class PIQADataset(Dataset):
    train_sep = "\n\n"
    mixed_set = False

    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)
        
    def load_dataset(self, path, **kwargs):

        train_examples = load_dataset('piqa')["train"]
        valid_examples = load_dataset('piqa')["validation"]
    
        train_samples = [self.build_sample(example) for example in train_examples]
        valid_samples = [self.build_sample(example) for example in valid_examples]
        self.samples = {"train": train_samples, "valid": valid_samples}
   
    # for generative tasks, candidates are []
    def build_sample(self, example):
        candidates = [example['sol1'], example['sol2']]
        sample = \
            Sample(
                data=example,
                candidates=candidates,
                correct_candidate=candidates[example['label']]
            )
        
        return sample
        
    def get_template(self, template_version=0):
        return {0: PIQATemplate}[template_version]() 


class SIQADataset(Dataset):
    train_sep = "\n\n"
    mixed_set = False

    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)
        
    def load_dataset(self, path, **kwargs):

        train_examples = load_dataset('social_i_qa')["train"]
        valid_examples = load_dataset('social_i_qa')["validation"]
    
        train_samples = [self.build_sample(example) for example in train_examples]
        valid_samples = [self.build_sample(example) for example in valid_examples]
        self.samples = {"train": train_samples, "valid": valid_samples}
   
    # for generative tasks, candidates are []
    def build_sample(self, example):
        candidates = [example['answerA'], example['answerB'], example['answerC']]
        sample = \
            Sample(
                data=example,
                candidates=candidates,
                correct_candidate=candidates[int(example['label']) - 1]
            )
        
        return sample
        
    def get_template(self, template_version=0):
        return {0: SIQATemplate}[template_version]() 


class MultiRCDataset(Dataset):
    
    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "multirc")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1],
                correct_candidate=example['label']
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: MultiRCTemplate}[template_version]()


class CBDataset(Dataset):
    
    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "cb")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1, 2],
                correct_candidate=example['label']
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: CBTemplate}[template_version]()


class WICDataset(Dataset):
    
    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "wic")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1],
                correct_candidate=example['label']
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: WICTemplate}[template_version]()


class WSCDataset(Dataset):
    
    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "wsc.fixed")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1],
                correct_candidate=example['label']
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: WSCTemplate}[template_version]()


class ReCoRDDataset(Dataset):
    
    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "record")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=example['entities'],
                correct_candidate=example['answers']
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: ReCoRDTemplate}[template_version]()


class MNLIDataset(Dataset):
    
    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        d = load_dataset("glue", "mnli")
        train_set = d["train"]
        valid_set = d["validation_matched"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1, 2],
                correct_candidate=example['label']
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: MNLITemplate}[template_version]()

class QNLIDataset(Dataset):
    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        d = load_dataset("glue", "qnli")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1],
                correct_candidate=example['label']
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: QNLITemplate}[template_version]()
    
    
class RTEDataset(Dataset):
    
    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        d = load_dataset("super_glue", "rte")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[0, 1],
                correct_candidate=example['label']
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: RTETemplate}[template_version]()


class WinograndeDataset(Dataset):
    
    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)
    
    def load_dataset(self, path, **kwargs):
        d = load_dataset("winogrande", "winogrande_debiased")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    def build_sample(self, example):
        sample = \
            Sample(
                data=example,
                candidates=[example['option1'], example['option2']],
                correct_candidate=example['option%s' % example['answer']]
            )
        
        return sample
    
    def get_template(self, template_version=0):
        return {0: WinograndeTemplate}[template_version]()


class SST5Dataset(Dataset):
    train_sep = "\n\n"
    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)
        
    def load_dataset(self, path, **kwargs):
        d = load_dataset("SetFit/sst5")
        train_d = d["train"]
        validation_d = d["validation"]
        
        train_samples = [self.build_sample(example, idx) for idx, example in enumerate(train_d)]
        valid_samples = [self.build_sample(example, idx) for idx, example in enumerate(validation_d)]
        
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    # for generative tasks, candidates are []
    def build_sample(self, example, idx):
        label = int(example["label"])
        return Sample(id=idx, data=example, correct_candidate=label, candidates=[0, 1, 2, 3, 4])
        
    def get_template(self, template_version=0):
        return {0: SST5Template}[template_version]()
    
class AGNewsDataset(Dataset):
    train_sep = "\n\n"
    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)
        
    def load_dataset(self, path, **kwargs):
        # d = load_dataset("ag_news")
        d = load_from_disk("/scratch/gpfs/mengzhou/space6/data/ag_news_short.pt")
        train_d = d["train"]
        validation_d = d["test"]
        
        train_samples = [self.build_sample(example, idx) for idx, example in enumerate(train_d)]
        valid_samples = [self.build_sample(example, idx) for idx, example in enumerate(validation_d)]
        
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    # for generative tasks, candidates are []
    def build_sample(self, example, idx):
        label = int(example["label"])
        return Sample(id=idx, data=example, correct_candidate=label, candidates=[0, 1, 2, 3])
        
    def get_template(self, template_version=1):
        return {0: AGNewsv2Template, 1: AGNewsSyntheticTemplate, 2: AGNewsTemplate, }[template_version]()

    
class SubjDataset(Dataset):
    train_sep = "\n\n"
    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)
        
    def load_dataset(self, path, **kwargs):
        try:
            d = load_dataset("SetFit/subj")
        except:
            d = datasets.load_from_disk("/scratch/gpfs/mengzhou/space6/data/Subj.pt")
        train_d = d["train"]
        validation_d = d["test"]
        
        train_samples = [self.build_sample(example, idx) for idx, example in enumerate(train_d)]
        valid_samples = [self.build_sample(example, idx) for idx, example in enumerate(validation_d)]
        
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    # for generative tasks, candidates are []
    def build_sample(self, example, idx):
        label = int(example["label"])
        return Sample(id=idx, data=example, correct_candidate=label, candidates=[0, 1])
        
    def get_template(self, template_version=0):
        return {0: SubjTemplate, 1: SubjSyntheticTemplate}[template_version]()

class DBPediaDataset(Dataset):
    train_sep = "\n\n"
    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)
        
    def load_dataset(self, path, **kwargs):
        d = load_dataset("dbpedia_14")
        train_d = d["train"]
        validation_d = d["test"]
        
        train_samples = [self.build_sample(example, idx) for idx, example in enumerate(train_d)]
        valid_samples = [self.build_sample(example, idx) for idx, example in enumerate(validation_d)]
        
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    # for generative tasks, candidates are []
    def build_sample(self, example, idx):
        label = int(example["label"])
        return Sample(id=idx, data=example, correct_candidate=label, candidates=list(range(14)))
        
    def get_template(self, template_version=1):
        return {0: DBPediaTemplate, 1: DBPediav2Template}[template_version]() 

class MRPCDataset(Dataset):
    train_sep = "\n\n"
    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)
        
    def load_dataset(self, path, **kwargs):
        d = load_dataset("glue", "mrpc")
        train_d = d["train"]
        validation_d = d["validation"]
        
        train_samples = [self.build_sample(example, idx) for idx, example in enumerate(train_d)]
        valid_samples = [self.build_sample(example, idx) for idx, example in enumerate(validation_d)]
        
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    # for generative tasks, candidates are []
    def build_sample(self, example, idx):
        label = int(example["label"])
        return Sample(id=idx, data=example, correct_candidate=label, candidates=list(range(2)))
        
    def get_template(self, template_version=0):
        return {0: MRPCTemplate}[template_version]() 

class QQPDataset(Dataset):
    train_sep = "\n\n"
    def __init__(self, subtask=None, **kwargs) -> None:
        super().__init__(subtask, **kwargs)
        self.load_dataset(subtask, **kwargs)
        
    def load_dataset(self, path, **kwargs):
        d = load_dataset("glue", "qqp")
        train_d = d["train"]
        validation_d = d["validation"]
        
        train_samples = [self.build_sample(example, idx) for idx, example in enumerate(train_d)]
        valid_samples = [self.build_sample(example, idx) for idx, example in enumerate(validation_d)]
        
        self.samples = {"train": train_samples, "valid": valid_samples}
    
    # for generative tasks, candidates are []
    def build_sample(self, example, idx):
        label = int(example["label"])
        return Sample(id=idx, data=example, correct_candidate=label, candidates=list(range(2)))
        
    def get_template(self, template_version=0):
        return {0: QQPTemplate}[template_version]()  