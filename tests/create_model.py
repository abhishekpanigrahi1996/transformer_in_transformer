import torch
from tqdm import tqdm
from tint_main.tint_creator import *
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import HfArgumentParser
from filelock import Timeout, FileLock
from .data_utils.data_utils import *
from tint_main.utils.all_arguments import *
      
parser = HfArgumentParser((ModelArguments, ConstructionArguments, DynamicArguments,))
model_args, construction_args, data_args, = parser.parse_args_into_dataclasses()
        
constructed_model, _, _, _ = TinT_Creator(model_args, construction_args)





