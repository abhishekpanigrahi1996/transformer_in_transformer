model_name = "opt-125m"
task = "MRPC"
eval_type = "valid"

# MRPC-train8-eval200-lr1e-03-labelcontext-dynamicFalse

import json
from pathlib import Path
import numpy as np
import os
def read_jsonl(file):
    lines = open(file, "r").readlines()
    d = []
    for line in lines:
        d.append(json.loads(line.strip()))
    return d
    
output_dir = Path(f"/n/fs/nlp-mengzhou/space9/out/llm_eval/dynamic_icl_eval_May3")
# for sfc_type in ["plain", "icl", "icl_sfc"]:
#     for shot in [16]:
#         for single_FT in [True, False]:
#             for label_type in ["label_only", "context"]:
#                 for num_train_layers in [3, 6, 12]:
#                     accuracies_all_seeds = []
#                     for seed in [1, 2, 3]:
#                         accuracies = []
#                         for lr in ["1e-05", "1e-04", "1e-03", "1e-02"]:
#                             file_name=f"{model_name}-ntrain{shot}-{task}-lr{lr}-seed{seed}-single_FT{single_FT}-layers{num_train_layers}-{label_type}-{sfc_type}-{eval_type}.json"
#                             # file_name=f"{model_name}-ntrain{shot}-{task}-seed{seed}-{sfc_type}-{eval_type}.json"
#                             file = output_dir / file_name
#                             if not os.path.exists(file):
#                                 continue
#                             d = read_jsonl(file)
#                             accuracies.append(np.array([dd["accuracy"] for dd in d]))
#                         accuracies = np.stack(accuracies).max(axis=0).tolist()
#                     accuracies_all_seeds.append(accuracies)
#                     accuracies_all_seeds = np.stack(accuracies_all_seeds)  
#                     accuracies = np.mean(accuracies_all_seeds, axis=0)
#                     re = " ".join(str(acc) for acc in accuracies)
#                     print(re)
        
accuracies = []
shot=16
for sfc_type in ["plain", "icl", "icl_sfc"]:
    for seed in [1, 2, 3]:
        file = output_dir / f"{model_name}-ntrain{shot}-{task}-seed{seed}-{sfc_type}-{eval_type}.json"
        if not os.path.exists(file):
            continue
        d = read_jsonl(file)
        accuracies.append(np.array([dd["accuracy"] for dd in d]))
    accuracies = np.stack(accuracies)
    accuracies = np.mean(accuracies, axis=0).tolist()
    re = " ".join(str(acc) for acc in accuracies)
    print(re) 
    
            

            