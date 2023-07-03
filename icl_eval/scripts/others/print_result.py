from pathlib import Path
import json
import numpy as np
import os


def print_numpy_result(result, models):
    for i in range(len(result)):
        # print(models[i], end="\t")
        for j in range(result.shape[1]):
            print(f"{result[i, j]:.4f}", end="\t")
        print()
    print()
    
out_dir = Path("/n/fs/nlp-mengzhou/space9/out/llm_eval/constructed-lm-results")

# rows
models = ["gpt2", "constructed-gpt2-l12-ns1-lr1e-06", "constructed-gpt2-l12-ns1-lr1e-05", "constructed-gpt2-l12-ns1-lr1e-04", "opt-125m", 
          "constructed-opt-l12-ns1-lr1e-06", "constructed-opt-l12-ns1-lr2e-06", "constructed-opt-l12-ns1-lr5e-07", "gpt2-large", "opt-1.3b"]

shots = [0, 2, 4, 8]

task = "AGNews"
icl_sfc = False 
sfc = False 
metric = "accuracy"

result = np.zeros((len(models), len(shots)))
for i, model in enumerate(models):
    for j, shot in enumerate(shots):
        if sfc: tag = "-sfc"
        elif icl_sfc: tag = "-icl_sfc"
        else: tag = ""
        if shot == 0: file = out_dir / model / f"{task}-{model}{tag}-sampleeval200-onetrainpereval.json"
        else: file = out_dir / model / f"{task}-{model}{tag}-sampleeval200-ntrain{shot}-onetrainpereval.json"
        if os.path.exists(file):
            re = json.load(open(file, "r"))
            result[i, j] = re[metric]
        
print(f"{task} {metric}")
print_numpy_result(result, models)        

