from pathlib import Path
import json
import numpy as np
import os

def read_jsonl(file):
    ds = []
    try:
        with open(file) as f:
            for i, line in enumerate(f):
                if line.strip() == "":
                    continue
                d = json.loads(line.strip())
                ds.append(d)
    except:
        import pdb
        pdb.set_trace()
    return ds

def print_numpy_result(result, models):
    for i in range(len(result)):
        # print(models[i], end="\t")
        for j in range(result.shape[1]):
            print(f"{result[i, j]:.4f}", end="\t")
        print()
    print()
    
out_dir = Path("/n/fs/nlp-mengzhou/space9/out/llm_eval/zeroshot_eval_May15")

  
shots = 0 
task = "MPQA"
metric = "accuracy"
eval_type = "eval"
seed = 1
# result = np.zeros((len(models), len(shots)))

print_model_name = False 
for tmp in [0]:
    for icl in ["plain", "icl_sfc"]:
        count = 0
        model = f"opt-1.3b-ntrain{shots}-{task}-seed{seed}-tmp{tmp}-{icl}-eval"
        task_dir = out_dir / model
        file = task_dir / "metrics.jsonl"
        if os.path.exists(file):
            re = read_jsonl(file)
            acc = re[0]["accuracy"]
            count += 1
        else:
            print(file)
        print(acc, end=" ")
        # print(seed_accs)
    print()
                    
                        
        
# print(f"{task} {metric}")
# print_numpy_result(result, models)        

