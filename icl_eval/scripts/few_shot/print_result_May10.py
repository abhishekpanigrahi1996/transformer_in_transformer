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
    
out_dir = Path("/n/fs/nlp-mengzhou/space9/out/llm_eval/fewshot_eval_May15")

shots = 32 
# task = sys.argv[1] 
metric = "accuracy"
eval_type = "eval"
# result = np.zeros((len(models), len(shots)))

all_seed_accs = []
for task in ["Subj", "AGNews", "SST2", "CR", "MR", "MPQA", "AmazonPolarity"]:
    # print(task)
    print_model_name = False 
    for tmp in [0]:
        for icl in ["icl_sfc"]:
            seed_accs = []
            seed_models = []
            for seed in range(1, 4):
                count = 0
                model = f"opt-1.3b-ntrain{shots}-{task}-seed{seed}-tmp{tmp}-{icl}-eval"
                task_dir = out_dir / model
                file = task_dir / "metrics.jsonl"
                if os.path.exists(file):
                    re = read_jsonl(file)
                    acc = re[0]["accuracy"]
                    count += 1
                    seed_accs.append(acc * 100)
                else:
                    print(file)
            all_seed_accs.append(np.array(seed_accs))
            acc = np.std(seed_accs)
            if print_model_name:
                print(" ".join(seed_models), end=" ")
            else:
                print(acc, end=" ")
            # print(seed_accs)
        print()
                    
all_seed_accs = np.stack(all_seed_accs, axis=1)
print("sigma:", all_seed_accs.mean(1).std())
        
# print(f"{task} {metric}")
# print_numpy_result(result, models)        

