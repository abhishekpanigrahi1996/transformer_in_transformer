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

def read_json(file):
    return json.load(open(file))
    
def print_numpy_result(result, models):
    for i in range(len(result)):
        # print(models[i], end="\t")
        for j in range(result.shape[1]):
            print(f"{result[i, j]:.4f}", end="\t")
        print()
    print()
    
out_dir = Path("/n/fs/nlp-mengzhou/space9/out/llm_eval/dynamic_icl_eval_May10")

  
shots = 32 
task = "Subj"
metric = "accuracy"
eval_type = "eval"
# result = np.zeros((len(models), len(shots)))

print_model_name = False 
all_accs = []
for task in ["Subj", "AGNews", "SST2", "CR", "MR", "MPQA", "AmazonPolarity"]:
    print(task)
    for tmp in [0, 1]:
        for single_FT in [True, False]:
            for label_type in ["context", "label_only"]:
                for icl in ["plain"]:
                    seed_accs = []
                    seed_models = []
                    for seed in range(1, 4):
                        acc = np.zeros((9, 2))
                        count = 0
                        for lr in ["1e-03", "1e-04", "1e-05"]:
                            for layer in [3, 6, 12]:
                                model = f"opt-125m-ntrain{shots}-{task}-lr{lr}-seed{seed}-single_FT{single_FT}-layers{layer}-{label_type}-tmp{tmp}-{icl}-{eval_type}"
                                task_dir = out_dir / model
                                file = task_dir / "metrics.jsonl"
                                if os.path.exists(file):
                                    re = read_json(file)
                                    acc[count, 0] = re["accuracy"]
                                else:
                                    print(file)
                                count += 1
                        model_name = f"lr{lr}-seed{seed}"
                        seed_models.append(model_name)
                        seed_acc = acc.max(axis=0)
                        seed_accs.append(seed_acc * 100)
                    acc = np.stack(seed_accs).std(axis=0)
                    if print_model_name:
                        print(" ".join(seed_models), end=" ")
                    else:
                        print(" ".join([str(a) for a in acc[0:1]]), end = " ")
                    # print(seed_accs)
                print()
    all_accs.append(seed_accs)

all_accs = np.stack(all_accs)
print("std:", all_accs[:, :, 0].mean(0).std())
                        
                        
        
# print(f"{task} {metric}")
# print_numpy_result(result, models)        

