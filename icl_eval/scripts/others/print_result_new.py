from pathlib import Path
import json
import numpy as np
import os
from cus_data.dataloader import read_jsonl


def print_numpy_result(result, models):
    for i in range(len(result)):
        # print(models[i], end="\t")
        for j in range(result.shape[1]):
            print(f"{result[i, j]:.4f}", end="\t")
        print()
    print()
    
out_dir = Path("/scratch/gpfs/mengzhou/space9/out/llm_eval/dynamic_icl_eval_May8_test")

# rows
models = ["constructed-opt-backprop-6-nstep-2-lr-1e-04", "constructed-opt-backprop-6-nstep-2-lr-1e-05", "constructed-opt-backprop-6-nstep-2-lr-1e-06",
          "constructed-opt-backprop-12-nstep-1-lr-1e-04", "constructed-opt-backprop-12-nstep-1-lr-1e-05", "constructed-opt-backprop-12-nstep-1-lr-1e-06",
          "constructed-opt-backprop-3-nstep-4-lr-1e-04", "constructed-opt-backprop-3-nstep-4-lr-1e-05", "constructed-opt-backprop-3-nstep-4-lr-1e-06"]

shots = 16
task = "MR"
metric = "accuracy"
eval_type = "eval"

# result = np.zeros((len(models), len(shots)))

for single_FT in [True, False]:
    for label_type in ["context", "label_only"]:
        for icl in ["plain", "icl_sfc"]:
            seed_accs = []
            for seed in range(1, 3):
                acc = np.zeros((len(models), 2))
                for i, model in enumerate(models):
                    model_dir = out_dir / model
                    lr = "-".join(model.split("-")[-2:])
                    layer = model.split("-")[3]
                    task_dir = model_dir / f"ntrain{shots}-{task}-lr{lr}-seed{seed}-single_FT{single_FT}-layers{layer}-{label_type}-{icl}-{eval_type}"
                    file = task_dir / "metrics.jsonl"
                    if os.path.exists(file):
                        re = read_jsonl(file)
                        acc[i, 0] = re[0]["accuracy"]
                        # acc[i, 1] = re[1]["accuracy"]
                    else:
                        print(file)
                seed_acc = acc.max(axis=0)
                seed_accs.append(seed_acc)
            acc = np.stack(seed_accs).mean(axis=0)
            print(" ".join([str(a) for a in acc]), end = " ")
            # print(seed_accs)
        print()
                    
                        
        
# print(f"{task} {metric}")
# print_numpy_result(result, models)        

