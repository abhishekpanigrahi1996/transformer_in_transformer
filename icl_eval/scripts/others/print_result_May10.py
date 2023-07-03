from pathlib import Path
import json
import numpy as np
import os

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

def print_numpy_result(result, models):
    for i in range(len(result)):
        # print(models[i], end="\t")
        for j in range(result.shape[1]):
            print(f"{result[i, j]:.4f}", end="\t")
        print()
    print()
    
out_dir = Path("/scratch/gpfs/smalladi/mengzhou/out/llm_eval/dynamic_icl_eval_May10")

# rows
def get_models(label_type):
    models = []
    for layer in [12, 6, 3]:
        steps = 12 // layer
        if label_type == "context": lrs = ["1e-05", "1e-06", "1e-07"]
        else: lrs = ["1e-03", "1e-04", "1e-05"]
        for lr in lrs:
            models.append(f"constructed-opt-backprop-{layer}-nstep-{steps}-lr-{lr}")
    return models
  
shots = 32 
task = "AmazonPolarity"
metric = "accuracy"
eval_type = "eval"
# result = np.zeros((len(models), len(shots)))

print_model_name = False 

print_res = []
all_accs = []
for task in ["Subj", "AGNews", "SST2", "CR", "MR", "MPQA", "AmazonPolarity"]:
    task_acc = []
    print(task)
    print_re = []
    for tmp in [0]:
        for single_FT in [True, False]:
            for label_type in ["context", "label_only"]:
                sub_models = get_models(label_type)
                for icl in ["icl_sfc"]:
                    seed_accs = []
                    seed_models = []
                    all_seed_accs = []
                    for seed in range(1, 4):
                        acc = np.zeros((len(sub_models), 2))
                        for i, model in enumerate(sub_models):
                            model_dir = out_dir / model
                            lr = "-".join(model.split("-")[-2:])
                            layer = model.split("-")[3]
                            task_dir = model_dir / f"ntrain{shots}-{task}-lr{lr}-seed{seed}-single_FT{single_FT}-layers{layer}-{label_type}-tmp{tmp}-{icl}-{eval_type}"
                            file = task_dir / "metrics.jsonl"
                            if os.path.exists(file):
                                re = read_jsonl(file)
                                acc[i, 0] = re[0]["accuracy"]
                                # acc[i, 1] = re[1]["accuracy"]
                            else:
                                pass
                                # print(file)
                        model_name = sub_models[acc[:, 0].argmax(axis=0)]
                        seed_models.append(model_name)
                        seed_acc = acc.max(axis=0)
                        seed_accs.append(seed_acc * 100)
                        all_seed_accs.append(seed_acc[0])
                    task_acc.append(all_seed_accs)
                    acc = np.stack(seed_accs).std(axis=0)
                    print_re.append(acc[0])
                    if print_model_name:
                        print(" ".join(seed_models), end=" ")
                    else:
                        print(" ".join([str(a) for a in acc[0:1]]), end = " ")
                    # print(seed_accs)
                print()
    print_res.append(print_re)
    all_accs.append(task_acc)
                # print(seed_accs)

print_res = np.array(print_res).transpose()
for line in print_res:
    print(' '.join(map(str, line)))

all_accs = np.array(all_accs) # num_tasks * num_settings * num_seeds
import pdb; pdb.set_trace()              
print((np.array(all_accs) * 100).max(1).mean(0).std())
                        
        
# print(f"{task} {metric}")
# print_numpy_result(result, models)        

