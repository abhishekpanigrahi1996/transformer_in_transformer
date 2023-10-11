#!/bin/sh
#SBATCH --job-name=icl
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=30gb

# $model_path $model_name $output_dir $ntrain $task $lr $dynamic_eval $seed $sf $layer $sfc_type $eval_type $label_type 

script=$base_dir/llm_eval_construction/run.py
model_path=$1 
model_name=$2 
output_dir=$3 
ntrain=$4 
task=$5
lr=$6
train_set_seed=$7
single_FT=$8
num_train_layers=${9}
label_type=${10}
sfc_type=${11}
eval_type=${12}
template=${13}
test=${14}
num_epochs=$(( 12 / $num_train_layers ))
dynamic_eval=${15}

# change the path of the constructed models
model_path=/scratch/gpfs/ap34/icl-as-ft/Dynamic_initialization/Constructed_models_withlnupdate_fastlin/model_opt_backprop_${num_train_layers}_nstep_${num_epochs}_lr_${lr}
model_name=constructed-opt-backprop-${num_train_layers}-nstep-${num_epochs}-lr-${lr}

restrict_attention_demonstration=False # whether to use restrict the attention of each demonstration to itself in icl experiments
position_modify_demonstration=False # whether to change the position ids of each demonstration to top; only relevant if restrict_attention_demonstration is true

if [[ $single_FT == True ]]; then
    restrict_attention_demonstration=True
    position_modify_demonstration=True
else
    restrict_attention_demonstration=False
    position_modify_demonstration=False
fi

if [[ $label_type == "label_only" ]]; then loss_label_only=True; 
else loss_label_only=False; fi

sub_dir=${model_name}/ntrain${ntrain}-${task}-lr${lr}-seed${train_set_seed}-single_FT${single_FT}-layers${num_train_layers}-${label_type}-tmp${template}-${sfc_type}-${eval_type}
output_dir=$output_dir/$sub_dir
mkdir -p $output_dir

echo "********** inside single script **********"
echo model_path=$model_path
echo model_name=$model_name
echo output_dir=$output_dir
echo ntrain=$ntrain
echo task=$task
echo train_set_seed=$train_set_seed
echo single_FT=$single_FT
echo label_type=$label_type
echo sfc_type=${sfc_type}
echo eval_type=${eval_type}
echo restrict_attention_demonstration=${restrict_attention_demonstration}
echo position_modify_demonstration=${position_modify_demonstration}
echo template=${template} 
echo "Outputting to $output_dir/${file_name}"
echo "********** inside single script **********"


if [[ -f $output_dir/metrics.jsonl ]]; then
    echo "File exists: $output_dir/metrics.jsonl"
    exit 0
fi

if [[ $test == True ]]; then num_eval=4; else num_eval=200; fi

cmd="python3 $script \
        --task_name $task \
        --num_train ${n_train} \
        --num_eval ${num_eval} \
        --model_path $model_path \
        --model_name $model_name \
        --load_float16 True \
        --pruned False \
        --output_dir=$output_dir \
        --train_set_seed $train_set_seed \
        --loss_type $label_type \
        --result_file $output_dir/metrics.jsonl \
        --exclusive_training True \
        --single_FT ${single_FT} \
        --num_train_layers ${num_train_layers} \
        --num_epochs ${num_epochs} \
        --eval_set_seed 0 \
        --restrict_attention_demonstration ${restrict_attention_demonstration} \
        --position_modify_demonstration ${position_modify_demonstration} \
        --loss_label_only ${loss_label_only} \
        --test_mode False \
        --template_id $template"

if [[ $eval_type == "test" ]]; then
    cmd="$cmd --test_set_seed 10"
fi
if [[ $sfc_type == "sfc" ]]; then
    cmd="$cmd --sfc True"
fi
if [[ $sfc_type == "icl_sfc" ]]; then
    cmd="$cmd --icl_sfc True"
fi 
$cmd 2>&1 | tee $output_dir/log.txt
