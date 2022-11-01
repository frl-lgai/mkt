rm -rf ~/.cache /w/data/hf_datasets/json/default*

port=$(shuf -i25000-30000 -n1)
base_dir=`pwd`

export CUDA_HOME='/usr/local/cuda'
export TOKENIZERS_PARALLELISM=false

deepspeed \
    --include localhost:0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
    --master_port $port \
    finetune_lingvo.py \
    --deepspeed \
    --deepspeed_config "deepspeed_configs/stage3.json" \
    --model_dir "/w/exaone_2022/model_1.7B_BI_MT_02" \
    --data_dir "/w/mkt/data/kobaco" \
    --output_dir "/w/exp/mkt/model_1.7B_BI_MT_02" \
    --num_epochs 20 \
    --per_device_batch_size 16 \
    --learning_rate 4e-5 \
    --save_steps 50 \
    --project "mkt" \
    --entity "dhlee347"

