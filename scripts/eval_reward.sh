rm -rf ~/.cache /w/data/hf_datasets/json/default*

export CUDA_HOME='/usr/local/cuda'
export TOKENIZERS_PARALLELISM=false
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

CUDA_VISIBLE_DEVICES=$1 python eval_reward.py --model_name_or_path $2 --data_dir $3