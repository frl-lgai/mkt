rm -rf ~/.cache /w/data/hf_datasets/json/default*

port=$(shuf -i25000-30000 -n1)
base_dir=`pwd`

export CUDA_HOME='/usr/local/cuda'
export TOKENIZERS_PARALLELISM=false
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# $1 - GPUs to use (ex - 1,2,3)
# $2 - config file (ex - configs/finetune_1.7B.yaml)
deepspeed --include localhost:$1 --master_port $port finetune.py --deepspeed --config $2
