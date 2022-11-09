#from transformers import AutoConfig, AutoModel, AutoTokenizer, GPTLingvoForCausalLM, AutoModelForCausalLM
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments, default_data_collator

import deepspeed

from mkt import data, config
from mkt.utils import *


def main(cfg):
    
    seed_everything(cfg.seed)
    wandb_init_distributed(cfg, __file__)

    train_args = TrainingArguments(**cfg.trainer)

    model_config = AutoConfig.from_pretrained(cfg.model_name_or_path)
    model_config.gradient_checkpointing = True
    model_config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path, config=model_config)
    model.resize_token_embeddings(len(tokenizer))

    dataset = data.load(cfg.data_dir, tokenizer=tokenizer, num_proc=cfg.num_proc, eos_token=cfg.eos_token)
    dataset = data.prepare_for_language_modeling(dataset, block_size=cfg.max_length, num_proc=cfg.num_proc)
    dataset = dataset.with_format('torch')

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset['train'],
        eval_dataset=dataset['valid'],
        data_collator=default_data_collator,
        args=train_args
    )

    train_results = trainer.train()
    trainer.save_model()

    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()


if __name__ == '__main__':
    
    import argparse
   
    parser = argparse.ArgumentParser(description="finetune_lingvo")
    parser.add_argument("--local_rank", type=int, default=0, help="local_rank for distributed training on gpus")
    parser.add_argument("--config", type=str, default="../configs/finetune_lingvo_1.7B.yaml")    
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    
    cfg = config.load(args.config, config.FinetuneConfig)
    cfg.local_rank = args.local_rank
    
    main(cfg)
