import os
import wandb

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments, default_data_collator

from mkt import data
from mkt.utils import *

def main(
    args,
    model_name: str = "skt/kogpt2-base-v2",
    data_dir: str = "/w/data/mkt",
    output_dir: str = "/w/exp/mkt",
    num_epochs: int = 10,
    per_device_batch_size: int = 8,
    project: str = "mkt",
    entity: str = "dhlee347"
):
    wandb.init(name=os.path.basename(__file__), project=project, entity=entity)

    train_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            learning_rate=4e-5,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            per_device_train_batch_size=per_device_batch_size,
            per_device_eval_batch_size=per_device_batch_size//2,
            evaluation_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=100,
            load_best_model_at_end=True,
            logging_strategy="steps",
            logging_steps=50,
            report_to="wandb",
    )

    config    = AutoConfig.from_pretrained(model_name)
    # config.gradient_checkpointing = True
    # config.use_cache              = False

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
    model.resize_token_embeddings(len(tokenizer))

    train_dataset = data.get(data_dir, split='train', tokenizer=tokenizer)
    train_dataset = data.prepare_for_language_modeling(train_dataset, block_size=1024, num_processes=8)
    train_dataset = train_dataset.with_format('torch')

    eval_dataset = data.get(data_dir, split='valid', tokenizer=tokenizer)
    eval_dataset = data.prepare_for_language_modeling(eval_dataset, block_size=1024, num_processes=8)
    eval_dataset = eval_dataset.with_format('torch')

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        args=train_args
    )

    train_results = trainer.train()

    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()


if __name__ == '__main__':
    seed_everything(42)
    main(args=None)