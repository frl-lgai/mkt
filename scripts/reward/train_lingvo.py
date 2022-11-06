import os
import wandb
import rich
import deepspeed

from dataclasses import dataclass

from transformers import AutoConfig, AutoTokenizer, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from rlhf.utils import *
from rlhf.models import GPT2ForRewardModel, GPTJForRewardModel
from rlhf.data import tldr


@dataclass
class DataCollatorWithPadding:

    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features):
        def pad(features):
            batch = self.tokenizer.pad(
                features,
                padding=True,
                return_tensors='pt',
            )
            if "label" in batch:
                batch["labels"] = batch["label"]
                del batch["label"]
            if "label_ids" in batch:
                batch["labels"] = batch["label_ids"]
                del batch["label_ids"]
            return batch

        features = {key: [f[key] for f in features] for key in features[0].keys()}

        # For padding, change the key 
        # ex) 'input_ids_pos', 'attention_mask_pos' => 'input_ids', 'attention_mask
        pos = pad({k[:-4]: features[k] for k in features if k.endswith('_pos')})
        neg = pad({k[:-4]: features[k] for k in features if k.endswith('_neg')})
        # after padding, bring the key back to its previous state
        # ex) 'input_ids', 'attention_mask => 'input_ids_pos', 'attention_mask_pos' 
        pos = {f'{k}_pos': pos[k] for k in pos}
        neg = {f'{k}_neg': neg[k] for k in neg}

        return {**pos, **neg}


def GPTForRewardModel(model_name_or_path):
    config = AutoConfig.from_pretrained(model_name_or_path)
    config.gradient_checkpointing = True
    config.use_cache              = False

    if config.model_type == "gpt2":
        model = GPT2ForRewardModel.from_pretrained(model_name_or_path, config=config)
    elif config.model_type == "gptj":
        model = GPTJForRewardModel.from_pretrained(model_name_or_path, config=config)

    return model


def main(
         args,
         model_name: str = "gptj",
         model_name_or_path: str = "EleutherAI/gpt-j-6B",
         data_dir: str = "/w/data/summarize-from-feedback/comparisons",
         output_dir: str = "/home/kyungmin.lee/clean_code/rlhf/save",
         num_epochs: int = 1,
         per_device_batch_size: int = 12,
         project: str = "tldr_reward",
         entity: str = "lkm2835"):

    train_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            learning_rate=1.5e-5,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            per_device_train_batch_size=per_device_batch_size,
            per_device_eval_batch_size=per_device_batch_size,
            evaluation_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=200,
            load_best_model_at_end=True,
            logging_steps=200,
            report_to="wandb",
            label_names=["input_ids_neg"], # we can see eval_loss, when label_names given
            deepspeed="stage3_offload.json",
            local_rank=args.local_rank,
    )

    wandb.init(name=os.path.basename(__file__), project=project, entity=entity)

    rich.print("\n[bold magenta] * Loading tokenizer and model... [/bold magenta]")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = GPTForRewardModel(model_name_or_path)

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(len(tokenizer))

    rich.print("\n[bold magenta] * Dataset preparation... [/bold magenta]")

    dataset = tldr.prepare_human_feedback(data_dir, tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset['train'],
        eval_dataset=dataset['valid'],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        args=train_args
    )

    # Train!
    train_result = trainer.train()
    trainer.save_model()

    metrics = train_result.metrics

    metrics["train_samples"] = len(dataset['train'])

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Evaluate
    rich.print("[bold magenta] **** Evaluate *** [/bold magenta]")
    metrics = trainer.evaluate()

    metrics["eval_samples"] = len(dataset['valid'])

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


if __name__ == '__main__':
    seed_everything(42)

    import argparse
    parser = argparse.ArgumentParser(description="GPTJ-finetune")

    parser.add_argument("--local_rank", type=int,
                    default=0,
                    help="local_rank for distributed training on gpus")

    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    main(args)