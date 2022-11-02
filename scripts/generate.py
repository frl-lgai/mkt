import os
import json
from random import random

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import deepspeed



with open("/w/mkt/data/kobaco/valid.jsonl", "r") as f:
    data = [json.loads(line) for line in f]


model_name = "/w/exp/mkt/model_8.8B_BI_MT_02-4th/checkpoint-20"

config = AutoConfig.from_pretrained(model_name)
config.gradient_checkpointing = True
config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
model.resize_token_embeddings(len(tokenizer))


local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

print(local_rank, world_size)

sharded_data = [example for i, example in enumerate(data) if i % world_size == local_rank]

ds_model = deepspeed.init_inference(model,
                                    mp_size=world_size,
                                    dtype=torch.float,
                                    replace_method='auto',
					                replace_with_kernel_inject=True)


with open(f"/w/exp/mkt/generated/gen_{local_rank}.jsonl", "w") as fw:

    for example in sharded_data:
        input = example['input']
        prompts = [input]*8

        tokenizer.padding_side = 'left'
        prompts_tensor = tokenizer(prompts, padding="longest", return_tensors='pt').to(model.device)
        tokenizer.padding_side = 'right'

        temp = random()*2.0

        gen_ids_batch = ds_model.generate(**prompts_tensor,
            max_length=256,
            repetition_penalty=2.0,
            do_sample=True,
            temperature=temp,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
        )

        def get_label(gen_ids):
            generated = tokenizer.decode(gen_ids)
            start_index = generated.find("[마케팅 문구]") + 9
            label = generated[start_index:].replace("[EOS]", "").strip()
            return label

        labels = [get_label(gen_ids) for gen_ids in gen_ids_batch]

        fw.write(json.dumps({
            "input": input,
            "labels": labels,
            "model": "/w/exp/mkt/model_8.8B_BI_MT_02-4th/checkpoint-20",
            "temperature": temp,
        }, ensure_ascii=False)+"\n")


# gen_ids_batch = generate(
#     model,
#     length=128,
#     do_sample=True,
#     temperature=1.5,
#     **prepare_inputs_for_generation(**prompts_tensor),
# )
