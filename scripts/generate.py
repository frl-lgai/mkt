import os
import wandb

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

model_name = "/w/exp/mkt/model_1.7B_BI_MT_02/checkpoint-150"

config = AutoConfig.from_pretrained(model_name)
config.gradient_checkpointing = True
config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
model.resize_token_embeddings(len(tokenizer))

prompt = """다음의 상품을 홍보하는 창의적인 마케팅 문구를 작성해 보세요.
[회사] 동아오츠카
[카테고리] 식품, 음료, 주류
[분류]
[상품명] 오로나민C
[마케팅 문구] """
input_ids = tokenizer.encode(prompt, return_tensors='pt')

for i in range(10):
    gen_ids = model.generate(input_ids,
                            max_length=256,
                            repetition_penalty=2.0,
                            do_sample=True,
                            temperature=1.5,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            bos_token_id=tokenizer.bos_token_id,
                            use_cache=True)

    generated = tokenizer.decode(gen_ids[0][:-1])
    print(generated)