import json
import random

import streamlit as st

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

model_name = "/w/exp/mkt/model_8.8B_BI_MT_02-4th/checkpoint-20"

config = AutoConfig.from_pretrained(model_name)
config.gradient_checkpointing = True
config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, config=config).to("cuda")
model.resize_token_embeddings(len(tokenizer))


st.title(f"마케팅 문구 생성 데모")

with st.form(key='submit'):

    prompt = st.text_input('Prompt', """다음의 상품을 홍보하는 창의적인 마케팅 문구를 작성해 보세요.
[회사] 동아오츠카
[카테고리] 식품, 음료, 주류
[분류]
[상품명] 오로나민C
[마케팅 문구] """)

    st.form_submit_button(label='출력하기',  on_click=lambda: infer(prompt))


def infer(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to("cuda")

    for i in range(3):
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

        st.write('-'*50)
        st.write(generated)

