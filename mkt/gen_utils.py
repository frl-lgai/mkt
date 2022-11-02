import torch
from torch import softmax

def generate(
        model, 
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        do_sample=True,
        temperature=1.0,
        length=20,
        end_token_id=50256,
        **kwargs
    ):
    """Sample text from language model."""

    end_indexes = [length-1]*len(input_ids)

    for gen_step in range(length):
        
        # Get Logits for the last token
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True
        ).logits[:, -1, :]

        if do_sample:
            next_input_ids = torch.multinomial(softmax(logits/temperature, dim=-1), num_samples=1)
        else:
            next_input_ids = torch.argmax(logits, dim=-1).unsqueeze(-1)
        
        next_attention_mask = attention_mask[:, -1:]
        next_position_ids   = position_ids[:, -1:] + 1

        for i, end_index in enumerate(end_indexes):

            if end_index < gen_step:
                next_attention_mask[i] = 0
                next_position_ids[i] = 1

            elif next_input_ids[i] == end_token_id:
                end_indexes[i] = gen_step

        input_ids      = torch.cat([input_ids,      next_input_ids     ], dim=-1)
        attention_mask = torch.cat([attention_mask, next_attention_mask], dim=-1)
        position_ids   = torch.cat([position_ids,   next_position_ids  ], dim=-1)

    for i, end_index in enumerate(end_indexes):
        if end_index < length - 1:
            input_ids[i,      -length+end_index+1:] = end_token_id
            attention_mask[i, -length+end_index+1:] = 0
            position_ids[i,   -length+end_index+1:] = 1

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'position_ids': position_ids,
    }, end_indexes


def prepare_inputs_for_generation(input_ids, past=None, **kwargs):
    token_type_ids = kwargs.get("token_type_ids", None)
    # only last token for inputs_ids if past is defined in kwargs
    if past:
        input_ids = input_ids[:, -1].unsqueeze(-1)
        if token_type_ids is not None:
            token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

    attention_mask = kwargs.get("attention_mask", None)
    position_ids = kwargs.get("position_ids", None)

    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        if past:
            position_ids = position_ids[:, -1].unsqueeze(-1)
    else:
        position_ids = None
    return {
        "input_ids": input_ids,
        "past_key_values": past,
        "use_cache": kwargs.get("use_cache"),
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }
