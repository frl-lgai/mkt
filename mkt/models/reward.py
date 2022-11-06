import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import GPT2ForSequenceClassification, GPTLingvoForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutputWithPast


class GPT2ForRewardModel(GPT2ForSequenceClassification):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias"]
    
    def __init__(self, config):
        config.num_labels = 1
        self.main_input_name = 'input_ids_pos'

        super().__init__(config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        input_ids_pos=None,
        attention_mask_pos=None,
        input_ids_neg=None,
        attention_mask_neg=None,
        past_key_values=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None):

        if input_ids is None:
            input_ids = input_ids_pos
            attention_mask = attention_mask_pos

        outputs_pos = super().forward(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        loss = None

        if input_ids_neg is not None:

            outputs_neg = super().forward(
                input_ids_neg,
                attention_mask=attention_mask_neg,
                past_key_values=past_key_values,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

            loss = - F.logsigmoid(outputs_pos.logits - outputs_neg.logits).mean()

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=outputs_pos.logits,
            past_key_values=past_key_values,
            hidden_states=outputs_pos.hidden_states,
            attentions=outputs_pos.attentions,
        )



class GPTLingvoForRewardModel(GPTLingvoForSequenceClassification):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias"]
    
    def __init__(self, config):
        config.num_labels = 1
        self.main_input_name = 'input_ids_pos'

        super().__init__(config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        input_ids_pos=None,
        attention_mask_pos=None,
        input_ids_neg=None,
        attention_mask_neg=None,
        past_key_values=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None):

        if input_ids is None:
            input_ids = input_ids_pos
            attention_mask = attention_mask_pos

        outputs_pos = super().forward(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        loss = None

        if input_ids_neg is not None:

            outputs_neg = super().forward(
                input_ids_neg,
                attention_mask=attention_mask_neg,
                past_key_values=past_key_values,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

            loss = - F.logsigmoid(outputs_pos.logits - outputs_neg.logits).mean()

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=outputs_pos.logits,
            past_key_values=past_key_values,
            hidden_states=outputs_pos.hidden_states,
            attentions=outputs_pos.attentions,
        )

