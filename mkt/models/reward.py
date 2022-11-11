import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import GPT2ForSequenceClassification, AutoModelForSequenceClassification
from transformers.models.gpt_lingvo import GPTLingvoForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutputWithPast


class RewardModelMixin:
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias"]

    def __init__(self, config):
        config.num_labels = 1
        super().__init__(config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        input_ids_neg=None,
        attention_mask_neg=None,
        past_key_values=None, # necessary for SequenceClassifierOutputWithPast
        **kwargs):

        kwargs['return_dict'] = True

        outputs = super().forward(input_ids, attention_mask=attention_mask, **kwargs)
        loss = None

        if input_ids_neg is not None:
            outputs_neg = super().forward(input_ids_neg, attention_mask=attention_mask_neg, **kwargs)
            loss = - F.logsigmoid(outputs.logits - outputs_neg.logits).mean()

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=outputs.logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            past_key_values=past_key_values,
        )


class GPT2ForRewardModel(RewardModelMixin, GPT2ForSequenceClassification):
    pass

class GPTLingvoForRewardModel(RewardModelMixin, GPTLingvoForSequenceClassification):
    pass


class AutoModelForRewardModel(AutoModelForSequenceClassification):
    _model_mapping = {
        "gpt2": GPT2ForRewardModel,
        "gpt_lingvo": GPTLingvoForRewardModel,
    }

    @classmethod
    def from_pretrained(cls, model_name_or_path, config):

        if config.model_type not in cls._model_mapping:
            raise Exception("Now \'gpt2\' and \'gpt_lingvo\' in config.model_type are only supported.")
        
        model_class = cls._model_mapping[config.model_type]
        return model_class.from_pretrained(model_name_or_path, config=config)
