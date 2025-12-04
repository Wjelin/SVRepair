from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.transformers import T5ForConditionalGeneration

from model.config import ModelConfig
from model.transformers.models.t5.configuration_t5 import T5Config


class SVRepairModel(nn.Module):
    def __init__(
            self,
            model_name_or_path,
            config: Union[str, ModelConfig] = "full"
        ):
        super().__init__()

        if isinstance(config, str):
            self.config = ModelConfig.from_preset(config)
        else:
            self.config = config
        t5_config = T5Config.from_pretrained(model_name_or_path)
        t5_config.use_struct_attn_bias = self.config.use_struct_attn_bias
        t5_config.struct_bias_init = self.config.struct_bias_init

        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name_or_path, config = t5_config)
        self.encoder = self.t5.encoder

        if self.config.use_auxiliary_task:
            self.dep_predictor = nn.Sequential(
                nn.Linear(self.t5.config.d_model, self.t5.config.d_model),
                nn.ReLU(),
                nn.Linear(self.t5.config.d_model, 1)
            )

    def forward(self, input_ids, attention_mask=None, labels=None, dep_distances=None, dep_scores=None,
                do_generate=False, num_beams=1):
        if not self.config.use_struct_attn_bias:
            dep_distances = None

        if do_generate:
            beam_outputs = self.t5.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                max_length=256,
                dep_distances=dep_distances
            )
            return beam_outputs

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
            dep_distances=None
        )
        encoder_hidden_states = encoder_outputs.last_hidden_state

        dep_loss = None
        if self.config.use_auxiliary_task:
            dep_logits = self.dep_predictor(encoder_hidden_states).squeeze(-1)
            dep_loss = F.smooth_l1_loss(dep_logits, dep_scores, reduction='none')
            dep_loss = dep_loss * attention_mask
            dep_loss = dep_loss.sum() / attention_mask.sum()

        outputs = self.t5(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
            dep_distances=dep_distances,
        )
        loss = outputs.loss

        return {
            "loss": loss,
            "dep_loss": dep_loss
        }
