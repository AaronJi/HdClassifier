"""Llama with the HdLM Depth2 output head and stage-specific supervision."""

import torch
from torch import nn
from torch.nn import functional as F
from transformers import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..data.hdlm_wos import role_header


def header_position(tokens: list[int], header: list[int], start: int = 0) -> int:
    for position in range(start, len(tokens) - len(header) + 1):
        if tokens[position : position + len(header)] == header:
            return position
    raise ValueError("A required HdLM role header is missing from the supervised sequence.")


class WoSHdLMForCausalLM(LlamaForCausalLM):
    accepts_loss_kwargs = False

    def __init__(self, config):
        super().__init__(config)
        self.think_layer_index = config.hdlm_depth2["think_layer_index"]
        self.new_lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def set_stage_headers(self, tokenizer) -> None:
        self.stage_headers = [
            tokenizer.encode(role_header(role), add_special_tokens=False)
            for role in ("thought", "subtask", "assistant")
        ]

    def stage_targets(self, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        thought_header, subtask_header, assistant_header = self.stage_headers
        thought_mask = torch.zeros_like(labels, dtype=torch.bool)
        final_mask = torch.zeros_like(labels, dtype=torch.bool)
        for row, tokens in enumerate(labels.tolist()):
            thought_start = header_position(tokens, thought_header) + len(thought_header)
            subtask_start = header_position(tokens, subtask_header, thought_start)
            assistant_start = header_position(
                tokens, assistant_header, subtask_start + len(subtask_header)
            ) + len(assistant_header)
            if thought_start == subtask_start or not any(token >= 0 for token in tokens[assistant_start:]):
                raise ValueError("Each HdLM stage must supervise at least one token.")
            thought_mask[row, thought_start:subtask_start] = True
            final_mask[row, assistant_start:] = True
        valid = labels != -100
        return thought_mask & valid, final_mask & valid

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        logits_to_keep=0,
        use_intermediate_head=False,
        **kwargs,
    ):
        inputs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
        )
        if labels is None and not use_intermediate_head:
            return super().forward(**inputs, logits_to_keep=logits_to_keep, **kwargs)

        kwargs["output_hidden_states"] = True
        outputs = self.model(**inputs, **kwargs)
        intermediate = outputs.hidden_states[self.think_layer_index]
        loss = None
        if labels is None:
            positions = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.new_lm_head(intermediate[:, positions, :])
        else:
            thought, final = self.stage_targets(labels)
            thought, final = thought[:, 1:], final[:, 1:]
            targets = labels[:, 1:]
            # Select the supervised positions before the intermediate vocabulary projection.
            thought_logits = self.new_lm_head(intermediate[:, :-1, :][thought])
            logits = self.lm_head(outputs.last_hidden_state)
            thought_loss = F.cross_entropy(thought_logits.float(), targets[thought])
            final_loss = F.cross_entropy(logits[:, :-1, :][final].float(), targets[final])
            config = self.config.hdlm_depth2
            loss = config["think_loss_weight"] * thought_loss + config["final_loss_weight"] * final_loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def load_wos_model(tokenizer, model_args, **init_kwargs):
    config = init_kwargs["config"]
    if config.model_type != "llama" or model_args.train_from_scratch:
        raise ValueError("HdLM WoS expects pretrained Llama weights.")

    saved = getattr(config, "hdlm_depth2", None)
    if saved is None:
        layer = model_args.hdlm_think_layer_index
        if layer is None or not 0 < layer < config.num_hidden_layers:
            raise ValueError("hdlm_think_layer_index must select an intermediate Llama layer.")
        config.hdlm_depth2 = {
            "think_layer_index": layer,
            "think_loss_weight": model_args.hdlm_think_loss_weight,
            "final_loss_weight": model_args.hdlm_final_loss_weight,
        }
    model, info = WoSHdLMForCausalLM.from_pretrained(**init_kwargs, output_loading_info=True)
    expected_missing = {"new_lm_head.weight"} if saved is None else set()
    if set(info["missing_keys"]) != expected_missing or any(
        info.get(key) for key in ("unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise ValueError(f"Incomplete HdLM checkpoint load: {info}")
    if saved is None:
        with torch.no_grad():
            model.new_lm_head.weight.copy_(model.lm_head.weight)

    model.set_stage_headers(tokenizer)
    return model
