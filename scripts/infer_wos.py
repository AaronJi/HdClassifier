"""Run the two HdLM decoding stages using a LlamaFactory inference YAML."""

import json
from collections import defaultdict
from pathlib import Path

import torch
from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.data.hdlm_wos import EOT, role_header
from llamafactory.data.hdlm_wos_prompts import Area_dict, Domain_dict, area_task_prompt
from llamafactory.data.parser import get_dataset_list
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_model, load_tokenizer


def encode(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def parse_id(text, candidates):
    text = text.strip()
    if text.isascii() and text.isdecimal() and int(text) in candidates:
        return int(text)
    return None


def generate_label(model, tokenizer, prefix, candidates, intermediate, generation):
    inputs = torch.tensor([prefix], dtype=torch.long, device=model.device)
    stop = tokenizer.convert_tokens_to_ids(EOT)
    options = {
        **generation,
        "input_ids": inputs,
        "attention_mask": torch.ones_like(inputs),
        "use_intermediate_head": intermediate,
        "eos_token_id": [tokenizer.convert_tokens_to_ids("<|end_of_text|>"), stop],
        "pad_token_id": stop,
    }

    def generate(**overrides):
        ids = model.generate(**{**options, **overrides})[0, len(prefix):]
        return tokenizer.decode(ids.tolist(), skip_special_tokens=True)

    raw = generate()
    label = parse_id(raw, candidates)
    if label is not None:
        return label, raw, False

    paths = [encode(tokenizer, str(label)) + [stop] for label in candidates]
    next_tokens = defaultdict(set)
    for path in paths:
        for position, token in enumerate(path):
            next_tokens[tuple(path[:position])].add(token)

    def allowed_tokens(_batch, token_ids):
        return sorted(next_tokens[tuple(token_ids[len(prefix):].tolist())])

    constrained = generate(
        max_new_tokens=max(map(len, paths)),
        prefix_allowed_tokens_fn=allowed_tokens,
    )
    label = parse_id(constrained, candidates)
    if label is None:
        raise ValueError(f"HdLM failed to generate a candidate ID: {constrained!r}")
    return label, raw, True


@torch.inference_mode()
def main():
    model_args, data_args, finetuning_args, generating_args = get_infer_args()
    dataset, = get_dataset_list(data_args.dataset, data_args.dataset_dir)
    source = Path(data_args.dataset_dir) / dataset.dataset_name
    with source.open(encoding="utf-8") as handle:
        samples = json.load(handle)[:data_args.max_samples]

    tokenizer = load_tokenizer(model_args)["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    model = load_model(tokenizer, model_args, finetuning_args, is_trainable=False)
    generation = generating_args.to_dict(obey_generation_config=True)
    print(f"Decoding {len(samples)} WoS samples on {model.device}.", flush=True)

    predictions = []
    for index, sample in enumerate(samples, 1):
        messages = [{"role": "user", "content": sample["input"]}, {"role": "assistant", "content": ""}]
        prefix, _ = template.encode_oneturn(tokenizer, messages, system=sample["system"])
        domain, raw_domain, domain_retry = generate_label(
            model, tokenizer, prefix + encode(tokenizer, role_header("thought")), Domain_dict, True, generation
        )
        areas = Area_dict[Domain_dict[domain]]
        subtask = area_task_prompt.format(Area_subdict=areas)
        prefix += encode(tokenizer, role_header("thought") + str(domain) + EOT)
        prefix += encode(tokenizer, role_header("subtask") + subtask + EOT + role_header("assistant"))
        area, raw_area, area_retry = generate_label(model, tokenizer, prefix, areas, False, generation)
        gold = [int(sample["think"]), int(sample["assistant"])]
        predictions.append(
            {
                "sample": index,
                "predicted": [domain, area],
                "gold": gold,
                "raw_domain": raw_domain,
                "raw_area": raw_area,
                "domain_constrained": domain_retry,
                "area_constrained": area_retry,
            }
        )
        print(f"{index}/{len(samples)} predicted={[domain, area]} gold={gold}", flush=True)

    checkpoint = Path(model_args.model_name_or_path)
    summary = {
        "checkpoint": checkpoint.as_posix(),
        "dataset": source.as_posix(),
        "sample_count": len(predictions),
        "domain_matches": sum(row["predicted"][0] == row["gold"][0] for row in predictions),
        "joint_matches": sum(row["predicted"] == row["gold"] for row in predictions),
        "domain_constrained": sum(row["domain_constrained"] for row in predictions),
        "area_constrained": sum(row["area_constrained"] for row in predictions),
    }
    output = checkpoint.parent / "wos_test_predictions.json"
    output.write_text(json.dumps({"summary": summary, "predictions": predictions}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
