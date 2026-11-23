# HdClassifier

HdClassifier provides a runnable HdLM example for two-level hierarchical text classification on Web of Science (WoS), using `Llama-3.2-3B-Instruct` as the example backbone. The default configuration runs ten training updates and ten test examples so you can verify the full workflow before adapting it to another model or dataset.

## Setup

Run the commands below from the repository root in PowerShell with your Python environment active. The example was tested on Python 3.13 with PyTorch 2.11.0 built for CUDA 12.8. Use a BF16-capable GPU and install the matching CUDA builds of PyTorch, torchvision, and torchaudio for your platform before installing the remaining dependencies.

```powershell
python -m pip install -r requirements.txt

$env:PYTHONPATH = 'vendor/LlamaFactory/src'
$env:HF_HUB_OFFLINE = '1'
$env:HF_DATASETS_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:USE_V1 = '0'
```

The commands use the vendored LlamaFactory source and local model and data files. The Windows launcher `scripts/run_wos.ps1` provides `-Stage train` and `-Stage infer`. It selects `../../anaconda3/python.exe` and includes the local PEFT dependency directory in its Python search path.

## Training

1. In `configs/wos_llama32_3b_train.yaml`, replace `<BASE_MODEL_ROOT>` with the parent directory of your local `Llama-3.2-3B-Instruct` model.
2. Start supervised fine-tuning through the LlamaFactory CLI.

```powershell
python -m llamafactory.cli train configs/wos_llama32_3b_train.yaml
```

LlamaFactory reads `WoS-HdLM` from `data/dataset_info.json`. Dataset file paths in the registry are relative to `data/`. The WoS records contain `system` and `input` for the prompt, `think` for the domain ID, `subtask` for the domain-specific area prompt, and `assistant` for the area ID.

The example trains the last four Transformer blocks and the intermediate output head. The intermediate head reads `hidden_states[25]` and predicts the domain. The final language-model head predicts the area. The training objective weights the domain and area losses 2:1. The default configuration uses BF16 and a per-device batch size of 1. The checkpoint and optimizer state at step 10 are saved to `runs/wos_llama32_3b_yaml/checkpoint-10/`.

## Inference

Set `model_name_or_path` in `configs/wos_llama32_3b_infer.yaml` to the trained checkpoint directory. For the default training run, use `runs/wos_llama32_3b_yaml/checkpoint-10`.

```powershell
python scripts/infer_wos.py configs/wos_llama32_3b_infer.yaml
```

The script reads the first ten records registered as `WoS-HTC-Test`. It decodes the domain with the intermediate head, then uses that predicted domain to build the area candidate list for the final head. Both stages use greedy decoding and retry with candidate-constrained decoding when the generated value is invalid. The script writes `wos_test_predictions.json` one directory above the checkpoint directory.

## Evaluation

The prediction report stores the predicted and gold `[domain, area]` pairs, the raw output from each stage, and whether candidate-constrained decoding was used. Divide `domain_matches` and `joint_matches` by `sample_count` to get domain accuracy and exact-path accuracy, respectively.

To compute Micro-F1 (MiF1) and Macro-F1 (MaF1), use the entire `[domain, area]` pair as a single label. The following code reads the report produced by the default run.

```python
import json
from pathlib import Path

from sklearn.metrics import f1_score

report_path = Path("runs/wos_llama32_3b_yaml/wos_test_predictions.json")
rows = json.loads(report_path.read_text(encoding="utf-8"))["predictions"]
gold = ["/".join(map(str, row["gold"])) for row in rows]
predicted = ["/".join(map(str, row["predicted"])) for row in rows]

print(f"MiF1: {100 * f1_score(gold, predicted, average='micro'):.2f}")
print(f"MaF1: {100 * f1_score(gold, predicted, average='macro'):.2f}")
```

Set `max_samples` to `null` in the inference YAML to score the entire selected split.

## Adapting the example

- **Backbone.** Try another pretrained Llama model by changing `model_name_or_path` and setting `hdlm_think_layer_index` for its depth. The bundled template expects the Llama 3 chat special tokens. For another model family, adapt the model loader, stage-specific heads, and tokenizer template.
- **Dataset.** For another two-level task, convert and register the splits, update both YAML dataset names, and adapt the prompt and parent-to-child label lists in `vendor/LlamaFactory/src/llamafactory/data/hdlm_wos_prompts.py`. The example decoder expects integer IDs and one label path per record. The repository includes converted DBpedia data. Running its three-level task requires a three-stage model and decoder.
