# Pretrained Checkpoints

This folder stores Hugging Face-style metadata for the paper-aligned full-scale **12b768** checkpoints. Model weights are not bundled in git at this stage.

Model checkpoints will be made available upon publication. For peer review, access can be provided to editors and reviewers upon request.

| ID / alias | Family | Metadata | Expected weight path |
|------------|--------|----------|----------------------|
| `ted` / `ted-hls-12b768` | Temporal Earth Distillation | `ted-hls-12b768/config.json` | `ted-hls-12b768/pytorch_model.bin` |
| `msm` / `msm-hls-12b768` | Masked Sequence Modeling | `msm-hls-12b768/config.json` | `msm-hls-12b768/pytorch_model.bin` |
| `ntp` / `ntp-hls-12b768` | Next-Token Prediction | `ntp-hls-12b768/config.json` | `ntp-hls-12b768/pytorch_model.bin` |

Each folder currently contains `config.json` with architecture and release metadata. When weights are available, place `pytorch_model.bin` under the matching folder. The helper script can restore weights from a local release bundle:

```bash
bash scripts/ensure_pretrained_weights.sh
# optional: PRETRAINED_BUNDLE=/path/to/pretrained_release
```

## Downstream Usage

```bash
# short alias
CHECKPOINT=ted MODEL=TED bash scripts/eval_downstream.sh

# zoo folder name
CHECKPOINT=msm-hls-12b768 MODEL=MSM bash scripts/eval_downstream.sh

# or pass the directory / weight file
python eval_downstream.py --checkpoint pretrained/ted-hls-12b768 --model TED ...
python eval_downstream.py --checkpoint pretrained/ntp-hls-12b768/pytorch_model.bin --model NTP ...
```

```python
from utils.pretrained import resolve_pretrained, list_pretrained
from eval.load_pretrained import from_pretrained

print(list_pretrained())
weight, config, model_dir = resolve_pretrained("ted")

model, config, family = from_pretrained("ted", device="cuda:0")
# MSM / NTP: from_pretrained("msm") / from_pretrained("ntp")
```

Naming follows common model-zoo practice: `{method}-hls-12b768`.