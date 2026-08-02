# Pretrained checkpoints

Hugging Face-style model zoo for paper-aligned full-scale **12b768** checkpoints.

| ID / alias | Family | Released checkpoint | Weights |
|------------|--------|---------------------|---------|
| `ted` / `ted-hls-12b768` | Temporal Earth Distillation | paper release | `ted-hls-12b768/pytorch_model.bin` |
| `msm` / `msm-hls-12b768` | Masked Sequence Modeling | paper baseline | `msm-hls-12b768/pytorch_model.bin` |
| `ntp` / `ntp-hls-12b768` | Next-Token Prediction | paper baseline | `ntp-hls-12b768/pytorch_model.bin` |

Each folder contains:

- `config.json` - architecture and release metadata
- `pytorch_model.bin` - `torch.save` state dict
- `checkpoint.pth` - symlink to the weight file

Weights are large and gitignored. Restore / detach hardlinks with:

```bash
bash scripts/ensure_pretrained_weights.sh
```

## Downstream usage

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
