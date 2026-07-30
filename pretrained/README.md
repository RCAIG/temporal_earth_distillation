# Pretrained checkpoints

HuggingFace-style model zoo for paper table anchors (full-scale **12b768**, best epoch by Glance+CropHarvest linear spatial nocap).

| ID / alias | Family | Epoch | Job | Weights |
|------------|--------|-------|-----|---------|
| `ted` / `ted-hls-12b768` | TED + cXattnB | 72 | 9616 | `ted-hls-12b768/pytorch_model.bin` |
| `msm` / `msm-hls-12b768` | MSM (reg4) | 25 | 10909 | `msm-hls-12b768/pytorch_model.bin` |
| `ntp` / `ntp-hls-12b768` | NTP | 25 | 10685 | `ntp-hls-12b768/pytorch_model.bin` |

Each folder contains:

- `config.json` — architecture + historical `model_id` (used by downstream flag inference)
- `pytorch_model.bin` — `torch.save` state dict (same format as `checkpoint_epoch_*.pth`)
- `checkpoint.pth` — symlink to the weight file

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

Naming follows common HF / vision SSL practice (`org`-optional + domain + size), e.g. `dinov2-base`, `vit-base-patch16`, here: `{method}-hls-12b768`.
