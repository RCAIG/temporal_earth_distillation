import os
import json
import random
import time
import warnings
import threading
import queue

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import optim
from torch.cuda.amp import GradScaler

from utils.amp_utils import amp_autocast_ctx, resolve_amp_dtype
from torch.nn.parallel import DistributedDataParallel as DDP

from data.factory import data_provider
from engine.base_trainer import BaseTrainer
from models.imputator import load_pretrained_imputator
from utils.losses import TEDCriterion
from utils.schedulers import apply_optim_scheduler, build_schedulers
from utils.tools import save_model_periodically, EarlyStopping, resolve_fft_align_lambda_for_epoch


warnings.filterwarnings('ignore')

"""
Self-supervised / representation pretraining main loop (TED, MSM, NTP).
"""


def get_total_params(model):
    """Count total model parameters"""
    # unwrap nn.DataParallel model
    model = model.module if isinstance(model, nn.DataParallel) else model
    return sum(p.numel() for p in model.parameters())

class SSLTrainer(BaseTrainer):
    def __init__(self, args):
        super(SSLTrainer, self).__init__(args)
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        # Downstream probe/eval entry points intentionally omitted in this package.
        # --- 1. Initialize Loss module (globalunique, shared state) ---
        self.criterion = TEDCriterion(self.args, self.device)

    def _compute_loss(self, outputs):
        """
        Compute loss from model output. For MSM / NTP SSL models,
        use outputs['ssl_loss'] directly; else use TEDCriterion.
        """
        if 'ssl_loss' in outputs:
            loss = outputs['ssl_loss']
            log_vars = outputs.get('log_vars', {})
            return loss, log_vars
        return self.criterion(outputs)

    def _build_model(self):
        # 1. initialize model
        model = self.model_dict[self.args.model].Model(self.args).float()
        
        # 2. move to target GPU
        model.to(self.device)

        # 3. DDP wrap
        if self.args.use_multi_gpu:
            # SyncBN (if BatchNorm layers present)
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            fu = bool(int(getattr(self.args, 'ddp_find_unused_parameters', 0)))
            if self.args.local_rank == 0 and not fu:
                print('[DDP] find_unused_parameters=False (speed); set --ddp_find_unused_parameters 1 if backward complains)')
            model = DDP(model, device_ids=[self.args.local_rank], find_unused_parameters=fu)

        return model

    @staticmethod
    def _normalize_compiled_state_dict_keys(state_dict):
        """
        strip torch.compile `_orig_mod` prefix for compatibility.
        only strip compile prefixes, not DDP `module.` (caller decides).
        """
        if not isinstance(state_dict, dict):
            return state_dict, False

        normalized = {}
        changed = False
        for k, v in state_dict.items():
            nk = k
            if nk.startswith('_orig_mod.'):
                nk = nk[len('_orig_mod.'):]
                changed = True
            while '._orig_mod.' in nk:
                nk = nk.replace('._orig_mod.', '.')
                changed = True
            normalized[nk] = v
        return normalized, changed

    def _get_data(self, flag, data=None):
        data_set, data_loader = data_provider(self.args, flag, data)
        return data_set, data_loader

    def _select_optimizer(self):
        # param groups: regular (decay) vs no-decay (bias, norm, head)
        regular_groups = []
        noreg_groups = []
        last_layer_groups = []  # last layer (special handling)
        
        # The self.model here has been DDP wrapped, so use .named_parameters() iterate
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            is_last_layer = "last_layer.weight" in name
            
            # 1. Bias, LayerNorm, BatchNorm do not decay
            if name.endswith(".bias") or ".bn" in name or ".norm" in name:
                if is_last_layer:
                    last_layer_groups.append({
                        'params': [param],
                        'is_last_layer': True,
                        'weight_decay': 0.0,
                        'scheduled_weight_decay': False,
                    })
                else:
                    noreg_groups.append(param)
            # 2. The last categorical-state layer should not use weight decay
            elif is_last_layer:
                last_layer_groups.append({
                    'params': [param],
                    'is_last_layer': True,
                    'weight_decay': 0.0,
                    'scheduled_weight_decay': False,
                })
            # 3. Embedding and Token do not decay either
            elif "embedding" in name or "cls_token" in name or "mask_token" in name:
                noreg_groups.append(param)
            else:
                regular_groups.append(param)
        
        param_groups = [
            {
                'params': regular_groups,
                'weight_decay': self.args.weight_decay,
                'is_last_layer': False,
                'scheduled_weight_decay': True,
            },
            {
                'params': noreg_groups,
                'weight_decay': 0.0,
                'is_last_layer': False,
                'scheduled_weight_decay': False,
            },
        ]
        
        # Add the last layer group
        param_groups.extend(last_layer_groups)
        
        # recommend AdamW for correct weight decay
        model_optim = optim.AdamW(param_groups, lr=self.args.learning_rate) 
        return model_optim

    def load_pretrained_model(self, checkpoint_path=None):
        """
        Load pretrained model weights.

        Args:
            checkpoint_path: checkpoint file path (optional)
                - if None, look for checkpoint.pth under args.pretrain_model
                - if str, use path directly (file or directory)
                - if directory, look for checkpoint.pth inside
                - if file, load directly
        """
        # resolve checkpoint path
        if checkpoint_path is None:
            if self.args.pretrain_model is None:
                raise ValueError("args.pretrain_model not set; cannot load pretrained model")
            
            # check file vs directory
            if os.path.isfile(self.args.pretrain_model):
                checkpoint_path = self.args.pretrain_model
            elif os.path.isdir(self.args.pretrain_model):
                checkpoint_path = os.path.join(self.args.pretrain_model, 'checkpoint.pth')
            else:
                raise FileNotFoundError(f"pretrained model path does not exist: {self.args.pretrain_model}")
        else:
            # If checkpoint_path is provided, check file vs directory
            if os.path.isfile(checkpoint_path):
                # direct file path
                pass
            elif os.path.isdir(checkpoint_path):
                # directory; find checkpoint.pth
                checkpoint_path = os.path.join(checkpoint_path, 'checkpoint.pth')
            else:
                raise FileNotFoundError(f"checkpoint path does not exist: {checkpoint_path}")
        
        # check file exists
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"checkpoint file does not exist: {checkpoint_path}")
        
        if self.args.local_rank == 0:
            print(f"Loading pretrained weights: {checkpoint_path}") 
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        except Exception as e:
            raise RuntimeError(f"failed to load checkpoint: {e}")

        # handle checkpoint dict with 'model' key
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            checkpoint = checkpoint['model']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']

        state_dict, changed_compile = self._normalize_compiled_state_dict_keys(checkpoint)
        if changed_compile and self.args.local_rank == 0:
            print("[LoadPretrained] Detected compiled checkpoint; removed _orig_mod prefix.")

        model_has_module = hasattr(self.model, 'module')
        checkpoint_has_module = any(k.startswith('module.') for k in state_dict.keys())

        if model_has_module and not checkpoint_has_module:
            state_dict = {f'module.{k}': v for k, v in state_dict.items()}
        elif not model_has_module and checkpoint_has_module:
            state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

        # get current model state_dict for shape checks
        model_state_dict = self.model.state_dict()
        
        # filterArgs:position encoding parameter related to skipdecoder and shapemismatch
        # The downstream task (probe task) does not need decoder, so you can skip all decoder related parameters.
        filtered_state_dict = {}
        skipped_keys = []
        
        # needsskip's parametermode (downstream tasks do not need these)
        skip_patterns = [
            'decoder_pos_embed',           # decoderpositionencoding
            'decoder.',                    # decoderlayer (all decoder related layers)
            'pixel_decoder',               # pixel-leveldecoder (used in reconstruction)
        ]
        
        # position encoding parameter (supports dynamic processing, shapemismatchwhenskip)
        position_encoding_keys = ['position_encoding.pos_table']
        
        for k, v in state_dict.items():
            # Check whether it is a decoder-related parameter (downstream task does not need)
            is_decoder_related = any(pattern in k for pattern in skip_patterns)
            
            if is_decoder_related:
                # skip all decoder related parameters
                skipped_keys.append(k)
                # Do not print the decoderparameter of each skip, only count it at the end
            elif k in model_state_dict:
                # Check whether it is a parameter related to position encoding
                is_pos_encoding = any(pos_key in k for pos_key in position_encoding_keys)
                
                if is_pos_encoding and v.shape != model_state_dict[k].shape:
                    # skipshapemismatch's position encoding parameter (forwardwhen already supports dynamic processing)
                    skipped_keys.append(k)
                    if self.args.local_rank == 0:
                        print(f"skipposition encoding parameter (shapemismatch, forwardwhen supports dynamic processing): {k} "
                              f"checkpoint shape: {v.shape}, model shape: {model_state_dict[k].shape}")
                elif v.shape == model_state_dict[k].shape:
                    # shapematch, can load
                    filtered_state_dict[k] = v
                else:
                    # Other shapemismatch parameters are also skipped.
                    skipped_keys.append(k)
                    if self.args.local_rank == 0:
                        print(f"skipparameter（shapemismatch）: {k} "
                              f"checkpoint shape: {v.shape}, model shape: {model_state_dict[k].shape}")
            else:
                # There is no such key in the model, skip
                skipped_keys.append(k)

        load_result = self.model.load_state_dict(filtered_state_dict, strict=False)

        if self.args.local_rank == 0:
            matched_keys_count = len(filtered_state_dict) - len(load_result.missing_keys)
            total_keys = len(state_dict)
            decoder_skipped = sum(1 for k in skipped_keys if any(p in k for p in ['decoder', 'pixel_decoder']))
            pos_encoding_skipped = sum(1 for k in skipped_keys if 'position_encoding' in k or 'decoder_pos_embed' in k)
            other_skipped = len(skipped_keys) - decoder_skipped - pos_encoding_skipped
            
            print(f"parameterload statistics: {matched_keys_count}/{total_keys}keymatchsuccess")
            if decoder_skipped > 0:
                print(f"Args related to skip's decoder: {decoder_skipped} (downstream task does not need decoder)")
            if pos_encoding_skipped > 0:
                print(f"skip position encoding Args: {pos_encoding_skipped} (shapemismatch, forwardwhen support dynamic processing)")
            if other_skipped > 0:
                print(f"Other Args of skip: {other_skipped} (None in shapemismatch or model)")
            if len(load_result.missing_keys) > 0:
                print(f"missing key count: {len(load_result.missing_keys)} (Top 3: {load_result.missing_keys[:3]})")
            if len(load_result.unexpected_keys) > 0:
                print(f"Unexpected key count: {len(load_result.unexpected_keys)} (Top 3: {load_result.unexpected_keys[:3]})")

    def _get_rng_state(self):
        """The RNG status of savecurrentprocess is used to resume training at breakpoints and try to keep random sequence consistent."""
        rngState = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch_cpu': torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            rngState['torch_cuda_all'] = torch.cuda.get_rng_state_all()
        return rngState

    def _set_rng_state(self, rngState):
        """The RNG status of resumecurrentprocess."""
        if not isinstance(rngState, dict):
            return
        try:
            if 'python' in rngState:
                random.setstate(rngState['python'])
            if 'numpy' in rngState:
                np.random.set_state(rngState['numpy'])
            if 'torch_cpu' in rngState:
                torch.set_rng_state(rngState['torch_cpu'])
            if torch.cuda.is_available() and 'torch_cuda_all' in rngState:
                torch.cuda.set_rng_state_all(rngState['torch_cuda_all'])
        except Exception as e:
            if getattr(self.args, 'local_rank', 0) == 0:
                print(f"[Resume] RNG state restore warning: {e}")

    def _resolve_resume_path(self, resumePath):
        """Parse resume path. Supports transferring directories or files; priority is given to rank-specific files under DDP."""
        if resumePath is None:
            return None
        if isinstance(resumePath, str) and resumePath.strip().lower() in ('', 'none', 'null'):
            return None

        rankId = int(getattr(self.args, 'local_rank', 0))
        isDdp = bool(getattr(self.args, 'use_multi_gpu', False))

        if os.path.isdir(resumePath):
            if isDdp:
                rankPath = os.path.join(resumePath, f"train_state_rank{rankId}.pth")
                if os.path.exists(rankPath):
                    return rankPath
            singlePath = os.path.join(resumePath, "train_state.pth")
            if os.path.exists(singlePath):
                return singlePath
            return None

        if os.path.isfile(resumePath):
            if isDdp:
                baseDir = os.path.dirname(resumePath)
                rankPath = os.path.join(baseDir, f"train_state_rank{rankId}.pth")
                if os.path.exists(rankPath):
                    return rankPath
            return resumePath

        return None

    def train(self, setting):
        # --- 1. data loading and environment initialization ---
        train_data, train_loader = self._get_data(flag='train')

        if self.args.local_rank == 0:
            total_params = get_total_params(self.model)
            print(f"modelArgs: {total_params:,} ({total_params / 1e6:.2f}M)")
        
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path) and self.args.local_rank == 0:
            os.makedirs(path)
            full_name = getattr(self.args, "checkpoint_setting_full", None)
            if full_name:
                try:
                    meta_path = os.path.join(path, "setting_full.txt")
                    with open(meta_path, "w", encoding="utf-8") as f:
                        f.write(full_name + "\n")
                except OSError as e:
                    print(f"[Checkpoints] Could not write setting_full.txt: {e}")

        # rank0: write all indicators async to JSONL in each step
        step_log_queue = None
        step_log_thread = None
        step_log_stop_token = object()
        step_log_drop_counter = 0
        if self.args.local_rank == 0 and bool(int(getattr(self.args, "train_step_log_enable", 1))):
            step_log_file_cfg = str(getattr(self.args, "train_step_log_file", "auto"))
            if step_log_file_cfg == "auto":
                logs_dir = os.path.join(".", "logs")
                os.makedirs(logs_dir, exist_ok=True)
                safe_model_id = str(getattr(self.args, "model_id", "train")).replace("/", "_")
                step_log_file = os.path.join(logs_dir, f"{safe_model_id}.jsonl")
            elif os.path.isabs(step_log_file_cfg):
                step_log_file = step_log_file_cfg
            else:
                step_log_file = os.path.join(".", step_log_file_cfg)
                parent = os.path.dirname(step_log_file)
                if parent:
                    os.makedirs(parent, exist_ok=True)
            step_log_queue = queue.Queue(maxsize=8192)

            def _step_log_worker():
                with open(step_log_file, "a", encoding="utf-8") as fout:
                    while True:
                        item = step_log_queue.get()
                        if item is step_log_stop_token:
                            break
                        fout.write(json.dumps(item, ensure_ascii=False) + "\n")

            step_log_thread = threading.Thread(
                target=_step_log_worker,
                name="train-step-jsonl-writer",
                daemon=True,
            )
            step_log_thread.start()

        time_now = time.time()
        train_steps = len(train_loader)
        total_iterations = train_steps * self.args.train_epochs  # Calculate the total number of iterations, used in curriculum learning
        
        # Inputator use early stop (based on train loss), TED/DINO class modeldo not use
        is_imputator = (self.args.model in ['Imputator', 'Transformer'])
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True) if is_imputator else None
        scaler = GradScaler() 
        model_optim = self._select_optimizer()

        if self.args.local_rank == 0 and self.args.use_amp and torch.cuda.is_available():
            _dt = resolve_amp_dtype(self.args)
            print(
                f"[AMP] amp_dtype={getattr(self.args, 'amp_dtype', 'auto')} -> autocast {_dt}"
                if _dt is not None
                else "[AMP] disabled"
            )

        # --- 2. Dynamic scheduler initialization ---
        # Build all schedulers（LR, WD, Momentum, Teacher Temp）
        lr_schedule, wd_schedule, momentum_schedule, teacher_temp_schedule, last_layer_lr_schedule = build_schedulers(
            self.args, train_steps
        )
        
        if self.args.local_rank == 0:
            print(f"Total iterations: {total_iterations}")
            print(f"Steps per epoch: {train_steps}")
            print(f"scheduler_version: {getattr(self.args, 'scheduler_version', 'cosine')}")
            print(f"Warmup epochs: {self.args.warmup_epochs}")
            print(f"LR: {self.args.learning_rate} -> {getattr(self.args, 'min_lr', self.args.learning_rate * 1e-6)}")
            print(f"WD: {self.args.weight_decay} -> {getattr(self.args, 'weight_decay_end', self.args.weight_decay * 10)}")
            print(f"Momentum: {getattr(self.args, 'momentum_teacher', 0.992)} -> {getattr(self.args, 'final_momentum_teacher', 1.0)}")
            print(f"Teacher Temp: {getattr(self.args, 'warmup_teacher_temp', 0.04)} -> {getattr(self.args, 'teacher_temp', 0.07)}")
            print(
                "Loss λ: fft_align={} cls={} patch={} koleo={} temporal={} cls_cons={}".format(
                    getattr(self.args, "lambda_fft_align", 0),
                    getattr(self.args, "lambda_cls_proto", 0),
                    getattr(self.args, "lambda_patch_proto", 0),
                    getattr(self.args, "lambda_koleo", 0),
                    getattr(self.args, "lambda_temporal", 0),
                    getattr(self.args, "lambda_cls_cons", 0),
                )
            )
            # printcurriculum learningstrategy
            curriculum_strategy = getattr(self.model.module if hasattr(self.model, 'module') else self.model, 'curriculum_strategy', 'fast')
            print(f"Curriculum Strategy: {curriculum_strategy}")

        if self.args.local_rank == 0:
            print(f"use_pretrained_imputator: {self.args.use_pretrained_imputator}")
            
        if self.args.use_pretrained_imputator:
            # Imputator is frozen auxiliary; DDP places it on each local GPU in load_pretrained_imputator.
            self.imputator = load_pretrained_imputator(
                self.args,
                self.args.pretrained_imputator_path,
                target_device=getattr(self.args, 'device', None),  # single-GPUwhenuse args.device, DDP when will be ignored
                use_ddp=bool(self.args.use_multi_gpu),
                local_rank=getattr(self.args, 'local_rank', None) if self.args.use_multi_gpu else None,
            )
            if self.args.local_rank == 0:
                imp_dev = next(self.imputator.parameters()).device
                print(f"load pretrained Imputator success (on {imp_dev},DDP when each process is on its own GPU)")
        else:
            self.imputator = None

        if self.args.pretrain_model is not None:
            self.load_pretrained_model()

        # --- 3. Breakpoint resume (training state) ---
        start_epoch = 0
        best_train_loss = float('inf')
        resume_path = self._resolve_resume_path(getattr(self.args, 'resume_checkpoint', None))
        if resume_path is not None:
            resume_state = torch.load(resume_path, map_location='cpu', weights_only=False)
            model_to_load = self.model.module if hasattr(self.model, 'module') else self.model
            resume_model_state = resume_state.get('model', {})
            resume_model_state, changed_compile = self._normalize_compiled_state_dict_keys(resume_model_state)
            # train_state is saved according to self.model.module (excluding module.), but we will deal with it here.
            if any(k.startswith('module.') for k in resume_model_state.keys()):
                resume_model_state = {
                    (k.replace('module.', '', 1) if k.startswith('module.') else k): v
                    for k, v in resume_model_state.items()
                }
            try:
                model_to_load.load_state_dict(resume_model_state)
            except RuntimeError as e:
                raise RuntimeError(
                    f"[Resume] modelweightsresumefailure: {e}\\n"
                    f"resume_path={resume_path}"
                )
            if changed_compile and self.args.local_rank == 0:
                print("[Resume] Detected compiled train_state, resumesuccess after automatically removing _orig_mod prefix.")
            if 'optimizer' in resume_state:
                model_optim.load_state_dict(resume_state['optimizer'])
                # Avoid optimizer state remaining on the CPU and ensure that training continues on the current device after resume.
                for st in model_optim.state.values():
                    for k, v in st.items():
                        if torch.is_tensor(v):
                            st[k] = v.to(self.device)
            if 'scaler' in resume_state and isinstance(resume_state['scaler'], dict):
                scaler.load_state_dict(resume_state['scaler'])
            best_train_loss = float(resume_state.get('best_train_loss', float('inf')))
            start_epoch = int(resume_state.get('next_epoch', 0))
            self._set_rng_state(resume_state.get('rng_state', None))
            if self.args.local_rank == 0:
                print(
                    f"[Resume] Loaded training state from {resume_path} | "
                    f"start_epoch={start_epoch}, best_train_loss={best_train_loss:.7f}"
                )
        elif self.args.local_rank == 0 and getattr(self.args, 'resume_checkpoint', None):
            print(f"[Resume] resume_checkpoint not found: {self.args.resume_checkpoint}, start from scratch.")

        if self.args.use_multi_gpu:
            dist.barrier()

        # --- 4. training loop ---
        def _save_train_state(nextEpoch, globalIter):
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
            state_payload = {
                'model': model_to_save.state_dict(),
                'optimizer': model_optim.state_dict(),
                'scaler': scaler.state_dict(),
                'best_train_loss': best_train_loss,
                'next_epoch': int(nextEpoch),
                'global_iter': int(globalIter),
                'rng_state': self._get_rng_state(),
                'setting': setting,
            }
            if self.args.use_multi_gpu:
                save_path = os.path.join(path, f"train_state_rank{self.args.local_rank}.pth")
            else:
                save_path = os.path.join(path, "train_state.pth")
            torch.save(state_payload, save_path)

        max_train_steps = int(getattr(self.args, 'max_train_steps', 0) or 0)
        smoke_steps_done = 0
        for epoch in range(start_epoch, self.args.train_epochs):
            iter_count = 0
            train_loss_epoch = []
            self.model.train()
            epoch_time = time.time()
            # Optional epoch-gated FFT align (1-based epoch indexing).
            model_ref = self.model.module if hasattr(self.model, 'module') else self.model
            if hasattr(model_ref, "lambda_fft_align"):
                ep1 = int(epoch + 1)
                target_fft, fft_info = resolve_fft_align_lambda_for_epoch(ep1, self.args)
                model_ref.lambda_fft_align = float(target_fft)
                if self.args.local_rank == 0:
                    if fft_info.get("mode") == "warmup":
                        print(
                            f"[FFT Align Warmup] epoch={ep1}/{self.args.train_epochs} "
                            f"lambda_fft_align={float(target_fft):.6f} "
                            f"(warmup_epochs={fft_info['warmup_epochs']}, "
                            f"start={fft_info['lambda_start']:.6f}, peak={fft_info['lambda_peak']:.6f}, "
                            f"end={fft_info['lambda_end']:.6f})",
                            flush=True,
                        )
                    elif fft_info.get("mode") == "gate":
                        print(
                            f"[FFT Align Gate] epoch={ep1} lambda_fft_align={float(target_fft):.6f} "
                            f"(start={fft_info['start_ep']}, end={fft_info['end_ep']}, "
                            f"active={fft_info['active']:.6f}, inactive={fft_info['inactive']:.6f})",
                            flush=True,
                        )
                    else:
                        print(
                            f"[FFT Align] epoch={ep1} lambda_fft_align={float(target_fft):.6f} "
                            f"(constant; set fft_align_warmup_epochs or fft_align_epoch_start for scheduling)",
                            flush=True,
                        )

            # DDP: IterableDataset does not need set_epoch because we handle shuffle in __iter__
            # if self.args.use_multi_gpu: train_loader.sampler.set_epoch(epoch)
            
            # key fix: record the batch count actually processed by each process, used for debugging
            actual_batches = 0
            stop_smoke = False

            global_iter = epoch * train_steps - 1
            for i, batch_tuple in enumerate(train_loader):
                batch_x, batch_x_mark, next_batch_x = batch_tuple[0], batch_tuple[1], batch_tuple[2]
                batch_lon_lat = batch_tuple[3] if len(batch_tuple) > 3 else None
                actual_batches += 1
                batch_x = batch_x.float().to(self.device, non_blocking=True)
                batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=True)
                if next_batch_x is not None:
                    next_batch_x = next_batch_x.float().to(self.device, non_blocking=True)
                if batch_lon_lat is not None:
                    batch_lon_lat = batch_lon_lat.float().to(self.device, non_blocking=True)
                
                iter_count += 1
                
                # Calculate the number of currentglobaliteration
                global_iter = epoch * train_steps + i
                
                # get currentiteration parameter value
                current_lr = lr_schedule[global_iter]
                current_wd = wd_schedule[global_iter]
                current_momentum = momentum_schedule[global_iter]
                current_teacher_temp = teacher_temp_schedule[global_iter]
                current_last_layer_lr = last_layer_lr_schedule[global_iter]
                
                # Apply useschedule to optimizer
                apply_optim_scheduler(model_optim, current_lr, current_wd, current_last_layer_lr)
                
                model_optim.zero_grad(set_to_none=True)

                # [gradient accumulation strategy]: Check whether usemixed_batchstrategy
                curriculum_strategy = getattr(self.model.module if hasattr(self.model, 'module') else self.model, 'curriculum_strategy', 'fast')
                use_gradient_accumulation = (curriculum_strategy == 'mixed_batch')
                
                gradAccumDivisor = None
                mixed_batch_effective_total = None
                if use_gradient_accumulation:
                    # Gradient accumulation mode: split the batch into multiple groups along the sample dimension, each group is independent forward+backward
                    # note: when interdimensional crop is performed by TED._forward in imputator thendone (guaranteed position encoding align),
                    # Here we only do batch dimension splitting, not when dimension cropping.
                    B, T, C = batch_x.shape
                    device = batch_x.device
                    weight_by_nv = bool(int(getattr(self.args, 'mixed_batch_weight_by_valid_samples', 1)))
                    
                    min_group_size = 32
                    max_groups = max(1, int(getattr(self.args, 'mixed_batch_groups', 2)))
                    num_groups = min(max_groups, max(1, B // min_group_size))
                    
                    if i == 0 and epoch == 0 and self.args.local_rank == 0:
                        print(
                            f"[Gradient Accumulation] Batch size: {B}, Groups: {num_groups}, "
                            f"Group sizes: {[B // num_groups + (1 if j < B % num_groups else 0) for j in range(num_groups)]}"
                            f"{'; grad weighted by ssl_num_valid_samples' if weight_by_nv else ''}"
                        )
                    
                    group_size = B // num_groups
                    remainder = B % num_groups
                    group_sizes = [group_size + (1 if j < remainder else 0) for j in range(num_groups)]
                    
                    total_loss = None
                    total_nv_sum = 0
                    valid_groups = 0
                    accumulated_log_vars = {}
                    
                    start_idx = 0
                    for group_idx in range(num_groups):
                        end_idx = start_idx + group_sizes[group_idx]
                        
                        batch_x_sub = batch_x[start_idx:end_idx]
                        batch_x_mark_sub = batch_x_mark[start_idx:end_idx] if batch_x_mark is not None else None
                        batch_lon_lat_sub = batch_lon_lat[start_idx:end_idx] if batch_lon_lat is not None else None
                        
                        with amp_autocast_ctx(self.args):
                            outputs_sub = self.model(
                                batch_x_sub, 
                                time_mark=batch_x_mark_sub, 
                                next_x_enc=next_batch_x[start_idx:end_idx] if next_batch_x is not None else None,
                                mode='train',
                                mask_rate_v1=self.args.mask_rate_v1,
                                mask_rate_v2=self.args.mask_rate_v2,
                                imputator=self.imputator,
                                teacher_temp=current_teacher_temp,
                                iteration=global_iter,
                                total_iterations=total_iterations,
                                current_epoch=epoch,
                                lon_lat=batch_lon_lat_sub,
                            )

                            nv_raw = outputs_sub.get('ssl_num_valid_samples', None)
                            if nv_raw is None:
                                nv = int(batch_x_sub.shape[0])
                            else:
                                nv = int(nv_raw)
                            if nv <= 0:
                                if self.args.local_rank == 0:
                                    print(
                                        f"[Warn] ssl_num_valid_samples=0 at step={global_iter}, "
                                        f"group={group_idx}, skip this group."
                                    )
                                del outputs_sub
                                start_idx = end_idx
                                continue
                            
                            loss_sub, log_vars_sub = self._compute_loss(outputs_sub)
                            
                            if not torch.isfinite(loss_sub):
                                if self.args.local_rank == 0:
                                    print(
                                        f"[Warn] Non-finite loss_sub detected at step={global_iter}, "
                                        f"group={group_idx}, skip this group."
                                    )
                                del outputs_sub
                                start_idx = end_idx
                                continue

                            valid_groups += 1
                            if weight_by_nv:
                                total_nv_sum += nv
                                lt = loss_sub.detach() * nv
                                if total_loss is None:
                                    total_loss = lt
                                else:
                                    total_loss = total_loss + lt
                                for k, v in log_vars_sub.items():
                                    try:
                                        fv = float(v.detach().item()) if isinstance(v, torch.Tensor) else float(v)
                                        accumulated_log_vars[k] = accumulated_log_vars.get(k, 0.0) + fv * nv
                                    except Exception:
                                        accumulated_log_vars[k] = v
                            else:
                                if total_loss is None:
                                    total_loss = loss_sub
                                else:
                                    total_loss = total_loss + loss_sub
                                for k, v in log_vars_sub.items():
                                    if k not in accumulated_log_vars:
                                        accumulated_log_vars[k] = v
                                    else:
                                        accumulated_log_vars[k] = accumulated_log_vars[k] + v
                        
                        if weight_by_nv:
                            scaler.scale(loss_sub * nv).backward()
                        else:
                            scaler.scale(loss_sub).backward()
                        del outputs_sub
                        start_idx = end_idx
                    
                    if valid_groups == 0 or total_loss is None:
                        if self.args.local_rank == 0:
                            print(f"[Warn] All sub-groups invalid at step={global_iter}, skip optimizer step.")
                        model_optim.zero_grad(set_to_none=True)
                        continue

                    if weight_by_nv:
                        mixed_batch_effective_total = float(total_nv_sum)
                        loss = total_loss / total_nv_sum
                        log_vars = {}
                        for k, v in accumulated_log_vars.items():
                            if isinstance(v, float):
                                log_vars[k] = v / total_nv_sum
                            else:
                                log_vars[k] = v
                    else:
                        # Only normalize by valid subgroup to avoid being extra reduced by skip subgroup.
                        gradAccumDivisor = float(valid_groups)
                        loss = total_loss / gradAccumDivisor
                        log_vars = {}
                        for k, v in accumulated_log_vars.items():
                            try:
                                if isinstance(v, torch.Tensor):
                                    log_vars[k] = v / gradAccumDivisor
                                elif isinstance(v, (int, float)):
                                    log_vars[k] = float(v) / gradAccumDivisor
                                else:
                                    log_vars[k] = v
                            except Exception:
                                log_vars[k] = v
                    
                else:
                    # Standard mode: do not use gradient accumulation
                    with amp_autocast_ctx(self.args):
                        # 1. Forward
                        outputs = self.model(
                            batch_x, 
                            time_mark=batch_x_mark, 
                            next_x_enc=next_batch_x,
                            mode='train',
                            mask_rate_v1=self.args.mask_rate_v1,
                            mask_rate_v2=self.args.mask_rate_v2,
                            imputator=self.imputator,
                            teacher_temp=current_teacher_temp,
                            iteration=global_iter,
                            total_iterations=total_iterations,
                            current_epoch=epoch,
                            lon_lat=batch_lon_lat,
                        )
                        
                        # 2. Loss Calculation (useshare criterion or ssl_loss)
                        loss, log_vars = self._compute_loss(outputs)
                        nv_raw = outputs.get('ssl_num_valid_samples', None)
                        if nv_raw is not None and int(nv_raw) <= 0:
                            if self.args.local_rank == 0:
                                print(
                                    f"[Warn] ssl_num_valid_samples=0 at step={global_iter}, skip this step."
                                )
                            model_optim.zero_grad(set_to_none=True)
                            continue
                        if not torch.isfinite(loss):
                            if self.args.local_rank == 0:
                                print(f"[Warn] Non-finite loss detected at step={global_iter}, skip this step.")
                            model_optim.zero_grad(set_to_none=True)
                            continue
                        
                        # backward propagation (use is adjusted here in standard mode)
                        scaler.scale(loss).backward()
                    
                    # Standard mode: Release outputs quotas as early as possible to facilitate GC recycling of large tensors (reducing GPU memory climbing)
                    try:
                        del outputs
                    except NameError:
                        pass
                loss_scalar = float(loss.detach().item())
                train_loss_epoch.append(loss_scalar)

                # Step indicators are fully recorded to the file at each step; the console prints at intervals.
                last_loss = train_loss_epoch[-1]
                if self.args.local_rank == 0:
                    step_id = global_iter + 1
                    tensor_item_interval = max(1, int(getattr(self.args, "train_step_tensor_item_interval", 1)))
                    collect_detail_this_step = (step_id % tensor_item_interval == 0)
                    scalar_log_vars = {}
                    if collect_detail_this_step:
                        for k, v in log_vars.items():
                            try:
                                if isinstance(v, torch.Tensor):
                                    scalar_log_vars[k] = float(v.detach().item())
                                elif isinstance(v, (int, float)):
                                    scalar_log_vars[k] = float(v)
                                else:
                                    scalar_log_vars[k] = str(v)
                            except Exception:
                                scalar_log_vars[k] = str(v)
                        keep_log_keys = {
                            "cls",
                            "patch",
                            "fft_align",
                            "koleo",
                            "temporal",
                            "cls_cons",
                            "cls_global_ce",
                            "cls_short_cond_ce",
                            "cls_short_crop_cond_ce",
                            "cls_short_random_cond_ce",
                            "cls_short_anchor_cond_ce",
                        }
                        scalar_log_vars = {
                            k: v
                            for k, v in scalar_log_vars.items()
                            if k in keep_log_keys or k.startswith("condition_")
                        }

                    if step_log_queue is not None:
                        step_record = {
                            "step": int(step_id),
                            "epoch": int(epoch + 1),
                            "iter_in_epoch": int(i + 1),
                            "total": float(last_loss),
                        }
                        step_record.update(scalar_log_vars)
                        try:
                            step_log_queue.put_nowait(step_record)
                        except queue.Full:
                            step_log_drop_counter += 1

                    if bool(int(getattr(self.args, "train_step_console_log_enable", 0))):
                        parts = [f"step={step_id}", f"total={last_loss:.6f}"]
                        for k, v in scalar_log_vars.items():
                            if isinstance(v, float):
                                parts.append(f"{k}={v:.6f}")
                            else:
                                parts.append(f"{k}={v}")
                        print("\t" + " | ".join(parts))

                # Low-frequency printing of progress lines (default every 200 steps), only used to observe ETA and speed
                progress_log_interval = max(1, int(getattr(self.args, 'progress_log_interval', 200)))
                if (i + 1) % progress_log_interval == 0:
                    if self.args.local_rank == 0:
                        progress = global_iter / max(total_iterations, 1) * 100
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print(f"\t[progress {progress:.2f}%] LR: {current_lr:.6f} | T_Temp: {current_teacher_temp:.4f} | speed: {speed:.4f}s/iter | left: {left_time:.0f}s")
                    iter_count = 0
                    time_now = time.time()

                # note: backward has been adjusted above (standard mode is in with autocast, gradient accumulation mode is in loop)
                scaler.unscale_(model_optim)
                if mixed_batch_effective_total is not None and mixed_batch_effective_total > 0:
                    inv_eff = 1.0 / float(mixed_batch_effective_total)
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad.mul_(inv_eff)
                elif gradAccumDivisor is not None and gradAccumDivisor > 0:
                    gradScale = 1.0 / gradAccumDivisor
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad.mul_(gradScale)
                
                max_grad_norm = float(getattr(self.args, 'max_grad_norm', 1.0))
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
                
                scaler.step(model_optim)
                scaler.update()
                
                # key fix：updateTeachermodel（EMAupdate）
                # Must be used in optimizer.step()then to ensure that studentparameter has been updated
                # use dynamic momentum
                if hasattr(self.model, 'module') and hasattr(self.model.module, '_update_teacher'):
                    # DDPwrap model
                    self.model.module._update_teacher(m=current_momentum)
                elif hasattr(self.model, '_update_teacher'):
                    # Non-DDPmodel
                    self.model._update_teacher(m=current_momentum)

                if max_train_steps > 0:
                    smoke_steps_done += 1
                    if smoke_steps_done >= max_train_steps:
                        if self.args.local_rank == 0:
                            print(
                                f"[Smoke] max_train_steps={max_train_steps} reached, stopping.",
                                flush=True,
                            )
                        stop_smoke = True
                        break

                # Periodically release the CUDA cache to alleviate the gradual growth of GPU memory caused by dynamic sequence length (PyTorch will cache released blocks, and different lengths lead to fragmentation)
                empty_cache_interval = getattr(self.args, 'empty_cache_interval', 0)
                if empty_cache_interval > 0 and (global_iter + 1) % empty_cache_interval == 0:
                    torch.cuda.empty_cache()

            if stop_smoke:
                if self.args.use_multi_gpu:
                    dist.barrier()
                break

            # Interrupt resume train_state (full rank); reduce default frequency and reduce disk; first epoch / last epoch must be written
            _tsi = max(1, int(getattr(self.args, "train_state_save_interval", 5)))
            _epn = epoch + 1
            if _epn == 1 or (_epn % _tsi == 0) or (_epn >= self.args.train_epochs):
                _save_train_state(nextEpoch=epoch + 1, globalIter=global_iter)
            # Because IterableDataset may cause different processes to process differentcount batches
            # If sync is not used, it will cause the collective operation countmismatch, causing NCCL to exceed when
            if self.args.use_multi_gpu:
                # sync all processes and ensure that the training loop is done
                dist.barrier()
                # Optional: print the batch count processed by each process (used for debugging)
                if epoch == 0:  # Only print in the first epoch to avoid excessive logs
                    print(f"[Rank {self.args.local_rank}] Processed {actual_batches} batches in epoch {epoch + 1}")
            
            if self.args.local_rank == 0:
                print("Epoch: {} consumption when: {}".format(epoch + 1, time.time() - epoch_time))
            
            train_loss_avg = np.average(train_loss_epoch)

            # --- 4. Probe evaluation (only KNN Probe, no validate set loss) ---
            if self.args.use_multi_gpu:
                dist.barrier()

            if self.args.local_rank == 0:
                print("Epoch: {0}, Steps: {1} | Train Loss (avg): {2:.7f}".format(
                    epoch + 1, train_steps, train_loss_avg))

                # ifcurrent train loss is better than the historical optimal, thensave is checkpoint.pth (for downstream test use)
                if train_loss_avg < best_train_loss:
                    best_train_loss = train_loss_avg
                    model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                    best_ckpt_path = os.path.join(path, 'checkpoint.pth')
                    torch.save(model_to_save.state_dict(), best_ckpt_path)
                    print(f"[Train] New best train loss {best_train_loss:.7f}, saved to {best_ckpt_path}")

                # --- 5. Periodic checkpointsave (numbered according to epoch); best checkpoint.pth still according to train loss ---
                save_model_periodically(
                    self.model,
                    path,
                    epoch + 1,
                    save_interval=max(1, int(getattr(self.args, "checkpoint_epoch_save_interval", 1))),
                    verbose=True,
                )
                current_lr = model_optim.param_groups[0]['lr']
                print('Current learning rate: {:.8f}'.format(current_lr))

            # Early stop: only Imputator takes effect, TED/DINO skip (DDP requires all ranks to participate in broadcast)
            if early_stopping is not None:
                early_stop_signal = torch.tensor(0.0, device=self.device)
                if self.args.local_rank == 0:
                    # Pass the unpacked model to EarlyStopping to avoid state_dict with module. prefix
                    _model_unwrap = self.model.module if hasattr(self.model, 'module') else self.model
                    early_stopping(train_loss_avg, _model_unwrap, path)
                    if early_stopping.early_stop:
                        early_stop_signal += 1.0
                if self.args.use_multi_gpu:
                    dist.broadcast(early_stop_signal, src=0)
                if early_stop_signal.item() > 0.5:
                    if self.args.local_rank == 0:
                        print(f"[EarlyStopping] Triggered at epoch {epoch + 1}")
                    break

        # Return the best checkpoint after early stop triggering/after trainingend (only Imputator)
        if early_stopping is not None and self.args.local_rank == 0:
            best_model_path = os.path.join(path, 'checkpoint.pth')
            if os.path.exists(best_model_path):
                state = torch.load(best_model_path, map_location='cpu')
                state, _ = self._normalize_compiled_state_dict_keys(state)
                # compatible DDP checkpoint: Remove possible remaining module. prefix
                clean_state = {
                    (k.replace('module.', '', 1) if k.startswith('module.') else k): v
                    for k, v in state.items()
                }
                model_to_load = self.model.module if hasattr(self.model, 'module') else self.model
                model_to_load.load_state_dict(clean_state)
                print(f"[EarlyStopping] Loaded best checkpoint from {best_model_path}")
        
        # The last sync before exiting to prevent Rank 0 from exiting after loading, while other Ranks are still waiting.
        if self.args.local_rank == 0 and step_log_queue is not None:
            if step_log_drop_counter > 0:
                print(f"[StepLog] dropped {step_log_drop_counter} step records due to full queue.")
            step_log_queue.put(step_log_stop_token)
            if step_log_thread is not None:
                step_log_thread.join(timeout=5)

        if self.args.use_multi_gpu:
            dist.barrier()

        return self.model
