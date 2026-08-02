from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os
import torch as t
import torch.nn.functional as F
from utils.mcar import mcar
import math

plt.switch_backend('agg')


def fs_safe_checkpoint_dir_component(
    name: str,
    max_component_bytes: int = 220,
) -> str:
    """
    Shorten a single directory name so it fits typical NAME_MAX (255 bytes on Linux).

    Preserves uniqueness via a SHA-256 digest suffix when truncation is required.
    """
    if not isinstance(name, str):
        name = str(name)
    encoded = name.replace(os.sep, "_").replace("/", "_").encode("utf-8")
    if len(encoded) <= max_component_bytes:
        return name.replace(os.sep, "_").replace("/", "_")

    digest = hashlib.sha256(encoded).hexdigest()[:16]
    suffix = f"__h{digest}"
    suffix_b = suffix.encode("utf-8")
    budget = max_component_bytes - len(suffix_b)
    if budget < 1:
        return suffix
    prefix_b = encoded[:budget]
    while prefix_b and (prefix_b[-1] & 0xC0) == 0x80:
        prefix_b = prefix_b[:-1]
    return prefix_b.decode("utf-8") + suffix


def clamp_experiment_setting_for_checkpoint(
    setting: str,
    max_total_bytes: int = 220,
) -> str:
    """
    Ensure experiment ``setting`` (used as one path component under checkpoints/) is
    short enough for the filesystem. Keeps the trailing ``_{des}_{itr}`` segments so
    ``setting.rsplit('_', 1)`` iteration logic in ``train.py`` stays valid.
    """
    total_b = setting.encode("utf-8")
    if len(total_b) <= max_total_bytes:
        return setting
    try:
        head, _des, _itr = setting.rsplit("_", 2)
    except ValueError:
        return fs_safe_checkpoint_dir_component(setting, max_component_bytes=max_total_bytes)
    tail = setting[len(head) :]
    tail_b = tail.encode("utf-8")
    head_budget = max_total_bytes - len(tail_b)
    if head_budget < 8:
        return fs_safe_checkpoint_dir_component(setting, max_component_bytes=max_total_bytes)
    head_short = fs_safe_checkpoint_dir_component(head, max_component_bytes=head_budget)
    return head_short + tail


def resolve_fft_align_lambda_for_epoch(ep1: int, args) -> tuple[float, dict]:
    """
    Effective FFT Gram multiplier for 1-based training epoch ``ep1``.

    If ``fft_align_warmup_epochs`` > 0: linear ramp from ``fft_align_lambda_start``
    to ``lambda_fft_align`` (peak) during warmup epochs, then linear decay to
    ``fft_align_lambda_end`` by the final epoch (ignores ``fft_align_epoch_*`` gate).

    Otherwise: legacy epoch window gate, or constant ``lambda_fft_align``.
    """

    warmup = int(getattr(args, "fft_align_warmup_epochs", 0) or 0)
    peak = float(getattr(args, "lambda_fft_align", 0.0))
    train_e = max(1, int(getattr(args, "train_epochs", 1)))

    if warmup > 0:
        lam_s = float(getattr(args, "fft_align_lambda_start", 0.0))
        lam_e = float(getattr(args, "fft_align_lambda_end", 0.0))
        if ep1 <= warmup:
            if warmup <= 1:
                lam = float(peak)
            else:
                w = float(ep1 - 1) / float(warmup - 1)
                lam = lam_s + w * (peak - lam_s)
        else:
            denom = float(max(train_e - warmup, 1))
            t = float(ep1 - warmup) / denom
            t = min(1.0, max(0.0, t))
            lam = peak + t * (lam_e - peak)
        return float(lam), {
            "mode": "warmup",
            "warmup_epochs": int(warmup),
            "lambda_start": lam_s,
            "lambda_peak": float(peak),
            "lambda_end": lam_e,
        }

    start_ep = int(getattr(args, "fft_align_epoch_start", -1))
    end_raw = int(getattr(args, "fft_align_epoch_end", -1))
    active_cfg = getattr(args, "fft_align_lambda_active", None)
    base_fft = float(peak)
    active_fft = base_fft if active_cfg is None else float(active_cfg)
    inactive_fft = float(getattr(args, "fft_align_lambda_inactive", 0.0))
    if start_ep > 0 and end_raw <= 0:
        use_gate = True
        end_ep = train_e
    elif start_ep > 0 and end_raw > 0 and end_raw >= start_ep:
        use_gate = True
        end_ep = end_raw
    else:
        use_gate = False
        end_ep = end_raw
    if use_gate:
        lam = active_fft if (start_ep <= ep1 <= end_ep) else inactive_fft
    else:
        lam = base_fft
    return float(lam), {
        "mode": "gate" if use_gate else "constant",
        "use_gate": use_gate,
        "start_ep": start_ep,
        "end_ep": int(end_ep) if use_gate else end_raw,
        "active": float(active_fft),
        "inactive": float(inactive_fft),
    }


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'PEMS':
        lr_adjust = {epoch: args.learning_rate * (0.95 ** (epoch // 1))}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))



class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


def save_model_periodically(model, path, epoch, save_interval=5, verbose=True):
    """
    Save model every N epochs
    
    Args:
        model: model to save
        path: save directory
        epoch: current epoch (1-based)
        save_interval: save interval, default every 5 epochs
        verbose: whether to print save message
    
    Returns:
        bool: whether save was performed
    """
    if epoch % save_interval == 0:
        import os
        if not os.path.exists(path):
            os.makedirs(path)
        
        # unwrap DataParallel model
        model_to_save = model.module if hasattr(model, 'module') else model
        
        save_path = os.path.join(path, f'checkpoint_epoch_{epoch}.pth')
        torch.save(model_to_save.state_dict(), save_path)
        
        if verbose:
            print(f'Saving model checkpoint at epoch {epoch} to {save_path}')
        return True
    return False


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def save_to_csv(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    data = pd.DataFrame({'true': true, 'preds': preds})
    data.to_csv(name, index=False, sep=',')


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def visual_weights(weights, name='./pic/test.pdf'):
    """
    Weights visualization
    """
    fig, ax = plt.subplots()
    # im = ax.imshow(weights, cmap='plasma_r')
    im = ax.imshow(weights, cmap='YlGnBu')
    fig.colorbar(im, pad=0.03, location='top')
    plt.savefig(name, dpi=500, pad_inches=0.02)
    plt.close()


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

# def apply_mask(ori_batch_x, ori_valid_mask, p, device, mode=None, min_p=0.25):
#     if p == 0:
#         # when p is 0, return raw data without masking
#         ori_batch_x = torch.nan_to_num(ori_batch_x, nan=0.0)
#         missing_mask = ori_valid_mask
#         indicating_mask = torch.zeros_like(ori_batch_x).int().to(device)
#         return ori_batch_x, ori_batch_x, missing_mask, indicating_mask

#     if mode == 'test':
#         torch.manual_seed(42)  # or another fixed value
#         np.random.seed(42)
#     else:
#         # in training, uniform in [min_p, p]
#         p = torch.FloatTensor(1).uniform_(min_p, p).item()

#     batch_x = mcar(ori_batch_x, p)
#     missing_mask = (1 - torch.isnan(batch_x).int()).to(device)
#     indicating_mask = (ori_valid_mask - missing_mask).to(device)
#     ori_batch_x = torch.nan_to_num(ori_batch_x, nan=0.0)
#     batch_x = torch.nan_to_num(batch_x, nan=0.0)

#     return ori_batch_x, batch_x, missing_mask, indicating_mask

def apply_mask(ori_batch_x, ori_valid_mask, p, device, mode=None, min_p=0.25, use_random_p=False):
    """
    Args:
        min_p (float): lower bound for random mask ratio.
        use_random_p (bool): whether to randomly pick mask ratio in [min_p, p].
    """
    
    # 1. if p==0, return immediately (no random logic)
    if p == 0:
        ori_batch_x = torch.nan_to_num(ori_batch_x, nan=0.0)
        missing_mask = ori_valid_mask
        indicating_mask = torch.zeros_like(ori_batch_x).int().to(device)
        return ori_batch_x, ori_batch_x, missing_mask, indicating_mask

    # 2. set current mask ratio
    current_p = p
    
    # randomize only in train when use_random_p is enabled
    if use_random_p and mode != 'test':
        # ensure low <= high to avoid errors
        low = min(min_p, p)
        high = max(min_p, p)
        # uniform sample in [min_p, p]
        current_p = np.random.uniform(low, high)

    # 3. fixed seed in test mode
    if mode == 'test':
        # fixed seed for test/pred modes
        torch.manual_seed(42)  
        np.random.seed(42)
        # test mode usually keeps fixed p (current_p stays p)
        current_p = p 

    # 4. apply mask with current_p via mcar
    batch_x = mcar(ori_batch_x, current_p)
    
    # 5. build mask tensors
    missing_mask = (1 - torch.isnan(batch_x).int()).to(device)
    indicating_mask = (ori_valid_mask - missing_mask).to(device)
    
    # 6. replace NaN with 0
    ori_batch_x = torch.nan_to_num(ori_batch_x, nan=0.0)
    batch_x = torch.nan_to_num(batch_x, nan=0.0)

    return ori_batch_x, batch_x, missing_mask, indicating_mask

def apply_mask_seasons(
    ori_batch_x: torch.Tensor,
    ori_valid_mask: torch.Tensor,
    p: float,
    device: torch.device,
    mode: str = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply season-level full masking on pre-chunked input with valid_mask.

    Args:
        ori_batch_x (torch.Tensor): input tensor, shape [batch_size, season, steps_per_season, in_chans]
        ori_valid_mask (torch.Tensor):validity mask, shape is[batch_size, season, steps_per_season]
        p (float): mask probability in [0, 1]
        device (torch.device): device
        mode (str, optional): 'test' or None to control random seed

    Returns:
        tuple: (
            ori_batch_x: raw tensor (NaN->0), shape [batch_size, season, steps_per_season, in_chans]
            batch_x: masked tensor (NaN->0), shape [batch_size, season, steps_per_season, in_chans]
            missing_mask: missing mask, shape [batch_size, season, steps_per_season]
            indicating_mask: indicating mask, shape [batch_size, season, steps_per_season, in_chans]
        )
    """
    # get input shape
    batch_size, season, steps_per_season, in_chans = ori_batch_x.shape
    assert ori_valid_mask.shape == (batch_size, season, steps_per_season), \
        f"Expected valid_mask shape {(batch_size, season, steps_per_season)}, but got {ori_valid_mask.shape}"

    # if p==0, return raw data without masking
    if p == 0:
        ori_batch_x = torch.nan_to_num(ori_batch_x, nan=0.0)
        missing_mask = ori_valid_mask
        indicating_mask = torch.zeros_like(ori_batch_x).int().to(device)
        return ori_batch_x, ori_batch_x, missing_mask, indicating_mask

    # set random seed
    if mode == 'test':
        torch.manual_seed(42)
        np.random.seed(42)
    else:
        # in training, random mask prob in [0, p]
        p = torch.FloatTensor(1).uniform_(1e-6, p).item()

    # Bernoulli mask decision per season
    season_mask = torch.bernoulli(torch.full((batch_size, season, 1), 1 - p, device=device))  # [batch_size, season, 1]
    season_mask = season_mask.expand(-1, -1, steps_per_season)  # [batch_size, season, steps_per_season]

    # build missing_mask from valid_mask
    missing_mask = ori_valid_mask * season_mask  # [batch_size, season, steps_per_season]

    # build batch_x with season-level mask
    batch_x = ori_batch_x.clone()
    mask_applied = missing_mask.unsqueeze(-1).expand(-1, -1, -1, in_chans)  # [batch_size, season, steps_per_season, in_chans]
    batch_x = batch_x.where(mask_applied == 1, torch.tensor(float('nan'), device=device))

    # indicating_mask: masked points (valid_mask=1 but season_mask=0)
    indicating_mask = (ori_valid_mask - missing_mask).clamp(min=0).unsqueeze(-1).expand(-1, -1, -1, in_chans).int().to(device)

    # replace NaN with 0
    ori_batch_x = torch.nan_to_num(ori_batch_x, nan=0.0)
    batch_x = torch.nan_to_num(batch_x, nan=0.0)

    return ori_batch_x, batch_x, missing_mask, indicating_mask

def visual_results(dec_ori, dec_out, valid_mask, anomaly_mask, epoch, batch_idx, plot_anomaly=False,
                    plot_atten=None, anomalies_prob=None):  # added plot_atten parameter
    """
    Visualize raw data, reconstruction, anomaly detection, and attention heatmap
    Args:
        dec_ori: raw data [B, T, N]
        dec_out: reconstruction [B, T, N]
        valid_mask: valid-value mask [B, T, N]
        anomaly_mask: anomaly detection mask [B, T, N]
        epoch: current training epoch
        plot_atten: attention heatmap matrix [T, N]（optional）
    """
    channels = ['Blue', 'Green','Red','NIR','SWIR1','SWIR2','NDVI']
    # create plots folder
    plot_dir = 'plots'
    os.makedirs(plot_dir, exist_ok=True)

    # build save path
    save_path = os.path.join(plot_dir, f'epoch_{epoch}_batch_{batch_idx}.png')

    mask = anomaly_mask.detach().cpu().numpy()
    # valid_mask = valid_mask.detach().cpu().numpy()

    # take last sample in batch
    sample_ori = dec_ori[-1]  # [T, N]
    sample_out = dec_out[-1]  # [T, N]
    sample_mask = mask[-1]  # [T, N]
    valid_sample_mask = valid_mask[-1]
    # if batch_idx == 1:
    #     print("True")
    #     save_path1 = os.path.join('plots', f'epoch_{epoch}_batch_{batch_idx}_sample_out.npy')
    #     np.save(save_path1, sample_out)
    # compute reconstruction error
    reconstruction_error = np.abs(sample_ori - sample_out) * valid_sample_mask  # [T, N]

    # subplots: 2 rows per channel
    n_channels = sample_ori.shape[1]
    if anomalies_prob is not None:
        n_plots = n_channels * 2 + 1
        fig, axes = plt.subplots(n_plots, 1, figsize=(15, 4 * n_plots))
    else:
        fig, axes = plt.subplots(n_channels * 2, 1, figsize=(15, 4 * n_channels))
    if n_channels == 1:
        axes = axes.reshape(-1)

    time_steps = np.arange(sample_ori.shape[0])

    for i in range(n_channels):
        # subplot for raw vs reconstructed
        ax1 = axes[i * 2]
        # subplot for reconstruction error
        ax2 = axes[i * 2 + 1]

        # plot raw and reconstructed
        ax1.plot(time_steps, sample_ori[:, i], 'b-', label='Original', alpha=0.5)
        ax1.plot(time_steps, sample_out[:, i], 'r--', label='Reconstructed', alpha=0.7)
        # ax1.plot(time_steps, sample_out[:, i], 'r--', label='Season', alpha=0.7)
        if plot_anomaly:
            anomaly_points = np.where(sample_mask[:, 0] > 0.9)[0]
            if len(anomaly_points) > 0:
                ax1.scatter(anomaly_points, sample_ori[anomaly_points, i],
                            c='red', marker='x', s=100, label='Anomaly')

        ax1.set_title(f'{channels[i]} - Original vs Reconstructed')
        ax1.legend()
        ax1.grid(True)

        # plot reconstruction error
        ax2.plot(time_steps, reconstruction_error[:, i], 'g-', label='Reconstruction Error')
        # if plot_anomaly and len(anomaly_points) > 0:
        #     ax2.scatter(anomaly_points, reconstruction_error[anomaly_points, i],
        #                 c='red', marker='x', s=100, label='Anomaly')

        ax2.set_title(f'{channels[i]} - Reconstruction Error')
        ax2.legend()
        ax2.grid(True)
    # plot anomaly probability
    if anomalies_prob is not None:
        ax_prob = axes[-1]  # use last subplot for anomaly probability
        anomalies_prob = anomalies_prob[-1].detach().cpu().numpy().flatten()  # flatten anomaly probability for current batch
        ax_prob.plot(time_steps, anomalies_prob, 'm-', label='Anomaly Probability')
        ax_prob.set_title(f'Anomaly Probability across Time Steps')
        # ax_prob.set_ylim([0, 10])  # anomaly probability in [0, 1]
        ax_prob.legend()
        ax_prob.grid(True)

    # plot attention heatmap if provided
    if plot_atten is not None:
        plot_atten = plot_atten[-1]
        fig_atten, ax_atten = plt.subplots(figsize=(10, 6))
        cax = ax_atten.imshow(plot_atten, aspect='auto', cmap='viridis', origin='lower')
        fig_atten.colorbar(cax, ax=ax_atten)
        ax_atten.set_title('Attention Heatmap')
        ax_atten.set_xlabel('Time Step')
        ax_atten.set_ylabel('Time Step')
        fig_atten.tight_layout()
        atten_save_path = os.path.join(plot_dir, f'epoch_{epoch}_batch_{batch_idx}_atten.png')
        plt.savefig(atten_save_path)
        plt.close(fig_atten)

    # save main figure
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def nll_t_per_step(mu, log_sigma, df_raw, labels, mask, df_mode='band', df_lower=3.0):
    """
    Calculate Negative Log-Likelihood per time step with Student’s t-distribution for anomaly detection.

    Args:
        mu: (batch_size, steps, bands), Student’s t-distribution mean
        log_sigma: (batch_size, steps, bands), log of scale parameter
        df_raw:
            - per_step: (batch_size, steps, bands)
            - sequence: (batch_size, 1, 1)
            - band: (batch_size, 1, bands)
        labels: (batch_size, steps, bands), true data
        mask: (batch_size, steps, bands), 1 for valid, 0 for invalid
        df_mode: 'per_step', 'sequence', or 'band' for df sharing
        df_lower: Minimum degrees of freedom for stability

    Returns:
        nll_per_step: (batch_size, steps, bands), Negative Log-Likelihood for each time step and band
    """
    # Ensure positive sigma
    sigma = t.exp(log_sigma) + 1e-6

    # Check shapes
    if mu.shape != labels.shape:
        raise ValueError(f"Expected mu shape {labels.shape}, got {mu.shape}")
    if log_sigma.shape != labels.shape:
        raise ValueError(f"Expected log_sigma shape {labels.shape}, got {log_sigma.shape}")
    if mask.shape != labels.shape:
        raise ValueError(f"Expected mask shape {labels.shape}, got {mask.shape}")

    # Handle df_raw based on df_mode
    if df_mode == 'per_step':
        if df_raw.shape != labels.shape:
            raise ValueError(f"Expected df_raw shape {labels.shape}, got {df_raw.shape}")
        df = F.softplus(df_raw) + 3
    elif df_mode == 'sequence':
        if df_raw.shape != (labels.shape[0], 1, 1):
            raise ValueError(f"Expected df_raw shape {(labels.shape[0], 1, 1)}, got {df_raw.shape}")
        df = F.softplus(df_raw) + 3
        df = df.expand(-1, labels.shape[1], labels.shape[2])
    elif df_mode == 'band':
        if df_raw.shape != (labels.shape[0], 1, labels.shape[2]):
            raise ValueError(f"Expected df_raw shape {(labels.shape[0], 1, labels.shape[2])}, got {df_raw.shape}")
        df = F.softplus(df_raw) + 3
        df = df.expand(-1, labels.shape[1], -1)
    else:
        raise ValueError(f"Invalid df_mode: {df_mode}. Choose 'per_step', 'sequence', or 'band'.")

    # Define Student’s t-distribution
    distribution = t.distributions.StudentT(df=df, loc=mu, scale=sigma)
    log_likelihood = distribution.log_prob(labels)  # (batch_size, steps, bands)

    # Calculate NLL (negative log-likelihood)
    nll_per_step = -log_likelihood * mask  # (batch_size, steps, bands)
    # Where mask is 0, NLL is set to 0 (invalid points)
    return nll_per_step

def z_score_detector(anomaly_prob, mask, threshold=2.5):
    """
    Unsupervised z-score anomaly detection; threshold adapts to valid-data ratio.

    Args:
    - anomaly_prob: (batch, steps) anomaly probability in [0,1]; higher means more anomalous.
    - mask: (batch, steps) valid-point mask: 1 valid, 0 invalid.
    - threshold: base z-score threshold, default 3.0.

    Returns:
    - anomaly_label: (batch, steps) anomaly labels: 1 anomaly, 0 normal.
    """
    # compute on valid points only
    valid_points = anomaly_prob * mask  # anomaly prob on valid points
    valid_mask = mask  # valid-point mask

    # mean and std over valid points
    count_valid = valid_mask.sum(dim=1, keepdim=True)
    mean = (valid_points.sum(dim=1, keepdim=True) / count_valid).nan_to_num()
    std = torch.sqrt(((valid_points - mean) ** 2 * valid_mask).sum(dim=1, keepdim=True) / count_valid).nan_to_num()

    # compute z-score
    z_scores = (valid_points - mean) / (std + 1e-8)  # avoid division by zero

    # valid ratio per sample
    steps = mask.size(1)
    p = count_valid / steps  # fraction of valid timesteps

    # dynamic threshold: if valid ratio < 1/3, log-increase threshold
    adjusted_threshold = torch.where(
        p < 1/3,
        threshold + torch.log(1.0 / (3 * p + 1e-8)),  # log-smoothed adjustment
        threshold
    )

    # detect anomalies and apply mask
    anomaly_label = (z_scores.abs() > adjusted_threshold).float() * valid_mask

    return anomaly_label.unsqueeze(-1)  # keep output shape consistent


def process_attention(attn_outputs, register_tokens, seasons, steps_per_season=None):
    """
Handle intra and inter attention, remove register tokens, normalize, and average inter-attention.

    Args:
        attn_outputs: List of (attn_type, attn) tuple; attn_type intra/inter, attn tensor
        register_tokens: number of register tokens
        seasons: number of seasons
        steps_per_season: timesteps per season (optional; inferred if omitted)

    Returns:
        inter_attn: processed inter-attention, shape (batch, seasons, seasons)
    """
    inter_attn = None

    for attn_type, attn in attn_outputs:
        if attn_type == "intra":
            # print('intra attn',attn.shape)
            # # infer batch
            # if attn.size(0) % seasons != 0:
            # raise ValueError(f"Intra-attention first dimension {attn.size(0)} not divisible by seasons={seasons}")
            # batch = attn.size(0) // seasons
            # # remove register tokens
            # intra_attn = attn[:, register_tokens:, register_tokens:]  # shape: (batch * seasons, steps_per_season, steps_per_season)
            # intra_attn = torch.softmax(intra_attn, dim=-1)  # normalize
            # intra_attn = intra_attn.view(batch, seasons, intra_attn.size(1), intra_attn.size(2))
            # print(f"processed intra-attention shape: {intra_attn.shape}")
            pass
        elif attn_type == "inter":

            # infer steps_per_season if not provided
            if steps_per_season is None:
                # if attn.size(0) % seasons != 0:
                # raise ValueError(f"Inter-attention first dimension {attn.size(0)} not divisible by seasons={seasons}; cannot infer steps_per_season")
                steps_per_season = attn.size(0) // seasons  # assume batch=1 to infer steps_per_season
            # # validate first dimension
            # if attn.size(0) % steps_per_season != 0:
            # raise ValueError(f"Inter-attention first dimension {attn.size(0)} not divisible by steps_per_season={steps_per_season}")
            batch = attn.size(0) // steps_per_season
            # # validate seasons
            # if attn.size(1) != attn.size(2) or attn.size(1) != (register_tokens + seasons):
            #     raise ValueError(f"Inter-attention shape {attn.shape} mismatches register_tokens={register_tokens} and seasons={seasons}")
            # remove register tokens
            inter_attn = attn[:, register_tokens:, register_tokens:]  # shape: (batch * steps_per_season, seasons, seasons)
            inter_attn = torch.softmax(inter_attn, dim=-1)  # normalize
            inter_attn = inter_attn.view(batch, steps_per_season, seasons, seasons)
            # mean over steps_per_season
            inter_attn = inter_attn.mean(dim=1)  # shape: (batch, seasons, seasons)
            # print(f"processed inter-attention shape: {inter_attn.shape}")

    if inter_attn is None:
        raise ValueError("inter-attention data not found")

    return inter_attn

def restore_data(preds, dataset, batch_indices=None, window_indices=None, stride=None, seq_len=None, time_steps=None, num_pixels=None, bands=None):
    """
    Restore predictions to [time_steps, bands, num_pixels].
    
    Args:
        preds: predictions [total_samples, seq_len, bands]
        dataset: IterableDataset with defaults and raw data
        batch_indices: optional batch indices overriding dataset.batch_indices
        window_indices: optional window indices overriding dataset.window_indices
        stride: optional stride overriding dataset.stride
        seq_len: optional seq_len overriding dataset.seq_len
        time_steps: optional time_steps overriding dataset.time_steps
        num_pixels: optional num_pixels overriding dataset.num_pixels
        bands: optional band count overriding dataset.data_x.shape[1]
    
    Returns:
        restored_preds: restored array [time_steps, bands, num_pixels]
    """
    # defaults from dataset or custom overrides
    time_steps = time_steps if time_steps is not None else dataset.time_steps
    seq_len = seq_len if seq_len is not None else dataset.seq_len
    batch_size = dataset.batch_size  # batch_size usually fixed from dataset
    num_pixels = num_pixels if num_pixels is not None else dataset.num_pixels
    bands = bands if bands is not None else dataset.data_x.shape[1]
    batch_indices = batch_indices if batch_indices is not None else dataset.batch_indices
    window_indices = window_indices if window_indices is not None else dataset.window_indices
    stride = stride if stride is not None else dataset.stride

    # initialize output arrays
    restored_preds = np.zeros((time_steps, bands, num_pixels))
    count = np.zeros((time_steps, bands, num_pixels))  # overlap counts

    # current sample index
    sample_idx = 0

    # iterate batches
    for batch_idx in batch_indices:
        start = batch_idx * batch_size
        end = min(start + batch_size, num_pixels)
        batch_pixels = list(range(start, end))
        batch_size_local = len(batch_pixels)

        # iterate windows
        for w_idx in window_indices:
            s_begin = w_idx * stride
            s_end = s_begin + seq_len

            if s_end > time_steps:
                continue  # skip invalid windows

            # extract predictions for batch/window
            batch_pred = preds[sample_idx:sample_idx + batch_size_local, :, :]  # [batch_size_local, seq_len, bands]

            # write predictions to time steps and pixels
            for i, pixel_idx in enumerate(batch_pixels):
                restored_preds[s_begin:s_end, :, pixel_idx] += batch_pred[i, :, :]
                count[s_begin:s_end, :, pixel_idx] += 1

            sample_idx += batch_size_local

    # average overlapping windows
    count[count == 0] = 1  # avoid divide by zero
    restored_preds /= count

    return restored_preds

def sequence2seasons(
    x: torch.Tensor, 
    season: int
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """
    Preprocess time series: split by season, zero-pad; infer seq_len and in_chans.
    Also build padding and validity masks (NaN and padding).

    Args:
        x (torch.Tensor): input tensor, shape [batch_size, seq_len, in_chans]
        season (int): number of seasons

    Returns:
        tuple: (
            processed tensor [batch_size, season, steps_per_season, in_chans],
            padding mask [season, steps_per_season] or None,
            validity mask [batch_size, season, steps_per_season]
        )
    """
    # shapes from input tensor
    B, seq_len, in_chans = x.shape
    # batch_x = replace_nan_with_zero(x)
    # validity mask: 1 non-NaN, 0 NaN
    valid_mask = (~torch.isnan(x)).float()  # [batch_size, seq_len, in_chans]

    # steps per season and padding amount
    steps_per_season = math.ceil(seq_len / season)
    padding = steps_per_season * season - seq_len

    # zero padding
    if padding > 0:
        x = torch.nn.functional.pad(x, (0, 0, 0, padding), mode='constant', value=0)
        # pad valid_mask; padded region 0
        valid_mask = torch.nn.functional.pad(valid_mask, (0, 0, 0, padding), mode='constant', value=0)

    # reshape tensor to [batch_size, season, steps_per_season, in_chans]
    x = x.view(B, season, steps_per_season, in_chans)
    valid_mask = valid_mask.view(B, season, steps_per_season, in_chans)

    # build padding mask
    padding_mask = None
    if padding > 0:
        padding_mask = torch.ones(season, steps_per_season, device=x.device)
        padding_mask[:, -padding:] = 0

    # merge channel dim: valid only if all channels valid
    valid_mask = valid_mask.min(dim=-1)[0]  # [batch_size, season, steps_per_season]

    # apply padding mask to valid_mask if present
    if padding_mask is not None:
        valid_mask = valid_mask * padding_mask  # broadcast padding_mask to batch

    return x, padding_mask, valid_mask


# def replace_nan_with_zero(x: torch.Tensor) -> torch.Tensor:
#     """
#     Clone input and replace NaN with 0, same shape.

#     Args:
#         x (torch.Tensor): input tensor, shape [batch_size, seq_len, in_chans]

#     Returns:
#         torch.Tensor: cloned tensor with NaN->0
#     """
#     # clone to avoid mutating input
#     x_clone = x.clone()
#     # replace NaN with 0
#     x_clone = torch.where(torch.isnan(x_clone), torch.zeros_like(x_clone), x_clone)
#     return x_clone


import torch
import torch.nn.functional as F
import math


# ==========================================
# 1. New: non-linear time warping (core augmentation)
# ==========================================
def apply_time_warp(x, warp_strength=0.2, num_control_points=5, p_warp=0.5):
    """
    [GPU vectorized] non-linear time warping
    Simulate nonlinear phenology (e.g. cold spring slows early growth, heat accelerates later)
    
    Random smooth warp field as time flow, resample via grid_sample.
    
    Args:
        x: [B, T, C]
        warp_strength: warp strength; larger = stronger deformation (try 0.1-0.3)
        num_control_points: control points; fewer = smoother (large-scale climate variation)
    """
    B, T, C = x.shape
    device = x.device
    
    # 1. Bernoulli: which samples to warp
    do_warp = torch.bernoulli(torch.full((B,), p_warp, device=device)).bool()
    if not do_warp.any():
        return x
    
    # 2. build smooth warp field
    # use faster bilinear instead of bicubic
    # few random control points interpolated to length T
    # noise shape: [B, 1, num_control, 1] -> dims for interpolate 
    noise = torch.randn(B, 1, num_control_points, 1, device=device) * warp_strength
    
    # bilinear warp offsets [B,1,T,1] (much faster than bicubic)
    # align_corners=True align corners at boundaries
    warp_field = F.interpolate(noise, size=(T, 1), mode='bilinear', align_corners=True)
    
    # 3. build base grid
    # grid range [-1,1] maps to time [0,T]
    # shape: [1, T, 1, 2] -> last dim (x,y); we warp y (time)
    base_grid = torch.zeros(1, T, 1, 2, device=device)
    base_grid[:, :, 0, 1] = torch.linspace(-1, 1, T, device=device) # set y coordinate (time)
    
    # 4. add warp offset
    #final_grid: [B, T, 1, 2]
    # expand not clone (view, no copy)
    final_grid = base_grid.expand(B, -1, -1, -1)
    # add offset only for warped samples
    # warp_field warp_field [B,1,T,1] -> [B,T,1] on y axis
    offset = warp_field.permute(0, 2, 3, 1) # [B, T, 1, 1]
    # must clone final_grid (expand is read-only)
    final_grid = final_grid.clone()
    final_grid[do_warp, :, :, 1] += offset[do_warp, :, :, 0]
    
    # 5. clamp grid to avoid out-of-bounds sampling
    final_grid = torch.clamp(final_grid, -1, 1)
    
    # 6. grid_sample resampling
    # linear interpolation may be faster for series
    # x reshape to [B,C,T,1] for grid_sample (1-pixel-wide image)
    x_in = x.permute(0, 2, 1).unsqueeze(-1) # [B, C, T, 1]
    
    # sampled: [B, C, T, 1]
    # padding_mode border faster than reflection
    x_warped = F.grid_sample(x_in, final_grid, mode='bilinear', padding_mode='border', align_corners=True)
    
    # restore shape [B, T, C]
    x_out = x_warped.squeeze(-1).permute(0, 2, 1)
    
    # blend: replace only selected samples
    # where instead of clone+index may be faster
    # do_warp: [B] -> [B, 1, 1] to match [B, T, C]
    x_final = torch.where(do_warp.view(B, 1, 1), x_out, x)
    
    return x_final

# ==========================================
# 2. New: amplitude shift (baseline drift)
# ==========================================
def apply_amplitude_shift(x, shift_range=(-0.1, 0.1), p_shift=0.5):
    """
    [GPU vectorized] random additive shift
    simulate atmospheric correction residual or sensor baseline drift
    """
    B, T, C = x.shape
    device = x.device
    
    # which samples get shift
    do_shift = torch.bernoulli(torch.full((B, 1, 1), p_shift, device=device))
    
    # sample shift values [B, 1, 1]
    low, high = shift_range
    shifts = torch.rand(B, 1, 1, device=device) * (high - low) + low
    
    return x + (shifts * do_shift)

# ==========================================
# 3. Optimized: scaling and noise (logic fix)
# ==========================================
def apply_scaling_and_noise(x, sigma=0.01, scale_range=(0.95, 1.05), p_scale=0.5):
    """
    fixed order: scale first, then noise
    """
    B, T, C = x.shape
    device = x.device
    
    # 1. multiplicative scaling
    if scale_range[0] != 1.0 or scale_range[1] != 1.0:
        do_scale = torch.bernoulli(torch.full((B, 1, 1), p_scale, device=device, dtype=x.dtype))
        low, high = scale_range
        scale_factors = torch.rand(B, 1, 1, device=device, dtype=x.dtype) * (high - low) + low
        final_scale = scale_factors * do_scale + (1.0 - do_scale)
        x = x * final_scale
    
    # 2. additive noise - noise magnitude should not scale with signal
    if sigma > 0:
        noise = torch.randn_like(x) * sigma
        x = x + noise
        
    return x

# ==========================================
# 4. channel masking (unchanged)
# ==========================================
def apply_channel_masking(x, mask_prob=0.3):
    if mask_prob <= 0:
        return x
    B, T, C = x.shape
    device = x.device
    
    keep_prob = 1 - mask_prob
    mask = torch.bernoulli(torch.full((B, 1, C), keep_prob, device=device, dtype=x.dtype))
    
    # safety: avoid all-zero channels
    channel_sums = mask.sum(dim=-1, keepdim=True)
    all_zeros = (channel_sums == 0).squeeze(-1)
    
    if all_zeros.any():
        zero_batch_indices = all_zeros.nonzero(as_tuple=False)[:, 0]
        if len(zero_batch_indices) > 0:
            rand_channels = torch.randint(0, C, (len(zero_batch_indices),), device=device)
            mask[zero_batch_indices, 0, rand_channels] = 1.0
            
    return x * mask


def apply_gaussian_noise(x, noise_std=0.01, p_noise=0.5):
    """Add Gaussian noise to the whole view."""
    if noise_std <= 0:
        return x
    if torch.rand(1, device=x.device) >= p_noise:
        return x
    noise = torch.randn_like(x) * float(noise_std)
    return x + noise

# ==========================================
# 5. Student/Teacher input generation
# ==========================================
_MISSING_FILL_SENTINEL = 0.0


# Compatibility no-op: missing positions are augmented after zero fill.
def _restore_missing_after_aug(x_aug, obs_valid, sentinel=_MISSING_FILL_SENTINEL):
    """Compatibility no-op; missing positions are augmented after zero fill."""
    return x_aug


def get_student_input(x_raw, aug_type='weak', is_train=True):
    """
    build weak/strong view strategy
    """
    if not is_train:
        return x_raw.nan_to_num(_MISSING_FILL_SENTINEL)

    x_in = x_raw.nan_to_num(_MISSING_FILL_SENTINEL)

    if aug_type == 'none':
        return x_in

    if aug_type == 'weak':
        # Student weak: keep only light observation noise.
        return apply_gaussian_noise(
            x_in, noise_std=0.012, p_noise=0.5
        )

    if aug_type == 'weak_local':
        # Student local/random: light noise plus weak channel masking.
        x_in = apply_gaussian_noise(
            x_in, noise_std=0.015, p_noise=0.5
        )
        x_in = apply_channel_masking(x_in, mask_prob=0.12)
        return x_in

    elif aug_type == 'strong':
        # Student global: observation noise + channel masking only.
        x_in = apply_gaussian_noise(
            x_in, noise_std=0.02, p_noise=0.6
        )
        x_in = apply_channel_masking(x_in, mask_prob=0.38) 
        return x_in
        
    return x_in


def get_teacher_input(x_raw, aug_type='weak', is_train=True):
    """
    Teacher view stays numerically stable; only content differences should matter.
    """
    if not is_train:
        return x_raw.nan_to_num(_MISSING_FILL_SENTINEL)

    x_in = x_raw.nan_to_num(_MISSING_FILL_SENTINEL)
    return x_in


def imputator_sliding_window_overlap(
    x_enc,
    time_mark,
    missing_mask_orig,
    imputator,
    window_len: int,
    stride: int,
    device: torch.device,
    imp_device: torch.device,
):
    """
    Long sequences: sliding imputator windows of window_len (usually imputator.pred_len), stride (e.g. 244 ~ two years);
    average imputations on overlaps at missing positions; observed positions use x_clean_filled, excluded from average numerator.

    Args:
        x_enc: [B, T, C]，may contain nan
        time_mark: [B, T, 2] or None
        missing_mask_orig: [B, T] bool，True=needs imputation
        imputator: pred_len usually equals window_len
        window_len: segment length for imputator (pad by repeating last timestep if shorter)
        stride: window start stride

    Returns:
        imputed_out: [B, T, C]，fused prediction only where missing_mask_orig True and covered by a window;
                     else same as x_clean_filled (caller merges with mask for perfect target).
    """
    B, T, C = x_enc.shape
    x_clean_filled = x_enc.nan_to_num(0.0)
    if T <= window_len:
        if imp_device != device:
            x_in_imp = x_enc.to(imp_device, non_blocking=True)
            time_imp = time_mark.to(imp_device, non_blocking=True) if time_mark is not None else None
            imputed_out = imputator(x_in_imp, time_mark=time_imp, mode="pred")
            imputed_out = imputed_out.to(device, non_blocking=True)
        else:
            imputed_out = imputator(x_enc, time_mark=time_mark, mode="pred")
        return imputed_out

    acc_sum = torch.zeros(B, T, C, device=device, dtype=x_enc.dtype)
    acc_count = torch.zeros(B, T, device=device, dtype=x_enc.dtype)

    starts = []
    s = 0
    while s < T:
        starts.append(s)
        if s + window_len >= T:
            break
        s += stride

    for start in starts:
        end = min(start + window_len, T)
        actual_len = end - start
        x_seg = x_enc[:, start:end, :]
        time_seg = time_mark[:, start:end, :] if time_mark is not None else None

        if actual_len < window_len:
            pad_len = window_len - actual_len
            last = x_seg[:, -1:, :].nan_to_num(0.0)
            x_input = torch.cat([x_seg, last.expand(B, pad_len, C)], dim=1)
            if time_seg is not None:
                last_t = time_seg[:, -1:, :]
                time_input = torch.cat([time_seg, last_t.expand(B, pad_len, time_seg.shape[-1])], dim=1)
            else:
                time_input = None
        else:
            x_input = x_seg
            time_input = time_seg

        if imp_device != device:
            xi = x_input.to(imp_device, non_blocking=True)
            ti = time_input.to(imp_device, non_blocking=True) if time_input is not None else None
            out = imputator(xi, time_mark=ti, mode="pred")
            out = out.to(device, non_blocking=True)
        else:
            out = imputator(x_input, time_mark=time_input, mode="pred")

        out = out[:, :actual_len, :]
        m = missing_mask_orig[:, start:end].float()
        acc_sum[:, start:end, :] += out * m.unsqueeze(-1)
        acc_count[:, start:end] += m

    has_pred = missing_mask_orig & (acc_count > 1e-6)
    fused = acc_sum / acc_count.unsqueeze(-1).clamp(min=1e-6)
    imputed_out = torch.where(has_pred.unsqueeze(-1), fused, x_clean_filled)
    return imputed_out


def valid_sample_keep_mask(x_enc: torch.Tensor, threshold: float) -> torch.Tensor | None:
    """
    Per-sample SSL loss gating aligned with TED.valid_sample_threshold.
    Returns [B,1,1] float mask (1=keep) or None when threshold<=0 (no gating).
    """
    thr = float(threshold)
    if thr <= 0.0:
        return None
    valid_ratio = (~torch.isnan(x_enc).any(dim=-1)).float().mean(dim=-1)
    return (valid_ratio >= thr).float().view(-1, 1, 1)


def patchify(x, patch_len, stride):
    """
    Split time series into patch tokens.

    Notes:
    - N tokens: N = ceil((T - patch_len) / stride) + 1 when T > patch_len
    - zero-pad tail if indivisible so unfold does not fail
    - unpatchify truncates padding using original_seq_len

    Args:
        x: [B, T, C]
        patch_len: patch length
        stride: stride

    Returns:
        patches: [B, N, patch_len * C]
    """
    B, T, C = x.shape

    # compute token count and padded length for unfold
    # N = floor((T_pad - patch_len)/stride) + 1  =>  T_pad = (N-1)*stride + patch_len
    if T <= patch_len:
        nPatches = 1
    else:
        nPatches = int(math.ceil((T - patch_len) / stride) + 1)
    tPadded = (nPatches - 1) * stride + patch_len
    padLen = max(0, tPadded - T)

    if padLen > 0:
        x = F.pad(x.permute(0, 2, 1), (0, padLen)).permute(0, 2, 1)

    x = x.permute(0, 2, 1)
    xUnfold = x.unfold(dimension=2, size=patch_len, step=stride)
    xUnfold = xUnfold.permute(0, 2, 1, 3).contiguous()
    return xUnfold.view(B, xUnfold.shape[1], -1)

def unpatchify(x_patches, original_seq_len, patch_len, c_out):
    B, N, PC = x_patches.shape
    C = c_out
    x = x_patches.view(B, N, C, patch_len)
    x = x.permute(0, 2, 1, 3).contiguous() 
    x = x.view(B, C, -1).permute(0, 2, 1)  
    if x.shape[1] > original_seq_len:
        x = x[:, :original_seq_len, :]
    return x


def random_patch_masking_patchtst_uniform(B, num_patches, mask_ratio, device):
    """
    Uniform random patch masking for the MSM baseline:
    uniform random mask ratio on patch indices; MSE reconstruct masked patches only.
    same return tuple as the block-biased masking helper for MSM swap.

    Args:
        B: batch size
        num_patches: N（patches per sample）
        mask_ratio: (0,1] ~mask_ratio*N patches masked (0.4 matches paper ~40% patches)
        device: torch device

    Returns:
        collated_masks: [B, N] bool，True = patch masked for SSL
        mask_indices_list: flattened indices where True（same as dinov3 version）
        masks_weight: row-normalized weights（same as dinov3 version）
    """
    N = int(num_patches)
    if N <= 0:
        collated_masks = torch.zeros(B, 0, dtype=torch.bool, device=device)
        return collated_masks, torch.empty(0, dtype=torch.long, device=device), torch.empty(0, device=device)

    ratio = float(mask_ratio)
    ratio = max(0.0, min(1.0, ratio))
    n_masked = int(round(N * ratio))
    n_masked = max(0, min(N, n_masked))

    if n_masked == 0:
        collated_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
    else:
        noise = torch.rand(B, N, device=device)
        perm = noise.argsort(dim=-1)
        idx = perm[:, :n_masked]
        collated_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
        collated_masks.scatter_(1, idx, True)

    mask_indices_list = collated_masks.flatten().nonzero().flatten()
    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]
    return collated_masks, mask_indices_list, masks_weight


def random_patch_masking_dinov3_style(
    B,
    mask_ratio_tuple,
    mask_sample_probability,
    num_patches,
    device,
    block_ratio: float = 0.8,
    block_size: int = 5,
):
    """
    Block-biased patch mask generation (optimized: avoid CPU-GPU sync)
    Args:
        B: batch size
        mask_ratio_tuple: (min_ratio, max_ratio)，e.g. (0.1, 0.5)
            mask_sample_probability: fraction of B rows with non-empty patch mask (0-1); 0.5 keeps about half rows unmasked
        num_patches: number of patches
        device: device
    Returns:
        masks: [B, num_patches] bool tensor; True=masked
        mask_indices_list: flattened mask indices
        masks_weight: weight per masked patch
    """
    N = num_patches
    # expected masked rows rounded to [0,B]; avoids int(B*p)==0 for small B
    p = float(mask_sample_probability)
    p = max(0.0, min(1.0, p))
    n_samples_masked = max(0, min(B, int(B * p + 0.5)))
    
    # linear spread of mask ratios
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1, device=device)
    
    # build all masks on GPU, no loops/sync
    masks_tensor = torch.zeros(B, N, dtype=torch.bool, device=device)
    
    # masks for rows that should be masked
    # avoid .item() in loop; stay on GPU
    for i in range(n_samples_masked):
        prob_max = probs[i + 1]  # stay on GPU, no .item()
        n_masked = int((N * prob_max).item())  # call .item() only when int needed
        # mask with block+random mix
        mask = generate_single_mask(N, n_masked, device, block_ratio=block_ratio, block_size=block_size)
        masks_tensor[i] = mask
    
    # unmasked rows stay False
    
    # GPU randperm shuffle, no CPU sync
    indices = torch.randperm(B, device=device)
    collated_masks = masks_tensor[indices]  # [B, N]
    
    # build flattened mask_indices_list
    mask_indices_list = collated_masks.flatten().nonzero().flatten()
    
    # masks_weight to balance mask counts across samples
    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]
    
    return collated_masks, mask_indices_list, masks_weight


def generate_single_mask(N, n_masked, device, block_ratio: float = 0.8, block_size: int = 5):
    """
    single-sample mask (block+random, optimized)
    Args:
        N: number of patches
        n_masked: number of patches to mask
    Returns:
        mask: [N] bool tensor
    """
    if n_masked == 0:
        return torch.zeros(N, dtype=torch.bool, device=device)
    
    mask = torch.zeros(N, dtype=torch.bool, device=device)
    
    # block_ratio fraction of mask is contiguous
    target_block_mask = int(n_masked * block_ratio)
    
    # vectorized blocks, fewer loops
    # build block mask
    num_blocks = int(math.ceil(target_block_mask / block_size * 1.2))
    if num_blocks > 0 and N >= block_size:
        # vectorized batch block mask instead of loops
        rand_starts = torch.randint(0, max(1, N - block_size + 1), (num_blocks,), device=device)
        # vectorized advanced indexing for all blocks
        # index ranges per block
        block_indices = rand_starts.unsqueeze(1) + torch.arange(block_size, device=device).unsqueeze(0)  # [num_blocks, block_size]
        block_indices = block_indices.clamp(max=N-1)  # clamp boundaries
        block_indices = block_indices.flatten()  # [num_blocks * block_size]
        # set all masks at once via unique indices
        unique_indices = torch.unique(block_indices)
        mask[unique_indices] = True
    
    # fill remaining with random mask
    # call .item() only when needed
    current_masked = mask.sum().item()
    to_fill = n_masked - current_masked
    if to_fill > 0:
        available_indices = (~mask).nonzero().flatten()
        if len(available_indices) > 0:
            n_to_select = min(to_fill, len(available_indices))
            selected = available_indices[torch.randperm(len(available_indices), device=device)[:n_to_select]]
            mask[selected] = True
    
    return mask


def random_patch_masking(B, mask_rate, device, num_patches):
    """
    [mixed] block + random masking
    fully vectorized, no CPU loops
    1 (True) = masked: student cannot see, must predict.
    0 (False) = visible: student context.
    """
    N = num_patches
    
    # parameters
    block_size = 5     # block size
    block_ratio = 0.8  # 80% of mask is contiguous
    
    # 1. target count for block mask
    target_mask_total = int(N * mask_rate)
    target_block_mask = int(target_mask_total * block_ratio)
    
    # 2. build contiguous block mask
    num_blocks = int(math.ceil(target_block_mask / block_size * 1.2)) 
    rand_starts = torch.randint(0, max(1, N - block_size + 1), (B, num_blocks), device=device)
    offsets = torch.arange(block_size, device=device).view(1, 1, -1)
    block_indices = rand_starts.unsqueeze(-1) + offsets
    block_indices = block_indices.view(B, -1)
    
    # 3. build mask matrix
    mask = torch.zeros((B, N), device=device)
    src = torch.ones_like(block_indices, dtype=torch.float)
    mask.scatter_(1, block_indices, src) # scatter blocks
    
    # 4. fill random scattered mask
    current_masked_count = mask.sum(dim=-1) # [B]
    to_fill = target_mask_total - current_masked_count
    to_fill = to_fill.clamp(min=0).long()
    
    noise = torch.rand((B, N), device=device)
    noise.masked_fill_(mask.bool(), 1e9) # exclude already masked positions

    max_fill = to_fill.max().item()
    if max_fill > 0:
        _, indices = torch.topk(noise, k=max_fill, dim=-1, largest=False)
        
        # mask indicating valid fill indices
        batch_range = torch.arange(max_fill, device=device).unsqueeze(0)
        valid_indices_mask = batch_range < to_fill.unsqueeze(1) # [B, max_fill]
        
        src_fill = valid_indices_mask.float()
        mask.scatter_(1, indices, src_fill)

    mask = mask.clamp(0, 1)
    return mask

def create_smooth_target(x_filled, missing_mask_bool, kernel_size=5):
    """
    [adapted: 1=invalid, 0=valid]
    AvgPool1d approximates linear interpolation to fill zeros
    """
    # x_filled: [B, T, C] (filled with 0 or original)
    # missing_mask_bool: [B, T] (True/1=missing/invalid, False/0=valid)
    
    B, T, C = x_filled.shape
    
    # 1. prepare tensors
    x = x_filled.permute(0, 2, 1) # [B, C, T]
    
    # Convert bool mask to float: 1.0=invalid, 0.0=valid
    mask_invalid = missing_mask_bool.float().unsqueeze(1)  # [B, 1, T]
    mask_valid = 1.0 - mask_invalid
    x_masked = x * mask_valid  # broadcast multiply

    # 2. smooth (merged ops)
    padding = kernel_size // 2
    x_smooth = F.avg_pool1d(x_masked, kernel_size, stride=1, padding=padding, count_include_pad=False)
    weight_smooth = F.avg_pool1d(mask_valid, kernel_size, stride=1, padding=padding, count_include_pad=False)
    
    # 3. interpolate (merged)
    x_interpolated = x_smooth / (weight_smooth + 1e-6)
    
    # 4. fuse (where instead of expand*)
    mask_invalid_expanded = mask_invalid.expand(-1, C, -1)  # [B, C, T]
    x_final = torch.where(mask_invalid_expanded, x_interpolated, x)

    return x_final.permute(0, 2, 1)  # [B, T, C]


def generate_local_view_crop(x, target_len, device):
    """
    local view via crop (vectorized)
    Args:
        x: [B, T, C] raw sequence
        target_len: target length in timesteps
        device: device
    Returns:
        x_crop: [B, target_len, C]
        start_indices: [B] start index per sample
    """
    B, T, C = x.shape
    max_start = max(0, T - target_len)
    start_indices = torch.randint(0, max_start + 1, (B,), device=device)
    
    # vectorized gather/indexing
    # build index matrix [B, target_len]
    batch_indices = torch.arange(B, device=device).unsqueeze(1)  # [B, 1]
    time_indices = start_indices.unsqueeze(1) + torch.arange(target_len, device=device).unsqueeze(0)  # [B, target_len]
    
    # extract via gather/indexing
    x_crop = x[batch_indices, time_indices]  # [B, target_len, C]
    return x_crop, start_indices


def generate_local_view_random_sample(x, num_tokens, device):
    """
    local view via random token sample (vectorized)
    Args:
        x: [B, T, C] raw sequence（patchified tokens; T is token count）
        num_tokens: number of tokens to sample
        device: device
    Returns:
        x_sampled: [B, num_tokens, C]
        token_indices: [B, num_tokens] sampled token indices
    """
    B, T, C = x.shape
    
    # vectorized random indices per batch
    # random matrix [B, T]，then topk
    rand_vals = torch.rand(B, T, device=device)
    _, token_indices = torch.topk(rand_vals, k=num_tokens, dim=1)  # [B, num_tokens]
    token_indices, _ = torch.sort(token_indices, dim=1)  # sort to preserve time order
    
    # gather extract
    batch_indices = torch.arange(B, device=device).unsqueeze(1)  # [B, 1]
    x_sampled = x[batch_indices, token_indices]  # [B, num_tokens, C]
    
    return x_sampled, token_indices
