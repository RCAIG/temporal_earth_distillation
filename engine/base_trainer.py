import os

import torch

from models import imputator, msm, ntp, ted


class BaseTrainer(object):
    """Minimal trainer base: device setup + model registry."""

    def __init__(self, args):
        self.args = args
        # Canonical names first; legacy aliases kept for older scripts/checkpoints.
        self.model_dict = {
            'TED': ted,
            'MSM': msm,
            'NTP': ntp,
            'Imputator': imputator,
            'TED_modular': ted,
            'Patch_Masked': msm,
            'Patch_NTP_TED': ntp,
            'Transformer': imputator,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            import platform
            if platform.system() == 'Darwin':
                device = torch.device('mps')
                if getattr(self.args, 'local_rank', 0) == 0:
                    print('Use MPS')
                return device
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            if getattr(self.args, 'local_rank', 0) == 0:
                if self.args.use_multi_gpu:
                    print('Use GPU: cuda{}'.format(self.args.device_ids))
                else:
                    print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            if getattr(self.args, 'local_rank', 0) == 0:
                print('Use CPU')
        return device

    def _get_data(self):
        pass

    def train(self):
        pass
