# Categorical-state projection head used by the sequence-state and patch-state branches.
import torch
import torch.nn as nn
import torch.nn.functional as F


class CategoricalStateHead(nn.Module):
    """
    Projection head for categorical state distributions.
    MLP (in_dim -> hidden_dim -> hidden_dim -> bottleneck_dim) -> L2 normalize -> Linear (bottleneck_dim -> out_dim)
    """
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        mlp_bias=True,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.mlp = self._build_mlp(
            nlayers,
            in_dim,
            bottleneck_dim,
            hidden_dim=hidden_dim,
            use_bn=use_bn,
            bias=mlp_bias,
        )
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)

    def _build_mlp(self, nlayers, in_dim, bottleneck_dim, hidden_dim=None, use_bn=False, bias=True):
        """
        Build MLP layers
        For time series, use LayerNorm not BatchNorm1d
BatchNorm1d in batch dimensionnormalize, may not be reasonable for time sequence
LayerNorm in feature dimensionnormalize, more suitable for time sequence tasks
        """
        if nlayers == 1:
            return nn.Linear(in_dim, bottleneck_dim, bias=bias)
        else:
            layers = [nn.Linear(in_dim, hidden_dim, bias=bias)]
            if use_bn:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
                if use_bn:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
            return nn.Sequential(*layers)

    def init_weights(self) -> None:
        """Initialize weights"""
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """weights initialization function"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, no_last_layer=False, only_last_layer=False):
        """
        forward pass
        Args:
            x: input features [..., in_dim]
            no_last_layer: if True, skip last layer
            only_last_layer: if True, only last layer (after MLP)
        """
        if not only_last_layer:
            x = self.mlp(x)
            eps = 1e-6 if x.dtype == torch.float16 else 1e-12
            x = F.normalize(x, dim=-1, p=2, eps=eps)
        if not no_last_layer:
            x = self.last_layer(x)
        return x
