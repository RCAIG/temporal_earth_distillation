# TED components: categorical-state heads and temporal backbone.
from .heads import CategoricalStateHead, DINOHead, ProtoHead
from .backbone import Backbone

__all__ = ["CategoricalStateHead", "DINOHead", "ProtoHead", "Backbone"]
