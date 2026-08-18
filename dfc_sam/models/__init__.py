"""DFC-SAM model components."""

from .factory import build_dfc_sam
from .feature_bridge import DifferentiableFeatureBridge, SoftBoxGate
from .sam_instance_decoder import SAMInstanceDecoder
from .ugca import UGCA, UGCAOutput

__all__ = [
    "DifferentiableFeatureBridge",
    "SAMInstanceDecoder",
    "SoftBoxGate",
    "UGCA",
    "UGCAOutput",
    "build_dfc_sam",
]
