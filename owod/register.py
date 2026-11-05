"""Make RT-DETRv2 (`src` package) importable and register the open-world modules."""
import os
import sys

RTDETR_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'third_party', 'RT-DETR', 'rtdetrv2_pytorch')
if RTDETR_ROOT not in sys.path:
    sys.path.insert(0, RTDETR_ROOT)

import src  # noqa: E402,F401  (registers upstream modules)
from .data import mowodb  # noqa: E402,F401
from .zoo import ow_decoder, ow_criterion, ow_postprocessor, ow_rtdetr, simota_matcher  # noqa: E402,F401
