# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Kota Nakajima
"""adaptFVC: RGB画像からの植被率(FVC)推定 対話型セグメンテーションツール。"""

import random

import numpy as np

__version__ = "3.2.0"
__author__ = "Kota Nakajima"
__copyright__ = "Copyright (C) 2026 Kota Nakajima"
__license__ = "GPL-3.0-or-later"
__url__ = "https://github.com/KotaNakajima/adaptFVC"

# Reproducibility: fix seeds for deterministic training
random.seed(42)
np.random.seed(42)

from adaptfvc.core import (  # noqa: E402,F401
    calculate_coverage,
    calculate_exg_mask,
    create_overlay_from_mask,
    create_overlay_from_mask_colored,
    get_pixel_features,
    load_image,
    predict_mask,
    segment_image,
    segment_image_with_method,
)
