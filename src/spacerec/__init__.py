"""spacerec: webcam-based real-time 3D space and object recognition.

Import this package before torch/cv heavy deps: it sets env vars required
on Apple Silicon (OpenMP duplicate runtime, MPS op fallback).
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
