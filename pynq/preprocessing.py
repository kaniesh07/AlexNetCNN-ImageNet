"""
preprocessing.py -- image -> kernel input words (runs on the ARM, numpy + Pillow).

Steps (identical to AlexNet_Weights.IMAGENET1K_V1.transforms on a PIL image):
  1. RGB, resize shorter side to 256 (bilinear), centre-crop 224x224
  2. /255, subtract mean (0.485,0.456,0.406), divide std (0.229,0.224,0.225)
CPU-side tensor preparation for the FPGA (part of the measured end-to-end time):
  3. int8 quantisation with the calibrated input scale
  4. zero pad by 2 (conv1 padding) -> 3x228x228
  5. space-to-depth (4x4) -> 48x57x57, so conv1 (11x11, stride 4) runs on the
     FPGA as an exactly equivalent 3x3 / stride-1 convolution
  6. pack to C8HW 64-bit words (19,494 words)
"""
import time

import numpy as np
from PIL import Image

import alexnet_common as ac


def load_and_prepare(path, input_scale):
    t0 = time.perf_counter()
    x = ac.preprocess_pil(Image.open(path))
    t1 = time.perf_counter()
    words = ac.prepare_input(x, input_scale)
    t2 = time.perf_counter()
    return x, words, {"preprocess": t1 - t0, "quantize_pack": t2 - t1}
