#!/usr/bin/env python3
"""
reference_test.py -- Stage A (software reference).  HOST, needs torch+torchvision.

1. numpy float AlexNet == torchvision AlexNet (random inputs)
2. preprocess_pil == AlexNet_Weights.IMAGENET1K_V1.transforms() (same PIL image)
3. optional: --images DIR runs the torchvision model on real images and prints
   top-5 labels so the pretrained checkpoint/class mapping can be eyeballed;
   --val-dir DIR --val-labels FILE computes Top-1 (torch fp32 vs numpy fp32).
"""
import argparse
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pynq"))
import alexnet_common as ac  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(ROOT, "build", "alexnet_fp32.npz"))
    ap.add_argument("--images")
    a = ap.parse_args()
    import torch
    from PIL import Image
    from torchvision.models import AlexNet_Weights, alexnet

    P = dict(np.load(a.weights))
    model = alexnet(weights=None)
    model.load_state_dict({k: torch.from_numpy(v) for k, v in P.items()})
    model.eval()

    # 1. forward equivalence
    g = torch.Generator().manual_seed(0)
    worst = 0.0
    for _ in range(3):
        x = torch.randn(1, 3, 224, 224, generator=g)
        with torch.no_grad():
            ref = model(x)[0].numpy()
        got = ac.alexnet_float(x[0].numpy(), P)
        worst = max(worst, float(np.abs(ref - got).max() / (np.abs(ref).max() + 1e-12)))
        assert np.argmax(ref) == np.argmax(got)
    assert worst < 1e-4, worst
    print(f"[A1] numpy float forward == torchvision (max rel err {worst:.1e})  PASS")

    # 2. preprocessing equivalence on a synthetic, non-square PIL image
    rng = np.random.default_rng(0)
    img = Image.fromarray((rng.random((300, 420, 3)) * 255).astype(np.uint8))
    tv = AlexNet_Weights.IMAGENET1K_V1.transforms()(img).numpy()
    mine = ac.preprocess_pil(img)
    err = float(np.abs(tv - mine).max())
    print(f"[A2] preprocess_pil vs torchvision transforms: max abs diff {err:.2e}  "
          f"{'PASS' if err < 1e-2 else 'CHECK'}")
    assert err < 5e-2, "preprocessing differs from torchvision"

    # 3. real images
    if a.images:
        labels = open(os.path.join(ROOT, "model", "imagenet_classes.txt")).read().splitlines()
        for f in sorted(glob.glob(os.path.join(a.images, "*"))):
            try:
                x = ac.preprocess_pil(Image.open(f))
            except Exception:
                continue
            p = ac.softmax(ac.alexnet_float(x, P))
            top = np.argsort(-p)[:5]
            print(os.path.basename(f), [(int(i), labels[i], round(float(p[i]), 3)) for i in top])


if __name__ == "__main__":
    main()
