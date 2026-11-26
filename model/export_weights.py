#!/usr/bin/env python3
"""
export_weights.py -- HOST PC script (needs torch + torchvision).

Loads the official torchvision AlexNet checkpoint
    torchvision.models.AlexNet_Weights.IMAGENET1K_V1
    URL  https://download.pytorch.org/models/alexnet-owt-7be5be79.pth
    ImageNet-1k val: acc@1 56.522 / acc@5 79.066 (torchvision documentation)
and writes
    build/alexnet_fp32.npz      float32 weights keyed by state_dict name
    model/imagenet_classes.txt  1000 labels, line i = class index i
It then checks that the numpy float reference (pynq/alexnet_common.py) matches
PyTorch on a random input, so every later stage starts from a verified
reference.

Usage
    python model/export_weights.py                     # download (cached in ~/.cache/torch)
    python model/export_weights.py --checkpoint alexnet-owt-7be5be79.pth   # offline
    python model/export_weights.py --random-init       # pipeline test only (NOT pretrained)
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "pynq"))
import alexnet_common as ac  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", help="local .pth file (offline use)")
    ap.add_argument("--random-init", action="store_true",
                    help="use randomly initialised weights (pipeline testing only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "build", "alexnet_fp32.npz"))
    args = ap.parse_args()

    import torch
    import torchvision
    from torchvision.models import AlexNet_Weights, alexnet

    torch.manual_seed(args.seed)
    weights_enum = AlexNet_Weights.IMAGENET1K_V1
    if args.random_init:
        model = alexnet(weights=None)
        print("WARNING: random-init weights -- results are NOT ImageNet-accurate")
    elif args.checkpoint:
        model = alexnet(weights=None)
        sd = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(sd)
        print("loaded checkpoint", args.checkpoint)
    else:
        model = alexnet(weights=weights_enum)
        print("loaded", weights_enum, "from", weights_enum.url)
    model.eval()

    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, **sd)
    print("wrote", args.out)
    print("torch", torch.__version__, "torchvision", torchvision.__version__)
    for k, v in sd.items():
        print(f"  {k:24s} {str(v.shape):22s}")
    print("total parameters:", sum(v.size for v in sd.values()))

    cls_path = os.path.join(HERE, "imagenet_classes.txt")
    with open(cls_path, "w") as f:
        for c in weights_enum.meta["categories"]:
            f.write(c + "\n")
    print("wrote", cls_path, "(1000 classes)")

    # ---- verify numpy reference == torch -------------------------------
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        ref = model(x)[0].numpy()
    got = ac.alexnet_float(x[0].numpy(), sd)
    err = np.abs(ref - got).max() / (np.abs(ref).max() + 1e-12)
    print(f"numpy-vs-torch max relative error: {err:.2e}")
    assert err < 1e-4, "numpy float reference does not match torchvision"
    assert np.argmax(ref) == np.argmax(got)
    print("PASS: numpy float reference matches torchvision AlexNet")


if __name__ == "__main__":
    main()
