import platform
import sys

import torch


def main():
    print("=" * 60)
    print("VITO SYSTEM CHECK")
    print("=" * 60)

    print(f"Python        : {sys.version.split()[0]}")
    print(f"Platform      : {platform.platform()}")
    print(f"Architecture  : {platform.machine()}")
    print(f"PyTorch       : {torch.__version__}")

    print()
    print("Hardware")
    print("-" * 60)

    print(f"MPS available : {torch.backends.mps.is_available()}")
    print(f"MPS built     : {torch.backends.mps.is_built()}")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Device        : {device}")
        print()
        print("VITO local GPU environment: READY")
    else:
        print()
        print("VITO local GPU environment: CPU ONLY")

    print("=" * 60)


if __name__ == "__main__":
    main()
