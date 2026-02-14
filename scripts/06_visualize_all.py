from __future__ import annotations

import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

SCRIPTS = [
    "visualize_metrics.py",
    "visualize_radar.py",
    "visualize_tradeoff.py",
]


def main():
    script_dir = Path(__file__).resolve().parent
    for name in SCRIPTS:
        path = script_dir / name
        if path.exists():
            print(f"\n--- {name} ---")
            subprocess.run([sys.executable, str(path)], check=True)
        else:
            print(f"Skip {name} (not found)")
    print("\nDone. Check reports/{data_dir}_{hash}/ for PNGs.")


if __name__ == "__main__":
    main()
