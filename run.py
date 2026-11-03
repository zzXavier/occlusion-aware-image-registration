import argparse
from pathlib import Path

from src.experiment import demo, evaluate, reproduce

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["demo", "reproduce", "evaluate"])
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    {"demo": demo, "reproduce": reproduce, "evaluate": evaluate}[args.action](args.out)
