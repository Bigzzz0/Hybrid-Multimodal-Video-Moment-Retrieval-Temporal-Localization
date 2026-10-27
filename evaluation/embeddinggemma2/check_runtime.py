"""Check dependency pins without loading a model or changing any environment."""
import json
import sys
from importlib.metadata import version
from pathlib import Path

def main():
    lock = json.loads(Path(__file__).with_name("runtime-lock.json").read_text())
    mismatches = []
    if f"{sys.version_info.major}.{sys.version_info.minor}" != lock["python"]:
        mismatches.append("Python version")
    for package, wanted in lock["packages"].items():
        actual = version(package)
        if actual != wanted:
            mismatches.append(f"{package}: {actual} != {wanted}")
    if mismatches:
        raise RuntimeError("Runtime drift: " + "; ".join(mismatches))
    print("Pinned Gemma runtime verified; no inference executed.")

if __name__ == "__main__":
    main()
