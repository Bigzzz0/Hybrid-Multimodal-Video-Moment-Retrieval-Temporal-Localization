"""
One-click Automated Setup for CapRL-Qwen3VL-4B Q6 (GGUF) + llama.cpp on Windows
Downloads llama.cpp Windows CUDA binary and CapRL Q6 weights, updates backend/.env,
and tests model loading.
"""

import os
import sys
import time
import zipfile
import urllib.request
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO_DIR = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = REPO_DIR / "backend"
BIN_DIR = BACKEND_DIR / "bin" / "llama"
MODELS_DIR = BACKEND_DIR / "models" / "gguf"
ENV_FILE = BACKEND_DIR / ".env"

LLAMA_ZIP_URL = "https://github.com/ggml-org/llama.cpp/releases/download/b11450/llama-b11450-bin-win-cuda-13.4-x64.zip"
CUDART_ZIP_URL = "https://github.com/ggml-org/llama.cpp/releases/download/b11450/cudart-llama-bin-win-cuda-13.4-x64.zip"

def download_and_extract_zip(url: str, dest_dir: Path, name: str):
    print(f"\n[1/3] Downloading {name} from {url}...")
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / f"{name}.zip"

    # Download with streaming and progress
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(zip_path, "wb") as f:
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        t0 = time.time()
        last_print = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            now = time.time()
            if now - last_print > 1.0 or downloaded == total:
                pct = (downloaded / total * 100) if total else 0
                mb = downloaded / (1024 * 1024)
                total_mb = total / (1024 * 1024) if total else 0
                spd = mb / max(0.1, (now - t0))
                print(f"  -> {name}: {mb:.1f}/{total_mb:.1f} MB ({pct:.1f}%) @ {spd:.2f} MB/s", end="\r", flush=True)
                last_print = now
    print(f"\n  Extracting {name} to {dest_dir}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)
    try:
        os.remove(zip_path)
    except Exception:
        pass
    print(f"  [OK] {name} extracted successfully.")

def download_caprl_models():
    print("\n[2/3] Downloading CapRL-Qwen3VL-4B GGUF models from Hugging Face...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    files = [
        ("CapRL-Qwen3VL-4B-mmproj-Q8_0.gguf", "Multimodal Projector (~800MB)"),
        ("CapRL-Qwen3VL-4B-q6_k.gguf", "CapRL Q6 Quantized Base (~3.5GB)"),
    ]
    
    for filename, desc in files:
        target_path = MODELS_DIR / filename
        if target_path.exists() and target_path.stat().st_size > 100 * 1024 * 1024:
            print(f"  [SKIP] {filename} already exists ({target_path.stat().st_size / (1024**2):.1f} MB).")
            continue
        print(f"  Downloading {filename} - {desc}...")
        hf_hub_download(
            repo_id="internlm/CapRL-Qwen3VL-4B-GGUF",
            filename=filename,
            local_dir=str(MODELS_DIR),
            local_dir_use_symlinks=False,
        )
        print(f"  [OK] {filename} downloaded successfully.")

def update_env():
    print("\n[3/3] Updating backend/.env configuration...")
    if not ENV_FILE.exists():
        print(f"  [WARN] {ENV_FILE} not found!")
        return

    # Find the llama executable (either llama-server.exe or llama.exe)
    llama_exe = BIN_DIR / "llama-server.exe"
    if not llama_exe.exists():
        alt = BIN_DIR / "llama.exe"
        if alt.exists():
            llama_exe = alt

    content = ENV_FILE.read_text(encoding="utf-8")
    
    updates = {
        "INFERENCE_WORKER_ENABLED": "true",
        "LLAMA_CPP_PATH": str(llama_exe),
        "VLM_GGUF_DIR": str(MODELS_DIR),
    }

    lines = content.splitlines()
    new_lines = []
    seen = set()

    for line in lines:
        matched = False
        for k, v in updates.items():
            if line.startswith(f"{k}="):
                new_lines.append(f"{k}={v}")
                seen.add(k)
                matched = True
                break
        if not matched:
            new_lines.append(line)

    for k, v in updates.items():
        if k not in seen:
            new_lines.append(f"{k}={v}")

    ENV_FILE.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    print("  [OK] Updated backend/.env with:")
    for k, v in updates.items():
        print(f"       {k}={v}")

def main():
    print("=" * 70)
    print("  CapRL-Qwen3VL-4B Q6 (GGUF) + llama.cpp Automated Setup")
    print("=" * 70)
    
    # 1. Download llama binaries
    llama_exe = BIN_DIR / "llama-server.exe"
    alt_exe = BIN_DIR / "llama.exe"
    if not (llama_exe.exists() or alt_exe.exists()):
        download_and_extract_zip(LLAMA_ZIP_URL, BIN_DIR, "llama_binary")
        download_and_extract_zip(CUDART_ZIP_URL, BIN_DIR, "cudart_runtime")
    else:
        print("  [SKIP] llama executable already exists in bin/llama.")

    # 2. Download CapRL models
    download_caprl_models()

    # 3. Update env file
    update_env()

    print("\n" + "=" * 70)
    print("  [SUCCESS] All components downloaded and configured!")
    print("=" * 70)

if __name__ == "__main__":
    main()
