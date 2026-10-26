#!/usr/bin/env python3
"""Prompt2Rig — Kaggle & Cloudflare Tunnel Master Runner.

Automatically checks GPU, downloads model checkpoints (SMPL-X, GVHMR, WiLoR),
sets up Cloudflare Tunnel, launches the unified server, and renders an interactive
web app link directly inside your Kaggle / Colab notebook.

Usage:
    python kaggle_run.py [--port 5177] [--fal-key YOUR_KEY] [--no-tunnel]
"""

import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HQ = ROOT / "hq"
WEIGHTS = HQ / "weights"
BODY_MODELS = HQ / "body_models"
SMPLX_DIR = BODY_MODELS / "smplx"
SMPLX_FILE = SMPLX_DIR / "SMPLX_NEUTRAL.npz"
WILOR_DIR = WEIGHTS / "wilor" / "pretrained_models"

# URLs for automated asset downloads
SMPLX_URL = "https://huggingface.co/camenduru/SMPLer-X/resolve/main/SMPLX_NEUTRAL.npz"
MP_TASK_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
MANO_RIGHT_URL = "https://huggingface.co/camenduru/HandRefiner/resolve/main/MANO_RIGHT.pkl"
WILOR_SPACE = "https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main"

CLOUDFLARED_LINUX = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
CLOUDFLARED_WIN = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"


def log(msg: str, tag: str = "kaggle"):
    print(f"[{tag} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def is_notebook() -> bool:
    """Check if code is running inside a Jupyter / Kaggle / Colab notebook."""
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            return True
    except Exception:
        pass
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ or "COLAB_GPU" in os.environ


def download_file(url: str, dest: Path, desc: str = ""):
    """Download a file with clean progress reporting."""
    if dest.exists() and dest.stat().st_size > 1024:
        log(f"Already have {dest.name} ({dest.stat().st_size / (1024*1024):.1f} MB)")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_dest = dest.with_suffix(dest.suffix + ".part")
    log(f"Downloading {desc or dest.name} from {url[:60]}... ")

    def reporthook(count, block_size, total_size):
        if total_size > 0 and count % max(1, int(1e7 // max(1, block_size))) == 0:
            pct = min(100.0, (count * block_size / total_size) * 100)
            sys.stdout.write(f"\r  -> {pct:.1f}% ({count * block_size / (1024*1024):.1f} / {total_size / (1024*1024):.1f} MB)")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, temp_dest, reporthook=reporthook)
        sys.stdout.write("\n")
        temp_dest.replace(dest)
        log(f"Saved {dest.name} ({dest.stat().st_size / (1024*1024):.1f} MB)")
    except Exception as e:
        if temp_dest.exists():
            temp_dest.unlink(missing_ok=True)
        log(f"Failed to download {dest.name}: {e}", tag="warn")


def ensure_models():
    """Ensure all required model checkpoints exist, downloading if missing."""
    log("Verifying model checkpoints...")

    # 1. SMPL-X v1.1 neutral body model (mandatory for GVHMR)
    if not SMPLX_FILE.exists():
        log("SMPLX_NEUTRAL.npz missing — downloading mirror from HuggingFace...")
        download_file(SMPLX_URL, SMPLX_FILE, "SMPLX Neutral Model")

    # 2. GVHMR demo checkpoints
    gvhmr_ckpt = WEIGHTS / "gvhmr.ckpt"
    if not gvhmr_ckpt.exists():
        log("Downloading GVHMR demo weights...")
        try:
            os.environ["GVHMR_CHECKPOINTS"] = str(WEIGHTS)
            os.environ["GVHMR_BODY_MODELS"] = str(BODY_MODELS)
            subprocess.run([sys.executable, "-m", "gvhmr.cli.download", "demo"], check=True)
        except Exception as e:
            log(f"gvhmr download command failed: {e}. You can download manually if needed.", tag="warn")

    # 3. MediaPipe Hand Landmarker task
    mp_task = WEIGHTS / "hand_landmarker.task"
    if not mp_task.exists():
        download_file(MP_TASK_URL, mp_task, "MediaPipe Hand Landmarker")

    # 4. WiLoR / MANO models (finger tracking)
    WILOR_DIR.mkdir(parents=True, exist_ok=True)
    mano_right = WILOR_DIR / "MANO_RIGHT.pkl"
    if not mano_right.exists():
        download_file(MANO_RIGHT_URL, mano_right, "MANO Right Hand Model")

    wilor_files = {
        "detector.pt": f"{WILOR_SPACE}/pretrained_models/detector.pt",
        "mano_mean_params.npz": f"{WILOR_SPACE}/mano_data/mano_mean_params.npz",
    }
    for fname, url in wilor_files.items():
        dst = WILOR_DIR / fname
        if not dst.exists():
            download_file(url, dst, f"WiLoR {fname}")


def get_cloudflared_binary() -> Path:
    """Find or download the cloudflared executable."""
    # Check if already in PATH
    which_cf = shutil.which("cloudflared")
    if which_cf:
        return Path(which_cf)

    # Local binary in root
    is_win = sys.platform == "win32"
    bin_name = "cloudflared.exe" if is_win else "cloudflared"
    local_bin = ROOT / bin_name

    if local_bin.exists():
        return local_bin

    url = CLOUDFLARED_WIN if is_win else CLOUDFLARED_LINUX
    log(f"Downloading cloudflared binary from {url}...")
    download_file(url, local_bin, "Cloudflare Tunnel binary")

    if not is_win:
        local_bin.chmod(0o755)

    return local_bin


def display_kaggle_banner(public_url: str, gpu_name: str, vram_gb: float):
    """Render a rich interactive HTML display widget in Kaggle / Jupyter notebooks."""
    if not is_notebook():
        return

    from IPython.display import HTML, display

    html_code = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #120e22 0%, #090710 100%);
                border: 1px solid rgba(168, 85, 247, 0.4);
                box-shadow: 0 10px 40px rgba(0, 0, 0, 0.7), inset 0 1px 0 rgba(255, 255, 255, 0.1);
                border-radius: 16px; padding: 24px 28px; max-width: 780px; margin: 16px 0; color: #f8fafc;">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px; margin-bottom: 18px;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <span style="font-size: 28px;">⚡</span>
                <div>
                    <h2 style="margin: 0; font-size: 22px; font-weight: 700; letter-spacing: -0.5px;
                               background: linear-gradient(90deg, #c084fc, #ec4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                        Prompt2Rig is Live on Cloudflare Tunnel
                    </h2>
                    <div style="font-size: 13px; color: #94a3b8; margin-top: 2px;">
                        3D Motion Lift & Mixamo Retargeting running on Kaggle GPU
                    </div>
                </div>
            </div>
            <div style="display: flex; gap: 8px;">
                <span style="background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3);
                             font-size: 11px; font-weight: 600; padding: 4px 10px; border-radius: 999px; display: inline-flex; align-items: center; gap: 5px;">
                    <span style="width: 6px; height: 6px; background: #10b981; border-radius: 50%; display: inline-block;"></span>
                    TUNNEL ACTIVE
                </span>
                <span style="background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.3);
                             font-size: 11px; font-weight: 600; padding: 4px 10px; border-radius: 999px;">
                    {gpu_name} ({vram_gb:.1f} GB)
                </span>
            </div>
        </div>

        <div style="background: rgba(255, 255, 255, 0.04); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 16px 20px; margin-bottom: 20px;">
            <div style="font-size: 12px; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;">
                Public Web Application URL
            </div>
            <div style="display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap;">
                <a href="{public_url}" target="_blank" rel="noopener noreferrer"
                   style="font-family: monospace; font-size: 16px; font-weight: 600; color: #38bdf8; text-decoration: none; word-break: break-all;">
                    {public_url}
                </a>
                <a href="{public_url}" target="_blank" rel="noopener noreferrer"
                   style="display: inline-flex; align-items: center; gap: 8px; padding: 10px 22px;
                          background: linear-gradient(135deg, #a855f7, #ec4899); color: #ffffff;
                          text-decoration: none; border-radius: 8px; font-weight: 600; font-size: 14px;
                          box-shadow: 0 4px 14px rgba(168, 85, 247, 0.4); transition: transform 0.2s;">
                    Open Web App ↗
                </a>
            </div>
        </div>

        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; font-size: 12.5px; color: #cbd5e1;">
            <div style="background: rgba(0, 0, 0, 0.25); padding: 10px 14px; border-radius: 8px; border-left: 3px solid #10b981;">
                <strong>🎬 Video Upload:</strong> Drag & drop your own dance / action clips into the web UI.
            </div>
            <div style="background: rgba(0, 0, 0, 0.25); padding: 10px 14px; border-radius: 8px; border-left: 3px solid #38bdf8;">
                <strong>🤖 AI Prompt:</strong> Generate clips using MiniMax H3 or FLUX 3 (configure FAL key in UI).
            </div>
            <div style="background: rgba(0, 0, 0, 0.25); padding: 10px 14px; border-radius: 8px; border-left: 3px solid #ec4899;">
                <strong>📦 Export GLB:</strong> Download 3D animated Mixamo rigs directly into Blender or Unity.
            </div>
        </div>
    </div>
    """
    display(HTML(html_code))


def main():
    parser = argparse.ArgumentParser(description="Prompt2Rig Kaggle & Cloudflare Tunnel Runner")
    parser.add_argument("--port", type=int, default=5177, help="Server port (default: 5177)")
    parser.add_argument("--fal-key", type=str, default="", help="FAL.ai API Key")
    parser.add_argument("--no-tunnel", action="store_true", help="Disable Cloudflare Tunnel")
    parser.add_argument("--weights-only", action="store_true", help="Download weights and exit")
    args = parser.parse_args()

    port = args.port
    if args.fal_key:
        os.environ["FAL_KEY"] = args.fal_key

    # Check GPU environment
    gpu_name = "CPU"
    vram_gb = 0.0
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            log(f"Detected GPU: {gpu_name} ({vram_gb:.1f} GB VRAM) — CUDA {torch.version.cuda}")
        else:
            log("No CUDA GPU detected; running in CPU fallback mode", tag="warn")
    except Exception as e:
        log(f"PyTorch check: {e}", tag="warn")

    # Step 1: Ensure models are downloaded
    ensure_models()
    if args.weights_only:
        log("Model checkpoints verified. Exiting (--weights-only).")
        return 0

    # Step 2: Start the unified FastAPI server
    server_cmd = [sys.executable, str(ROOT / "server.py")]
    server_env = os.environ.copy()
    server_env["PORT"] = str(port)

    log(f"Launching unified server on port {port}...")
    server_proc = subprocess.Popen(
        server_cmd,
        env=server_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def stream_server_logs():
        for line in iter(server_proc.stdout.readline, ""):
            if line:
                print(f"[server] {line.rstrip()}", flush=True)

    server_log_thread = threading.Thread(target=stream_server_logs, daemon=True)
    server_log_thread.start()

    # Wait briefly for server startup
    time.sleep(2.0)

    # Step 3: Start Cloudflare Tunnel
    tunnel_proc = None
    public_url = ""

    if not args.no_tunnel:
        cf_bin = get_cloudflared_binary()
        log("Starting Cloudflare Tunnel...")

        cf_cmd = [
            str(cf_bin),
            "tunnel",
            "--url", f"http://127.0.0.1:{port}",
            "--no-autoupdate",
        ]

        tunnel_proc = subprocess.Popen(
            cf_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        tunnel_url_found = threading.Event()

        def stream_tunnel_logs():
            nonlocal public_url
            pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
            # Cloudflared prints quick tunnel output to stderr
            for line in iter(tunnel_proc.stderr.readline, ""):
                m = pattern.search(line)
                if m and not public_url:
                    public_url = m.group(0)
                    os.environ["TUNNEL_URL"] = public_url
                    try:
                        (ROOT / ".tunnel_url").write_text(public_url, encoding="utf-8")
                    except Exception:
                        pass
                    tunnel_url_found.set()
                # Suppress noisy heartbeat logs, keep connection info
                if "Registered tunnel connection" in line or "error" in line.lower():
                    log(f"[cloudflared] {line.strip()}", tag="tunnel")

        tunnel_thread = threading.Thread(target=stream_tunnel_logs, daemon=True)
        tunnel_thread.start()

        log("Waiting for Cloudflare Tunnel URL...")
        tunnel_url_found.wait(timeout=25)

        if public_url:
            log("=" * 68)
            log(f"🎉 CLOUDFLARE TUNNEL LIVE: {public_url}")
            log("=" * 68)
            display_kaggle_banner(public_url, gpu_name, vram_gb)
        else:
            log("Cloudflare tunnel did not return a URL within 25s. You can still use http://127.0.0.1:{port}", tag="warn")

    # Step 4: Keep-alive loop and graceful shutdown
    def cleanup(signum=None, frame=None):
        log("Shutting down Prompt2Rig services...")
        if tunnel_proc and tunnel_proc.poll() is None:
            tunnel_proc.terminate()
        if server_proc and server_proc.poll() is None:
            server_proc.terminate()
        (ROOT / ".tunnel_url").unlink(missing_ok=True)
        log("Services stopped successfully.")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        while True:
            if server_proc.poll() is not None:
                log(f"Server process terminated with code {server_proc.returncode}", tag="err")
                break
            if tunnel_proc and tunnel_proc.poll() is not None:
                log(f"Tunnel process terminated with code {tunnel_proc.returncode}", tag="warn")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    sys.exit(main())
