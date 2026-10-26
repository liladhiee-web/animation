"""Prompt2Rig Unified Server — High-performance FastAPI server.
Serves the web application (./public), handles video uploads (./takes/uploads),
proxies fal.ai requests, and runs the GVHMR + WiLoR GPU motion pipeline in-process.

Works seamlessly on Kaggle, Google Colab, Linux servers, and local machines.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
PUBLIC = ROOT / "public"
TAKES = ROOT / "takes"
UPLOADS = TAKES / "uploads"
HQ_DIR = ROOT / "hq"

# Ensure directories exist
PUBLIC.mkdir(exist_ok=True)
TAKES.mkdir(exist_ok=True)
UPLOADS.mkdir(parents=True, exist_ok=True)

# Add hq to python path so hands and worker modules can be imported
if str(HQ_DIR) not in sys.path:
    sys.path.insert(0, str(HQ_DIR))

# --- Load .env if present ---
env_file = ROOT / ".env"
if env_file.exists():
    try:
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception as e:
        print(f"[prompt2rig] Warning reading .env: {e}")

PORT = int(os.environ.get("PORT", 5177))
TUNNEL_URL = os.environ.get("TUNNEL_URL", "")

app = FastAPI(title="Prompt2Rig", version="1.0.0")

# Enable CORS for Cloudflare Tunnel and local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Video Models Configuration ---
VIDEO_MODELS = {
    "h3": {
        "t2v": "minimax/h3-max-turbo/text-to-video",
        "i2v": "minimax/h3-max-turbo/image-to-video",
        "input": lambda b, i2v: {
            "duration": min(15, max(1, int(b.get("duration", 5)))),
            "resolution": b.get("resolution") if b.get("resolution") in ("480P", "768P", "1080P") else "480P",
            "prompt_expansion_mode": "disabled",
            **({} if i2v else {"aspect_ratio": b.get("aspect_ratio", "16:9")}),
        },
    },
    "flux3": {
        "t2v": "blackforestlabs/flux-3-max/text-to-video",
        "i2v": "blackforestlabs/flux-3-max/image-to-video",
        "input": lambda b, i2v: {
            "duration": min(20, max(5, round(float(b.get("duration", 5))))),
            "resolution": "1080p" if b.get("resolution") == "1080P" else "720p",
            "generate_audio": False,
            "reasoning_level": "high" if b.get("reasoning_level") == "high" else "low",
            "aspect_ratio": "auto" if i2v else b.get("aspect_ratio", "16:9"),
        },
    },
}
IMAGE_MODEL = "fal-ai/flux/schnell"


def get_fal_key(request: Request, body_key: Optional[str] = None) -> Optional[str]:
    """Retrieve FAL key from request header, body, or environment."""
    return (
        request.headers.get("x-fal-key")
        or request.headers.get("authorization", "").replace("Key ", "").replace("Bearer ", "").strip()
        or body_key
        or os.environ.get("FAL_KEY")
    )


def fal_run(model: str, payload: dict, fal_key: str) -> dict:
    t0 = time.perf_counter()
    url = f"https://fal.run/{model}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Key {fal_key}",
            "Content-Type": "application/json",
            "User-Agent": "Prompt2Rig-Unified/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            text = resp.read().decode("utf-8")
            try:
                res_json = json.loads(text)
            except Exception:
                res_json = {"raw": text}
            res_json["_elapsed"] = time.perf_counter() - t0
            return res_json
    except urllib.error.HTTPError as e:
        err_text = e.read().decode("utf-8", errors="replace")
        try:
            err_json = json.loads(err_text)
            detail = err_json.get("detail", err_text[:500])
        except Exception:
            detail = err_text[:500]
        raise HTTPException(status_code=e.code, detail=f"fal {model} -> {e.code}: {detail}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"fal network error: {e}")


# --- Lazy Pipeline Loader for GPU Worker ---
worker_module = None
worker_init_attempted = False
worker_error = None


def get_worker():
    global worker_module, worker_init_attempted, worker_error
    if worker_module is not None:
        return worker_module
    if not worker_init_attempted:
        worker_init_attempted = True
        try:
            import worker as wm
            worker_module = wm
            # Launch GPU pipeline loading in background thread pool if not yet loaded
            if hasattr(wm, "GPU") and hasattr(wm, "load_pipeline"):
                wm.GPU.submit(wm.load_pipeline)
            return worker_module
        except Exception as e:
            worker_error = f"GVHMR worker failed to import: {e}"
            print(f"[prompt2rig] {worker_error}")
    return None


# --- API Routes ---

@app.get("/api/system")
def get_system_info():
    """Returns telemetry on GPU, memory, Cloudflare tunnel, and server status."""
    gpu_info = "None (CPU Mode)"
    vram_used_gb = 0.0
    vram_total_gb = 0.0
    cuda_available = False

    try:
        import torch
        if torch.cuda.is_available():
            cuda_available = True
            gpu_info = torch.cuda.get_device_name(0)
            vram_used_gb = round(torch.cuda.memory_allocated(0) / (1024 ** 3), 2)
            vram_total_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
    except Exception:
        pass

    tunnel_url = os.environ.get("TUNNEL_URL", TUNNEL_URL)
    if not tunnel_url and (ROOT / ".tunnel_url").exists():
        try:
            tunnel_url = (ROOT / ".tunnel_url").read_text().strip()
        except Exception:
            pass

    wm = get_worker()
    worker_ok = bool(wm and getattr(wm, "state", {}).get("pipe") is not None)
    err = worker_error or (wm and getattr(wm, "state", {}).get("error"))

    return {
        "status": "online",
        "gpu": gpu_info,
        "cuda": cuda_available,
        "vram_used_gb": vram_used_gb,
        "vram_total_gb": vram_total_gb,
        "worker_ready": worker_ok,
        "worker_error": err,
        "tunnel_url": tunnel_url,
        "has_fal_key": bool(os.environ.get("FAL_KEY")),
        "platform": sys.platform,
        "python_version": sys.version.split()[0],
    }


@app.get("/api/hq/health")
def hq_health():
    """Returns the health status of the GVHMR motion recovery engine."""
    wm = get_worker()
    if wm is None:
        return {"ok": False, "device": None, "error": worker_error or "Worker not initialized"}
    state = getattr(wm, "state", {})
    return {
        "ok": state.get("pipe") is not None,
        "device": state.get("device"),
        "error": state.get("error") or worker_error,
    }


HQ_JOBS: dict[str, dict] = {}


class HQProcessRequest(BaseModel):
    video_url: str
    hands: Optional[str] = None
    poll: Optional[bool] = True


@app.get("/api/hq/job/{key}")
def get_hq_job(key: str):
    """Retrieve the progress or result of an async HQ motion job."""
    if key in HQ_JOBS:
        job = HQ_JOBS[key]
        if job["status"] == "running":
            return {
                "status": "processing",
                "elapsed": round(time.perf_counter() - job["start_time"], 1),
            }
        elif job["status"] == "done":
            return {
                "status": "done",
                "result": job["result"],
                "elapsed": round(job.get("elapsed", 0), 1),
            }
        elif job["status"] == "error":
            return {
                "status": "error",
                "error": job["error"],
            }
    # Check if results already saved in cache on disk
    wm = get_worker()
    if wm:
        cache_dir = wm.CACHE / key
        if (cache_dir / "input.mp4").exists() and (cache_dir / "hmr4d_results.pt").exists():
            return {"status": "processing", "message": "finishing on disk"}
    return {"status": "not_found"}


@app.post("/api/hq")
def hq_process(body: HQProcessRequest):
    """Processes video motion with GVHMR + WiLoR.

    Supports asynchronous background polling to avoid Cloudflare 100s proxy timeouts.
    """
    wm = get_worker()
    if wm is None:
        raise HTTPException(503, f"HQ Worker unavailable: {worker_error}")

    state = getattr(wm, "state", {})
    if state.get("pipe") is None:
        wm.GPU.submit(wm.load_pipeline).result()
        if state.get("pipe") is None:
            raise HTTPException(503, state.get("error") or "GVHMR not loaded on GPU")

    key = hashlib.sha1(body.video_url.encode()).hexdigest()[:12]

    # Return cached result immediately if job already completed
    if key in HQ_JOBS and HQ_JOBS[key]["status"] == "done":
        return HQ_JOBS[key]["result"]

    # If already running, return 202 with processing status
    if key in HQ_JOBS and HQ_JOBS[key]["status"] == "running":
        return JSONResponse(
            status_code=202,
            content={
                "status": "processing",
                "key": key,
                "elapsed": round(time.perf_counter() - HQ_JOBS[key]["start_time"], 1),
            },
        )

    # Intercept local /takes/ path
    local_path = None
    v_url = body.video_url
    if "/takes/" in v_url or v_url.startswith("takes/"):
        rel = v_url.split("takes/", 1)[1].lstrip("/")
        candidate = TAKES / rel
        if candidate.exists():
            local_path = candidate

    if local_path and local_path.exists():
        out_dir = wm.CACHE / key
        out_dir.mkdir(parents=True, exist_ok=True)
        cached_source = out_dir / "source.mp4"
        if not cached_source.exists():
            shutil.copyfile(local_path, cached_source)

    process_in = wm.ProcessIn(video_url=body.video_url, hands=body.hands)

    # If synchronous mode requested explicitly
    if body.poll is False:
        try:
            return wm.process(process_in)
        except Exception as e:
            raise HTTPException(500, f"GVHMR error: {e}")

    # Launch in background thread for async polling
    t0 = time.perf_counter()
    HQ_JOBS[key] = {
        "status": "running",
        "start_time": t0,
        "result": None,
        "error": None,
    }

    def execute_job():
        try:
            print(f"[hq] Starting GPU motion extraction for {key}...")
            res = wm.process(process_in)
            HQ_JOBS[key] = {
                "status": "done",
                "result": res,
                "elapsed": time.perf_counter() - t0,
                "start_time": t0,
            }
            print(f"[hq] Job {key} finished successfully in {time.perf_counter() - t0:.1f}s")
        except Exception as e:
            import traceback
            traceback.print_exc()
            HQ_JOBS[key] = {
                "status": "error",
                "error": str(e),
                "start_time": t0,
            }

    threading.Thread(target=execute_job, daemon=True).start()

    return JSONResponse(
        status_code=202,
        content={
            "status": "processing",
            "key": key,
            "message": "Motion processing in progress on GPU",
        },
    )


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Uploads a local user video file for motion retargeting."""
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    safe_name = "".join(c for c in file.filename if c.isalnum() or c in "._- ")
    safe_name = safe_name.replace(" ", "_")
    name_stem = Path(safe_name).stem[:40]
    ext = Path(safe_name).suffix.lower() or ".mp4"
    unique_name = f"{name_stem}_{int(time.time())}_{hashlib.md5(safe_name.encode()).hexdigest()[:6]}{ext}"

    dest = UPLOADS / unique_name
    try:
        with dest.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_size = dest.stat().st_size
        print(f"[upload] Saved video {dest.name} ({file_size / (1024*1024):.2f} MB)")
        return {
            "ok": True,
            "video_url": f"/takes/uploads/{dest.name}",
            "filename": dest.name,
            "size": file_size,
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to save uploaded video: {e}")


@app.post("/api/video")
async def generate_video(request: Request):
    """Proxies video generation to fal.ai."""
    try:
        b = await request.json()
    except Exception:
        b = {}

    fal_key = get_fal_key(request, b.get("fal_key"))
    if not fal_key:
        raise HTTPException(
            401,
            "FAL_KEY is not set. Please enter your Fal API key in Settings or add it to .env / environment."
        )

    model_key = b.get("model") if b.get("model") in VIDEO_MODELS else "h3"
    mode = "i2v" if b.get("image_url") else "t2v"
    model_conf = VIDEO_MODELS[model_key]

    prompt = str(b.get("prompt") or "").strip()[:4000]
    input_data = {
        "prompt": prompt,
        **model_conf["input"](b, mode == "i2v"),
    }
    if b.get("seed") is not None and b.get("seed") != "":
        try:
            input_data["seed"] = int(b["seed"])
        except ValueError:
            pass
    if mode == "i2v" and b.get("image_url"):
        input_data["image_url"] = b["image_url"]

    fal_target = model_conf[mode]
    out = fal_run(fal_target, input_data, fal_key)
    print(f"[video] {model_key} {mode} completed in {out.get('_elapsed', 0):.1f}s")

    return {
        "model": model_key,
        "mode": mode,
        "video_url": out.get("video", {}).get("url"),
        "timings": out.get("timings"),
        "elapsed": out.get("_elapsed"),
    }


@app.post("/api/image")
async def generate_image(request: Request):
    """Proxies performer reference image generation to fal.ai."""
    try:
        b = await request.json()
    except Exception:
        b = {}

    fal_key = get_fal_key(request, b.get("fal_key"))
    if not fal_key:
        raise HTTPException(
            401,
            "FAL_KEY is not set. Please enter your Fal API key in Settings or add it to .env / environment."
        )

    prompt = str(b.get("prompt") or "").strip()[:4000]
    input_data = {
        "prompt": prompt,
        "image_size": b.get("image_size", "landscape_16_9"),
        "num_inference_steps": 4,
        "enable_safety_checker": True,
    }

    out = fal_run(IMAGE_MODEL, input_data, fal_key)
    images = out.get("images", [])
    first_url = images[0].get("url") if images else None
    print(f"[image] Performer generated in {out.get('_elapsed', 0):.1f}s")
    return {"image_url": first_url, "elapsed": out.get("_elapsed")}


@app.post("/api/takes")
async def save_take(request: Request):
    """Saves an exported take (GLB binary + metadata JSON) into ./takes."""
    name_param = request.query_params.get("name") or "take"
    safe_name = "".join(c for c in name_param if c.isalnum() or c in "_-")[:80] or "take"
    source = request.query_params.get("source")
    prompt = request.query_params.get("prompt") or ""

    body = await request.body()
    if len(body) > 100 * 1024 * 1024:
        raise HTTPException(413, "Take payload too large (>100MB)")

    glb_file = TAKES / f"{safe_name}.glb"
    json_file = TAKES / f"{safe_name}.json"

    glb_file.write_bytes(body)
    meta = {
        "name": safe_name,
        "source": source,
        "prompt": prompt,
        "saved": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "size_bytes": len(body),
    }
    json_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[take] Saved takes/{safe_name}.glb ({len(body) / (1024*1024):.2f} MB)")
    return {"path": f"takes/{safe_name}.glb"}


# --- Mount Static Files ---
app.mount("/takes", StaticFiles(directory=str(TAKES)), name="takes")
app.mount("/", StaticFiles(directory=str(PUBLIC), html=True), name="public")


if __name__ == "__main__":
    import uvicorn
    # Start GPU pipeline in background immediately
    get_worker()
    print(f"[prompt2rig] Unified server starting on http://0.0.0.0:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
