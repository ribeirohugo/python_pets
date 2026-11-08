"""FastAPI server: serves the web UI and a /api/predict endpoint.

Run:
    python -m python_pets.app            # then open http://127.0.0.1:8010
"""
from __future__ import annotations

import io

import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

from python_pets.config import CLASSES, STATIC_DIR, server_port
from python_pets.inference import detect, summarize
from python_pets.model import load_detector

model, device, mode = load_detector()
print(f"Loaded detector in '{mode}' mode on {device}")

app = FastAPI(title="Cat or Dog?")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/info")
def info():
    return {"mode": mode, "device": device, "classes": CLASSES[1:]}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), threshold: float = Query(0.5, ge=0.0, le=1.0)):
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("RGB")  # respect phone rotation
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "That file doesn't look like an image.")

    detections = detect(model, device, mode, img, threshold)
    verdict, counts = summarize(detections)
    return {"verdict": verdict, "counts": counts, "detections": detections,
            "width": img.width, "height": img.height, "mode": mode}


def main():
    uvicorn.run(app, host="127.0.0.1", port=server_port())


if __name__ == "__main__":
    main()
