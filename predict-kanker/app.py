from ultralytics import YOLO
import cv2
import numpy as np

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn


app = FastAPI()

# Tambahkan middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ganti dengan ["http://localhost"] jika ingin lebih aman
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = YOLO("best.pt")

# Endpoint untuk inference
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    results = model.predict(img)
    # Contoh: ambil bounding box
    detections = []
    for result in results:
        for box in result.boxes.xyxy:
            detections.append(box.tolist())
    
    return JSONResponse(content={"detections": detections})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
