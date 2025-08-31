from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
import cv2

app = FastAPI(title="People Counter API")

# CORS (allow local HTML to call API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load a YOLO model pretrained on COCO (contains 'person' class)
# If you have v8 or v11: choose the appropriate variant available in your env
# ultralytics will download weights on first run
MODEL_NAME = "yolov8n.pt"  # small and fast; alternatives: yolov8s.pt, yolov11n.pt
model = YOLO(MODEL_NAME)

@app.post("/count")
async def count_people(file: UploadFile = File(...)):
    """Return number of persons and bounding boxes for each detection."""
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})

    results = model.predict(img, conf=0.3)

    person_class_ids = {0}  # in COCO, class 0 is 'person'
    total = 0
    boxes = []  # [x1,y1,x2,y2,conf]

    for r in results:
        names = r.names  # id -> name
        for b, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            if int(c) in person_class_ids or names.get(int(c), "") == "person":
                total += 1
                boxes.append([float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(s)])

    return {"count": total, "boxes": boxes}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)
