from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
import io

app = FastAPI()

# Load model YOLO
model = YOLO("weights/your_model.pt")
confidence_threshold = 0.5

@app.get("/")
def root():
    return {"message": "YOLO Mask Detection API ready"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    results = model.predict(img, conf=confidence_threshold, verbose=False)
    result = results[0]

    counter = {
        'with_mask': 0,
        'without_mask': 0,
        'mask_weared_incorrect': 0
    }

    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            cls = int(box.cls[0])
            class_name = model.names[cls]
            if class_name in counter:
                counter[class_name] += 1

    return JSONResponse(content=counter)
