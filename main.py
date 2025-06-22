import cv2
from ultralytics import YOLO

model = YOLO("weights/model_best_mask.pt")
confidence_threshold = 0.3

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=confidence_threshold, verbose=False)
    result = results[0]

    counter = {
        'proper_mask': 0,
        'no_mask': 0,
        'improper_mask': 0
    }

    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = model.names[cls]
            label = f"{class_name} {conf:.2f}"

            # Counter
            if class_name in counter:
                counter[class_name] += 1

            # Bounding box
            if class_name == 'proper_mask':
                color = (0, 255, 0)
            elif class_name == 'no_mask':
                color = (0, 0, 255)
            elif class_name == 'improper_mask':
                color = (0, 255, 255)
            else:
                color = (255, 255, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Tampilkan counter di frame
    y_offset = 30
    for key, value in counter.items():
        text = f"{key}: {value}"
        cv2.putText(frame, text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_offset += 30

    cv2.imshow("Realtime Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
