import cv2
from ultralytics import YOLO

# Load pretrained YOLO model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run object detection
    results = model(frame, verbose=False)

    # Draw detections on frame
    annotated_frame = results[0].plot()

    # Show frame
    cv2.imshow("Object Detection", annotated_frame)

    # Exit on q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()