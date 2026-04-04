import cv2
import numpy as np
import csv

# -------- CONFIG --------
CSV_FILE = "gesture_data.csv"
IMG_SIZE = 500
# ------------------------

# Hand connections (same as MediaPipe)
connections = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]

# Load CSV data
data = []
with open(CSV_FILE, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        values = list(map(float, row[:-1]))  # all except label
        label = row[-1]
        data.append((values, label))

index = 0

while True:
    # Get current sample
    values, label = data[index]

    # Convert to (x, y) points
    points = [(values[i], values[i+1]) for i in range(0, len(values), 2)]

    # Create blank image
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

    # Draw connections
    for start, end in connections:
        x1, y1 = points[start]
        x2, y2 = points[end]

        p1 = (int(x1 * IMG_SIZE), int(y1 * IMG_SIZE))
        p2 = (int(x2 * IMG_SIZE), int(y2 * IMG_SIZE))

        cv2.line(img, p1, p2, (255, 0, 0), 2)

    # Draw points
    for (x, y) in points:
        px = int(x * IMG_SIZE)
        py = int(y * IMG_SIZE)
        cv2.circle(img, (px, py), 5, (0, 255, 0), -1)

    # Show label + index
    cv2.putText(
        img,
        f"{label} ({index+1}/{len(data)})",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow("Dataset Viewer", img)

    key = cv2.waitKey(0) & 0xFF

    # Controls
    if key == ord('q'):
        break
    elif key == ord('d'):  # next
        index = (index + 1) % len(data)
    elif key == ord('a'):  # previous
        index = (index - 1) % len(data)

cv2.destroyAllWindows()