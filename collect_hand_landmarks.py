import cv2
import mediapipe as mp
import csv

# Initialize file and gesture recording
output_file = "gesture_data.csv"
current_label = "none"

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error opening camera")
    exit()

# Open CSV file for appending data
with open(output_file, mode="a", newline='') as f:
    writer = csv.writer(f)

    while True:
        # Get frame
        ret, frame = cap.read()
        if not ret:
            print("Error capturing frame")
            break
        
        # Process frame as hand
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        # Go through all hands (we set max to 1)
        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                # Get landmark coordinates
                coordinates = []
                for lm in hand.landmark:
                    coordinates.append(lm.x)
                    coordinates.append(lm.y)

                # Save to CSV
                if current_label != "none":
                    writer.writerow(coordinates + [current_label])
                    current_label = "none"      # makes it so it's like taking a picture :)

        # Controls
        controls = ["none", "thumbs_up", "thumbs_down", "okay", "peace"]
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif ord('0') <= key <= ord('4'):
            current_label = controls[key - ord('0')]
        
        # Display controls
        for i, label in enumerate(controls):
            cv2.putText(
                frame,
                f"{i}: {label}",                # text
                (10, 50 + i*50),                # position
                cv2.FONT_HERSHEY_SIMPLEX,
                2,                              # scale
                (255, 255, 255),                # color
                2                               # thickness
            )
        
        # Show frame
        cv2.imshow("Webcam", frame)

cap.release()
cv2.destroyAllWindows()