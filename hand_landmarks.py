import cv2
import mediapipe as mp

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

    # Draw landmarks
    if result.multi_hand_landmarks:
        for landmark in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, landmark, mp_hands.HAND_CONNECTIONS)

    # Show frame with landmarks
    cv2.imshow("Webcam", frame)

    # Exit on q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()