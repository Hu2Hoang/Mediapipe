import cv2
import mediapipe as mp
from mediapipe.modules import hand_landmark
from pynput.keyboard import Key, Controller

mp_drawing_util = mp.solutions.drawing_utils
mp_drawing_style = mp.solutions.drawing_styles

kb = Controller()

mp_hand = mp.solutions.hands
hands = mp_hand.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(1)
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
while cap.isOpened():
    success, img = cap.read()
    if not success:
        break
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks:
        for idx, hand in enumerate(result.multi_hand_landmarks):
            if hand.landmark[8].y < hand.landmark[7].y and hand.landmark[12].y < hand.landmark[11].y and \
                    hand.landmark[16].y < hand.landmark[15].y:
                kb.press(Key.space)
            else:
                kb.release(Key.space)
            mp_drawing_util.draw_landmarks(
                img,
                hand,
                mp_hand.HAND_CONNECTIONS,
                mp_drawing_style.get_default_hand_landmarks_style(),
                mp_drawing_style.get_default_hand_connections_style()
            )

            mp_drawing_util.draw_landmarks(img, hand, mp_hand.HAND_CONNECTIONS)

            lbl = result.multi_handedness[idx].classification[0].label
            if lbl == "Left":
                for id, lm in enumerate(hand.landmark):
                    h, w, _ = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if id == 8:
                        cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

    cv2.imshow("Game", img)
    key = cv2.waitKey(1)
    if key == 27:  # esc
        break

cap.release()
