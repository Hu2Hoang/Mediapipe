import cv2
import mediapipe as mp
from mediapipe.modules import hand_landmark
from pynput.keyboard import Key, Controller

x_previous = 320
mark_x = 320
y_previous = 240
mark_y = 240
dead_zone_left_right = 40
dead_zone_up_down = 80
chatter_protection = 20

mp_drawing_util = mp.solutions.drawing_utils
mp_drawing_style = mp.solutions.drawing_styles

keyboard = Controller()

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
        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            x, y = hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y
            x = int(x * 640)
            y = int(y * 480)

            if y > (y_previous + dead_zone_up_down):
                keyboard.press(Key.down)
                # keyboard.release(Key.down)
                continue

            if y < (y_previous - dead_zone_up_down) :
                keyboard.press(Key.up)
                # keyboard.release(Key.up)
                continue

            if x > (x_previous + dead_zone_left_right) :
                keyboard.press(Key.left)
                # keyboard.release(Key.left)
                continue

            if x < (x_previous - dead_zone_left_right):
                keyboard.press(Key.right)
                # keyboard.release(Key.right)
                continue

            x_9, y_9 = hand_landmarks.landmark[9].x, hand_landmarks.landmark[9].y
            y_9 = int(y_9 * 480)
            x_9 = int(x_9 * 640)

            _, y_12 = hand_landmarks.landmark[12].x, hand_landmarks.landmark[12].y
            y_12 = int(y_12 * 480)

            mp_drawing_util.draw_landmarks(
                img,
                hand_landmarks,
                mp_hand.HAND_CONNECTIONS,
                mp_drawing_style.get_default_hand_landmarks_style(),
                mp_drawing_style.get_default_hand_connections_style()
            )

            mp_drawing_util.draw_landmarks(img, hand_landmarks, mp_hand.HAND_CONNECTIONS)

            lbl = result.multi_handedness[idx].classification[0].label
            if lbl == "Left":
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, _ = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if id == 8:
                        cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

    cv2.imshow("Nhan dang ban tay", img)
    key = cv2.waitKey(1)
    if key == 27:  # esc
        break

cap.release()
