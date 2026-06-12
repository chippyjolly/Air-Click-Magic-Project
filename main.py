import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

stars = []
pinch_active = False

mp_draw = mp.solutions.drawing_utils

while True:
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        h, w, _ = frame.shape

        for hand in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            thumb = hand.landmark[4]
            index = hand.landmark[8]

            tx = int(thumb.x * w)
            ty = int(thumb.y * h)

            ix = int(index.x * w)
            iy = int(index.y * h)

            cv2.circle(frame, (tx, ty), 10, (0, 255, 0), -1)
            cv2.circle(frame, (ix, iy), 10, (0, 255, 0), -1)

            distance = ((tx - ix) ** 2 + (ty - iy) ** 2) ** 0.5

            cv2.putText(
                frame,
                f"Distance: {int(distance)}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

            if distance < 40:
                if not pinch_active:
                    stars.append((ix, iy))
                    pinch_active = True
            else:
                pinch_active = False

    for x, y in stars:
        cv2.drawMarker(
            frame,
            (x, y),
            (0, 255, 255),
            markerType=cv2.MARKER_STAR,
            markerSize=20,
            thickness=2
        )

    cv2.imshow("Magic Touch", frame)
    

    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()