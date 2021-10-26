import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

model = load_model('mp_hand_gesture')
classNames = ['okay', 'peace', 'thumbs up', 'thumbs down', 'call me', 'stop', 'rock', 'live long', 'fist', 'smile']


def main():
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()

        x, y, _ = frame.shape
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(framergb)
        className = ''

        if result.multi_hand_landmarks:
            landmarks = []
            for hand_landmarks in result.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    landmarks.append([landmark.x * x, landmark.y * y])

                mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
                prediction = model.predict([landmarks])
                className = classNames[np.argmax(prediction)]

        cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Output", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
