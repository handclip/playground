import argparse

import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

model = load_model('mp_hand_gesture')
classNames = ['okay', 'peace', 'thumbs up', 'thumbs down', 'call me', 'stop', 'rock', 'live long', 'fist', 'smile']


def start(video_path, show_capture):
    cap = cv2.VideoCapture(video_path or 0)
    while True:
        _, frame = cap.read()

        if frame is None:
            break

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

                prediction = model.predict([landmarks])
                className = classNames[np.argmax(prediction)]

                if show_capture:
                    mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
                else:
                    print(f'Found {className} @ {int(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)}s')

        if show_capture:
            cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("Output", frame)
            if cv2.waitKey(1) == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', help='Path to a video.')
    parser.add_argument('--show-capture', default=True, action=argparse.BooleanOptionalAction,
                        help='Whether to display a window of the capture.')
    args = parser.parse_args()

    start(args.video_path, args.show_capture)


if __name__ == '__main__':
    main()
