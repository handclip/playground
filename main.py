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


def start(video_path):
    cap = cv2.VideoCapture(video_path or 0)
    while True:
        _, frame = cap.read()

        if frame is None:
            break

        frame_width, frame_height, _ = frame.shape
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(framergb)
        className = ''

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                print(hand_landmarks)
                landmarks = [[landmark.x * frame_width, landmark.y * frame_height]
                             for landmark in hand_landmarks.landmark]
                prediction = model.predict([landmarks])
                className = classNames[np.argmax(prediction)]
                mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)

        cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Output", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', help='Path to a video.')
    args = parser.parse_args()

    start(args.video_path)


if __name__ == '__main__':
    main()
