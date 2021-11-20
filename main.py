import argparse
import pickle

import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils


REQUIRED_HAND_COUNT = 2


class ModelClass:
    OK = 0
    NOT_OK = 1


with open('model.pickle', 'rb') as model_file:
    model = pickle.load(model_file)


def is_marked(multi_hand_landmarks):
    for hand_landmarks in multi_hand_landmarks:
        hand_landmarks = [(landmark.x, landmark.y, landmark.z)
                          for landmark in hand_landmarks.landmark]
        flattened_landmarks = np.array(hand_landmarks).flatten()
        if model.predict([flattened_landmarks]) == ModelClass.NOT_OK:
            return False
    return True


def start(video_path):
    cap = cv2.VideoCapture(video_path or 0)

    if not cap.isOpened():
        print('could not open video')
        return

    while True:
        _, frame = cap.read()

        if frame is None:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(frame_rgb)
        if result.multi_hand_landmarks and len(result.multi_hand_landmarks) == REQUIRED_HAND_COUNT:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.putText(frame, str(is_marked(result.multi_hand_landmarks)),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

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
