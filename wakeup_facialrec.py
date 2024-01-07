# original drowsy detector code from https://www.youtube.com/watch?v=OCJSJ-anywc&t=788s
# minor changes made by me

import cv2
import dlib
from scipy.spatial import distance
from datetime import datetime

import contextlib
with contextlib.redirect_stdout(None):
    import pygame


def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A + B) / (2.0 * C)
    return ear_aspect_ratio


cap = cv2.VideoCapture(0)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
eyes_open = True
closed_eyes_time = None  # why not 0
pygame.mixer.init()
alert_sound = pygame.mixer.Sound('VineBoomSoundEffect.wav')
audio_delay = 0 # sound plays every 5 loops
cv2.namedWindow("Wake up")
cv2.setWindowProperty("Wake up", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Wake up", cv2.WND_PROP_TOPMOST, 1)  # Set window to be on top

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)
    for face in faces:
        face_landmarks = dlib_facelandmark(gray, face)
        leftEye = []
        rightEye = []

        for n in range(36, 42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x, y))
            next_point = n + 1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        for n in range(42, 48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x, y))
            next_point = n + 1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        left_ear = calculate_EAR(leftEye)
        right_ear = calculate_EAR(rightEye)

        EAR = (left_ear + right_ear) / 2
        EAR = round(EAR, 2)

        if EAR < 0.20:
            if eyes_open:
                closed_eyes_time = datetime.now()
                eyes_open = False
            else:
                time_diff = datetime.now() - closed_eyes_time
                if time_diff.total_seconds() >= 3:
                    cv2.putText(frame, "WAKE UP", (20, 100),
                                cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 4)
                    cv2.putText(frame, "YOU'RE GOING TO DIE", (20, 400),
                                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 8)
                    if audio_delay % 5 == 0:
                        alert_sound.play() # how to make it so it only plays every like 0.5 seconds not each run of the loop
                    audio_delay += 1
        else:  # if eyes reopen
            eyes_open = True
            time_diff = 0
        #print(EAR)

    cv2.imshow("Wake up", frame)

    key = cv2.waitKey(1)
    if key == 27: # presses esc
        break
    if cv2.getWindowProperty("Wake up", cv2.WND_PROP_VISIBLE) < 1: #clicks x
        break


cap.release()
cv2.destroyAllWindows()
