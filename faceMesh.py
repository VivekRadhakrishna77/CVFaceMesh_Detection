import cv2 as cv 
import mediapipe as mp
import time


cap = cv.VideoCapture(0)
mpDraw = mp.solutions.drawing_utils
fm = mp.solutions.face_mesh
faceMesh = fm.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness = 2, circle_radius=2)

while True:
    ret, img = cap.read()

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLMs in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLMs, fm.FACE_CONNECTIONS, drawSpec, drawSpec)
        #for id, lm in enumerate(results.mult_face_landmarks):

    cv.imshow("Image", img)
    cv.waitKey(1)