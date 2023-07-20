import numpy as np
import cv2
# cap = cv2.VideoCapture(0)
# while True:
#     ret, frame = cap.read()
#     cv2.imshow('frame',frame)
#     if cv2.waitKey(1) == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()

def draw_boundry(img,classifier,scaleFactor,minNeighbors,color,text):
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ftrs = classifier.detectMultiScale(gray_img,scaleFactor,minNeighbors)
    crdnts = []
    for(x,y,w,h) in ftrs:
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        cv2.putText(img,text,(x,y-3),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1,cv2.LINE_AA)
        crdnts = [x,y,w,h]

    return crdnts,img

def detect(img,face_cascade,eye_cascade,nose_cascade,mouth_cascade):
    color = {"blue":(255,0,0),"red":(0,0,255),"green":(0,255,0)}
    crdnts,img = draw_boundry(img,face_cascade,1.1,10,color['blue'],"Face")

    if len(crdnts)==4:
        roi_img = img[crdnts[1]:crdnts[1]+crdnts[3], crdnts[0]:crdnts[0]+crdnts[2]]
        crdnts = draw_boundry(roi_img,eye_cascade,1.1,20,color['red'],"Eye")
        crdnts = draw_boundry(roi_img,nose_cascade,1.1,25,color['green'],"Nose")
        crdnts = draw_boundry(roi_img,mouth_cascade,1.1,24,color['blue'],"Mouth")
    return img
    

face_cascade = cv2.CascadeClassifier("raw.githubusercontent.com_opencv_opencv_master_data_haarcascades_haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("raw.githubusercontent.com_opencv_opencv_master_data_haarcascades_haarcascade_eye.xml")
nose_cascade = cv2.CascadeClassifier("raw.githubusercontent.com_the-javapocalypse_Face-Detection-Recognition-Using-OpenCV-in-Python_master_Nariz.xml")
mouth_cascade = cv2.CascadeClassifier("raw.githubusercontent.com_the-javapocalypse_Face-Detection-Recognition-Using-OpenCV-in-Python_master_Mouth.xml")

video_cap = cv2.VideoCapture(0)                #for external webcam write -1 inside VideoCapture func.
while True:
    ret, img = video_cap.read()
    img = detect(img,face_cascade,eye_cascade,nose_cascade,mouth_cascade)
    cv2.imshow("Face Detection",img)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

video_cap.release()
cv2.destroyAllWindows()

