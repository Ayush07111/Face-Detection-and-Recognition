import cv2
import numpy as np
import face_recognition

imgElon = face_recognition.load_image_file('Sample Images/Elon Musk.jpg')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgElonTest = face_recognition.load_image_file('Sample Images/Bill Gates.jpeg')
imgElonTest = cv2.cvtColor(imgElonTest,cv2.COLOR_BGR2RGB)
facLoc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(facLoc[3],facLoc[0]),(facLoc[1],facLoc[2]),(255,0,255),2)
facLocTest = face_recognition.face_locations(imgElonTest)[0]
encodeElonTest = face_recognition.face_encodings(imgElonTest)[0]
cv2.rectangle(imgElonTest,(facLocTest[3],facLocTest[0]),(facLocTest[1],facLocTest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeElon],encodeElonTest)
print(results)

# cv2.imshow('Elon Musk',imgElon)
# cv2.imshow('Elon Musk Test',imgElonTest)
cv2.waitKey(0)