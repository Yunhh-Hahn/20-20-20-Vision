import FaceModule as Fm
from eye import Eye
import numpy as np
from mediapipe import solutions as mp
import cv2 as cv
# Landmark points corresponding to left eye's iris
all_left_iris_point = list(mp.face_mesh.FACEMESH_LEFT_IRIS)
# Flatten and remove duplicates to loop into 
all_left_iris_point = set(np.ravel(all_left_iris_point))

# Landmark points corresponding to right eye's iris
all_right_iris_point = list(mp.face_mesh.FACEMESH_RIGHT_IRIS)
# Flatten and remove duplicates to loop into
all_right_iris_point = set(np.ravel(all_right_iris_point))

# Combined for plotting - landmark points for both iris
all_iris_point = all_left_iris_point.union(all_right_iris_point)
all_iris_point = list(all_iris_point)
# All iris point included pupil also, for some reason it doesn't included in ???
all_iris_point = [468,469, 470, 471, 472, 473, 474, 475, 476, 477]
chosen_left_eye_lmList  = [362, 385, 387, 263, 373, 380]
chosen_right_eye_lmList = [33,  160, 158, 133, 153, 144]
all_chosen_eye_lmList = chosen_left_eye_lmList + chosen_right_eye_lmList

# capture = cv.VideoCapture(0)
# while True:
#     success, img = capture.read()
#     if not success:
#         break

img = cv.imread("open-eye-asian-man.jpg")
# height(y), width(x), c: color chanel
h, w, c = img.shape
utils = Fm.Utility()
#You have to declare faceDetector first to activate the init and the "self" and the default value you put in the FaceModule
#Without it, it won't have the argument to put in the place of the parameter self
faceDetector = Fm.faceDetector()
landmarksList = faceDetector.findLandMarks(img=img)

blinkDetetor = Fm.BlinkingDetector()
EAR_value, eye_cor = blinkDetetor.calculate_average_EAR(landmarksList,chosen_left_eye_lmList,chosen_right_eye_lmList,w,h)
print(EAR_value)
if EAR_value > 0.2:
    print("Eye is open")

cor = utils.getCoordinate(landmarksList,all_iris_point,w,h)
print(cor)
faceDetector.drawLandmarks(img,all_iris_point,landmarksList) 
cv.imshow("test",img)
cv.waitKey(0)


