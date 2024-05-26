import FaceModule as Fm
import cv2 as cv

chosen_left_eye_lmList  = [362, 385, 387, 263, 373, 380]
chosen_right_eye_lmList = [33,  160, 158, 133, 153, 144]
all_chosen_eye_lmList = chosen_left_eye_lmList + chosen_right_eye_lmList

# capture = cv.VideoCapture(0)
# while True:
#     success, img = capture.read()
#     if not success:
#         break

img = cv.imread("squiting-eye-woman.jpg")
# height(y), width(x), c: color chanel
h, w, c = img.shape
#You have to declare faceDetector first to activate the init and the "self" and the default value you put in the FaceModule
#Without it, it won't have the argument to put in the place of the parameter self
faceDetector = Fm.faceDetector()
landmarksList = faceDetector.findLandMarks(img=img)

blinkDetetor = Fm.BlinkingDetector()
EAR_value, eye_cor = blinkDetetor.calculate_average_EAR(landmarksList,chosen_left_eye_lmList,chosen_right_eye_lmList,w,h)
print(EAR_value)
print(eye_cor)
if EAR_value > 0.2:
    print("Eye is open")
    
faceDetector.drawLandmarks(img,all_chosen_eye_lmList,landmarksList) 
cv.imshow("test",img)
cv.waitKey(0)


