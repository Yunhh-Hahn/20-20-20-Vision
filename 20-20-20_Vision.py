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

img = cv.imread("16297800391_5c6e812832-400x267.jpg")
# height(y), width(x), c: color chanel
h, w, c = img.shape
#You have to declare detector first to activate the init and the "self" and the default value you put in the FaceModule
#Without it, it won't have the argument to put in the place of the parameter self
detector = Fm.faceDetector()
landmarks = detector.findLandMarks(img=img)
detector.drawLandmarks(img,all_chosen_eye_lmList,landmarks)
cv.imshow("test",img)
cv.waitKey(0)
# landmark_x = landmark.x * w
# landmark_y = landmark.y * h
# landmark_z = landmark.z * w #documentary said so according to the tutorial guy, don't fuckin see it in the documentary though
