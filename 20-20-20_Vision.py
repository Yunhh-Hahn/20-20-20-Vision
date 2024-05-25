import FaceModule as Fm
import cv2 as cv


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
landmark = detector.findLandMark(img=img,draw=True)
cv.imshow("test",img)
cv.waitKey(0)
# landmark_x = landmark.x * w
# landmark_y = landmark.y * h
# landmark_z = landmark.z * w #documentary said so according to the tutorial guy, don't fuckin see it in the documentary though
