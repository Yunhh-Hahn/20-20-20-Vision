import cv2 as cv
import mediapipe as mp
import math
from abc import ABC, abstractmethod


class AngleCalculationScenario(ABC):
    @abstractmethod
    def calculate_angle(self, img, lmlist):
        pass

    @property
    @abstractmethod
    def key_landmarks(self):
        pass
    # @abstractmethod
    # def display_data(self, img, lmList, angle_data):
    #     pass



class FaceAngle(AngleCalculationScenario):
    def calculate_angle(self, img, lmlist):
        return super().calculate_angle(img, lmlist)
    
class faceDetector:
    def __init__(self,mode=False,num_face=1,refine_landmarks=False,detectionCon=0.5,trackCon=0.5):
        self.mode = mode
        self.num_face = num_face
        self.refine = refine_landmarks
        self.minConDetection =  detectionCon
        self.minConTracking = trackCon
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(self.mode, self.num_face, self.refine, self.minConDetection,self.face_mesh)
    
    def findLandMark(self,img, draw = False):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.result = self.face_mesh.process(imgRGB)
        pass

     
capture = cv.VideoCapture(0)
while True:
    success, img = capture.read()
    if not success:
        break
    mp.solutions.face_mesh.FaceMesh()