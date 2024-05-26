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
        self.face_mesh = self.mp_face_mesh.FaceMesh(self.mode, self.num_face, self.refine, self.minConDetection)

        self.mp_draw = mp.solutions.drawing_utils
    
    def findLandMarks(self,img):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.result = self.face_mesh.process(imgRGB)
        # multi_face_landmarks have the attribute landmark (I don't see it the documentary but it has it)
        # The index 0 is for the number of face (1 in this case), landmark[index] is the index number landmark point (see all points visualization on mediapipe web)
        landmarks = self.result.multi_face_landmarks[0].landmark
        return landmarks
    
    def drawLandmarks(self,img,lmList,landmarks, num_face = 1,draw_all = False):
        # Multi_face_landmark
        num__face_index = num_face - 1
        if draw_all:
            self.mp_draw.draw_landmarks(img,self.result.multi_face_landmarks[num__face_index],self.mp_face_mesh.FACEMESH_TESSELATION)
            return 
        h,w,c = img.shape
        for lmIndex,lm in enumerate(landmarks):
            if lmIndex in lmList:
                cx, cy = int(lm.x*w), int(lm.y*h)
                cv.circle(img, (cx,cy) , 2, (255,0,0), cv.FILLED)





     
