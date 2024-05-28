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
    def __init__(self,mode=False,num_face=1,refine_landmarks=True,detectionCon=0.5,trackCon=0.5):
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
        
        # Reminder to create the exception in the case that the model didn't managed to detect a face
        landmarksList = self.result.multi_face_landmarks[0].landmark
        return landmarksList
    
    def drawLandmarks(self,img,lmList,landmarks, num_face = 1,draw_all = False):
        get_pixel_coordinates = self.mp_draw._normalized_to_pixel_coordinates
        # Multi_face_landmark can detect multiple face and put into an array with each index a face
        num__face_index = num_face - 1
        if draw_all:
            self.mp_draw.draw_landmarks(img,self.result.multi_face_landmarks[num__face_index],self.mp_face_mesh.FACEMESH_TESSELATION)
            return 
        h,w,c = img.shape
        for lmIndex,lm in enumerate(landmarks):
            if lmIndex in lmList:
                # landmark_x = lm.x * w
                # landmark_y = lm.y * h
                # landmark_z = lm.z * w #documentary said so according to the tutorial guy, don't fuckin see it in the documentary though
                pixel_cor = get_pixel_coordinates(lm.x,lm.y,w,h)
                cv.circle(img, pixel_cor , 2, (0,0,255), cv.FILLED)

class BlinkingDetector():
    def distance(self,point1,point2):
        dist = sum( [ (i-j)**2 for i,j in zip(point1,point2) ] ) **0.5
        return dist
        '''
            Formula:
            distance between two point or the norm of a vector = √ x^2 + y^2 + ... (depend on dimension) 
            Simplified:
            sum = 0
            for i in range (len(point1)):
                sum+= (point1[i] -point2[i])**2
            distance = sum ** 0.5

            Remember it is point with cor so we subtract it to get vector
            zip allows for multi array looping link using index number between array
            sum function take all value within an array and get the sum of all value
            ==> Loop inside each array to get x_cor, y_cor --> square each value--> put inside a list --> sum function --> square root 
        '''
    def getEAR(self,lmList,targetLandmarksList, frame_width, frame_height):
        get_pixel_coordinates = mp.solutions.drawing_utils._normalized_to_pixel_coordinates
        """
        Calculate Eye Aspect Ratio for one eye.

        Args:
            landmarksList: (list) Detected landmarks list
            targetLandmarksList: (list) Index positions of the chosen landmarks
                                in order P1, P2, P3, P4, P5, P6
            frame_width: (int) Width of captured frame
            frame_height: (int) Height of captured frame

        Returns:
            ear: (float) Eye aspect ratio
        """
        try:
        # Get the distance of needed point
            coords_points = []
            for i in targetLandmarksList:
                lm = lmList[i]
                cor = get_pixel_coordinates(lm.x,lm.y,frame_width,frame_height)
                coords_points.append(cor)

            P2_P6 = self.distance(coords_points[1], coords_points[5])
            P3_P5 = self.distance(coords_points[2], coords_points[4])
            P1_P4 = self.distance(coords_points[0], coords_points[3])

            EAR = (P2_P6 + P3_P5) / (2.0*(P1_P4)) 
        except:
            EAR = 0.0
            coords_points = None
        return EAR, coords_points

    def calculate_average_EAR(self,lmList,left_eye_lmList,right_eye_lmList,frame_width,frame_height):
        left_EAR, left_lm_cor = self.getEAR(lmList,left_eye_lmList,frame_width,frame_height)
        right_EAR, right_lm_cor = self.getEAR(lmList,right_eye_lmList,frame_width,frame_height)
        Avg_EAR = (left_EAR+right_EAR) / 2.0
        return Avg_EAR, (left_lm_cor,right_lm_cor)
    
    def getCoordinate(self,lmList,targetLandmarksList, frame_width, frame_height):
        get_pixel_coordinates = mp.solutions.drawing_utils._normalized_to_pixel_coordinates
        # Get the distance of needed point
        coords_points = []
        for i in targetLandmarksList:
            lm = lmList[i]
            cor = get_pixel_coordinates(lm.x,lm.y,frame_width,frame_height)
            coords_points.append(cor)
        return coords_points






     
