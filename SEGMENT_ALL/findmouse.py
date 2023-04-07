import cv2
import numpy as np
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh=mp_face_mesh.FaceMesh(
    min_detection_confidence=0.2,
    min_tracking_confidence=0.2)
#嘴巴
mouse=[62,96,89,179,86,15,316,403,319,325,292,407,272,271,268,12,38,41,42,183]
lip=[78,95,88,178,87,14,317,402,318,324,308,415,310,311,312,13,82,81,80,191]#嘴唇


def cut(img, pol):
    pol=np.array([pol], np.int32)
    #遮片
    mask=np.zeros(img.shape[:2], np.uint8)
    #多邊形填上白色
    cv2.polylines(mask, [pol], isClosed=True,color=(255,255,255), thickness=1)
    cv2.fillPoly(mask,pol,255)
    
    dst=cv2.bitwise_and(img, img, mask=mask)
    return dst

def find_mouse(img):
    #h, w, d = read.shape
    #img=cv2.resize(read, (1000, int(1000*h/w)), interpolation=cv2.INTER_AREA)#固定大小
    h, w, d = img.shape
    
    

    #########################openCV辨識嘴 #########################
    RGBim = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(RGBim)
    mousep=[]
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            #for index in mouse:
            for index in lip:
                x = int(face_landmarks.landmark[index].x * w)
                y = int(face_landmarks.landmark[index].y * h)
                mousep.append([x,y])
    mousep=np.array(mousep)
    
    mousep_b=[]
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for index in mouse:
                x = int(face_landmarks.landmark[index].x * w)
                y = int(face_landmarks.landmark[index].y * h)
                mousep_b.append([x,y])
    mousep_b=np.array(mousep_b)

    
    #找 marker
    umos=min(mousep_b[:,1])#嘴上緣
    dmos=max(mousep_b[:,1])#嘴下緣
    lmos=min(mousep_b[:,0])#嘴左緣
    rmos=max(mousep_b[:,0])#嘴右緣
    wmos=rmos-lmos#嘴寬
    hmos=dmos-umos#嘴高
    mmos=[int((lmos+rmos)/2),int((umos+dmos)/2)]#嘴中心
    
    
    
    cuted=cut(img, mousep_b)
    return cuted,np.array([lmos,umos,rmos,dmos])
