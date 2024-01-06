import cv2
import numpy as np
import imutils
import copy
import pickle
from skimage.transform import resize
from skimage.io import imread

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))


### Initialize Kalman filter ###
X = np.array([[0], # Position along the x-axis
              [0], # Velocity along the x-axis
              [0], # Position along the y-axis
              [0]])# Velocity along the y-axis

P = np.array([[1000, 0, 0, 0],
              [0, 1000, 0, 0],
              [0, 0, 1000, 0],
              [0, 0, 0, 1000]
             ])

# The external motion. Set to 0 here.
u = np.array([[0],
              [0],
              [0],
              [0]])

# The transition matrix. 
F = np.array([[1, 1, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 1],
              [0, 0, 0, 1]
              ])

# The observation matrix. We only get the position as measurement.
H = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])

# The measurement uncertainty
R = np.array([[1],
              [1]])

# The identity matrix. Simply a matrix with 1 in the diagonal and 0 elsewhere.
I = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

# Optical flow parameters
f_param = dict(pyr_scale = 0.5, 
               levels = 5, 
               winsize = 15, 
               iterations = 5, 
               poly_n = 1, 
               poly_sigma = 0, 
               flags = 0)


def classification(img, model):
    Categories = ['Cup','Book','Box']
    img_resize = resize(img, (150,150,3))
    l = np.array([img_resize.flatten()]).reshape(1,-1)
    probability = model.predict_proba(l)
    return Categories[model.predict(l)[0]]

def update(x, P, Z, H, R): # Kalman filter update
    y = Z - H.dot(x)
    s = H.dot(P).dot(np.transpose(H)) + R
    k = P.dot(np.transpose(H)).dot(np.linalg.pinv(s))
    x_ = x + k.dot(y)
    p_ = (I - k.dot(H)).dot(P)
    
    return x_, p_

def predict(x, P, F, u): # Kalman filter predict
    x_ = F.dot(x) + u
    p_ = F.dot(P).dot(np.transpose(F))
    
    return x_, p_

def tracking(frames, ret, preframe, h, w, mapl1, mapl2, mapr1, mapr2, model):
    global X, P, u, F, H, R, I, f_param

    # Set the obtained fps
    framerate = 1
    c = 1

    # Set array to store data
    trackp = []
    Area = []
    ROI = []
    obser = []
    Z = []
    flag = 0
    X_l = []
    P_l = []

    prel = cv2.remap(preframe, mapl1, mapl2, cv2.INTER_LINEAR)
    prel_g = cv2.cvtColor(prel, cv2.COLOR_BGR2GRAY)
    obser = []

    for i in range(int(frames)-1):
        ret, nextframe = cap.read()
        
        if ret:
            if(c % framerate == 0):
                posl = cv2.remap(nextframe, mapl1, mapl2, cv2.INTER_LINEAR)
                
                img = copy.deepcopy(posl)

                posl_g = cv2.cvtColor(posl, cv2.COLOR_BGR2GRAY)
        
                flow = cv2.calcOpticalFlowFarneback(prel_g, posl_g, None, **f_param)
                distance = np.sqrt((flow[...,0]**2 + flow[...,1]**2))
                
                idx = cv2.threshold(distance, 1, 255, cv2.THRESH_BINARY)[1]
#               idx = cv2.erode(idx, np.ones((,17)))
#               idx = cv2.dilate(idx, np.ones((15,15)))
                idx = np.array(idx, dtype = np.uint8)
                cnts = cv2.findContours(idx.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)
                
                maxArea = 0
                obj = 0
                roi = (0,0,0,0)
                roi_p = (0,0,0,0)
                center = np.array([[0],[0]])
                
                for co in cnts:
                    if cv2.contourArea(co) > maxArea and cv2.contourArea(co) > 7000:
                        maxArea = cv2.contourArea(co)
                        obj = co
                        
                        (x,y,w,h) = cv2.boundingRect(obj)
                        center = np.array([[int(x+w/2)], [int(y+h/2)]])
                        roi = (x, y, x+w, y+h)
                        
                        if (center[1][0] < 300):
                            maxArea = 0
                            center = np.array([[0], [0]])
                            roi = (0, 0, 0, 0)
                        
                        else:
                            pass
                        
                Area.append([maxArea])
                
                if len(Area) >= 2:
                    if (Area[-1][0] - Area[-2][0]) > 5800:
                        center = np.array([[0], [0]])
                        roi = (0, 0, 0, 0)
                    else:
                        pass
                    
                xloc = str(center[0][0])
                yloc = str(center[1][0])
#               area = maxArea
                text = f'location: ({xloc}, {yloc}), MaxArea: {maxArea}, Area: {Area[-1][0]}'
                
                z = np.array([[np.int16(center[0][0])],
                            [np.int16(center[1][0])]])
                
                Z.append(z)
                
                obser.append(z)
                
                if (len(obser) > 3):
                    # state before occlusion
                    if ((Area[-2][0] + Area[-1][0])/ 2) >= 7000 and z[0] > 1100 and z[1] < 400:
                        if (obser[-1][0][0] - obser[-2][0][0]) < 0 and (obser[-1][1][0] - obser[-2][1][0]) > 0:
#                       if X[1][0] < -3 and X[3][0] > 2:
                            X, P = update(X, P, z, H, R)
                        
                        X, P = predict(X, P, F, u)

                        if(X[1][0] > -2.5):
                            X[1][0] = -1.5
                            P = P_l[-1]
                            X, P = predict(X, P, F, u)
                            
                        if(X[3][0] < 1.5):
                            X[3][0] = 0.2
                            P = P_l[-1]
                            X, P = predict(X, P, F, u)
                            
                    if ((Area[-2][0] - Area[-1][0]) >=  7000 and 700 < Z[-2][0][0] < 1200 and Z[-2][1][0] < 450):
                        flag = 1
                        
                    if flag == 1:
                        X, P = predict(X, P, F, u)
                        if(X[1][0] > -2.5):
                            X[1][0] = -1.5
                            P = P_l[-1]
                            X, P = predict(X, P, F, u)
                            
                        if(X[3][0] < 1.5):
                            X[3][0] = 0.2
                            P = P_l[-1]
                            X, P = predict(X, P, F, u)
                            
                        if maxArea != 0:
                            flag = 0
                    
                    if (Area[-1][0]) > 7500 and z[0] < 750 and z[1] > 450:
                        if (obser[-1][0][0] - obser[-2][0][0]) < 0 and (obser[-1][1][0] - obser[-2][1][0]) > 0:
                            X, P = update(X, P, z, H, R)
                        
                        X, P = predict(X, P, F, u)
                        
                    if (np.sqrt(obser[-1][0][0]**2 + obser[-1][1][0]**2) - np.sqrt(obser[-2][0][0]**2 + obser[-2][1][0]**2)) > 500 or Area[-1][0] < 2000 and z[0] < 500 and z[1] > 450:
                        
                        P = np.array([[1000, 0, 0, 0],
                                    [0, 1000, 0, 0],
                                    [0, 0, 1000, 0],
                                    [0, 0, 0, 1000]])
                        
                        R = np.array([[1],
                                    [1]])
                        
                    if X[0][0] < 200:
                        
                        P = np.array([[1000, 0, 0, 0],
                                    [0, 1000, 0, 0],
                                    [0, 0, 1000, 0],
                                    [0, 0, 0, 1000]])
                        
                        R = np.array([[1],
                                    [1]])
                        
                        X = np.array([[0], 
                                    [0], 
                                    [0], 
                                    [0]])

                else:
                    pass
                
                X_l.append(X)
                P_l.append(P)
                
                x_p = np.int16(X[0][0])
                y_p = np.int16(X[2][0])
                
                vx_p = np.int16(X[1][0])
                vy_p = np.int16(X[3][0])
                
                text_p = f'predict_location: ({x_p}, {y_p})'
                text_v = f'predict_velocity: ({vx_p}, {vy_p})'
                
                roi_p = (np.int16(X[0][0]-w/2), np.int16(X[2][0]-h/2), np.int16(X[0][0]+w/2), np.int16(X[2][0]+h/2))

                img_obj = None
                if np.sum(roi) != 0:
                    img_obj = img[roi[1]:roi[3], roi[0]:roi[2], 0:3]
                
                if img_obj is not None:
                    category = classification(img_obj, model)
                    text_o = f'{category}'
                    cv2.putText(img, text_o, (center[0][0]-20, center[1][0]-int((roi[3]-roi[1])/2)-10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
                            
                cv2.putText(img, text, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.putText(img, text_p, (10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.putText(img, text_v, (10,85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.circle(img, (center[0][0], center[1][0]), 20, (0,255,0), -1)
                cv2.circle(img, (np.int16(X[0][0]), np.int16(X[2][0])), 10, (0,0,255), -1)
                cv2.rectangle(img, (roi[0], roi[1]), (roi[2], roi[3]), (255,0,0), 3)
                cv2.rectangle(img, (roi_p[0], roi_p[1]), (roi_p[2], roi_p[3]), (155,155,100), 3)
            
                cv2.imshow('binary', idx)
                cv2.imshow('original', img)
                cv2.waitKey(50)
                
                prel = posl
                prel_g = posl_g
                
                print(roi)
                print('object: ', center)
                print('measure:', z)
                print('maxArea: ',maxArea)
                print('========================')
                trackp.append(center)
                ROI.append([roi])
                
        c += 1
        
    cv2.destroyAllWindows()
    cap.release()


# Dealing with video
cap = cv2.VideoCapture('data/stereo_conveyor_with_occlusions.mp4')

fps = cap.get(cv2.CAP_PROP_FPS)
frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
ret, preframe = cap.read()
h, w = preframe.shape[:2]

# Load SVM Model
filename = 'classification_model.sav'
model = pickle.load(open('model/' + filename, 'rb'))

# Load calibration and rectification parameters
R1 = np.load('parameters/R1.npy')
R2 = np.load('parameters/R2.npy')
P1 = np.load('parameters/P1.npy')
P2 = np.load('parameters/P2.npy')
mtxl = np.load('parameters/mtxl.npy')
mtxr = np.load('parameters/mtxr.npy')
distl = np.load('parameters/distl.npy')
distr = np.load('parameters/distr.npy')

mapl1, mapl2 = cv2.initUndistortRectifyMap(mtxl, distl, R1, P1, (w,h), cv2.CV_32FC1)
mapr1, mapr2 = cv2.initUndistortRectifyMap(mtxr, distr, R2, P2, (w,h), cv2.CV_32FC1)

tracking(frames, ret, preframe, h, w, mapl1, mapl2, mapr1, mapr2, model)
