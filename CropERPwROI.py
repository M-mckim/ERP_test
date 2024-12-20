import cv2
import numpy as np
import math
import time
import sys
import os
import argparse
import os
import sys
import cv2
import numpy as np

class Equirectangular:
    def __init__(self, img):
        self._img = img
        [self._height, self._width, _] = self._img.shape
        print(self._img.shape)
    

    def GetPerspective(self, FOV, THETA, PHI, height, width):
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #

        equ_h = self._height
        equ_w = self._width
        equ_cx = (equ_w - 1) / 2.0
        equ_cy = (equ_h - 1) / 2.0

        wFOV = FOV
        hFOV = float(height) / width * wFOV

        w_len = np.tan(np.radians(wFOV / 2.0))
        h_len = np.tan(np.radians(hFOV / 2.0))


        x_map = np.ones([height, width], np.float32)
        y_map = np.tile(np.linspace(-w_len, w_len,width), [height,1])
        z_map = -np.tile(np.linspace(-h_len, h_len,height), [width,1]).T

        D = np.sqrt(x_map**2 + y_map**2 + z_map**2)
        xyz = np.stack((x_map,y_map,z_map),axis=2)/np.repeat(D[:, :, np.newaxis], 3, axis=2)
        
        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)
        [R1, _] = cv2.Rodrigues(z_axis * np.radians(THETA))
        [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-PHI))

        xyz = xyz.reshape([height * width, 3]).T
        xyz = np.dot(R1, xyz)
        xyz = np.dot(R2, xyz).T
        lat = np.arcsin(xyz[:, 2])
        lon = np.arctan2(xyz[:, 1] , xyz[:, 0])

        lon = lon.reshape([height, width]) / np.pi * 180
        lat = -lat.reshape([height, width]) / np.pi * 180

        lon = lon / 180 * equ_cx + equ_cx
        lat = lat / 90  * equ_cy + equ_cy

        
            
        persp = cv2.remap(self._img, lon.astype(np.float32), lat.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        return persp
        
            
class Perspective:
    def __init__(self, img, FOV, THETA, PHI ):
        self._img = img
        [self._height, self._width, _] = self._img.shape
        self.wFOV = FOV
        self.THETA = THETA
        self.PHI = PHI
        self.hFOV = float(self._height) / self._width * FOV

        self.w_len = np.tan(np.radians(self.wFOV / 2.0))
        self.h_len = np.tan(np.radians(self.hFOV / 2.0))

    

    def GetEquirec(self,height,width):
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #

        x,y = np.meshgrid(np.linspace(-180, 180,width),np.linspace(90,-90,height))
        
        x_map = np.cos(np.radians(x)) * np.cos(np.radians(y))
        y_map = np.sin(np.radians(x)) * np.cos(np.radians(y))
        z_map = np.sin(np.radians(y))

        xyz = np.stack((x_map,y_map,z_map),axis=2)

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)
        [R1, _] = cv2.Rodrigues(z_axis * np.radians(self.THETA))
        [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-self.PHI))

        R1 = np.linalg.inv(R1)
        R2 = np.linalg.inv(R2)

        xyz = xyz.reshape([height * width, 3]).T
        xyz = np.dot(R2, xyz)
        xyz = np.dot(R1, xyz).T

        xyz = xyz.reshape([height , width, 3])
        inverse_mask = np.where(xyz[:,:,0]>0,1,0)

        xyz[:,:] = xyz[:,:]/np.repeat(xyz[:,:,0][:, :, np.newaxis], 3, axis=2)
        
        
        lon_map = np.where((-self.w_len<xyz[:,:,1])&(xyz[:,:,1]<self.w_len)&(-self.h_len<xyz[:,:,2])
                    &(xyz[:,:,2]<self.h_len),(xyz[:,:,1]+self.w_len)/2/self.w_len*self._width,0)
        lat_map = np.where((-self.w_len<xyz[:,:,1])&(xyz[:,:,1]<self.w_len)&(-self.h_len<xyz[:,:,2])
                    &(xyz[:,:,2]<self.h_len),(-xyz[:,:,2]+self.h_len)/2/self.h_len*self._height,0)
        mask = np.where((-self.w_len<xyz[:,:,1])&(xyz[:,:,1]<self.w_len)&(-self.h_len<xyz[:,:,2])
                    &(xyz[:,:,2]<self.h_len),1,0)

        persp = cv2.remap(self._img, lon_map.astype(np.float32), lat_map.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        
        mask = mask * inverse_mask
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        persp = persp * mask
        
        
        return persp , mask



if __name__ == '__main__':
    # anotate the input parameters Theta, Phi
    parser = argparse.ArgumentParser(description="Crop ERP video")

    parser.add_argument('-f','--File_path', type=str, default='../campus.mp4', help='Input file path')
    parser.add_argument('-o','--Output_path', type=str, default='../campusERPwROI_0D.mp4', help='Output file path')
    parser.add_argument('-t','--Theta', type=float, default=0.0, help='Theta(left/right) angle degree')
    parser.add_argument('-p','--Phi', type=float, default=0.0, help='Phi(up/down) angle degree')
    parser.add_argument('-wi','--Width', type=int, default=1440, help='Output video width')
    parser.add_argument('-he','--Height', type=int, default=810, help='Output video height')
    args = parser.parse_args()
    
    # 1080s(540x300), 1920s(960x540), 3840s(1920x1080)
    # 5.7K (5760x2880) Youtube(16:9)에 맞추면.. 1440x810(FoV 90)
    start_time = time.time()

    theta = args.Theta
    phi = args.Phi

    cap = cv2.VideoCapture(args.File_path)

    if not cap.isOpened():
        print("Failed to open!")
        sys.exit()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    wri = cv2.VideoWriter(args.Output_path, fourcc, fps, (1440,810)) # width, height
    print("Start converting...")
    
    f_cnt = 1
    x1,y1,w1,h1,x2,y2,w2,h2 = 0, 0, 0, 0, 0, 0, 0, 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if f_cnt == 1:
            f_cnt = 0
            equ = Equirectangular(frame)    # Load equirectangular image
            # get Perspective image
            img = equ.GetPerspective(90, theta, phi, 810, 1440)  # Specify parameters(FOV, theta, phi, height, width)
            # get Equirectangular image corresponding to the perspective image
            per = Perspective(img, 90, theta, phi)   # img , FOV, THETA, PHI
                            

            img2, mask = per.GetEquirec(2880,5760) # hight, width
            img2 = cv2.convertScaleAbs(img2)
            gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            conts, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # img2 = cv2.drawContours(frame.copy(), conts, -1, (0, 0, 255), 3) # 1440x810일때는 3, 5760x2880일때는 10 정도로해야 윤곽선이 잘 보임
            # all_conts = np.concatenate(conts)
            # x,y,w,h = cv2.boundingRect(all_conts)
            # roi = img2[y:y+h,x:x+w]
            
            if len(conts) > 1:
                x1,y1,w1,h1 = cv2.boundingRect(conts[0])
                x2,y2,w2,h2 = cv2.boundingRect(conts[1])
                h1 = max(h1,h2) # 수평 길이는 각각 다르게...
                y1 = min(y1,y2) # 수직 길이는 큰 값에 맞춰서...
                if(x1>x2):
                    roi1 = frame[y1:y1+h1,x1:x1+w1]
                    roi2 = frame[y1:y1+h1,x2:x2+w2]
                else:
                    roi1 = frame[y1:y1+h1,x2:x2+w2]
                    roi2 = frame[y1:y1+h1,x1:x1+w1]
                
                roi = np.concatenate((roi1,roi2),axis=1)  
            else:
                x1,y1,w1,h1 = cv2.boundingRect(conts[0])
                roi = frame[y1:y1+h1,x1:x1+w1]
            
                roi = cv2.resize(roi,(1440,810))
                cv2.imwrite('../croppedROI_0.jpg',roi)
                wri.write(roi)
        
        else:
            if len(conts) > 1:
                if(x1>x2):
                    roi1 = frame[y1:y1+h1,x1:x1+w1]
                    roi2 = frame[y1:y1+h1,x2:x2+w2]
                else:
                    roi1 = frame[y1:y1+h1,x2:x2+w2]
                    roi2 = frame[y1:y1+h1,x1:x1+w1]
                roi = np.concatenate((roi1,roi2),axis=1) 
            else:
                x1,y1,w1,h1 = cv2.boundingRect(conts[0])
                roi = frame[y1:y1+h1,x1:x1+w1]
            
            roi = cv2.resize(roi,(1440,810))
            wri.write(roi)

    wri.release()
    end_time = time.time()
    print(f"Time taken: {(end_time-start_time):.6f} seconds", flush=True)
