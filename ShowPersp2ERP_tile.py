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

    parser.add_argument('-f','--File_path', type=str, default='graysclae_test_img.png', help='Input file path')
    parser.add_argument('-o','--Output_path', type=str, default='showERP_0D.mp4', help='Output file path')
    parser.add_argument('-wi','--Width', type=int, default=5120, help='Output video width')
    parser.add_argument('-he','--Height', type=int, default=2560, help='Output video height')
    args = parser.parse_args()
    
    # 1080s(540x300), 1920s(960x540), 3840s(1920x1080)
    # 5.7K (5760x2880) Youtube(16:9)에 맞추면.. 1440x810(FoV 90)
    start_time = time.time()
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 9
    
    wri = cv2.VideoWriter(args.Output_path, fourcc, fps, (args.Width,args.Height))
    print("Start converting...")
        
    input_ERP = cv2.imread(args.File_path, cv2.IMREAD_COLOR)
    input_ERP = cv2.resize(input_ERP, (args.Width, args.Height))
    
    img_height, img_width = input_ERP.shape[:2]
    num_rows, num_columns = 10, 20
    tile_height, tile_width = img_height // num_rows, img_width // num_columns
    
    equ = Equirectangular(input_ERP)    # Load equirectangular image
    theta = 0 # horizontal angle
    phi = 45 # vertical angle
    fov = 90
    
    tiles = []
    for row in range(num_rows):
        for col in range(num_columns):
            x_start = col * tile_width
            y_start = row * tile_height
            x_end = x_start + tile_width
            y_end = y_start + tile_height
            tiles.append(((x_start, y_start), (x_end, y_end)))
    
    views = [(0,90)] # (0,0), (-90,45), (90,-45)
    
    for theta, phi in views:
        # get Perspective image
        img = equ.GetPerspective(fov, theta, phi, 720, 1280)  # Specify parameters(FOV, theta, phi, height, width)
        # get Equirectangular image corresponding to the perspective image
        per = Perspective(img, fov, theta, phi)   # img , FOV, THETA, PHI
        
        img2, mask = per.GetEquirec(args.Height,args.Width)
        img2 = cv2.convertScaleAbs(img2)
        # croppedERP = cv2.bitwise_and(input_ERP, mask)
        gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        conts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 새로운 프레임에 그리는 버전!
        img2 = cv2.drawContours(input_ERP.copy(), conts[0], -1, (0, 0, 255), 10)
        # 기존 프레임에 그리는 버전! copy()가 없으면 원본이 바뀜
        # img2 = cv2.drawContours(input_ERP, conts[0], -1, (0, 0, 255), 10)
        wri.write(img2)
    
        cv2.imwrite(args.Output_path.replace('.mp4',f'({theta},{phi}).png'), img2)
            
        tile_indices = set()
        
        for contour in conts[0]:
            for idx, ((x_start, y_start), (x_end, y_end)) in enumerate(tiles):
                # 타일 마스크
                tile_contour_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                cv2.rectangle(tile_contour_mask, (x_start, y_start), (x_end, y_end), 255, -1)

                # 윤곽선 마스크
                contour_mask = np.zeros_like(tile_contour_mask)
                cv2.drawContours(contour_mask, [contour], -1, 255, -1)

                if cv2.countNonZero(cv2.bitwise_and(tile_contour_mask, contour_mask)) > 0:
                    tile_indices.add(idx)
                    
        # 전체 이미지에 대해 20x10 tile 그리기
        output_img = img2.copy()
        for ((x_start, y_start), (x_end, y_end)) in tiles:
            cv2.rectangle(output_img, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)

        # 타일에 윤곽선 포함 여부 시각화
        output_img = output_img.copy()
        for idx in tile_indices:
            ((x_start, y_start), (x_end, y_end)) = tiles[idx]
            cv2.rectangle(output_img, (x_start, y_start), (x_end, y_end), (0, 255, 0), 4)

        output_img = output_img.copy()
        output_img = cv2.drawContours(output_img, conts[0], -1, (0, 0, 255), 10)

        cv2.imwrite(f"output_with_tiles({theta},{phi}).png", output_img)
        
        tile_coordinates = []

        # 타일 인덱스를 좌표로 변환
        for idx in sorted(tile_indices):
            tile_x = (idx % num_columns) + 1  # 열 번호
            tile_y = (idx // num_columns) + 1  # 행 번호
            tile_coordinates.append((tile_x, tile_y))

        print("(0,0) 타일 좌표 (1,1 ~ 20,10):", tile_coordinates)
    
    # wri.release()
    end_time = time.time()
    print(f"Time taken: {(end_time-start_time):.6f} seconds", flush=True)
