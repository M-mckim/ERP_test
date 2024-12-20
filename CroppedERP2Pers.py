import cv2
import numpy as np
import zlib

def RollCenter(img, THETA):
    equ_h, equ_w, _ = img.shape
    
    # Shift the image so that the desired longitude is in the center
    shift_pixels = int((THETA + 180) * equ_w / 360)
    img = np.roll(img, -shift_pixels, axis=1)
    
    return img

def GetPerspective(equ_h, equ_w, FOV, PHI, height, width):
    equ_cx = (equ_w - 1) / 2.0
    equ_cy = (equ_h - 1) / 2.0

    wFOV = FOV
    hFOV = float(height) / width * wFOV

    w_len = np.tan(np.radians(wFOV / 2.0))
    h_len = np.tan(np.radians(hFOV / 2.0))

    x_map = np.ones([height, width], np.float32)
    y_map = np.tile(np.linspace(-w_len, w_len, width), [height, 1])
    z_map = -np.tile(np.linspace(-h_len, h_len, height), [width, 1]).T

    D = np.sqrt(x_map**2 + y_map**2 + z_map**2)
    xyz = np.stack((x_map, y_map, z_map), axis=2) / np.repeat(D[:, :, np.newaxis], 3, axis=2)

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    [R, _] = cv2.Rodrigues(y_axis * np.radians(-PHI))

    xyz = xyz.reshape([height * width, 3]).T
    xyz = np.dot(R, xyz).T
    lat = np.arcsin(xyz[:, 2])
    lon = np.arctan2(xyz[:, 1], xyz[:, 0])

    lon = lon.reshape([height, width]) / np.pi * 180
    lat = -lat.reshape([height, width]) / np.pi * 180

    lon = lon / 180 * equ_cx + equ_cx
    lat = lat / 90 * equ_cy + equ_cy

    return lon, lat

### main ###
img = cv2.imread('first_frame_7680.png')
THETA = 180  # 예시로 150도를 중앙에 위치 longtitude (0~360)
PHI = 0
FOV = 90
height, width = 1080, 1920

# Shift the image so that the desired longitude is in the center
img = RollCenter(img, THETA)

equ_h, equ_w, _ = img.shape

lon, lat = GetPerspective(equ_h, equ_w, FOV, PHI, height, width)
print("lon.shape: ", lon.shape, "lat.shape: ", lat.shape)
print("lon size" , lon.size, "lat size: ", lat.size)
print("lon.tolist size: ", len(lon.tolist()), "lat.tolist size: ", len(lat.tolist()))
perspective_img = cv2.remap(img, lon.astype(np.float32), lat.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
cv2.imshow('Perspective', perspective_img)

lon_min = lon.min()
lon_max = lon.max()
lat_min = lat.min()
lat_max = lat.max()
# get ROI
cropped_img = img[int(lat_min):int(lat_max+1), int(lon_min):int(lon_max+1)]
print("ROI: ", lon_min, lon_max, lat_min, lat_max)
cv2.imshow('ROI', cropped_img)

lon = lon - abs(int(lon_min))
lat = lat - abs(int(lat_min))
lon_16 = lon.astype(np.float16)
lat_16 = lat.astype(np.float16)
persp = cv2.remap(cropped_img, lon_16.astype(np.float32), lat_16.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
cv2.imshow('Perspective_cropped', persp)

cv2.imwrite('Perspective_cropped.png', persp)
cv2.imwrite('Perspective.png', perspective_img)

cv2.waitKey(0)
cv2.destroyAllWindows()