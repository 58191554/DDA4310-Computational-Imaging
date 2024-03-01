import cv2
import numpy as np
from tqdm import tqdm

from utils import *

def RGB2YUV(img:np.ndarray):
    matrix = np.array([[66, 129, 25],
                       [-38, -74, 112],
                       [112, -94, -18]], dtype=np.int32).T  # x256
    bias = np.array([16, 128, 128], dtype=np.int32).reshape(1, 1, 3)
    rgb_image = img.astype(np.int32)

    ycrcb_image = (np.right_shift(rgb_image @ matrix, 8) + bias).astype(np.uint8)

    return ycrcb_image

def YUV2RGB(img: np.ndarray):
    # Y'UV to RGB conversion matrix
    yuv_to_rgb_matrix = np.array([[1.000, 1.000, 1.000],
                                  [0.000, -0.394, 2.032],
                                  [1.140, -0.581, 0.000]])
    
    # Convert YUV image to float32
    yuv_image_float = img.astype(np.float32) / 255.0
    
    # Apply matrix multiplication for conversion
    rgb_image_float = np.dot(yuv_image_float, yuv_to_rgb_matrix.T)
    
    # Clip values to the valid range [0, 255]
    rgb_image_float = np.clip(rgb_image_float, 0, 1)
    
    # Scale back to the range [0, 255] and convert to uint8
    rgb_image_uint8 = (rgb_image_float * 255).astype(np.uint8)
    
    return rgb_image_uint8

def EdgeEnhancement(yuv_img:np.ndarray, edge_filter:np.ndarray, ee_gain:list, ee_thres:list, ee_emclip:list):
    return EE(yuv_img, edge_filter, ee_gain, ee_thres, ee_emclip)
    
def BrightnessContrastControl(yuvimg_ee:np.ndarray, brightness, contrast, bcc_clip):
    return BCC(yuvimg_ee, brightness, contrast, bcc_clip)
    
def FalseColorSuppression(ee_img, em_img, fcs_edge, fcs_gain, fcs_intercept, fcs_slope):
    return FCS(ee_img, em_img, fcs_edge, fcs_gain, fcs_intercept, fcs_slope)
    
def HueSaturationControl(yuvimg_fcs, hue, saturation, hsc_clip):
    return HSC(yuvimg_fcs, hue, saturation, hsc_clip)
    
class EE:
    def __init__(self, img, edge_filter, ee_gain, ee_thres, ee_emclip):
        self.img = img
        self.edge_filter = edge_filter
        self.gain = ee_gain
        self.thres = ee_thres
        self.emclip = ee_emclip
    

    def execute(self):
        # img_pad = np.pad(self.img, ((1, 1), (2, 2)), 'reflect')
        # h, w = img_pad.shape[0], img_pad.shape[1]
        # ee_img = np.empty((h, w), np.int16)
        # em_img = np.empty((h, w), np.int16)
        # for y in tqdm(range(img_pad.shape[0] - 2)):
        #     for x in range(img_pad.shape[1] - 4):
        #         em_img[y,x] = np.sum(np.multiply(img_pad[y:y+3, x:x+5], self.edge_filter[:, :])) / 8

        #         lut = 0
        #         if em_img[y,x] < -self.thres[1]:
        #             lut = self.gain[1] * em_img[y,x]
        #         elif em_img[y,x] < -self.thres[0] and em_img[y,x] > -self.thres[1]:
        #             lut = 0
        #         elif em_img[y,x] < self.thres[0] and em_img[y,x] > -self.thres[1]:
        #             lut = self.gain[0] * em_img[y,x]
        #         elif em_img[y,x] > self.thres[0] and em_img[y,x] < self.thres[1]:
        #             lut = 0
        #         elif em_img[y,x] > self.thres[1]:
        #             lut = self.gain[1] * em_img[y,x]
        #         # np.clip(lut, clip[0], clip[1], out=lut)
        #         lut = max(self.emclip[0], min(lut / 256, self.emclip[1]))
                
                
        #         ee_img[y,x] = img_pad[y+1,x+2] + lut
        # np.clip(ee_img, 0, 255, out=ee_img)
        # return ee_img, em_img
        
        img_pad = np.pad(self.img, ((1, 1), (2, 2)), 'reflect')
        h, w = img_pad.shape[0], img_pad.shape[1]
        em_img = np.empty((h, w), np.int16)
        em_img = convolution3x5_grey(img_pad, self.edge_filter)

        # Vectorized computation
        lut = np.zeros_like(em_img)

        # Compute lut based on conditions
        mask1 = em_img < -self.thres[1]
        mask2 = (em_img >= -self.thres[1]) & (em_img <= -self.thres[0])
        mask3 = (em_img >= -self.thres[0]) & (em_img <= self.thres[0])
        mask4 = (em_img >= self.thres[0]) & (em_img <= self.thres[1])
        mask5 = em_img > self.thres[1]

        lut[mask1] = self.gain[1] * em_img[mask1]
        lut[mask2] = 0
        lut[mask3] = self.gain[0] * em_img[mask3]
        lut[mask4] = 0
        lut[mask5] = self.gain[1] * em_img[mask5]

        lut = np.clip(lut, self.emclip[0], self.emclip[1])        
        ee_img = self.img + lut   
        np.clip(ee_img, 0, 255, out=ee_img)
        return ee_img, em_img
        
           
class BCC:
    def __init__(self, img, brightness, contrast, bcc_clip):
        self.img = img
        self.brightness = brightness
        self.contrast = contrast
        self.clip = bcc_clip
    
    def execute(self):
        h, w = self.img.shape[0], self.img.shape[1]
        bcc_img = np.empty((h, w), np.int16)
        bcc_img = self.img + self.brightness
        bcc_img = bcc_img + (self.img - 127) * self.contrast
        np.clip(bcc_img, 0, self.clip, out=bcc_img)
        return bcc_img[1:-1, 2:-2]
        
class FCS:
    def __init__(self, ee_img, em_map, fcs_edge, fcs_gain, fcs_intercept, fcs_slope):
        self.img = ee_img
        self.edgemap = em_map
        self.fcs_edge = fcs_edge
        self.gain = fcs_gain
        self.intercept = fcs_intercept
        self.slope = fcs_slope
        
    def execute(self):
        h, w, c = self.img.shape
        fcs_img = np.empty((h, w, c), np.int16)
        for y in tqdm(range(h)):
            for x in range(w):
                if np.abs(self.edgemap[y,x]) <= self.fcs_edge[0]:
                    uvgain = self.gain
                elif np.abs(self.edgemap[y,x]) > self.fcs_edge[0] and np.abs(self.edgemap[y,x]) < self.fcs_edge[1]:
                    uvgain = self.intercept - self.slope * self.edgemap[y,x]
                else:
                    uvgain = 0
                fcs_img[y,x,:] = uvgain * (self.img[y,x,:]) / 256 + 128
        np.clip(fcs_img, 0, 255, out=fcs_img)
        return fcs_img
    
class HSC:
    def __init__(self, img, hue, saturation, hsc_clip):
        self.img = img
        self.hue = hue
        self.saturation = saturation
        self.clip = hsc_clip

    def execute(self):
        # 需要改。。。
        ind = np.array([i for i in range(360)])
        sin = np.sin(ind * np.pi / 180) * 256
        cos = np.cos(ind * np.pi / 180) * 256
        lut_sin = dict(zip(ind, [round(sin[i]) for i in ind])) # 需要改。。。
        lut_cos = dict(zip(ind, [round(cos[i]) for i in ind])) # 需要改。。。
        hsc_img = np.empty(self.img.shape, np.int16)
        hsc_img[:,:,0] = (self.img[:,:,0] - 128) * lut_cos[self.hue] + (self.img[:,:,1] - 128) * lut_sin[self.hue] + 128
        hsc_img[:,:,1] = (self.img[:,:,1] - 128) * lut_cos[self.hue] - (self.img[:,:,0] - 128) * lut_sin[self.hue] + 128
        hsc_img[:,:,0] = self.saturation * (self.img[:,:,0] - 128) / 256 + 128
        hsc_img[:,:,1] = self.saturation * (self.img[:,:,1] - 128) / 256 + 128
        np.clip(hsc_img, 0, self.clip, out=hsc_img)
        return hsc_img
        
if __name__ == "__main__":
    pass