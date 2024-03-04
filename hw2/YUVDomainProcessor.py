import cv2
import numpy as np
from tqdm import tqdm

from utils import *

def RGB2YUV(img:np.ndarray):
    matrix = np.array([[0.257, 0.504, 0.098],
                       [0.439, -0.368, -0.071],
                       [-0.148, -0.291, 0.439]], dtype=np.float32)  # x256
    bias = np.array([16, 128, 128], dtype=np.float32).reshape(1, 1, 3)
    rgb_image = img.astype(np.float32)

    ycrcb_image = (rgb_image @ matrix + bias).astype(np.uint8)

    return ycrcb_image

def YUV2RGB(img: np.ndarray):
    # Y'UV to RGB conversion matrix
    matrix = np.array([
        [1.164, 1.5958, -0.0018],
        [1.164, -0.8135, -0.3914],
        [1.164, -0.0012, 2.0178]], dtype=np.float32)
    bias = np.array([16, 128, 128], dtype=np.float32).reshape(1, 1, 3)
    yuv_image = img.astype(np.int32)
    
    rbg_image = ((yuv_image - bias) @ matrix).astype(np.uint8)
    return rbg_image

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
        self.edge_filter = edge_filter/8
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
        # return ee_img[1:-1, 2:-2], em_img[1:-1, 2:-2]
        
        em_img = conv3x5(self.img, self.edge_filter)
        em_img = np.transpose(em_img)
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
        lut[mask3] = -self.gain[0] * em_img[mask3]
        lut[mask4] = 0
        lut[mask5] = self.gain[1] * em_img[mask5]

        # 可能问题在这里。。
        # np.clip(lut, clip[0], clip[1], out=lut)
        lut = np.maximum(self.emclip[0], np.minimum(lut / 256, self.emclip[1]))
        ee_img = self.img + lut   
        np.clip(ee_img, 0, 255, out=ee_img)
        return ee_img, em_img
        
class BCC:
    def __init__(self, img, brightness, contrast, bcc_clip, saturation_values = 300):
        self.img = img
        self.brightness = brightness
        self.contrast = contrast
        self.clip = bcc_clip
        self.saturation_values = saturation_values
    
    def execute(self):
        # h, w = self.img.shape[0], self.img.shape[1]
        # bcc_img = np.empty((h, w), np.int16)
        # bcc_img = self.img + self.brightness + (self.img - 127) * self.contrast
        # np.clip(bcc_img, self.clip[0], self.clip[1], out=bcc_img)
        # return bcc_img
        
        y_image = self.img.astype(np.int32)

        # bcc_y_image = np.clip(y_image + self.brightness, 0, self.saturation_values)

        # y_median = np.median(bcc_y_image).astype(np.int32)
        # print(y_median)
        # bcc_y_image = (bcc_y_image - y_median) * self.contrast + y_median
        bcc_y_image = y_image * self.contrast + self.brightness
        bcc_y_image = np.clip(bcc_y_image, 0, self.saturation_values)

        bcc_y_image = bcc_y_image.astype(np.uint8)
        return bcc_y_image
        
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
        # for y in tqdm(range(h)):
        #     for x in range(w):
        #         if np.abs(self.edgemap[y,x]) <= self.fcs_edge[0]:
        #             uvgain = self.gain
        #         elif np.abs(self.edgemap[y,x]) > self.fcs_edge[0] and np.abs(self.edgemap[y,x]) < self.fcs_edge[1]:
        #             uvgain = self.intercept - self.slope * self.edgemap[y,x]
        #         else:
        #             uvgain = 0
        #         fcs_img[y,x,:] = uvgain * (self.img[y,x,:]) / 256 + 128
        # np.clip(fcs_img, 0, 255, out=fcs_img)
        # return fcs_img
    
        absEM = np.abs(self.edgemap)
        mask1 = absEM <= self.fcs_edge[0]
        mask2 = np.logical_and(
            absEM > self.fcs_edge[0], 
            absEM < self.fcs_edge[1])
        mask3 = absEM >= self.fcs_edge[1]
        
        fcs_img[mask1] = self.img[mask1]
        fcs_img[mask2, 0] = self.gain*(absEM[mask2]-128)/65536 + 128
        fcs_img[mask2, 1] = self.gain*(absEM[mask2]-128)/65536 + 128    
        fcs_img[mask3] = 0
        
        # np.clip(fcs_img, 0, 255, out=fcs_img)
        np.clip(fcs_img, 16,239, out=fcs_img)
        return fcs_img    
    
class HSC:
    def __init__(self, img, hue, saturation, hsc_clip):
        self.img = img
        self.hue = hue
        self.saturation = saturation
        self.clip = hsc_clip

    def execute(self):
        # ind = np.array([i for i in range(360)])
        # sin = np.sin(ind * np.pi / 180) * 256
        # cos = np.cos(ind * np.pi / 180) * 256
        # lut_sin = dict(zip(ind, [round(sin[i]) for i in ind]))
        # lut_cos = dict(zip(ind, [round(cos[i]) for i in ind]))
        # hsc_img = np.empty(self.img.shape, np.int16)
        # hsc_img[:,:,0] = (self.img[:,:,0] - 128) * lut_cos[self.hue] + (self.img[:,:,1] - 128) * lut_sin[self.hue] + 128
        # hsc_img[:,:,1] = (self.img[:,:,1] - 128) * lut_cos[self.hue] - (self.img[:,:,0] - 128) * lut_sin[self.hue] + 128
        # hsc_img[:,:,0] = self.saturation * (self.img[:,:,0] - 128) / 256 + 128
        # hsc_img[:,:,1] = self.saturation * (self.img[:,:,1] - 128) / 256 + 128
        # np.clip(hsc_img, 0, self.clip, out=hsc_img)
        # return hsc_img
       
        hsc_img = np.empty(self.img.shape, np.int16)
        sin_hue = round(np.sin(self.hue))
        cos_hue = round(np.cos(self.hue))
        hsc_img[:,:,0] = (self.img[:, :, 0]-128) * cos_hue + (self.img[:, :, 1]-128)*sin_hue + 128
        hsc_img[:,:,1] = (self.img[:, :, 1]-128) * cos_hue + (self.img[:, :, 0]-128)*sin_hue + 128
        
        hsc_img[:,:,0] = self.saturation * (self.img[:,:,0] - 128) / 256 + 128
        hsc_img[:,:,1] = self.saturation * (self.img[:,:,1] - 128) / 256 + 128
        
        # np.clip(hsc_img, 0, self.clip, out=hsc_img)
        np.clip(hsc_img, 16,239, out=hsc_img)
        
        return hsc_img
        
if __name__ == "__main__":
    pass