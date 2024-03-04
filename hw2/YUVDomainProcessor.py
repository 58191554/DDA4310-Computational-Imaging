import cv2
import numpy as np
from tqdm import tqdm

# from utils import *

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
    
def conv3x5(img, kernel:np.ndarray):
    img_pad = np.pad(img, ((1, 1), (2, 2)), 'reflect')
    y_indices, x_indices = np.meshgrid(np.arange(0, img.shape[0], dtype=np.int32), np.arange(0, img.shape[1], dtype=np.int32))
    a1 = img_pad[y_indices, x_indices]
    a2 = img_pad[y_indices, x_indices+1]
    a3 = img_pad[y_indices, x_indices+2]
    a4 = img_pad[y_indices, x_indices+3]
    a5 = img_pad[y_indices, x_indices+4]
    a6 = img_pad[y_indices+1, x_indices]
    a7 = img_pad[y_indices+1, x_indices+1]
    a8 = img_pad[y_indices+1, x_indices+2]
    a9 = img_pad[y_indices+1, x_indices+3]
    a10 = img_pad[y_indices+1, x_indices+4]
    a11 = img_pad[y_indices+2, x_indices]
    a12 = img_pad[y_indices+2, x_indices+1]
    a13 = img_pad[y_indices+2, x_indices+2]
    a14 = img_pad[y_indices+2, x_indices+3]
    a15 = img_pad[y_indices+2, x_indices+4]
    
    kernel = kernel.flatten()
    weighted_sum = np.sum(kernel[:, np.newaxis, np.newaxis] * np.array([a1, a2, a3, a4, a5,
                                                                    a6, a7, a8, a9, a10,
                                                                    a11, a12, a13, a14, a15]), axis=0)

    return np.transpose(weighted_sum)

def stat_draw(img:np.ndarray):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Plot histograms
    axs[0, 0].hist(img[:, :, 0].flatten(), bins=30, edgecolor='black', color='r')
    axs[0, 0].set_title('Red Channel Histogram')
    axs[0, 0].set_xlabel('Value')
    axs[0, 0].set_ylabel('Frequency')

    axs[0, 1].hist(img[:, :, 1].flatten(), bins=30, edgecolor='black', color='g')
    axs[0, 1].set_title('Green Channel Histogram')
    axs[0, 1].set_xlabel('Value')
    axs[0, 1].set_ylabel('Frequency')

    axs[1, 0].hist(img[:, :, 2].flatten(), bins=30, edgecolor='black', color='b')
    axs[1, 0].set_title('Blue Channel Histogram')
    axs[1, 0].set_xlabel('Value')
    axs[1, 0].set_ylabel('Frequency')

    # Plot the image
    axs[1, 1].imshow(img)
    axs[1, 1].set_title('RGB Image')

    # Hide the x and y ticks for the image plot
    axs[1, 1].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Show the plot
    plt.show()

def stat_draw_yuv(y:np.ndarray, uv:np.ndarray):
    fig, axs = plt.subplots(1, 3, figsize=(10, 8))

    # Plot histograms
    axs[0].hist(y[:, :].flatten(), bins=30, edgecolor='black')
    axs[0].set_title('Y Channel Histogram')
    axs[0].set_xlabel('Value')
    axs[0].set_ylabel('Frequency')

    axs[1].hist(uv[:, :, 0].flatten(), bins=30, edgecolor='black', color='r')
    axs[1].set_title('Cr Channel Histogram')
    axs[1].set_xlabel('Value')
    axs[1].set_ylabel('Frequency')

    axs[2].hist(uv[:, :, 1].flatten(), bins=30, edgecolor='black', color='b')
    axs[2].set_title('Cb Channel Histogram')
    axs[2].set_xlabel('Value')
    axs[2].set_ylabel('Frequency')


    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Show the plot
    plt.show()

def normalize(img, scale = 3000):
    # 找到每个通道的最小值和最大值
    img = img.astype(np.float32)
    # img /= scale
    min_vals = np.min(img, axis=(0, 1))
    max_vals = np.max(img, axis=(0, 1))
    
    # 归一化每个通道
    normalized_img = (img - min_vals) / (max_vals - min_vals + 1e-8)
    
    # 缩放到0-255范围
    normalized_img = normalized_img * 255.0
    
    return normalized_img.astype(np.uint8)


class EE:
    def __init__(self, img, edge_filter, ee_gain, ee_thres, ee_emclip):
        self.img = img
        self.edge_filter = edge_filter/8
        self.gain = ee_gain
        self.thres = ee_thres
        self.emclip = ee_emclip
    

    def execute(self):        
        em_img = conv3x5(self.img, self.edge_filter)
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