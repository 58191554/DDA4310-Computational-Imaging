import cv2
import numpy as np
from tqdm import tqdm

"""
input a raw image, the threshold, the dpc mode, and the dpc clip range
output a dpc Object that can execute the  
"""
def deadPixelCorrection(rawimg:np.ndarray, dpc_thres:int, dpc_mode:str, dpc_clip:int):
    return DPC(rawimg, dpc_thres, dpc_mode, dpc_clip)

def blackLevelCompensation(Bayer_dpc:np.ndarray, parameter:list, bayer_pattern:str, clip:int):
    return BLC(Bayer_dpc, parameter, bayer_pattern, clip)

def antiAliasingFilter(Bayer_blackLevelCompensation:np.ndarray):
    return AAF(Bayer_blackLevelCompensation)
    
def ChromaNoiseFiltering(Bayer_awb, bayer_pattern, thres, parameter, cfa_clip):
    return CNF(Bayer_awb, bayer_pattern, 0, parameter, cfa_clip)

def CFA_Interpolation(img:np.ndarray, cfa_mode:str, bayer_pattern:str, cfa_clip:int):
    return CFA(img, cfa_mode, bayer_pattern, cfa_clip)

def malvar(ctr_name, ctr_val, y, x, img):
    if ctr_name == 'r':
        r = ctr_val
        # G at R location
        g = 4 * img[y,x] - img[y-2,x] - img[y,x-2] - img[y+2,x] - img[y,x+2] \
            + 2 * (img[y+1,x] + img[y,x+1] + img[y-1,x] + img[y,x-1])
        g /= 8
        # B at red in R row, R column
        b = 6 * img[y,x] - 3 * (img[y-2,x] + img[y,x-2] + img[y+2,x] + img[y,x+2]) / 2 \
            + 2 * (img[y-1,x-1] + img[y-1,x+1] + img[y+1,x-1] + img[y+1,x+1])
        b /= 8
    elif ctr_name == 'gr':
        g = ctr_val
        # R at green in R row, B column
        r = 5 * img[y,x] - img[y,x-2] - img[y-1,x-1] - img[y+1,x-1] - img[y-1,x+1] - img[y+1,x+1] - img[y,x+2] \
            + (img[y-2,x] + img[y+2,x]) / 2 + 4 * (img[y,x-1] + img[y,x+1])
        r /= 8
        # B at green in R row, B column
        b = 5 * img[y,x] - img[y-2,x] - img[y-1,x-1] - img[y-1,x+1] - img[y+2,x] - img[y+1,x-1] - img[y+1,x+1] \
            + (img[y,x-2] + img[y,x+2]) / 2 + 4 * (img[y-1,x] + img[y+1,x])
        b /= 8
    elif ctr_name == 'gb':
        g = ctr_val
        # R at green in B row, R column
        r = 5 * img[y,x] - img[y-2,x] - img[y-1,x-1] - img[y-1,x+1] - img[y+2,x] - img[y+1,x-1] - img[y+1,x+1] \
            + (img[y,x-2] + img[y,x+2]) / 2 + 4 * (img[y-1,x] + img[y+1,x])
        r /= 8
        # B at green in B row, R column
        b = 5 * img[y,x] - img[y,x-2] - img[y-1,x-1] - img[y+1,x-1] - img[y-1,x+1] - img[y+1,x+1] - img[y,x+2] \
            + (img[y-2,x] + img[y+2,x]) / 2 + 4 * (img[y,x-1] + img[y,x+1])
        b /= 8
    elif ctr_name == 'b':
        b = ctr_val
        # R at blue in B row, B column
        r = 6 * img[y,x] - 3 * (img[y-2,x] + img[y,x-2] + img[y+2,x] + img[y,x+2]) / 2 \
            + 2 * (img[y-1,x-1] + img[y-1,x+1] + img[y+1,x-1] + img[y+1,x+1])
        r /= 8
        # G at blue in B row, B column
        g = 4 * img[y,x] - img[y-2,x] - img[y,x-2] - img[y+2,x] - img[y,x+2] \
            + 2 * (img[y+1,x] + img[y,x+1] + img[y-1,x] + img[y,x-1])
        g /= 8
    return [r, g, b]
    
class DPC:
    def __init__(self, img, thres, mode, clip):
        self.img = img
        self.thres = thres
        self.mode = mode
        self.clip = clip
    
    def execute(self):
        
        img_pad = np.pad(self.img, (2, 2), "reflect")
        h, w = img_pad.shape[:2]
        dpc_img = np.empty((h, w), np.uint64)
        for y in tqdm(range(h - 4)):
            for x in range(w - 4):
                """
                p1 p2 p3
                p4 p0 p5
                p6 p7 p8
                """
                p0 = img_pad[y + 2, x + 2]
                p1 = img_pad[y, x]
                p2 = img_pad[y, x + 2]
                p3 = img_pad[y, x + 4]
                p4 = img_pad[y + 2, x]
                p5 = img_pad[y + 2, x + 4]
                p6 = img_pad[y + 4, x]
                p7 = img_pad[y + 4, x + 2]
                p8 = img_pad[y + 4, x + 4]
                
                try:
                    abc = (abs(p1 - p0) > self.thres) and \
                          (abs(p2 - p0) > self.thres) and \
                          (abs(p3 - p0) > self.thres) and \
                          (abs(p4 - p0) > self.thres) and \
                          (abs(p5 - p0) > self.thres) and \
                          (abs(p6 - p0) > self.thres) and \
                          (abs(p7 - p0) > self.thres) and \
                          (abs(p8 - p0) > self.thres)
                except RuntimeWarning:
                    abc = True
                
                if abc:
                    if self.mode == 'gradient':
                        dv = abs(2 * p0 - p2 - p7)
                        dh = abs(2 * p0 - p4 - p5)
                        ddl = abs(2 * p0 - p1 - p8)
                        ddr = abs(2 * p0 - p3 - p6)
                        min_val = min(dv, dh, ddl, ddr)
                        
                        if min_val == dv:
                            p0 = (p2 + p7 + 1) // 2
                        elif min_val == dh:
                            p0 = (p4 + p5 + 1) // 2
                        elif min_val == ddl:
                            p0 = (p1 + p8 + 1) // 2
                        else:
                            p0 = (p3 + p6 + 1) // 2
                
                dpc_img[y, x] = p0
        
        np.clip(dpc_img, 0, self.clip, out=dpc_img)
        return dpc_img    
    
class BLC:
    def __init__(self, img, parameter, bayer_pattern, clip):
        self.img = img
        self.parameter = parameter
        self.bayer_pattern = bayer_pattern
        self.clip = clip
        
    def execute(self):
        bl_r = self.parameter[0]
        bl_gr = self.parameter[1]
        bl_gb = self.parameter[2]
        bl_b = self.parameter[3]
        alpha = self.parameter[4]
        beta = self.parameter[5]
        raw_h = self.img.shape[0]
        raw_w = self.img.shape[1]
        blc_img = np.empty((raw_h,raw_w), np.int16)
        if self.bayer_pattern == 'rggb':
            r = self.img[::2, ::2] + bl_r
            b = self.img[1::2, 1::2] + bl_b
            gr = self.img[::2, 1::2] + bl_gr + alpha * r / 256
            gb = self.img[1::2, ::2] + bl_gb + beta * b / 256
            blc_img[::2, ::2] = r
            blc_img[::2, 1::2] = gr
            blc_img[1::2, ::2] = gb
            blc_img[1::2, 1::2] = b
        np.clip(blc_img, 0, self.clip, out=blc_img)
        return blc_img
    
class AAF:
    def __init__(self, img, \
        kernel = np.array([
            [1, 0, 1, 0, 1],
            [0, 0, 0, 0, 0],
            [1, 0, 8, 0, 1],
            [0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1]])/16
        ):
        self.img = img
        self.kernel = kernel
                      
    def execute(self):
        img_pad = np.pad(self.img, ((2, 2), (2, 2)), 'reflect')
        kernel_height, kernel_width = self.kernel.shape
        img_height, img_width = self.img.shape
        result_height = img_height - kernel_height + 1
        result_width = img_width - kernel_width + 1
        aaf_img = np.zeros((result_height, result_width))
        
        # convolve
        for y in tqdm(range(result_height)):
            for x in range(result_width):
                patch = img_pad[y:y+kernel_height, x:x+kernel_width]
                aaf_img[y, x] = np.sum(patch * self.kernel)
        
        return aaf_img    
    
class AWB:
    def __init__(self, img, parameter, bayer_pattern, awb_clip):
        self.img = img
        self.parameter = parameter
        self.bayer_pattern = bayer_pattern
        self.clip = awb_clip
        
    def execute(self):
        r_gain = self.parameter[0]
        gr_gain = self.parameter[1]
        gb_gain = self.parameter[2]
        b_gain = self.parameter[3]
        raw_h = self.img.shape[0]
        raw_w = self.img.shape[1]
        awb_img = np.empty((raw_h, raw_w), np.int16)
        if self.bayer_pattern == 'rggb':
            r = self.img[::2, ::2] * r_gain
            b = self.img[1::2, 1::2] * b_gain
            gr = self.img[::2, 1::2] * gr_gain
            gb = self.img[1::2, ::2] * gb_gain
            awb_img[::2, ::2] = r
            awb_img[::2, 1::2] = gr
            awb_img[1::2, ::2] = gb
            awb_img[1::2, 1::2] = b
        np.clip(awb_img, 0, self.clip, out=awb_img)
        return awb_img
    
class CNF:
    def __init__(self, img, bayer_pattern, thres, gain, clip):
        self.img = img
        self.bayer_pattern = bayer_pattern
        self.thres = thres
        self.gain = gain
        self.clip = clip

    def execute(self):
        pass

class CFA:
    def __init__(self, img, cfa_mode, bayer_pattern, cfa_clip):
        self.img = img
        self.mode = cfa_mode
        self.bayer_pattern = bayer_pattern
        self.clip = cfa_clip
        
    def execute(self):
        img_pad = np.pad(self.img, ((2, 2), (2, 2)), 'reflect')
        img_pad = img_pad.astype(np.int32)
        raw_h = self.img.shape[0]
        raw_w = self.img.shape[1]
        cfa_img = np.empty((raw_h, raw_w, 3), np.int16)        
        
        for y in tqdm(range(0, img_pad.shape[0]-4-1, 2)):
            for x in range(0, img_pad.shape[1]-4-1, 2):
                # Default as rggb malvar
                r = img_pad[y+2,x+2]
                gr = img_pad[y+2,x+3]
                gb = img_pad[y+3,x+2]
                b = img_pad[y+3,x+3]        
                
                cfa_img[y,x,:] = malvar('r', r, y+2,x+2, img_pad)
                cfa_img[y,x+1,:] = malvar('gr', gr, y+2,x+3, img_pad)
                cfa_img[y+1,x,:] = malvar('gb', gb, y+3,x+2, img_pad)
                cfa_img[y+1,x+1,:] = malvar('b', b, y+3,x+3, img_pad)
        np.clip(cfa_img, 0, self.clip, out=cfa_img)
        return cfa_img
                
if __name__ == "__main__":
    
    
    pass