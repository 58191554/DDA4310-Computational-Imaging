import cv2
import numpy as np
from utils import *
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
        g //= 8
        # B at red in R row, R column
        b = 6 * img[y,x] - 3 * (img[y-2,x] + img[y,x-2] + img[y+2,x] + img[y,x+2]) / 2 \
            + 2 * (img[y-1,x-1] + img[y-1,x+1] + img[y+1,x-1] + img[y+1,x+1])
        b //= 8
    elif ctr_name == 'gr':
        g = ctr_val
        # R at green in R row, B column
        r = 5 * img[y,x] - img[y,x-2] - img[y-1,x-1] - img[y+1,x-1] - img[y-1,x+1] - img[y+1,x+1] - img[y,x+2] \
            + (img[y-2,x] + img[y+2,x]) / 2 + 4 * (img[y,x-1] + img[y,x+1])
        r //= 8
        # B at green in R row, B column
        b = 5 * img[y,x] - img[y-2,x] - img[y-1,x-1] - img[y-1,x+1] - img[y+2,x] - img[y+1,x-1] - img[y+1,x+1] \
            + (img[y,x-2] + img[y,x+2]) / 2 + 4 * (img[y-1,x] + img[y+1,x])
        b //= 8
    elif ctr_name == 'gb':
        g = ctr_val
        # R at green in B row, R column
        r = 5 * img[y,x] - img[y-2,x] - img[y-1,x-1] - img[y-1,x+1] - img[y+2,x] - img[y+1,x-1] - img[y+1,x+1] \
            + (img[y,x-2] + img[y,x+2]) / 2 + 4 * (img[y-1,x] + img[y+1,x])
        r //= 8
        # B at green in B row, R column
        b = 5 * img[y,x] - img[y,x-2] - img[y-1,x-1] - img[y+1,x-1] - img[y-1,x+1] - img[y+1,x+1] - img[y,x+2] \
            + (img[y-2,x] + img[y+2,x]) / 2 + 4 * (img[y,x-1] + img[y,x+1])
        b //= 8
    elif ctr_name == 'b':
        b = ctr_val
        # R at blue in B row, B column
        r = 6 * img[y,x] - 3 * (img[y-2,x] + img[y,x-2] + img[y+2,x] + img[y,x+2]) / 2 \
            + 2 * (img[y-1,x-1] + img[y-1,x+1] + img[y+1,x-1] + img[y+1,x+1])
        r //= 8
        # G at blue in B row, B column
        g = 4 * img[y,x] - img[y-2,x] - img[y,x-2] - img[y+2,x] - img[y,x+2] \
            + 2 * (img[y+1,x] + img[y,x+1] + img[y-1,x] + img[y,x-1])
        g //= 8
    return np.stack((r, g, b), axis=-1)
    
class DPC:
    def __init__(self, img, thres, mode, clip):
        self.img = img
        self.thres = thres
        self.mode = mode
        self.clip = clip
    
    def execute(self):
        padded_bayer = np.pad(self.img, ((2, 2), (2, 2)), 'reflect')
        padded_sub_arrays = split_bayer(padded_bayer, 'rggb')
        
        dpc_sub_arrays = []
        for padded_array in padded_sub_arrays:
            shifted_arrays = tuple(shift_array(padded_array, window_size=3))   # generator --> tuple

            mask = (np.abs(shifted_arrays[4] - shifted_arrays[1]) > self.thres) * \
                   (np.abs(shifted_arrays[4] - shifted_arrays[7]) > self.thres) * \
                   (np.abs(shifted_arrays[4] - shifted_arrays[3]) > self.thres) * \
                   (np.abs(shifted_arrays[4] - shifted_arrays[5]) > self.thres) * \
                   (np.abs(shifted_arrays[4] - shifted_arrays[0]) > self.thres) * \
                   (np.abs(shifted_arrays[4] - shifted_arrays[2]) > self.thres) * \
                   (np.abs(shifted_arrays[4] - shifted_arrays[6]) > self.thres) * \
                   (np.abs(shifted_arrays[4] - shifted_arrays[8]) > self.thres)

            dv = np.abs(2 * shifted_arrays[4] - shifted_arrays[1] - shifted_arrays[7])
            dh = np.abs(2 * shifted_arrays[4] - shifted_arrays[3] - shifted_arrays[5])
            ddl = np.abs(2 * shifted_arrays[4] - shifted_arrays[0] - shifted_arrays[8])
            ddr = np.abs(2 * shifted_arrays[4] - shifted_arrays[6] - shifted_arrays[2])
            indices = np.argmin(np.dstack([dv, dh, ddl, ddr]), axis=2)[..., None]

            neighbor_stack = np.right_shift(np.dstack([
                shifted_arrays[1] + shifted_arrays[7],
                shifted_arrays[3] + shifted_arrays[5],
                shifted_arrays[0] + shifted_arrays[8],
                shifted_arrays[6] + shifted_arrays[2]
            ]), 1)
            dpc_array = np.take_along_axis(neighbor_stack, indices, axis=2).squeeze(2)
            dpc_sub_arrays.append(
                mask * dpc_array + ~mask * shifted_arrays[4]
            )

        dpc_bayer = reconstruct_bayer(dpc_sub_arrays, 'rggb')

        dpc_bayer = dpc_bayer.astype(np.uint16)
        return dpc_bayer

    
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
        bayer = self.img.astype(np.uint32)

        padded_bayer = np.pad(bayer, ((2, 2), (2, 2)), 'reflect')

        padded_sub_arrays = split_bayer(padded_bayer, 'rggb')

        aaf_sub_arrays = []
        for padded_array in padded_sub_arrays:
            shifted_arrays = shift_array(padded_array, window_size=3)
            aaf_sub_array = 0
            for i, shifted_array in enumerate(shifted_arrays):
                mul = 8 if i == 4 else 1
                aaf_sub_array += mul * shifted_array

            aaf_sub_arrays.append(np.right_shift(aaf_sub_array, 4))

        aaf_bayer = reconstruct_bayer(aaf_sub_arrays, 'rggb')

        aaf_bayer = aaf_bayer.astype(np.uint16)
        return aaf_bayer

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
        self.channel_indices_and_weights = (
            [[1, 3, 4, 5, 7], [-0.125, -0.125, 0.5, -0.125, -0.125]],
            [[1, 3, 4, 5, 7], [-0.1875, -0.1875, 0.75, -0.1875, -0.1875]],
            [[1, 3, 4, 5, 7], [0.0625, -0.125, 0.625, -0.125, 0.0625]],
            [[1, 3, 4, 5, 7], [-0.125, 0.0625, 0.625, 0.0625, -0.125]],
            [[3, 4], [0.25, 0.25]],
            [[1, 4], [0.25, 0.25]],
            [[4, 7], [0.25, 0.25]],
            [[4, 5], [0.25, 0.25]],
            [[4, 5], [0.5, 0.5]],
            [[1, 3], [0.5, 0.5]],
            [[1, 4], [0.5, 0.5]],
            [[4, 7], [0.5, 0.5]],
            [[3, 4], [0.5, 0.5]],
            [[0, 1, 3, 4], [0.25, 0.25, 0.25, 0.25]],
            [[4, 5, 7, 8], [0.25, 0.25, 0.25, 0.25]],
            [[1, 2, 4, 5], [-0.125, -0.125, -0.125, -0.125]],
            [[3, 4, 6, 7], [-0.125, -0.125, -0.125, -0.125]]
        )

    def execute(self):
        # img_pad = np.pad(self.img, ((2, 2), (2, 2)), 'reflect')
        # img_pad = img_pad.astype(np.int32)
        # raw_h = self.img.shape[0]
        # raw_w = self.img.shape[1]
        # cfa_img = np.empty((raw_h, raw_w, 3), np.int16)        
        
        # for y in tqdm(range(0, img_pad.shape[0]-4-1, 2)):
        #     for x in range(0, img_pad.shape[1]-4-1, 2):
        #         # Default as rggb malvar
        #         r = img_pad[y+2,x+2]
        #         gr = img_pad[y+2,x+3]
        #         gb = img_pad[y+3,x+2]
        #         b = img_pad[y+3,x+3]        
                
        #         cfa_img[y,x,:] = malvar('r', r, y+2,x+2, img_pad)
        #         cfa_img[y,x+1,:] = malvar('gr', gr, y+2,x+3, img_pad)
        #         cfa_img[y+1,x,:] = malvar('gb', gb, y+3,x+2, img_pad)
        #         cfa_img[y+1,x+1,:] = malvar('b', b, y+3,x+3, img_pad)
        # np.clip(cfa_img, 0, self.clip, out=cfa_img)
        # return cfa_img

        img_pad = np.pad(self.img, ((2, 2), (2, 2)), 'reflect')
        img_pad = img_pad.astype(np.int32)
        raw_h = self.img.shape[0]
        raw_w = self.img.shape[1]
        cfa_img = np.empty((raw_h, raw_w, 3), np.int16)
        
        # Generate indices for accessing elements
        y_indices, x_indices = np.meshgrid(np.arange(0, img_pad.shape[0]-4-1, 2), np.arange(0, img_pad.shape[1]-4-1, 2))
        # Default as rggb malvar
        r = img_pad[y_indices + 2, x_indices + 2]
        gr = img_pad[y_indices + 2, x_indices + 3]
        gb = img_pad[y_indices + 3, x_indices + 2]
        b = img_pad[y_indices + 3, x_indices + 3]
        
        cfa_img[y_indices, x_indices, :] = malvar('r', r, y_indices + 2, x_indices + 2, img_pad)
        cfa_img[y_indices, x_indices + 1, :] = malvar('gr', gr, y_indices + 2, x_indices + 3, img_pad)
        cfa_img[y_indices + 1, x_indices, :] = malvar('gb', gb, y_indices + 3, x_indices + 2, img_pad)
        cfa_img[y_indices + 1, x_indices + 1, :] = malvar('b', b, y_indices + 3, x_indices + 3, img_pad)
        
        np.clip(cfa_img, 0, self.clip, out=cfa_img)
        return cfa_img
    

if __name__ == "__main__":
    
    
    pass