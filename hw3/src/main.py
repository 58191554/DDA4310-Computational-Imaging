"""
@author: Zhen Tong
"""
import rawpy
import os
import sys
import time
import argparse
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2 #only for save images to jpg
import exifread
from matplotlib import pyplot as plt #use it only for debug
import numpy as np
from tqdm import tqdm

from isp.BayerDomainProcessor import deadPixelCorrection, blackLevelCompensation, \
    antiAliasingFilter, AWB, CFA_Interpolation

# from isp.RGBDomainProcessor 
# from isp import deadPixelCorrection, 


def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draw-intermediate", type=bool, default=True)
    parser.add_argument("--set-dir", type=str, default="../data/set03")
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--itmd-dir", type=str, default="itmd", help=\
        "intermediate file root directory.")
    args = parser.parse_args()
    assert os.path.exists(args.set_dir)
    if os.path.exists(args.output_dir) == False:
        os.makedirs(args.output_dir)

    args.itmd_dir = os.path.join(args.itmd_dir, args.set_dir.strip("/")[-1])
    if os.path.exists(args.itmd_dir) == False:
        os.makedirs(args.itmd_dir)
    assert os.path.exists(args.output_dir)
    assert os.path.exists(args.itmd_dir)
    return args

if __name__ == "__main__":
    args = get_arg() 

    image_name_list = []
    for file in os.listdir(args.set_dir):
        if os.path.splitext(file)[1]=='.ARW': 
            file_dir = os.path.join(args.set_dir, file)
            image_name_list.append(file_dir)
            assert os.path.exists(file_dir)

    image_name_list.sort()
    image_name_list
    print(image_name_list)

    image_list = []
    exposure_times = []
    for name in image_name_list:
        raw = rawpy.imread(name)
        image_list.append(raw.raw_image)
        raw_file = open(name, 'rb')
        exif_file = exifread.process_file(raw_file, details=False, strict=True)
        ISO_str = exif_file['EXIF ISOSpeedRatings'].printable
        ExposureTime = exif_file['EXIF ExposureTime'].printable
        exposure_times.append(float(ExposureTime.split(
            '/')[0]) / float(ExposureTime.split('/')[1]) * float(ISO_str))
    rawimg = raw.raw_image

    clip_list = []
    for image in image_list:
        clip_list.append(np.clip(image, raw.black_level_per_channel[0], 2**14))
    print(50*'-' + '\nLoading RAW Image Done......')
    
    #####################################  Part 1: Bayer Domain Processing Steps  ###################################
    # Step 1. Dead Pixel Correction (10pts)
    Bayer_dpcs = []
    dpc_thres, dpc_mode, dpc_clip  = 200, 'gradient', raw.camera_white_level_per_channel[0]
    for raw_image in tqdm(clip_list):
        dpc = deadPixelCorrection(raw_image, dpc_thres, dpc_mode, dpc_clip)
        Bayer_dpcs.append(dpc.execute())
    print(50*'-' + '\n 1.1 Dead Pixel Correction Done......')
    if args.draw_intermediate:
        cv2.imwrite(args.itmd_dir+'Dead Pixel Correction.jpg',Bayer_dpcs[0])
    
    # Step 2.'Black Level Compensation' (5pts)
    alpha, beta = 0, 0
    parameter = raw.black_level_per_channel + [alpha, beta]
    Bayer_blackLevelCompensations = []
    for Bayer_dpc in tqdm(Bayer_dpcs):
        blkC = blackLevelCompensation(Bayer_dpc, parameter,"rggb",clip=2**14)
        Bayer_blackLevelCompensations.append(blkC.execute())
    print(50*'-' + '\n 1.2 Black Level Compensation Done......')
    
    # Step 4. Anti Aliasing Filter (10pts)
    Bayer_antiAliasingFilters = []
    for Bayer_blackLevelCompensation in Bayer_blackLevelCompensations:
        aaf = antiAliasingFilter(Bayer_blackLevelCompensation)
        Bayer_antiAliasingFilters.append(aaf.execute())
    print(50*'-' + '\n 1.4 Anti-aliasing Filtering Done......')
    if args.draw_intermediate:
        cv2.imwrite(args.itmd_dir+'Bayer_antiAliasingFilter.jpg',Bayer_antiAliasingFilters[0] )
    
    # Step 5. Auto White Balance and Gain Control (10pts)
    r_gain, gr_gain, gb_gain, b_gain = 0.6, 1.0, 1.0, 1/(2080/1024)
    parameter = [r_gain, gr_gain, gb_gain, b_gain]
    awb_clip = raw.camera_white_level_per_channel[0]
    Bayer_awbs = []
    for Bayer_antiAliasingFilter in Bayer_antiAliasingFilters:
        awb = AWB(Bayer_antiAliasingFilter, parameter, 'rggb', awb_clip)
        Bayer_awbs.append(awb.execute())
    print(50*'-' + '\n 1.5 White Balance Gain Done......')
    if args.draw_intermediate:
        cv2.imwrite(args.itmd_dir+'White Balance Gain.jpg',Bayer_awbs[0])
    
    # Step 6. Raw Exposure Fusion
    # your code here




    # # Step 7. 'Color Filter Array Interpolation'  Malvar (20pts)
    # cfa = CFA_Interpolation(Fusion, cfa_mode, bayer_pattern, cfa_clip)
    # rgbimg_cfa, tone_mapping_factor = cfa.execute()
    # print(tone_mapping_factor)
    # print(50*'-' + '\n 1.7 Demosaicing Done......')
    # if args.draw_intermediate:
    #     cv2.imwrite(args.itmd_dir+'CFA_Interpolation.hdr',cv2.cvtColor(rgbimg_cfa.astype(np.float32),cv2.COLOR_RGB2BGR))
    # #####################################  Bayer Domain Processing end    ###########################################
    # gamma = GammaCorrection(rgbimg_cfa,None,'gamma')
    # gamma_corrected = gamma.execute()
    # if args.draw_intermediate:
    #     cv2.imwrite(args.itmd_dir+'GammaCorrectionWithoutToneMapping.png',cv2.cvtColor(gamma_corrected.astype(np.uint8),cv2.COLOR_RGB2BGR))
    # #Step:  Your local tone mappibng here



    # YUV = RGB2YUV(gamma_corrected)
    # #####################################    Part 2: YUV Domain Processing Steps  ###################################
    # # Step Luma-2  Edge Enhancement  for Luma (20pts)
    # ee = EdgeEnhancement(YUV[:,:,0], edge_filter, ee_gain, ee_thres, ee_emclip)
    # yuvimg_ee,yuvimg_edgemap = ee.execute()
    # print(50*'-' + '\n 3.Luma.2  Edge Enhancement Done......')
    # # Step Luma-3 Brightness/Contrast Control (5pts)
    # contrast = contrast / pow(2,5)    #[-32,128]
    # bcc = BrightnessContrastControl(yuvimg_ee, brightness, contrast, bcc_clip)
    # yuvimg_bcc = bcc.execute()
    # print(50*'-' + '\nBrightness/Contrast Adjustment Done......')
    # # Step Chroma-1 False Color Suppresion (10pts)
    # fcs = FalseColorSuppression(YUV[:,:,1:3], yuvimg_edgemap, fcs_edge, fcs_gain, fcs_intercept, fcs_slope)
    # yuvimg_fcs = fcs.execute()
    # print(50*'-' + '\n 3.Chroma.1 False Color Suppresion Done......')
    # # Step Chroma-2 Hue/Saturation control (10pts)
    # hsc = HueSaturationControl(yuvimg_fcs, hue, saturation, hsc_clip)
    # yuvimg_hsc = hsc.execute()
    # print(50*'-' + '\n 3.Chroma.2  Hue/Saturation Adjustment Done......')
    # # Concate Y UV Channels
    # yuvimg_out = np.zeros_like(YUV) 
    # yuvimg_out[:,:,0] = yuvimg_bcc
    # yuvimg_out[:,:,1:3] = yuvimg_hsc
    # RGB = YUV2RGB(yuvimg_out) # Pay attention to the bits
    # #####################################   End YUV Domain Processing Steps  ###################################
    # result_dir = "result/"
    # cv2.imwrite(result_dir + sub_dir+'.png',cv2.cvtColor(RGB,cv2.COLOR_RGB2BGR)) 