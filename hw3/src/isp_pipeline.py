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
# from isp import deadPixelCorrection, 


def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draw-intermediate", type=bool, default=True)
    parser.add_argument("--set-dir", type=str, default="../../data/set03")
    parser.add_argument("--output-dir", type=str, default="../data")
    parser.add_argument("--itmd_dir", type=str, default="intermediate_files", help=\
        "intermediate file directory.")
    args = parser.parse_args()
    assert os.path.exists(args.set_dir)
    if os.path.exists(args.output_dir) == False:
        os.makedirs(args.output_dir)
    if os.path.exists(args.itmd_dir) == False:
        os.makedirs(args.itmd_dir)
    return args

if __name__ == "__main__":
    args = get_arg() 

    image_name_list = []
    for file in os.listdir(dir):
        if os.path.splitext(file)[1]=='.ARW': 
            image_name_list.append(dir+file)

    image_name_list.sort()
    image_name_list
    # file_name = image_name_list[4]

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
    # # Step 1. Dead Pixel Correction (10pts)
    # Bayer_dpcs = []
    # for raw_image in clip_list:
    #     dpc = deadPixelCorrection(rawimg, dpc_thres, dpc_mode, dpc_clip)
    #     Bayer_dpcs.append(dpc.execute())
    # print(50*'-' + '\n 1.1 Dead Pixel Correction Done......')
    # if draw_intermediate:
    #     cv2.imwrite(intermediate_files_folder+'Dead Pixel Correction.jpg',Bayer_dpcs[0])
    # # Step 2.'Black Level Compensation' (5pts)
    # parameter = raw.black_level_per_channel + [alpha, beta]
    # Bayer_blackLevelCompensations = []
    # for Bayer_dpc in Bayer_dpcs:
    #     blkC = blackLevelCompensation(Bayer_dpc, parameter,2**14, bayer_pattern)
    #     Bayer_blackLevelCompensations.append(blkC.execute())
    # print(50*'-' + '\n 1.2 Black Level Compensation Done......')
    # # Step 4. Anti Aliasing Filter (10pts)
    # Bayer_antiAliasingFilters = []
    # for Bayer_blackLevelCompensation in Bayer_blackLevelCompensations:
    #     antiAliasingFilter = AntiAliasingFilter(Bayer_blackLevelCompensation)
    #     Bayer_antiAliasingFilters.append(antiAliasingFilter.execute())
    # print(50*'-' + '\n 1.4 Anti-aliasing Filtering Done......')
    # if draw_intermediate:
    #     cv2.imwrite(intermediate_files_folder+'Bayer_antiAliasingFilter.jpg',Bayer_antiAliasingFilters[0] )
    # # Step 5. Auto White Balance and Gain Control (10pts)
    # parameter = [r_gain, gr_gain, gb_gain, b_gain]
    # Bayer_awbs = []
    # for Bayer_antiAliasingFilter in Bayer_antiAliasingFilters:
    #     awb = AWB(Bayer_antiAliasingFilter, parameter, bayer_pattern, awb_clip)
    #     Bayer_awbs.append(awb.execute())
    # print(50*'-' + '\n 1.5 White Balance Gain Done......')
    # if draw_intermediate:
    #     cv2.imwrite(intermediate_files_folder+'White Balance Gain.jpg',Bayer_awbs[0])
    # # Step 6. Raw Exposure Fusion
    # # your code here

    # # Step 7. 'Color Filter Array Interpolation'  Malvar (20pts)
    # cfa = CFA_Interpolation(Fusion, cfa_mode, bayer_pattern, cfa_clip)
    # rgbimg_cfa, tone_mapping_factor = cfa.execute()
    # print(tone_mapping_factor)
    # print(50*'-' + '\n 1.7 Demosaicing Done......')
    # if draw_intermediate:
    #     cv2.imwrite(intermediate_files_folder+'CFA_Interpolation.hdr',cv2.cvtColor(rgbimg_cfa.astype(np.float32),cv2.COLOR_RGB2BGR))
    # #####################################  Bayer Domain Processing end    ###########################################
    # gamma = GammaCorrection(rgbimg_cfa,None,'gamma')
    # gamma_corrected = gamma.execute()
    # if draw_intermediate:
    #     cv2.imwrite(intermediate_files_folder+'GammaCorrectionWithoutToneMapping.png',cv2.cvtColor(gamma_corrected.astype(np.uint8),cv2.COLOR_RGB2BGR))
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