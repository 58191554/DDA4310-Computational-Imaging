import rawpy
import cv2
import argparse
import numpy as np

def raw_processing(input_file, output_file, args):
    with rawpy.imread(input_file) as raw:
        params = {}
        if args.user_black is not None:
            params['user_black'] = args.user_black
            print('Enable user_black=', args.user_black)
        if args.use_auto_wb:
            params['use_auto_wb'] = args.use_auto_wb
            print("Enable use_auto_wb")
        if args.no_auto_scale:
            params['no_auto_scale'] = args.no_auto_scale
            print("Enable no_auto_scale")
        if args.no_auto_bright:
            params['no_auto_bright'] = args.no_auto_bright
            print("Enable no_auto_scale")
        if args.gamma is not None:
            # Set the Gamma Correction parameters
            params['gamma'] = args.gamma
            print("Enable gamma correction with parameters:", args.gamma)
        
        rgb = raw.postprocess(**params)
    
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(output_file, bgr)
    print("raw process complete...")
    
def rgb_to_yuv(input_file, output_file):
    rgb_image = cv2.imread(input_file)
    
    yuv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2YUV)
    
    cv2.imwrite(output_file, yuv_image)

def yuv_to_rgb(input_file, output_file):
    yuv_image = cv2.imread(input_file)
    
    rgb_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)
    
    cv2.imwrite(output_file, rgb_image)
    
def parse_args():
    parser = argparse.ArgumentParser(description='Perform raw processing on ARW files')
    parser.add_argument('input_file', help='Input ARW file')
    parser.add_argument('output_file', help='Output JPEG file')
    parser.add_argument('--task-no', help='Execute task no.')
    parser.add_argument('--user-black', type=int, help='Custom black level')
    parser.add_argument('--use-auto-wb', action='store_true', help='Enable auto white balance')
    parser.add_argument('--no-auto-scale', action='store_true', help='Disable pixel value scaling')
    parser.add_argument('--no-auto-bright', action='store_true', help='Disable automatic increase of brightness')    
    parser.add_argument('--RGB2YUV', action='store_true', help='RGB2YUV')
    parser.add_argument('--YUV2RGB', action='store_true', help='YUV2RGB')
    parser.add_argument('--gamma', type=float, nargs=2, default=None, help='Gamma values (power, slope)')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    print("Task ", args.task_no)
    if int(args.task_no) in range(1, 6):
        raw_processing(args.input_file, args.output_file, args)
    elif int(args.task_no) == 6:
        if args.YUV2RGB:
            yuv_to_rgb(args.input_file, args.output_file)
        elif args.RGB2YUV:
            rgb_to_yuv(args.input_file, args.output_file)
    elif int(args.task_no) == 7:
        raw_processing(args.input_file, args.output_file, args)
    else:
        raw_processing(args.input_file, args.output_file, args)
