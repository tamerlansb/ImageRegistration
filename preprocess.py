import matplotlib.pyplot as plt
from skimage.io import imread, imsave
import math
import argparse
import pandas as pd
import numpy as np
from tqdm import trange
import cv2
from os.path import join
import os

def MultiscaleRetinex(image, n = 3, sigma = 1.6):
    result = np.zeros(image.shape)
    for i in range(n):
        curS = (int)(sigma*(i+1)*6);
        s = 3 if curS < 1 else curS if curS % 2 == 1 else curS + 1;
        temp = cv2.GaussianBlur(image,(s, s), sigma*(i+1), sigma*(i+1)) + 0.1
        result = result + np.log(image / temp ) 
    return result / n 

def gamma_correction(img, correction):
    img = (img.astype('float32') - img.min()) / (img.max() - img.min())
    img = cv2.pow(img, correction)
    return np.uint8(img*255)


def preprocess(file_in = None, folder_in = None, folder_out = None):
    if file_in is None and folder_in is None:
        raise ValueError("Input data is None")
    
    align_illum_num_iter = 10 # 3
    start_sigma = 3 #25
    if file_in is not None:
        vidcap = cv2.VideoCapture(file_in)

        success, first_frame = vidcap.read() 
        if not success:
            print('Error read file %s ' % file_in)
            cv2.destroyAllWindows()
            return
        if not os.path.exists(folder_out):
            os.makedirs(folder_out)
            print("dir %s created" % folder_out)
        num = 0;
        first_frame_processed = gamma_correction(MultiscaleRetinex(cv2.cvtColor(first_frame ,cv2.COLOR_BGR2GRAY) + 10 ,
                                                                   align_illum_num_iter, start_sigma),0.65)
        mean_intensity = first_frame_processed.mean()
        var_intensity =  first_frame_processed.std()
        imsave(join(folder_out, 'img%d.png' % num) ,first_frame_processed)

        while True:
            num = num + 1
            success, frame = vidcap.read() 
            if not success:
                break
            frame = gamma_correction(MultiscaleRetinex(cv2.cvtColor(frame ,cv2.COLOR_BGR2GRAY) +10 ,align_illum_num_iter, start_sigma),0.65)
            cur_mean_intensity = frame.mean()
            cur_var_intensity = frame.std()
            frame = np.uint8( (frame -  cur_mean_intensity)*var_intensity/cur_var_intensity + mean_intensity )
            imsave(join(folder_out,'img%d.png') % num, frame )
    else:
        n = len(os.listdir(folder_in))
        first_frame = imread(join(folder_in, "img0.png")).astype("float32")
        first_frame = (first_frame - first_frame.min())/ (first_frame.max()-  first_frame.min())
        first_frame_processed = gamma_correction(MultiscaleRetinex(first_frame + 1,
                                                                   align_illum_num_iter, start_sigma),0.75)
        mean_intensity = first_frame_processed.mean()
        var_intensity =  first_frame_processed.std()
        imsave(join(folder_out, 'img%d.png' % 0) ,first_frame_processed)
        for i in trange(1,n):
            frame = imread(join(folder_in, "img%d.png" % i)).astype("float32")
            frame = (frame - frame.min()) / (frame.max() - frame.min())
            frame = gamma_correction(MultiscaleRetinex(frame +1 ,align_illum_num_iter, start_sigma),0.75)
            cur_mean_intensity = frame.mean()
            cur_var_intensity = frame.std()
            frame = np.uint8((frame -  cur_mean_intensity)*var_intensity/cur_var_intensity + mean_intensity )
            imsave(join(folder_out,'img%d.png') % i, frame )
    print('File %s processed, %d frames, result in %s', (file_in, num,  folder_out))
    return

def main():
    parser = argparse.ArgumentParser(description='preprocessing')
    parser.add_argument('--file-in', type=str, default=None, help=' input file ')
    parser.add_argument('--folder-in', type=str, default=None, help=' input folder with frames ')
    parser.add_argument('--folder-out', type=str, default='', help=' output path dir ')
    args = parser.parse_args()
    
    input_file = args.file_in
    input_dir = args.folder_in
    output_path_dir = args.folder_out
    if input_file is not None:
        preprocess(file_in = input_file, folder_out = output_path_dir)
    elif input_dir  is not None:
        preprocess(folder_in = input_dir, folder_out =  output_path_dir)
    else:
        print("Not input data")
    return

# python preprocess.py --file-in <full-path-to-file> --folder-out <path>
if __name__=="__main__":
    main()