import argparse
import math
import cv2
import os
from os.path import join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
import SimpleITK as sitk
import torch.multiprocessing as mp
from tqdm import trange


def MultiscaleRetinex(image, n = 3, sigma = 1.6):
    result = np.zeros(image.shape)
    for i in range(n):
        curS = (int)(sigma*(i+1)*6);
        s = 3 if curS < 1 else curS if curS % 2 == 1 else curS + 1;
        temp = cv2.GaussianBlur(image,(s, s), sigma, sigma) + 0.1
        result = result + np.log(image / temp ) 
    return result / n 

def gamma_correction(img, correction):
    img = (img.astype('float32') - img.min()) / (img.max() - img.min())
    img = cv2.pow(img, correction)
    return np.uint8(img*255)


def preprocess(file_in, folder_out):
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
    first_frame_processed = gamma_correction(MultiscaleRetinex(cv2.cvtColor(first_frame ,cv2.COLOR_BGR2GRAY) + 10 ,3, 25),0.65)
    mean_intensity = first_frame_processed.mean()
    var_intensity =  first_frame_processed.std()
    imsave(join(folder_out, 'img%d.png' % num) ,first_frame_processed)
    
    while True:
        num = num + 1
        success, frame = vidcap.read() 
        if not success:
            break
        frame = gamma_correction(MultiscaleRetinex(cv2.cvtColor(frame ,cv2.COLOR_BGR2GRAY) +10 ,3, 25),0.65)
        cur_mean_intensity = frame.mean()
        cur_var_intensity = frame.std()
        frame = np.uint8( (frame -  cur_mean_intensity)*var_intensity/cur_var_intensity + mean_intensity )
        imsave(join(folder_out,'img%d.png') % num, frame )
    print('File %s processed, %d frames, result in %s', (file_in, num,  folder_out))
    return

def child_proc(num, i):
    print('Init train child:%d, Proccess PID:%d\n' % (num, os.getpid()))
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(sitk.ReadImage('data\\aligned_illumination\img0.png'))
    elastixImageFilter.SetMovingImage(sitk.ReadImage('data\\aligned_illumination\img%d.png' % num))

    parameterMapVector = sitk.VectorOfParameterMap()
    parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))
    parameterMapVector.append(sitk.GetDefaultParameterMap("bspline"))
    elastixImageFilter.SetParameterMap(parameterMapVector)

    elastixImageFilter.Execute()
    res = elastixImageFilter.GetResultImage()
    img_reg = np.zeros((res.GetHeight(),res.GetWidth()))
    for i in range(img_reg.shape[0]):
        for j in range(img_reg.shape[1]):
            img_reg[i,j] = res.GetPixel(j,i)
    imsave('sitkRes\\img%d.png' % num, img_reg / img_reg.max())

def registr(from_file,file, input_path, output):
    mp.set_start_method('spawn',  force=True)
    num_workers = 5 
    if not from_file:
        frame_count = len(os.listdir(input_path))
        for k in range(frame_count // num_workers + 1):
            processes = []
            end =time.time()
            for i in range(num_workers*k,num_workers*(k+1)):
                proc = mp.Process(target=child_proc, args=(i,i))
                proc.start()
                processes.append(proc)

            print('\nCHILD Proccess started\n')
            for p in processes:
                p.join()
            print('Time:', time.time() - end)
    else:
        

def main():
    parser = argparse.ArgumentParser(description='preprocessing') 
    parser.add_argument('--file-in', type=str, default='', help=' input file process')
    parser.add_argument('--folder-in', type=str, default='', help=' input from folder if file not exist')
    parser.add_argument('--folder-out', type=str, default='', help=' output path dir ')
    parser.add_argument('--N-frame', type=int, default=-1, help = 'count frame process')
    args = parser.parse_args()
    
    input_file = args.file_in
    output_path_dir = args.folder_out
    preprocess(input_file, output_path_dir)
    return

# python preprocess.py --file-in <full-path-to-file> --folder-out <path>
if __name__=="__main__":
    main()
   
    