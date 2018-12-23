import SimpleITK as sitk
from tqdm import trange
import torch.multiprocessing as mp
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave



def child_proc(num, i):
    print('Init train child:%d, Proccess PID:%d\n' % (num, os.getpid()))
    f.write('Init train child:%d, Proccess PID:%d\n' % (num, os.getpid()))
    elastixImageFilter = sitk.ElastixImageFilter()
#     elastixImageFilter.SetFixedImage(sitk.ReadImage('data\\aligned_illumination\img0.png'))
#     elastixImageFilter.SetMovingImage(sitk.ReadImage('data\\aligned_illumination\img%d.png' % num))
    
    
    elastixImageFilter.SetFixedImage(sitk.ReadImage('correct_preproc\\blur\\correct_preproc%04d.png' % 0))
    elastixImageFilter.SetMovingImage(sitk.ReadImage('correct_preproc\\blur\\correct_preproc%04d.png' % num))
    elastixImageFilter.SetFixedMask(sitk.ReadImage('correct_preproc\\blur\\mask_v2\\mask_l%04d.png' % 0))
    elastixImageFilter.SetMovingMask(sitk.ReadImage('correct_preproc\\blur\\mask_v2\\mask_l%04d.png' % num))



    elastixImageFilter.PrintParameterMap()
    parameterMapVector = sitk.VectorOfParameterMap()
    affineParameterMap = sitk.GetDefaultParameterMap("affine")
    bsplineParameterMap = sitk.GetDefaultParameterMap("bspline")
    
    affineParameterMap['MaximumNumberOfIterations'] = ['128']
    bsplineParameterMap['MaximumNumberOfIterations'] = ['128']
    parameterMapVector.append(affineParameterMap)
    parameterMapVector.append(bsplineParameterMap)
    elastixImageFilter.SetParameterMap(parameterMapVector)

    elastixImageFilter.Execute()
#     res = elastixImageFilter.GetResultImage()
    transformParameterMap = elastixImageFilter.GetTransformParameterMap()
    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.SetTransformParameterMap(transformParameterMap)
    transformixImageFilter.SetMovingImage(sitk.ReadImage('denoised_imgs\\img%d.png' % num))
    transformixImageFilter.Execute()
    res = transformixImageFilter.GetResultImage()
    
    img_reg = np.zeros((res.GetHeight(),res.GetWidth()))
    for i in range(img_reg.shape[0]):
        for j in range(img_reg.shape[1]):
            img_reg[i,j] = res.GetPixel(j,i)
    imsave('sitkRes_mask_v2\\img%d.png' % num, img_reg / img_reg.max())
    
def main():
    mp.set_start_method('spawn',  force=True)
    wind = 30
    for k in range(0,35):
        processes = []
        end =time.time()
        for i in range(wind*k,wind*(k+1)):
            proc = mp.Process(target=child_proc, args=(i,i))
            proc.start()
            processes.append(proc)

        print('\nCHILD Proccess started\n')
        f.write('\nCHILD Proccess started\n')
        for p in processes:
            p.join()
        print('Time:', time.time() - end)
        f.write('Time:%f'% (time.time() - end))

f = open('log.txt', 'w')
if __name__ == '__main__':
    main()
    f.close()