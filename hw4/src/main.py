import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

import os

from scipy.sparse import csr_matrix, lil_matrix, linalg
from copy import deepcopy

import argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

import os
Lap3x3_1 = np.array([
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0]])
Lap3x3_2 = np.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]])
Lap3x3_3 = np.array([[1, -2, 1],
                    [-2, 4, -2],
                    [1, -2, 1]])
Lap5x5 = np.array([[0, 0, -1, 0, 0],
                   [0, -1, -2, -1, 0],
                   [-1, -2, 17, -2, -1],
                   [0, -1, -2, -1, 0],
                   [0, 0, -1, 0, 0]])

_show_detail = False

def show_freq_rgb(X, title = ""):
    plt.figure(figsize=(15, 5))
    plt.title(title)
    plt.subplot(131)
    plt.imshow(np.log(1 + np.abs(X[:, :, 0])), cmap='gray')
    plt.title('Red Channel Magnitude Spectrum')
    plt.colorbar()

    plt.subplot(132)
    plt.imshow(np.log(1 + np.abs(X[:, :, 1])), cmap='gray')
    plt.title('Green Channel Magnitude Spectrum')
    plt.colorbar()

    plt.subplot(133)
    plt.imshow(np.log(1 + np.abs(X[:, :, 2])), cmap='gray')
    plt.title('Red Channel Magnitude Spectrum')
    plt.colorbar()

    plt.show()

def show_freq(X, title="", show=True):
    img = np.log(1 + np.abs(X[:, :]))
    if show:
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.colorbar()
    else:
        return img.astype(np.float64)

def get_denoise(B, H, L, lbd, N = None, epsilon = 0.0001):
    X = np.zeros_like(B)
    H_2 = np.square(H)
    H_conj = np.conj(H)
    L_2 = np.square(L)
    for i in range(3):
        B_subChannle = B[:, :, i]
        if N is not None:
            X[:, :, i] = (H_conj*(B_subChannle+N[:, :, i])) / (H_2 + lbd*L_2 + epsilon) 
        else:
            X[:, :, i] = (H_conj*B_subChannle) / (H_2 + lbd*L_2 + epsilon) 
    return X

def normalize(X):
    X = np.clip(X, 0, 255, out=X)
    X_min = np.min(X)
    X_max = np.max(X)
    Y = (X-X_min)/(X_max-X_min)
    return Y

def normalize_rgb(X):
    Y = np.zeros_like(X)
    for i in range(3):
        Y[:, :, i] = X[:, :, i]
        Y = normalize(Y)*255
    return Y.astype(int)

def patchify(img:np.ndarray, p_size, m_size):
    """
    input the image, output the patches image with overlap margin
    """
    h, w, _ = img.shape
    x_pos = [i for i in range(0, h, p_size)]
    y_pos = [i for i in range(0, w, p_size)]

    patch_grid = []
    label_grid = []
    for x in x_pos:
        row = []
        row_lable = []
        for y in y_pos:
            patch_x0 = x-m_size
            patch_x1 = x+p_size+m_size
            patch_y0 = y-m_size
            patch_y1 = y+p_size+m_size
            ele_label = [True, True, True, True]
            if x-m_size < 0:
                patch_x0 = 0
                ele_label[0] = False
            if x+p_size+m_size > h:
                patch_x1 = h
                ele_label[1] = False
            if y-m_size < 0:
                patch_y0 = 0
                ele_label[2] = False
            if y+p_size+m_size > w:
                patch_y1 = w
                ele_label[3] = False
            row.append(img[patch_x0:patch_x1, patch_y0:patch_y1])
            row_lable.append(ele_label)
        patch_grid.append(row)
        label_grid.append(row_lable)
    return patch_grid, label_grid

def resize_image(image, target_size):
    if image.shape == target_size:
        return image.astype(np.int64)
    else:
        target_img = np.zeros(target_size)
        target_img[:image.shape[0], :image.shape[1], :] = image
        return target_img.astype(np.int64)

def plot_images(image_array, rows, cols, p_size, m_size):
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    target_size = (p_size+2*m_size, p_size+2*m_size, 3)
    
    for i in range(rows):
        for j in range(cols):

            axes[i, j].imshow(resize_image(image_array[i][j], target_size))
            axes[i, j].axis('off')

    plt.show()
    

def get_gaussian_kernel(kx ,ky, sigma = 3):
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(kx-1)/2)**2 + (y-(ky-1)/2)**2)/(2*sigma**2)), (kx, ky))
    return kernel / np.sum(kernel)

def get_laplacian_kernel(k_size):
    kernel = np.zeros((k_size, k_size))
    return kernel

def sysmetrix_shift(X:np.ndarray, rate = 0.1):
    Y = np.zeros_like(X)
    h = X.shape[0]; w = X.shape[1]
    rh, rw = int(rate*h), int(rate*w)
    fh, fw = h-rh, w-rw
    Y[:fh, :fw] = X[rh:, rw:]
    Y[:fh, fw:] = X[rh:, :rw]
    Y[fh:, :fw] = X[:rh, rw:]
    Y[fh:, fw:] = X[:rh, :rw]
    return Y

def add_gaussian_noise(img:np.ndarray, noise_add = True, k_size = 17, sigma0 = 3, \
                        sigma_r = 0.01, sigma_g = 0.03, sigma_b = 0.1):
    
    kernel = get_gaussian_kernel(img.shape[0], img.shape[1], sigma0)
    b_x = np.zeros_like(img).astype(np.float64)

    kernel_freq = np.fft.fft2(kernel, s=(img.shape[0], img.shape[1]))

    for i in range(3):
        b_x[:, :, i] = np.fft.ifft2(kernel_freq*np.fft.fft2(img[:, :, i]))
    b_x = sysmetrix_shift(b_x, rate = 0.5)

    h, w, c = img.shape    
    noise = np.zeros_like(img)
    if noise_add:
        noise[:, :, 0] = np.random.normal(0, sigma_r, (h, w))
        noise[:, :, 1] = np.random.normal(0, sigma_g, (h, w))
        noise[:, :, 2] = np.random.normal(0, sigma_b, (h, w))

    b_x += noise
    return b_x.astype(np.int32), kernel, noise
    
def get_spatial_domain_rbg(X, img_size):
    x = np.zeros(img_size)
    for i in range(3):
        x[:, :, i] = np.abs(np.fft.ifft2(X[:, :, i]))   
    x = sysmetrix_shift(x, 0.5)
    return x.astype(np.float64)

def get_frequency_domain_rgb(img, f_size, shift = False):
    if shift:
        X = [np.fft.fftshift(np.fft.fft2(img[:, :, i], s=f_size)) for i in range(3)]
    else:
        X = [np.fft.fft2(img[:, :, i], s=f_size) for i in range(3)]
    # TODO merge the X into a three channel rgb matrix in frquency domain
    merged_real = cv2.merge([np.real(channel) for channel in X])
    merged_imag = cv2.merge([np.imag(channel) for channel in X])
    merged_complex = merged_real + 1j * merged_imag
    return merged_complex

def get_pad_deblur(blur_list, pad_img, pad_mask_cal, LBD = 0.01):

    deblur_list = []

    pad_deblur_img = np.zeros_like(pad_img, dtype=np.float64)
    for i, row in tqdm(enumerate(blur_list)):
        deblur_row = []
        for j, blur in enumerate(row):
            # Compute deblur
            f_size = (blur.shape[0], blur.shape[1])
            B = get_frequency_domain_rgb(blur, f_size)
            G_matrix = np.fft.fft2(get_gaussian_kernel(blur.shape[0], blur.shape[1]), s=f_size)
            L_matrix = np.fft.fft2(Lap3x3_2, s=f_size)
            X = get_denoise(B, G_matrix, L_matrix, LBD)
            deblur = get_spatial_domain_rbg(X, patches[i][j].shape)
            deblur_row.append(deblur)

            # add it in the result
            label = label_list[i][j]
            x0, y0 = i*p_size, j*p_size
            if label[0]:
                x0-=m_size
            if label[2]:
                y0-=m_size
            x1 = x0+deblur.shape[0]
            y1 = y0+deblur.shape[1]
            pad_deblur_img[x0+pad_size:x1+pad_size, y0+pad_size:y1+pad_size] += deblur

        deblur_list.append(deblur_row)

    return pad_deblur_img, deblur_list

def seamlessClone(src,dst,mask,flag):

    laplacian = cv2.Laplacian(np.float64(src),ddepth=-1,ksize=1)
    if flag == 1:
        kernel = [np.array([[0, -1, 1]]),np.array([[1, -1, 0]]), np.array([[0], [-1], [1]]), np.array([[1], [-1], [0]])] 
        grads = []
        for i in range(4):
            srcGrad = cv2.filter2D(np.float64(src), -1, kernel[i])
            dstGrad = cv2.filter2D(np.float64(dst), -1, kernel[i])
            grads.append(np.where(np.abs(srcGrad) > np.abs(dstGrad), srcGrad, dstGrad)) 
        laplacian = np.sum(grads, axis=0) 
    result = [getX(a, mask, b) for a, b in zip(cv2.split(dst), cv2.split(laplacian))]
    final = cv2.merge(result)
    return final


def getX(dst, mask, lap):

    loc = np.nonzero(mask)
    num = loc[0].shape[0] 
    A = lil_matrix((num, num), dtype=np.float64)
    b = np.ndarray((num, ), dtype=np.float64) 
    hhash = {(x, y): i for i, (x, y) in enumerate(zip(loc[0], loc[1]))}

    dx = [1,0,-1,0]
    dy = [0,1,0,-1]

    for i, (x, y) in enumerate(zip(loc[0],loc[1])):
        A[i, i] = -4
        b[i] = lap[x, y]
        p = [(x + dx[j], y + dy[j]) for j in range(4)] 
        for j in range(4):
            if p[j] in hhash: 
                A[i, hhash[p[j]]] = 1
            else:
                b[i] -= dst[p[j]]
    print("="*15+" Solving Linear Equation Ax=b... "+"="*15)
    A = A.tocsc()
    X = linalg.splu(A).solve(b)

    result = np.copy(dst)
    for i, (x, y) in enumerate(zip(loc[0],loc[1])):
        if X[i] < 0:
            X[i] = 0
        if X[i] > 255:
            X[i] = 255
        result[x, y] = X[i]

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-path", type=str, default="Apollo_15_flag_rover_LM_Irwin-1200x1200.jpg")
    parser.add_argument("--m-size", type=int, default=4)
    parser.add_argument("--p-size", type=int, default=100)
    parser.add_argument("--lbd", type=float, default=0.05)
    
    # img_path = "Apollo_15_flag_rover_LM_Irwin-1200x1200.jpg"
    
    args = parser.parse_args()
    src_img = cv2.imread(args.img_path)
    src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB); 

    p_size = args.p_size
    m_size = args.m_size

    pad_size = 2

    patches, label_list = patchify(src_img, p_size, m_size)
    if _show_detail:
        plot_images(patches, len(patches), len(patches[0]), p_size, m_size)

    img =       patches[2][2]
    label = label_list [2][2]
    x0, x1, y0, y1 = 0, img.shape[0], 0, img.shape[1]
    if label[0]:
        x0 += m_size
    if label[1]:
        x1 -= m_size
    if label[2]:
        y0 += m_size
    if label[3]:
        y1 -= m_size
    bias_img, gaussian_kernel, noise = add_gaussian_noise(img)
    # img = img[x0:x1, y0:y1]
    # bias_img = bias_img[x0:x1, y0:y1]

    f_size = (img.shape[0], img.shape[1])
    B = get_frequency_domain_rgb(bias_img, f_size)
    S = get_frequency_domain_rgb(img, f_size)
    N = get_frequency_domain_rgb(noise, f_size)
    B_shift = np.fft.fftshift(B)
    S_shift = np.fft.fftshift(S)

    G_matrix = np.fft.fft2(gaussian_kernel, s=f_size)
    G_matrix_shift = np.fft.fftshift(G_matrix)

    L_matrix = np.fft.fft2(Lap3x3_1, s=f_size)
    L_matrix_shift = np.fft.fftshift(L_matrix)


    mask = np.zeros((src_img.shape[0], src_img.shape[1]))
    # pad_img = np.zeros((src_img.shape[0]+pad_size*2, src_img.shape[1]+pad_size*2, 3))
    pad_img, _, _ = add_gaussian_noise(src_img)
    pad_img = np.pad(pad_img, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode="constant", constant_values=1)
    blur_list = []
    print("="*15+" patch task distributing... "+"="*15)
    for i, patch_row in enumerate(patches):
        blur_row = []
        for j, patch in enumerate(patch_row):
            label = label_list[i][j]
            x0, y0 = i*p_size, j*p_size

            if label[0]:
                x0-=m_size
            if label[2]:
                y0-=m_size
            x1 = x0+patch.shape[0]
            y1 = y0+patch.shape[1]

            mask[x0:x1, y0:y1] +=1
            # bias_patch, _, _ = add_gaussian_noise(patch)
            # pad_img[x0+pad_size:x1+pad_size, y0+pad_size:y1+pad_size] = bias_patch
            bias_patch = pad_img[x0+pad_size:x1+pad_size, y0+pad_size:y1+pad_size]
            blur_row.append(bias_patch)
        blur_list.append(blur_row)


    pad_mask = np.pad(mask, ((pad_size, pad_size), (pad_size, pad_size)), mode="constant", constant_values=1)-1
    pad_mask_cal = deepcopy(pad_mask).astype(np.float64)+1
    pad_mask[pad_mask>0] = 1
    print("="*15+" patch deblur processing... "+"="*15)

    pad_deblur_img, deblur_list = get_pad_deblur(blur_list, pad_img, pad_mask_cal, LBD=args.lbd)
    
    print("="*15+" Poission Blending... "+"="*15)

    output = seamlessClone(pad_img, pad_deblur_img, pad_mask, 0)
    plt.axis("off");plt.title("blur");plt.imshow(pad_img.astype(np.int32));plt.show();
    pad_img = np.clip(pad_img, 0, 255, out=pad_img)
    plt.imsave("blur.png", pad_img.astype(np.uint8))
    plt.axis("off");plt.title("deblur");plt.imshow(pad_deblur_img.astype(np.int32));plt.show();
    pad_deblur_img = np.clip(pad_deblur_img, 0, 255, out=pad_deblur_img)
    plt.imsave("deblur_no_poission.png", pad_deblur_img.astype(np.uint8))
    plt.axis("off");plt.title("Poisson");plt.imshow(output.astype(np.int32));plt.show();
    output = np.clip(output, 0, 255, out=output)
    plt.imsave("deblur_poission.png", output.astype(np.uint8))

