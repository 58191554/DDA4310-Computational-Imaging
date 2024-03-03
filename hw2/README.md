# Homework 2 Image Signal Processing

> Zhen Tong 120090694

## Dead Pixel Correction



## Color Filter Array Interpolation

### Idea

We need to interpolate the `[height, width, 3]` RGB image from the raw `[height, width]` Matrix. The matrix is in `rggb` bayer pattern, which is:

![rggb](./data/rggb.png)

What we need to is to interpolate from the 2 dimension array to 3 dimension with *Malvar* (2004) demosaicing algorithm. 

There are several situation of interpolation:

| Color in Bayer/Target Color | R        | G        | B        |
| --------------------------- | -------- | -------- | -------- |
| **R**                       | $r$      | $g_r$    | $b_r$    |
| **Gr**                      | $r_{gr}$ | $g_{gr}$ | $b_{gr}$ |
| **Gb**                      | $r_{gb}$ | $g_{gb}$ | $b_{gb}$ |
| **B**                       | $r_b$    | $g_b$    | $b$      |

Following the weight of the algorithm in the figure, 

![cfa_rule](./data/cfa_rule.png)

 For example G at R location is:
$$
g_r = \frac{{4 \cdot X_{y,x} - X_{y-2,x} - X_{y,x-2} - X_{y+2,x} - X_{y,x+2} + 2 \cdot (X_{y+1,x} +X_{y,x+1} + X_{y-1,x} + X_{y,x-1})}}{{8}}\\
$$

### Acceleration

In the interpolation, intuitively we need to use the nested for loop, which is in $N^2$ time. To speedup the code in python, we can use the **Indices Generationi**: Meshgrid is used to generate indices for accessing elements of the 2D image in a structured manner. This generates arrays `y_indices` and `x_indices` representing the y and x coordinates of pixels in the padded image.

```python
y_indices, x_indices = np.meshgrid(np.arange(0, img_pad.shape[0]-4-1, 2), np.arange(0, img_pad.shape[1]-4-1, 2))
```

The meshgrid will generate the 2d coordinate of step-size = 2. Then, we can extract the rs, grs, gbs, and bs matrix from the bayer matrix by:

```python
r = img_pad[y_indices + 2, x_indices + 2]
gr = img_pad[y_indices + 2, x_indices + 3]
gb = img_pad[y_indices + 3, x_indices + 2]
b = img_pad[y_indices + 3, x_indices + 3]
```

Then, we can interpolate in matrix level for 4 times:

```python
cfa_img[y_indices, x_indices, :] = malvar('r', r, y_indices + 2, x_indices + 2, img_pad)
cfa_img[y_indices, x_indices + 1, :] = malvar('gr', gr, y_indices + 2, x_indices + 3, img_pad)
cfa_img[y_indices + 1, x_indices, :] = malvar('gb', gb, y_indices + 3, x_indices + 2, img_pad)
cfa_img[y_indices + 1, x_indices + 1, :] = malvar('b', b, y_indices + 3, x_indices + 3, img_pad)
```

After all, remember to clip the image.

### Normalization

Notably, without normalization, the `matplotlib.pyplot` can not transfrom the scale of CFA output to the 0-255, or 0-1 scale (all white in the RGB Image bellow), we need to manually normalize it into 0-255 or 0-1.

<img src="./data/cfa_no_norm.png" style="zoom:50%;" />

After normalization:
$$
X = \frac{X-min(X)}{max(X)-min(X)}
$$
<img src="./data/cfa_3000norm.png" style="zoom:50%;" />

## RGB2YUV

We need to change to YUV for the further process but not direct using RGB, because YUV can separate the Chroma(color information`Cr` , `Cb`) and Luma(intensity information `Y`)

### Acceleration

Change the pixel by pixel transform into matrix transform

```python
def RGB2YUV(img:np.ndarray):
    matrix = np.array([[66, 129, 25],
                       [-38, -74, 112],
                       [112, -94, -18]], dtype=np.int32).T  # x256
    bias = np.array([16, 128, 128], dtype=np.int32).reshape(1, 1, 3)
    np.right_shift(rgb_image @ matrix, 8) + bias
```

### Output

In the following content, when we show the `YUV` output, we are actually showing the `YUV[:, :, 0]`, the first channel Luma, but not the 3-Channel image. Next, we try to improve the edge, contrast, and brightness quality of the image one the first channel of YUV.

![](./data/YUV.png)

## Edge Enhancement

### Idea

The edge enhancement includes two parts: compute the convolution of edge filter, and compute the edge map loop up table (EMLUT). With gains: $g_1, g_2 = 32, 128$, threstholds: $x_1, x2 = 32, 64$, the transform function is in shape of:
$$
f(x) = \begin{cases} 
    128x & \text{if } x < -64 \\
    0 & \text{if } -64 < x < -32 \\
    -32x & \text{if } -32 < x < 32 \\
    0 & \text{if } 32 < x < 64 \\
    128x & \text{if } x > 64
\end{cases}
$$
<img src="./data/EMLUT(x).png" style="zoom:50%;" />

### Speed up

First, in the convolution part, the filter kernel is a 3x5 matrix. We can again use the meshgrid to divide the 15 channels of image from the first channel of YUV image. Each channel of the 15 channel corespond to one weight of the kernel filter. Then compute the weighted sum

```python
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
    weighted_sum = np.sum(kernel[:, np.newaxis, np.newaxis] * np.array([a1, a2, a3, a4, a5,a6, a7, a8, a9, a10,a11, a12, a13, a14, a15]), axis=0)
```

Then the piecewise function transform to the convolution output can be speed up with mask operation in numpy. The mask records all the pixels that in the range of the given condition. Then it can be used to assign the value by indices accessing.

```python
# Compute lut based on conditions
mask1 = em_img < -self.thres[1]
mask2 = (em_img >= -self.thres[1]) & (em_img <= -self.thres[0])
mask3 = (em_img >= -self.thres[0]) & (em_img <= self.thres[0])
mask4 = (em_img >= self.thres[0]) & (em_img <= self.thres[1])
mask5 = em_img > self.thres[1]
lut[mask1] = self.gain[1] * em_img[mask1]
lut[mask2] = 0
lut[mask3] = self.gain[0] * em_img[mask3]
lut[mask4] = 0
lut[mask5] = self.gain[1] * em_img[mask5]
```

### Output

After computing the `EMLUT(x)`, clip the `EMLUT(x)` and add  it to the image as an output:

![](./data/ee_ee.png)

## Brightness/Contrast Control

### Idea

Brightness control is :
$$
Y = Y+\text{brightness}+(Y-127)\times\text{contrast}
$$

### Output

In the output image, the sky is brighter, and the shade of tree is darker (left of the image)

![](./data/bcc.png)

















## References

**Github**

- FastOpenISP: https://github.com/QiuJueqin/fast-openISP?tab=readme-ov-file
- OpenISP: https://github.com/cruxopen/openISP

