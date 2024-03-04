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

### Idea

We need to change to YUV for the further process but not direct using RGB, because YUV can separate the Chroma(color information`Cr` , `Cb`) and Luma(intensity information `Y`). Notice that the matrix transform of range `[0, 1]` and `[0, 255]` is different, here our RGB range is `[0, 255]`



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

We can draw the histogram of the `YUV` channels, to monitor the shape change of the distribution later.

<img src="./data/cfa_yuv_stat.png" style="zoom:50%;" />

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

<img src="./data/ee_stat.png" style="zoom:50%;" />

After computing the `EMLUT(x)`, clip the `EMLUT(x)` and add  it to the image as an output:

![](./data/ee_ee.png)

## Brightness/Contrast Control

### Idea

Brightness control is :
$$
Y\times\text{contrast}+\text{brightess}\\
i.e.\ \alpha Y +\beta
$$

This operation is first move the distribution to the right of `brightness` scale, and then enlarge the variance with `contrast` scale. In the example image, I choose brightness as 12.3208, and contrast = 0.9249. 

Here we minus the median value to control the moving shape of the distribution. One tips here for the adjustment of the parameter is that you can first pin some point of the distribution, and the calculate the parameter using the [least square method](https://zh.wikipedia.org/wiki/%E6%9C%80%E5%B0%8F%E4%BA%8C%E4%B9%98%E6%B3%95).

In the histogram above, we can observe that there is a high frequent around 30-40, and we don't want the high value part moving too far from 150. Therefore we can set a group of function from the original distribution to the target distribution:
$$
40\alpha + \beta = 50\\
75\alpha + \beta = 80\\
125\alpha+ \beta = 130\\
150\alpha + \beta = 150
$$
 By the LSM, we can get the $\beta =  12.3208$ , and $\alpha = 0.9249$. 

Notice that now the clip range is no longer  `[16, 235]`, the `Y` channel is in range of `[16, 235]`. 

### Output

In the output image, the sky is brighter, and the shade of tree is darker (left of the image). In the histogram, the darker part is moved from around 30 to around 45.

<img src="./data/bcc_stat.png" style="zoom:50%;" />

![](./data/bcc.png)

## False Color Suppression

### Idea

False Color was caused from the CFA interpolation. In short, false color is introduced into the image when the predicted color is inaccurate, especially in the gray area.

First compute the gain of `u` dimension and `v` dimension by this piecewise function:
$$
Cb, Cr = \begin{cases} 
    Cb, Cr &      |\text{EM}(y, x)| \leq \text{fcsEdge}_1 \\
   \frac{\text{gain}\times(|EM|-128)}{65536}+128& 
    \text{fcsEdge}_1 < |\text{EM}(y, x)| < \text{fcsEdge}_2 \\
    0 & |\text{EM}(y, x)| \geq \text{fcsEdge}_2
\end{cases}\\
$$

In this figure the edge map distribution is:

<img src="./data/EM_stat.png" alt="EM_stat" style="zoom:67%;" />

Therefore, I set the $\text{fcsEdge}_1 = 16$, and $\text{fcsEdge}_2 = 32$

### Acceleration

Again use the mask boolean indices to assign the value in a vectorization way:

```python
absEM = np.abs(self.edgemap)
mask1 = absEM <= self.fcs_edge[0]
mask2 = np.logical_and(
    absEM > self.fcs_edge[0], 
    absEM < self.fcs_edge[1])
mask3 = absEM >= self.fcs_edge[1]
        
fcs_img[mask1] = self.img[mask1]
fcs_img[mask2, 0] = self.gain*(absEM[mask2]-128)/65536 + 128
fcs_img[mask2, 1] = self.gain*(absEM[mask2]-128)/65536 + 128    
fcs_img[mask3] = 0
        
# np.clip(fcs_img, 0, 255, out=fcs_img)
np.clip(fcs_img, 16,239, out=fcs_img)
```

### Output

In the output, we can see the `Cb` part improved, and the darker part in `Cb` become brighter. In the RGB level, the darker part become less red, and more blue.

<img src="./data/fcs_stat.png" style="zoom:50%;" />

![](./data/fcs.png)

## Hue/Saturation control

### Idea

HSC is in two step: the Hue control, and the Saturation control. In the Hue tuning, the two channels range are first moved from `[0-256]` to `[-128, 128] `, then using the sine of Hue parameter to get the new `Cr`, and `Cb` channel value.
$$
Y_{Cr}’ = (Y_{Cr}-128)\times\cos(h)+(Y_{Cb}-128)\times \sin(h) + 128\\
Y_{Cb}’ = (Y_{Cb}-128)\times\cos(h)+(Y_{Cr}-128)\times \sin(h) + 128
$$
Then in the Saturation control step, with a linear transform, we can output the tuned image of UV channels:
$$
Y = \frac{s(Y-128)}{256}+128
$$

### Output

In the image below, we can observe both Cr and Cb showed more details, and more contrast.

<img src="./data/hsc_stat.png" style="zoom:50%;" />

![](./data/hsc.png)

### Normalize and output the RBG

After all, assign the `Y` channel as the BCC output, and assign the `Cr`, `Cb` channel as the HSC output.

![](./data/output_stat.png)



## How to Run







## References

**Github**

- ISP Pipeline | camera成像原理: https://juejin.cn/post/7309301549154828323

- FastOpenISP: https://github.com/QiuJueqin/fast-openISP?tab=readme-ov-file
- OpenISP: https://github.com/cruxopen/openISP

- Opencv BCC: https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html

- False Color Suppression in Demosaiced Color Images: https://www.cse.iitb.ac.in/~sharat/icvgip.org/icvgip2004/proceedings/ip1.3_029.pdf

- FalseColor-Python: https://github.com/serrob23/falsecolor
