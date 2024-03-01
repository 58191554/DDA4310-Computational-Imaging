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

why we need to change to YUV for the further process but not direct using RGB???





```python
def RGB2YUV(img:np.ndarray):
    matrix = np.array([[66, 129, 25],
                       [-38, -74, 112],
                       [112, -94, -18]], dtype=np.int32).T  # x256
    bias = np.array([16, 128, 128], dtype=np.int32).reshape(1, 1, 3)
    rgb_image = img.astype(np.int32)

    ycrcb_image = (np.right_shift(rgb_image @ matrix, 8) + bias).astype(np.uint8)

    return ycrcb_image

```

## Edge Enhancement



## References

**Github**

- FastOpenISP: https://github.com/QiuJueqin/fast-openISP?tab=readme-ov-file
- OpenISP: https://github.com/cruxopen/openISP

