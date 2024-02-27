# Homework 2 Image Signal Processing

> Zhen Tong 120090694

## Dead Pixel Correction

这里定义了三个变量：

1. `dpc_thres`: 这是一个用于缺陷像素校正的阈值。通常，它表示允许的最大像素值差异，超过此阈值的像素将被视为缺陷像素。

2. `dpc_mode`: 这是缺陷像素校正的模式。在这里，它被设置为 `'gradient'`，这可能是指定了一种根据像素之间的梯度进行校正的方法。不同的模式可能会采用不同的校正算法或策略。

3. `dpc_clip`: 这是一个阈值，用于限制缺陷像素校正后像素的最大值。超过这个值的像素将被截断或裁剪到此值以内。在这里，它被设置为 `4095`，可能是指相机或图像的白平衡级别。





## References

**Github**

- FastOpenISP: https://github.com/QiuJueqin/fast-openISP?tab=readme-ov-file
- OpenISP: https://github.com/cruxopen/openISP

