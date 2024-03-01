import numpy as np
import matplotlib.pyplot as plt

def split_bayer(bayer_array, bayer_pattern):
    """
    Split R, Gr, Gb, and B channels sub-array from a Bayer array
    :param bayer_array: np.ndarray(H, W)
    :param bayer_pattern: 'gbrg' | 'rggb' | 'bggr' | 'grbg'
    :return: 4-element list of R, Gr, Gb, and B channel sub-arrays, each is an np.ndarray(H/2, W/2)
    """
    rggb_indices = get_bayer_indices(bayer_pattern)

    sub_arrays = []
    for idx in rggb_indices:
        x0, y0 = idx
        sub_arrays.append(
            bayer_array[y0::2, x0::2]
        )

    return sub_arrays

def get_bayer_indices(pattern):
    """
    Get (x_start_idx, y_start_idx) for R, Gr, Gb, and B channels
    in Bayer array, respectively
    """
    return {'gbrg': ((0, 1), (1, 1), (0, 0), (1, 0)),
            'rggb': ((0, 0), (1, 0), (0, 1), (1, 1)),
            'bggr': ((1, 1), (0, 1), (1, 0), (0, 0)),
            'grbg': ((1, 0), (0, 0), (1, 1), (0, 1))}[pattern.lower()]

def shift_array(padded_array, window_size):
    """
    Shift an array within a window and generate window_size**2 shifted arrays
    :param padded_array: np.ndarray(H+2r, W+2r)
    :param window_size: 2r+1
    :return: a generator of length (2r+1)*(2r+1), each is an np.ndarray(H, W), and the original
        array before padding locates in the middle of the generator
    """
    wy, wx = window_size if isinstance(window_size, (list, tuple)) else (window_size, window_size)
    assert wy % 2 == 1 and wx % 2 == 1, 'only odd window size is valid'

    height = padded_array.shape[0] - wy + 1
    width = padded_array.shape[1] - wx + 1

    for y0 in range(wy):
        for x0 in range(wx):
            yield padded_array[y0:y0 + height, x0:x0 + width, ...]

def reconstruct_bayer(sub_arrays, bayer_pattern):
    """
    Inverse implementation of split_bayer: reconstruct a Bayer array from a list of
        R, Gr, Gb, and B channel sub-arrays
    :param sub_arrays: 4-element list of R, Gr, Gb, and B channel sub-arrays, each np.ndarray(H/2, W/2)
    :param bayer_pattern: 'gbrg' | 'rggb' | 'bggr' | 'grbg'
    :return: np.ndarray(H, W)
    """
    rggb_indices = get_bayer_indices(bayer_pattern)

    height, width = sub_arrays[0].shape
    bayer_array = np.empty(shape=(2 * height, 2 * width), dtype=sub_arrays[0].dtype)

    for idx, sub_array in zip(rggb_indices, sub_arrays):
        x0, y0 = idx
        bayer_array[y0::2, x0::2] = sub_array

    return bayer_array

def rotate_to_rggb(array):
    k = {'rggb': 0,
         'bggr': 2,
         'grbg': 1,
         'gbrg': 3}['rggb']
    return np.rot90(array, k=k)

def index_weighted_sum(arrays, indices, weights):
    result = 0
    for i, w in zip(indices, weights):
        result += w * arrays[i]
    return result

def rotate_from_rggb(array):
    k = {'rggb': 0,
         'bggr': 2,
         'grbg': 3,
         'gbrg': 1}['rggb']
    return np.rot90(array, k=k)

def generic_filter(array, kernel):
    """
    Filter input image array with given kernel
    :param array: array to be filter: np.ndarray(H, W, ...), must be np.int dtype
    :param kernel: np.ndarray(h, w)
    :return: filtered array: np.ndarray(H, W, ...)
    """
    kh, kw = kernel.shape[:2]
    kernel = kernel.flatten()

    padded_array = pad(array, pads=(kh // 2, kw // 2))
    shifted_arrays = shift_array(padded_array, window_size=(kh, kw))

    filtered_array = np.zeros_like(array)
    weights = np.zeros_like(array)

    for i, shifted_array in enumerate(shifted_arrays):
        filtered_array += kernel[i] * shifted_array
        weights += kernel[i]

    filtered_array = (filtered_array / weights).astype(array.dtype)
    return filtered_array

def pad(array, pads, mode='reflect'):
    """
    Pad an array with given margins
    :param array: np.ndarray(H, W, ...)
    :param pads: {int, sequence}
        if int, pad top, bottom, left, and right directions with the same margin
        if 2-element sequence: (y-direction pad, x-direction pad)
        if 4-element sequence: (top pad, bottom pad, left pad, right pad)
    :param mode: padding mode, see np.pad
    :return: padded array: np.ndarray(H', W', ...)
    """
    if isinstance(pads, (list, tuple, np.ndarray)):
        if len(pads) == 2:
            pads = ((pads[0], pads[0]), (pads[1], pads[1])) + ((0, 0),) * (array.ndim - 2)
        elif len(pads) == 4:
            pads = ((pads[0], pads[1]), (pads[2], pads[3])) + ((0, 0),) * (array.ndim - 2)
        else:
            raise NotImplementedError

    return np.pad(array, pads, mode)

def gen_gaussian_kernel(kernel_size, sigma):
    if isinstance(kernel_size, (list, tuple)):
        assert len(kernel_size) == 2
        wy, wx = kernel_size
    else:
        wy = wx = kernel_size

    x = np.arange(wx) - wx // 2
    if wx % 2 == 0:
        x += 0.5

    y = np.arange(wy) - wy // 2
    if wy % 2 == 0:
        y += 0.5

    y, x = np.meshgrid(y, x)

    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


channel_indices_and_weights = (
    [[1, 3, 4, 5, 7], [-0.125, -0.125, 0.5, -0.125, -0.125]],
    [[1, 3, 4, 5, 7], [-0.1875, -0.1875, 0.75, -0.1875, -0.1875]],
    [[1, 3, 4, 5, 7], [0.0625, -0.125, 0.625, -0.125, 0.0625]],
    [[1, 3, 4, 5, 7], [-0.125, 0.0625, 0.625, 0.0625, -0.125]],
    [[3, 4], [0.25, 0.25]],
    [[1, 4], [0.25, 0.25]],
    [[4, 7], [0.25, 0.25]],
    [[4, 5], [0.25, 0.25]],
    [[4, 5], [0.5, 0.5]],
    [[1, 3], [0.5, 0.5]],
    [[1, 4], [0.5, 0.5]],
    [[4, 7], [0.5, 0.5]],
    [[3, 4], [0.5, 0.5]],
    [[0, 1, 3, 4], [0.25, 0.25, 0.25, 0.25]],
    [[4, 5, 7, 8], [0.25, 0.25, 0.25, 0.25]],
    [[1, 2, 4, 5], [-0.125, -0.125, -0.125, -0.125]],
    [[3, 4, 6, 7], [-0.125, -0.125, -0.125, -0.125]]
)

def stat_draw(img:np.ndarray):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Plot histograms
    axs[0, 0].hist(img[:, :, 0].flatten(), bins=30, edgecolor='black', color='r')
    axs[0, 0].set_title('Red Channel Histogram')
    axs[0, 0].set_xlabel('Value')
    axs[0, 0].set_ylabel('Frequency')

    axs[0, 1].hist(img[:, :, 1].flatten(), bins=30, edgecolor='black', color='g')
    axs[0, 1].set_title('Green Channel Histogram')
    axs[0, 1].set_xlabel('Value')
    axs[0, 1].set_ylabel('Frequency')

    axs[1, 0].hist(img[:, :, 2].flatten(), bins=30, edgecolor='black', color='b')
    axs[1, 0].set_title('Blue Channel Histogram')
    axs[1, 0].set_xlabel('Value')
    axs[1, 0].set_ylabel('Frequency')

    # Plot the image
    axs[1, 1].imshow(img)
    axs[1, 1].set_title('RGB Image')

    # Hide the x and y ticks for the image plot
    axs[1, 1].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Show the plot
    plt.show()

def normalize(img, scale = 3000):
    # 找到每个通道的最小值和最大值
    img = img.astype(np.float32)
    img /= scale
    min_vals = np.min(img, axis=(0, 1))
    max_vals = np.max(img, axis=(0, 1))
    
    # 归一化每个通道
    normalized_img = (img - min_vals) / (max_vals - min_vals + 1e-8)
    
    # 缩放到0-255范围
    normalized_img = normalized_img * 255.0
    
    return normalized_img.astype(np.uint8)

def index_weighted_sum(arrays, indices, weights):
    result = 0
    for i, w in zip(indices, weights):
        result += w * arrays[i]
    return result

def convolution3x5(image, kernel):
    # Get dimensions of the image and the kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    # Get the output dimensions
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1
    
    # Generate indices for all possible positions of the kernel
    rows = np.arange(kernel_height)[:, None] + np.arange(output_height)
    cols = np.arange(kernel_width)[:, None] + np.arange(output_width)
    
    # Extract regions from the image using the generated indices
    regions = image[rows, :, None, cols]
    
    # Perform element-wise multiplication and sum along specified axes
    output = np.tensordot(regions, kernel, axes=([1, 3], [0, 1]))
    
    return output

def convolution3x5_grey(image, kernel):
    # TODO
    pass

def convolve2d(image, kernel):
    # Get dimensions of the image and kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    # Calculate padding sizes
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    
    # Pad the image
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
    
    # Generate indices for all possible positions of the kernel
    indices_i = np.arange(kernel_height)[:, None] + np.arange(image_height)[None, :] - pad_height
    indices_j = np.arange(kernel_width)[None, :] + np.arange(image_width)[:, None] - pad_width
    print("indices_i")
    print(indices_i)
    print("indices_j")
    print(indices_j)

    # Extract regions from the padded image using the generated indices
    regions = padded_image[indices_i[:, :, None], indices_j[None, :, :]]
    print(regions)
    # Perform element-wise multiplication and sum along specified axes
    convolved_image = np.sum(regions * kernel[:, :, None, None], axis=(0, 2))
    
    return convolved_image

if __name__ == "__main__":
    # kernel = np.array([[1, 2, 3, 4, 5],
    #                   [6, 7, 8, 9, 10],
    #                   [11, 12, 13, 14, 15],
    #                   [16, 17, 18, 19, 20]])
    # img = np.array([[1, 0, -1],
    #                    [2, 0, -2],
    #                    [1, 0, -1]])
    
    # a = convolution3x5_grey(img, kernel)
    # print(a)
    
    # Example usage:
    image = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])

    kernel = np.array([[1, 0, -1],
                       [1, 0, -1],
                       [1, 0, -1]])

    result = convolve2d(image, kernel)
    print(result)