import numpy as np

def valid_cross_correlation_2d(image, filter):
    image_height, image_width = image.shape
    filter_height, filter_width = filter.shape
    output_height = image_height - filter_height + 1
    output_width = image_width - filter_width + 1
    output = np.zeros((output_height, output_width))

    for iy in range(output_height):
        for ix in range(output_width):
            for fy in range(filter_height):
                for fx in range(filter_width):
                    iv = image[iy+fy, ix+fx]
                    fv = filter[fy, fx]
                    output[iy, ix] += iv * fv

    return output


def fft_valid_cross_correlation_rfft_2d(image, filter):
    """
    Perform 2D cross-correlation using RFFT for valid range only.
    :param image: 2D NumPy array, input image
    :param filter: 2D NumPy array, filter filter (must have odd dimensions)
    :return: 2D NumPy array, resulting image after cross-correlation (valid region only)
    """
    # Check if filter dimensions are odd
    if filter.shape[0] % 2 == 0 or filter.shape[1] % 2 == 0:
        raise ValueError("Kernel size must be odd.")

    # Flip the filter for cross-correlation
    filter_flipped = filter[::-1, ::-1]
    # Get the valid output size
    output_shape = (image.shape[0] - filter.shape[0] + 1,
                    image.shape[1] - filter.shape[1] + 1)
    # Pad the filter to match the image size
    padded_filter = np.pad(filter_flipped,
                           ((0, image.shape[0] - filter.shape[0]),
                            (0, image.shape[1] - filter.shape[1])),
                           mode='constant')

    # Perform RFFT on image and filter
    rfft_image = np.fft.rfft2(image)
    rfft_filter = np.fft.rfft2(padded_filter)

    # Perform element-wise multiplication in Fourier domain
    rfft_result = rfft_image * rfft_filter

    # Perform inverse RFFT to get the spatial domain result
    full_result = np.fft.irfft2(rfft_result, s=image.shape)

    # Extract the valid region
    valid_result = full_result[filter.shape[0] - 1:, filter.shape[1] - 1:]
    return valid_result


if __name__ == "__main__":
    np.random.seed(0)

    image_height = 7
    image_width = 7
    image = np.random.rand(image_height, image_width)

    filter_height = 5
    filter_width = 5
    filter = np.random.rand(filter_height, filter_width)

    output = valid_cross_correlation_2d(image, filter)
    print("Direct Cross Correlation\n", output)

    output2 = fft_valid_cross_correlation_rfft_2d(image, filter)
    print("FFT Cross Correlation\n", output2)

