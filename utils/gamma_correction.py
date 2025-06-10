import numpy as np

def gamma_correction(image, gamma=2.2):
    """
    Apply gamma correction to the image.

    Parameters:
        image (np.ndarray): Input image array with pixel values in the range [0, 1].
        gamma (float): The gamma value for correction.

    Returns:
        np.ndarray: Gamma-corrected image.
    """

    # Apply gamma correction
    corrected_image = np.power(image, 1.0 / gamma)

    # Ensure the output image is clipped to the range [0, 1]
    corrected_image = np.clip(corrected_image, 0, 1)

    return corrected_image
