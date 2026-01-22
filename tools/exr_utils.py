import torch
import cv2
import numpy as np
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def save_tensor_to_exr(tensor, path):
    """
    Saves a PyTorch tensor (C, H, W) to an EXR file (H, W, C) with 32-bit float precision.
    """
    # 1. Move to CPU, detach gradients, and convert to numpy
    # resulting shape: (C, H, W)
    img_numpy = tensor.detach().cpu().numpy()

    # 2. Handle Batch Dimension if present (1, C, H, W) -> (C, H, W)
    if img_numpy.ndim == 4:
        img_numpy = img_numpy[0]

    # 3. Permute dimensions: (C, H, W) -> (H, W, C) for OpenCV
    img_numpy = np.transpose(img_numpy, (1, 2, 0))

    # 4. Convert RGB to BGR (OpenCV standard)
    # Only do this if we have 3 channels.
    if img_numpy.shape[2] == 3:
        img_numpy = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR)

    # 5. Save using specific EXR flags
    # IMWRITE_EXR_TYPE_FLOAT ensures it saves as 32-bit float (HDR)
    # IMWRITE_EXR_TYPE_HALF would save as 16-bit float (smaller file, less precision)
    cv2.imwrite(path, img_numpy, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])