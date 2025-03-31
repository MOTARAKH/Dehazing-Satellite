import cv2
import numpy as np

def single_scale_retinex(img, variance):
    return np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), variance))

def multi_scale_retinex(img, variance_list):
    retinex = np.zeros_like(img)
    for variance in variance_list:
        retinex += single_scale_retinex(img, variance)
    retinex = retinex / len(variance_list)
    return retinex

def MSR(img, variance_list):
    img = np.float64(img) + 1.0
    img_retinex = multi_scale_retinex(img, variance_list)
    for i in range(img_retinex.shape[2]):
        channel = img_retinex[:, :, i]
        channel = (channel - np.min(channel)) / (np.max(channel) - np.min(channel)) * 255
        img_retinex[:, :, i] = channel
    return np.uint8(img_retinex)

def SSR(img, variance):
    img = np.float64(img) + 1.0
    img_retinex = single_scale_retinex(img, variance)
    for i in range(img_retinex.shape[2]):
        channel = img_retinex[:, :, i]
        channel = (channel - np.min(channel)) / (np.max(channel) - np.min(channel)) * 255
        img_retinex[:, :, i] = channel
    return np.uint8(img_retinex)

def retinex_enhance(input_path, output_path, method="MSR", variance_list=[15, 80, 30], variance=300):
    img = cv2.imread(input_path)
    if img is None:
        print(f"Image could not be loaded.❌: {input_path}")
        return

    if method == "MSR":
        enhanced = MSR(img, variance_list)
    elif method == "SSR":
        enhanced = SSR(img, variance)
    else:
        raise ValueError("Method unknown, choose 'MSR' or 'SSR'")

    cv2.imwrite(output_path, enhanced)
    print(f"The image has been processed and saved in✅: {output_path}")
