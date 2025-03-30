import cv2
import math
import numpy as np
import os

def dark_channel(im, sz):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    return dark

def atmospheric_light(im, dark):
    h, w = im.shape[:2]
    imsz = h * w
    numpx = int(max(math.floor(imsz / 1000), 1))
    darkvec = dark.reshape(imsz)
    imvec = im.reshape(imsz, 3)

    indices = darkvec.argsort()
    indices = indices[imsz - numpx:]

    atmsum = np.zeros([1, 3])
    for ind in range(numpx):
        atmsum += imvec[indices[ind]]

    A = atmsum / numpx
    return A

def transmission_estimate(im, A, sz):
    omega = 0.95
    im3 = np.empty(im.shape, im.dtype)
    for ind in range(3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]
    transmission = 1 - omega * dark_channel(im3, sz)
    return transmission

def guided_filter(im, p, r, eps):
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * im + mean_b
    return q

def transmission_refine(im, et):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray) / 255
    r = 60
    eps = 0.0001
    t = guided_filter(gray, et, r, eps)
    return t

def recover(im, t, A, tx=0.1):
    res = np.empty(im.shape, im.dtype)
    t = cv2.max(t, tx)
    for ind in range(3):
        res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t + A[0, ind]
    return res

# ✅ الدالة المهمة التي يجب أن تكون موجودة!
def dehaze_image(path_to_image, output_dir):
    filename = os.path.basename(path_to_image)
    img = cv2.imread(path_to_image)
    if img is None:
        print(f"❌ تعذر تحميل الصورة: {filename}")

        return

    I = img.astype('float64') / 255
    dark = dark_channel(I, 15)
    A = atmospheric_light(I, dark)
    te = transmission_estimate(I, A, 15)
    t = transmission_refine(img, te)
    J = recover(I, t, A, 0.1)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, f"{filename.split('.')[0]}_dcp.png")
    cv2.imwrite(output_path, (J * 255).astype(np.uint8))
    print(f"✅ تمت معالجة الصورة وحفظها في: {output_path}")
