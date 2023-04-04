import cv2
import glob
import argparse
import numpy as np


def image_agcwd(img, a=0.25, truncated_cdf=False):
    hist, _ = np.histogram(img.flatten(), 256, [0, 256])
    prob_normalized = hist / hist.sum()

    unique_intensity = np.unique(img)

    prob_min = prob_normalized.min()
    prob_max = prob_normalized.max()
    pn_temp = (prob_normalized - prob_min) / (prob_max - prob_min)
    pn_temp[pn_temp > 0] = prob_max * (pn_temp[pn_temp > 0]**a)
    pn_temp[pn_temp < 0] = prob_max * (-((-pn_temp[pn_temp < 0])**a))
    prob_normalized_wd = pn_temp / pn_temp.sum()  # normalize to [0,1]
    cdf_prob_normalized_wd = prob_normalized_wd.cumsum()

    if truncated_cdf:
        inverse_cdf = np.maximum(0.5, 1 - cdf_prob_normalized_wd)
    else:
        inverse_cdf = 1 - cdf_prob_normalized_wd

    img_new = img.copy()
    for i in unique_intensity:
        img_new[img == i] = np.round(255 * (i / 255)**inverse_cdf[i])

    return img_new


def process_bright(img):
    img_negative = 255 - img
    agcwd = image_agcwd(img_negative, a=0.25, truncated_cdf=False)
    _reversed = 255 - agcwd
    return _reversed


def process_dimmed(img):
    agcwd = image_agcwd(img, a=0.75, truncated_cdf=True)
    return agcwd


def main():
    parser = argparse.ArgumentParser(description='IAGCWD')
    parser.add_argument('--input', dest='input_dir', default='./input/', type=str,
                        help='Input directory for image(s)')
    parser.add_argument('--output', dest='output_dir', default='./output/', type=str,
                        help='Output directory for image(s)')
    args = parser.parse_args()

    img_paths = glob.glob(args.input_dir+'*')
    for path in img_paths:
        img = cv2.imread(path, 1)
        name = path.split('\\')[-1].split('.')[0]

        # Extract intensity component of the image
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        Y = ycrcb[:, :, 0]
        # Determine whether image is bright or dimmed
        threshold = 0.3
        exp_in = 112  # Expected global average intensity
        M, N = img.shape[:2]
        mean_in = np.sum(Y/(M*N))
        t = (mean_in - exp_in) / exp_in

        # Process image for gamma correction
        img_output = None
        if t < -threshold:  # Dimmed Image
            print(name + ": Dimmed")
            result = process_dimmed(Y)
            ycrcb[:, :, 0] = result
            img_output = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        elif t > threshold:
            print(name + ": Bright Image")  # Bright Image
            result = process_bright(Y)
            ycrcb[:, :, 0] = result
            img_output = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        else:
            img_output = img

        cv2.imwrite(args.output_dir+name+'.jpg', img_output)


if __name__ == '__main__':
    main()
