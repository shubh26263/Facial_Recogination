import os
import cv2
import numpy as np
import argparse
import warnings
import time

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')

import collections
from itertools import chain
import urllib.request as request
import pickle 

import numpy as np

import scipy.signal as signal
import scipy.special as special
import scipy.optimize as optimize

import matplotlib.pyplot as plt

import skimage.io
import skimage.transform

import cv2

from libsvm import svmutil

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


SAMPLE_IMAGE_PATH = "./images/sample/"

def check_image(image):
    height, width, channel = image.shape
    if width/height != 3/4:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        return False
    else:
        return True


def test(image_name, model_dir, device_id):
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    image_org = cv2.imread(SAMPLE_IMAGE_PATH + image_name)
    image = cv2.resize(image_org , (int(image_org.shape[0] * 3/4) , image_org.shape[0]))
    result = check_image(image)
    if result is False:
        return
    image_bbox = model_test.get_bbox(image)
    prediction = np.zeros((1, 3))
    test_speed = 0

    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        start = time.time()
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))
        test_speed += time.time()-start

    # Check image metrics 
    def normalize_kernel(kernel):
        return kernel / np.sum(kernel)

    def gaussian_kernel2d(n, sigma):
        Y, X = np.indices((n, n)) - int(n/2)
        gaussian_kernel = 1 / (2 * np.pi * sigma ** 2) * np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2)) 
        return normalize_kernel(gaussian_kernel)

    def local_mean(image, kernel):
        return signal.convolve2d(image, kernel, 'same')
    
    def local_deviation(image, local_mean, kernel):
        "Vectorized approximation of local deviation"
        sigma = image ** 2
        sigma = signal.convolve2d(sigma, kernel, 'same')
        return np.sqrt(np.abs(local_mean ** 2 - sigma))
    
    def calculate_mscn_coefficients(image, kernel_size=6, sigma=7/6):
        C = 1/255
        kernel = gaussian_kernel2d(kernel_size, sigma=sigma)
        local_mean = signal.convolve2d(image, kernel, 'same')
        local_var = local_deviation(image, local_mean, kernel)
        return (image - local_mean) / (local_var + C)
    
    def generalized_gaussian_dist(x, alpha, sigma):
        beta = sigma * np.sqrt(special.gamma(1 / alpha) / special.gamma(3 / alpha))
    
        coefficient = alpha / (2 * beta() * special.gamma(1 / alpha))
        return coefficient * np.exp(-(np.abs(x) / beta) ** alpha)
    
    def calculate_pair_product_coefficients(mscn_coefficients):
        return collections.OrderedDict({
            'mscn': mscn_coefficients,
            'horizontal': mscn_coefficients[:, :-1] * mscn_coefficients[:, 1:],
            'vertical': mscn_coefficients[:-1, :] * mscn_coefficients[1:, :],
            'main_diagonal': mscn_coefficients[:-1, :-1] * mscn_coefficients[1:, 1:],
            'secondary_diagonal': mscn_coefficients[1:, :-1] * mscn_coefficients[:-1, 1:]
            })
    
    def asymmetric_generalized_gaussian(x, nu, sigma_l, sigma_r):
        def beta(sigma):
            return sigma * np.sqrt(special.gamma(1 / nu) / special.gamma(3 / nu))
        
        coefficient = nu / ((beta(sigma_l) + beta(sigma_r)) * special.gamma(1 / nu))
        f = lambda x, sigma: coefficient * np.exp(-(x / beta(sigma)) ** nu)
        return np.where(x < 0, f(-x, sigma_l), f(x, sigma_r))
    
    def asymmetric_generalized_gaussian_fit(x):
        def estimate_phi(alpha):
            numerator = special.gamma(2 / alpha) ** 2
            denominator = special.gamma(1 / alpha) * special.gamma(3 / alpha)
            return numerator / denominator

        def estimate_r_hat(x):
            size = np.prod(x.shape)
            return (np.sum(np.abs(x)) / size) ** 2 / (np.sum(x ** 2) / size)

        def estimate_R_hat(r_hat, gamma):
            numerator = (gamma ** 3 + 1) * (gamma + 1)
            denominator = (gamma ** 2 + 1) ** 2
            return r_hat * numerator / denominator

        def mean_squares_sum(x, filter = lambda z: z == z):
            filtered_values = x[filter(x)]
            squares_sum = np.sum(filtered_values ** 2)
            return squares_sum / ((filtered_values.shape))

        def estimate_gamma(x):
            left_squares = mean_squares_sum(x, lambda z: z < 0)
            right_squares = mean_squares_sum(x, lambda z: z >= 0)
            return np.sqrt(left_squares) / np.sqrt(right_squares)

        def estimate_alpha(x):
            r_hat = estimate_r_hat(x)
            gamma = estimate_gamma(x)
            R_hat = estimate_R_hat(r_hat, gamma)
            solution = optimize.root(lambda z: estimate_phi(z) - R_hat, [0.2]).x
            return solution[0]

        def estimate_sigma(x, alpha, filter = lambda z: z < 0):
            return np.sqrt(mean_squares_sum(x, filter))
    
        def estimate_mean(alpha, sigma_l, sigma_r):
            return (sigma_r - sigma_l) * constant * (special.gamma(2 / alpha) / special.gamma(1 / alpha))
    
        alpha = estimate_alpha(x)
        sigma_l = estimate_sigma(x, alpha, lambda z: z < 0)
        sigma_r = estimate_sigma(x, alpha, lambda z: z >= 0)
    
        constant = np.sqrt(special.gamma(1 / alpha) / special.gamma(3 / alpha))
        mean = estimate_mean(alpha, sigma_l, sigma_r)
    
        return alpha, mean, sigma_l, sigma_r
    
    def calculate_brisque_features(image, kernel_size=7, sigma=7/6):
        def calculate_features(coefficients_name, coefficients, accum=np.array([])):
            alpha, mean, sigma_l, sigma_r = asymmetric_generalized_gaussian_fit(coefficients)

            if coefficients_name == 'mscn':
                var = (sigma_l ** 2 + sigma_r ** 2) / 2
                return [alpha, var]
        
            return [alpha, mean, sigma_l ** 2, sigma_r ** 2]
    
        mscn_coefficients = calculate_mscn_coefficients(image, kernel_size, sigma)
        coefficients = calculate_pair_product_coefficients(mscn_coefficients)
    
        features = [calculate_features(name, coeff) for name, coeff in coefficients.items()]
        flatten_features = list(chain.from_iterable(features))
        return np.array(flatten_features)
    
    gray_image = skimage.color.rgb2gray(image_org)
    mscn_coefficients = calculate_mscn_coefficients(gray_image, 7, 7/6)
    coefficients = calculate_pair_product_coefficients(mscn_coefficients)

    brisque_features = calculate_brisque_features(gray_image, kernel_size=7, sigma=7/6)
    downscaled_image = cv2.resize(gray_image, None, fx=1/2, fy=1/2, interpolation = cv2.INTER_CUBIC)
    downscale_brisque_features = calculate_brisque_features(downscaled_image, kernel_size=7, sigma=7/6)

    brisque_features = np.concatenate((brisque_features, downscale_brisque_features))

    def scale_features(features):
        with open('normalize.pickle', 'rb') as handle:
            scale_params = pickle.load(handle)
    
        min_ = np.array(scale_params['min_'])
        max_ = np.array(scale_params['max_'])
    
        return -1 + (2.0 / (max_ - min_) * (features - min_))

    def calculate_image_quality_score(brisque_features):
        model = svmutil.svm_load_model('brisque_svm.txt')
        scaled_brisque_features = scale_features(brisque_features)
    
        x, idx = svmutil.gen_svm_nodearray(
            scaled_brisque_features,
            isKernel=(model.param.kernel_type == svmutil.PRECOMPUTED))
    
        nr_classifier = 1
        prob_estimates = (svmutil.c_double * nr_classifier)()
    
        return svmutil.libsvm.svm_predict_probability(model, x, prob_estimates)

    image_score =  calculate_image_quality_score(brisque_features)


    # draw result of prediction
    label = np.argmax(prediction)
    value = prediction[0][label]/2
    if label == 1 and image_score <= 40:
        print("Image '{}' is Real Face. Score: {:.2f} Image_quality: {:.2f}".format(image_name, value, image_score))
        result_text = "RealFace Score: {:.2f}".format(value)
        color = (255, 0, 0)
    else:
        print("Image '{}' is Fake Face. Score: {:.2f} Image_quality: {:.2f}".format(image_name, value, image_score))
        result_text = "FakeFace Score: {:.2f}".format(value)
        color = (0, 0, 255)
    print("Prediction cost {:.2f} s".format(test_speed))
    cv2.rectangle(
        image,
        (image_bbox[0], image_bbox[1]),
        (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
        color, 2)
    cv2.putText(
        image,
        result_text,
        (image_bbox[0], image_bbox[1] - 5),
        cv2.FONT_HERSHEY_COMPLEX, 0.5*image.shape[0]/1024, color)

    format_ = os.path.splitext(image_name)[-1]
    result_image_name = image_name.replace(format_, "_result" + format_)
    cv2.imwrite(SAMPLE_IMAGE_PATH + result_image_name, image)


if __name__ == "__main__":
    desc = "test"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="which gpu id, [0/1/2/3]")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./resources/anti_spoof_models",
        help="model_lib used to test")
    parser.add_argument(
        "--image_name",
        type=str,
        default="Portrait.jpg",
        help="image used to test")
    args = parser.parse_args()
    test(args.image_name, args.model_dir, args.device_id)
