import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from binarization_utils import threshold
from perspective_utils import birds_eye


def measure_curvature(warped):
    '''
    Calculates the curvature of polynomial functions in pixels.
    parameters:
        warped: Bird's eye view image
    return:
        left and right radius of curvatureof the lanes respectively
    '''
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(warped)
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
        
    ##### TO-DO: Implement the calculation of R_curve (radius of curvature) #####
    left_curverad = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    
    return left_curverad, right_curverad
    
    
if __name__ = '__main__'
    img = mpimg.imread(args.data_dir+ 'test_images/test1.jpg')
    binary = threshold(img)
    img, M = birds_eye(binary) 
    # Calculate the radius of curvature in pixels for both lane lines
    left_curverad, right_curverad = measure_curvature(img)

    # Visualization
    