from args import *
from binarization_utils import threshold
from perspective_utils import birds_eye
from lane_utils import find_lane_pixels

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
            
    ##### TO-DO: Implement the calculation of R_curve (radius of curvature) #####
    left_curverad = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    
    left_curverad = round(left_curverad, 2)
    right_curverad = round(right_curverad, 2)
    
    lane_mid = np.mean(right_fit + left_fit)
    car_mid = warped.shape[1]/2
    off_center = (lane_mid - car_mid)*xm_per_pix
    off_center = round(off_center, 2)
    return left_curverad, right_curverad, off_center
    
if __name__ == '__main__':
    img = mpimg.imread(data_dir+ 'test_images/test1.jpg')
    binary = threshold(img)
    img, M = birds_eye(binary) 
    # Calculate the radius of curvature in pixels for both lane lines
    left_curverad, right_curverad, off_center = measure_curvature(img)

    print(left_curverad, right_curverad, off_center)
    