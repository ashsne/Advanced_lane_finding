from args import *
from calibration_utils import undist_img

'''
Functions:
    threshold()
'''
def threshold(img, image_type='RGB', s_thresh=(50, 255), sx_thresh=(20, 100)):
    
    '''
    Input: RGB image
    Output: Color binary image 
    
    The function takes an RGB image and undergoes following steps:
    1. Undistortion 
    2. Cnvert the image into HLS color space
    3. Apply sobel x and threshold gradient
    4. Apply color gradient on s channel
    5. Stack the images for color binary
    '''
    
    # Convert the image into HLS color space
    if image_type == 'RGB':
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if image_type == 'BGR':
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:,:,2]
    
    # Sobel x
    sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0) # Gradient in the X direction
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx)) # Convert values into uint8
    
    # Threshold x gradient
    sx_binary = np.zeros_like(scaled_sobel)
    sx_binary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])]
    
    # Stack each channel to provide color binary
    color_binary = np.dstack((np.zeros_like(sx_binary), sx_binary, s_binary)) * 255
    color_binary = cv2.cvtColor(color_binary, cv2.COLOR_RGB2GRAY)
    return color_binary

if __name__ == '__main__':
    img = mpimg.imread(data_dir + 'test_images/test1.jpg')
    # Undistort the image
    img = undist_img(img)
    binary_img = threshold(img)
    
    # Visualization
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,8))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=20)
    ax2.imshow(binary_img)
    ax2.set_title('Binary Image', fontsize=20)
    