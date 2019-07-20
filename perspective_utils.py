import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def birds_eye(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_size = (gray.shape[1], gray.shape[0])
    width , height = img_size
    src = np.float32([[220, 700],
                    [600, 460],
                    [750, 460],
                    [1180, 700]])
    dst = np.float32([[220, 700],
                    [220, 0],
                    [1180, 0],
                    [1180, 700]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped_img = cv2.warpPerspective(gray, M, (img_size))
    return warped_img, M
	

def rev_warp(RGB):
    img = threshold(RGB, image_type='RGB', s_thresh=(50, 255), sx_thresh=(20, 100))

    warped, M = birds_eye(img)
    undist = undist_img(img)
    Minv = np.linalg.inv(M)
    Minv =  np.linalg.inv(M)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result


if __name__ = '__main__'
    RGB = mpimg.imread(args.data_dir + '/test_images/test1.jpg')
    binary = threshold(RGB)
    beye = birds_eye(img)
    rev_warped = rev_warp(RGB)
    # Visualization
    f, (ax1, ax2, ax2) = plt.subplots(1, 2, figsize=(10,8))
    ax1.imshow(RGB)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(binary)
    ax2.set_title('Binary Image', fontsize=30)
    ax3.imshow(binary)
    ax3.set_title('Colored lane Image', fontsize=30)    

    
if __name__ = '__main__'
    img = mpimg.imread(data_dir + 'test_images/test1.jpg')
    beye, M = birds_eye(img)
    
    # Visualization
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,8))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(img_undistorted)
    ax2.set_title("Bird's eye Image", fontsize=30)
    