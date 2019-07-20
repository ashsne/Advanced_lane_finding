from args import *
from binarization_utils import threshold
from calibration_utils import undist_img
from lane_utils import fit_polynomial
from perspective_utils import birds_eye
from moviepy.editor import VideoFileClip



def rev_warp(warped, M):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    left_fitx, right_fitx, ploty, image = fit_polynomial(warped)
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Inverse matrix for the reverse warping
    Minv = np.linalg.inv(M)
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (warped.shape[1], warped.shape[0])) 
    return newwarp


def pipeline(RGB):
    # Undistort the image
    undist = undist_img(RGB)
    
    # BInariza the image with sobel and color threshold
    img = threshold(RGB, image_type='RGB', s_thresh=(50, 255), sx_thresh=(20, 100))
    
    # Bird's eye of the image
    warped, M = birds_eye(img)
    
    # Get the lane path to original shape form the warped shape
    newwarp = rev_warp(warped, M)
    
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    #plt.imshow(result)

    return result

if __name__ == '__main__':
     
     ret, mtx, dist, rvecs, tvecs = get_camera_calibration(data_dir)
     
    mode = 'video'
    if mode == 'video':
        selector = 'project'
        clip = VideoFileClip('{}_video.mp4'.format(selector)).fl_image(pipeline)
        clip.write_videofile('{}_video_result_{}.mp4'.format(selector, time_window), audio=False)
                
    else:
        image_paths = glob.glob(data_dir + '/test_images/*.jpg')
        for path in image_paths:
            RGB = mpimg.imread(path)
            result = pipeline(RGB)
            write_name = data_dir + 'output_images/' + path.split('\\')[-1]
            
            cv2.imwrite(write_name, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            plt.imshow(result)
            plt.show()