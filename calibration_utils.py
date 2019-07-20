import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle

'''
Functions:
    calibrate_camera()
    get_camera_calibration()
    undist_img()
'''

class Args():
    data_dir = './'
args = Args()


def calibrate_camera(data_dir):
    '''
    Camera calibration
    parameters:
        data_dir:   Data directory 
                    where the calibation image forlder (camera_cal) is stored
    return:
        camera calibration parameters (ret, mtx, dist, rvecs, tvecs)
    '''
    # chess board 
    nx = 9
    ny = 6

    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

    # Arrays to store object points and image pointsa from all the images
    objpoints = [] # 3D ponits in real world
    imgpoints = [] # 2D points in image plane

    # Make a directory store all the images for camera calibration
    image_paths = glob.glob(data_dir + 'camera_cal/calibration*.jpg')

    # Step throught the list, find the corners and draw them
    for idx, fname in enumerate(image_paths):
        img = mpimg.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

        # If the corners are found draw them on the image and save
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw them and save
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            #write_name = data_dir + 'output_images/' + 'corners_found'+ str(idx) +'.jpg'
            #cv2.imwrite(write_name, img)
            #cv2.imshow('img', img)
            #cv2.waitKey(500)

    #cv2.destroyAllWindows()
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

    dist_pickle = {}
    dist_pickle["ret"] =ret
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    dist_pickle["rvecs"] = rvecs
    dist_pickle["tvecs"] = tvecs
    pickle.dump( dist_pickle, open(data_dir +  "camera_cal/wide_dist_pickle.p", "wb" ) )
    return ret, mtx, dist, rvecs, tvecs

def get_camera_calibration(data_dir):
    dist_pickle = pickle.load(open(data_dir +  "camera_cal/wide_dist_pickle.p", "rb" ) )
    ret = dist_pickle["ret"]
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    rvecs = dist_pickle["rvecs"]
    tvecs = dist_pickle["tvecs"]
    return ret, mtx, dist, rvecs, tvecs

def undist_img(img):
    '''
    Parameters:
        Distorted image
    return:
        Distortion corrected image
    '''
    # Calibrate the camera for the given images
    try:
        ret, mtx, dist, rvecs, tvecs = get_camera_calibration(args.data_dir)
    except:
            ret, mtx, dist, rvecs, tvecs = calibrate_camera(args.data_dir)
    # Undistort the image
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


if __name__ == '__main__':
    img = mpimg.imread(args.data_dir + '/camera_cal/calibration1.jpg')
    img_undistorted = undistort(img, mtx, dist)

    cv2.imwrite('img/test_calibration_before.jpg', img)
    cv2.imwrite('img/test_calibration_after.jpg', img_undistorted)
    
    # Visualize undistortion
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,8))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(img_undistorted)
    ax2.set_title('Undistorted Image', fontsize=30)
    
