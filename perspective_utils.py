from args import *

def birds_eye(img):
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_size = (img.shape[1], img.shape[0])
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
    warped_img = cv2.warpPerspective(img, M, (img_size))
    return warped_img, M
	

if __name__ == '__main__':
    RGB = mpimg.imread(data_dir + '/test_images/test1.jpg')
    beye, M = birds_eye(RGB)

    # Visualization
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,8))
    ax1.imshow(RGB)
    ax1.set_title('Original Image', fontsize=10)
    ax2.imshow(beye)
    ax2.set_title("Bird's eye Image", fontsize=10)
