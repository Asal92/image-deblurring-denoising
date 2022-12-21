import cv2
import os
from PIL import Image
from PIL import ImageFilter
import random
import numpy as np
import re


def blur_image(folder_path, new_path, img_f, new_img):

    im = Image.open(os.path.join(folder_path,img_f))

    #blur_num = random.randint(8,14)
    
    im_blur = im.filter(ImageFilter.GaussianBlur(17))

    im_blur_resized = im_blur.resize((806, 1000))
    
    im_blur_resized.save(os.path.join(new_path,new_img))


def noise_image(folder_path, new_path, img_f, new_img):

    img = cv2.imread(os.path.join(folder_path,img_f),0)

    row,col= img.shape
    mean = 0
    var = 70
    sigma = var
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    noisy_img = img + gauss

    img_rezied = cv2.resize(noisy_img, (806, 1000))

    # save image
    cv2.imwrite(os.path.join(new_path,new_img), img_rezied)


def to_jpg(folder_path, new_path, img_f, new_img_name):

    im = Image.open(os.path.join(folder_path, img_f))
    im.save(os.path.join(new_path, new_img_name))



#size - in pixels, size of motion blur
#angel - in degrees, direction of motion blur
def apply_motion_blur(folder_path, new_path, img_f, new_img, k_size):
        img = cv2.imread(os.path.join(folder_path, img_f), 0)

        #Specify the kernel size.
        #The greater the size, the more the motion.
        kernel_size = k_size # 75
        # Create the vertical kernel.
        kernel_v = np.zeros((kernel_size, kernel_size))
        # Create a copy of the same for creating the horizontal kernel.
        kernel_h = np.copy(kernel_v)

        # Fill the middle row with ones.
        kernel_v[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
        kernel_h[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)

        # Normalize.
        kernel_v /= kernel_size
        kernel_h /= kernel_size

        ran = random.randint(0,1)
        print(ran)
        if ran == 0:
            # Apply the vertical kernel.
            motion_blur_img = cv2.filter2D(img, -1, kernel_v)
        elif ran == 1:
            # Apply the horizontal kernel.
            motion_blur_img = cv2.filter2D(img, -1, kernel_h)

        # vertical_mb = cv2.filter2D(img, -1, kernel_v)
        # horizonal_mb = cv2.filter2D(img, -1, kernel_h)
        # Save the outputs.
        cv2.imwrite(os.path.join(new_path, new_img), motion_blur_img)


def resize(folder_path, new_path, img_f, new_img):

        img = cv2.imread(os.path.join(folder_path, img_f), 0)
        img_rezied = cv2.resize(img, (806, 1000))
        cv2.imwrite(os.path.join(new_path, new_img), img_rezied)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    folder_path = "/5527-DeepLearning/project/Datasets/dental-radiograms/Challenge1/RawImage/Test2Data"
    data_dir = os.listdir(folder_path)
    new_dir= folder_path+'/noise'


    try:
        os.mkdir(new_dir)
    except FileExistsError:
        pass

    # detec and crop for all images in the directory
    print("Reading and Blurring....")

    for img in sorted(data_dir):
        new_img_name = re.sub(".bmp", ".JPG", img)
        #blur_image(folder_path, new_dir, img, new_img_name)
        noise_image(folder_path, new_dir, img, new_img_name)
        #apply_motion_blur(folder_path, new_dir, img, new_img_name, 75)
        #resize(folder_path, new_dir, img, new_img_name)


    print("All Done!")

