"""
Author: Yorai Shaoul (yorai@cmu.edu)

A short script that reads two image sequences and produces, for each pair of images, a horizontally-stacked image.
"""
import os
import shutil
import cv2
import numpy as np
import imageio

# Folder to output images to.
out_dirp = './vis_output/nordicharbor_fish_stereo'
# Save images too?
is_save_images = False

# Folder path to the first sequence.
dirp1 = '/home/yoraish/code/tartanvo-fisheye/data/NordicHarbor/Data_easy/P000/image_rcam_fish'
dirp2 = '/home/yoraish/code/tartanvo-fisheye/data/NordicHarbor/Data_easy/P000/image_lcam_fish'

# Get the frame names from the directories.
fnames1 = os.listdir(dirp1)
fnames2 = os.listdir(dirp2)

# Sort by name. This assumes that image names are meaningful for sorting in this very particular way.
fnames1_zipped = sorted([(int(fname.split("_")[0]), fname) for fname in fnames1])
fnames2_zipped = sorted([(int(fname.split("_")[0]), fname) for fname in fnames2])

fnames1 = [f[1] for f in fnames1_zipped] 
fnames2 = [f[1] for f in fnames2_zipped] 

# Would you like your frames padded?
rpad = 20
cpad = 20

# Would you like to resize your images?
H = 300
W = 300

# Create output directory if needed.
if os.path.exists(out_dirp):
    print("The output folder already exists.")
    is_overwirte = input("Would you like to overwrite? [Y/n]")
    print(is_overwirte)
    if is_overwirte in [None, 'y', 'Y', '']:
        shutil.rmtree(out_dirp)
    else:
        raise FileExistsError("File exists and chosen to be kept. See you later!")

# Create output directory.
os.mkdir(out_dirp)

# List to save images.
img_list = []

for ix in range(min(len(fnames1), len(fnames2))):
    fname1, fname2 = fnames1[ix], fnames2[ix]
    print(fname1, fname2)
    img1 = cv2.imread(os.path.join(dirp1,fname1)) 
    img2 = cv2.imread(os.path.join(dirp2,fname2)) 

    img1 = np.pad(img1, [(rpad, rpad), (cpad, cpad), (0,0)])
    img2 = np.pad(img2, [(rpad, rpad), (cpad, cpad), (0,0)])

    if H and W:
        img1, img2 = cv2.resize(img1, (H, W)), cv2.resize(img2, (H, W))

    # Connect the images.
    out = np.hstack((img1, img2))
    
    # Show.
    cv2.imshow("out", out)
    cv2.waitKey(30)

    # Output file path.
    out_filep = os.path.join(out_dirp,str(ix)+".png")
    if is_save_images:
        cv2.imwrite(out_filep, out)    

    # Save to list.
    img_list.append(out)

imageio.mimsave(os.path.join(out_dirp, 'movie.gif'), img_list)

