import cv2
import os
import glob
import re


image_folder = "."

video_name = "flux_animation.mp4"


images = glob.glob(os.path.join(image_folder, "flux_cell_*.png"))


def sort_key(filename):
    # flux_cell_0_0.png -> (0, 0)
    match = re.search(r'flux_cell_(\d+)_(\d+)\.png', filename)
    if match:
        row = int(match.group(1))
        col = int(match.group(2))
        return (row, col)
    else:
        return (0, 0)

images = sorted(images, key=sort_key)

if not images:
    raise ValueError("Could not find png files!")


frame = cv2.imread(images[0])
height, width, layers = frame.shape

#fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fourcc = cv2.VideoWriter_fourcc(*'X264')  # H.264
video = cv2.VideoWriter(video_name, fourcc, 8 , (width, height))  # 5 fps

for image in images:
    img = cv2.imread(image)
    video.write(img)

video.release()
print(f"Video Saved: {video_name}")
