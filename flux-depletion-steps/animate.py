import cv2
import os
import natsort  


image_folder = os.getcwd()  
video_name = 'mesh_animation.mp4'


images = [img for img in os.listdir(image_folder) if img.startswith("flux_heat_step") and img.endswith(".png")]
images = natsort.natsorted(images)  

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape


video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 5, (width, height))
# 5: FPS
for img in images:
    print("Adding:", img)
    frame = cv2.imread(os.path.join(image_folder, img))
    video.write(frame)

video.release()
cv2.destroyAllWindows()

print(f"\n✅ Video created successfully: {video_name}")

