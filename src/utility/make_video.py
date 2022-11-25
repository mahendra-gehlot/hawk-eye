import glob
import cv2

img_array = []
all_files = sorted(glob.glob("reports/video_frames/*.jpg"))
for filename in all_files:
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter('reports/sr_videos.mp4', cv2.VideoWriter_fourcc(*'DIVX'),
                      15, size)

for i in range(len(img_array)):
    out.write(img_array[i])

out.release()
print("Video made successfully")
