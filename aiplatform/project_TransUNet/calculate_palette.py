import cv2
import matplotlib.pyplot as plt
import glob
import os

files = glob.glob("data/Synapse/ann_dir/val/*.jpg")
print(len(files))

while True:
    path = "data/Synapse/ann_dir/val/case0002_115.jpg"
    img=cv2.imread(path)
    filename = os.path.basename(path)
    cv2.imshow(f"{filename}",img)
    if cv2.waitKey(10)==ord('v'):
        break
cv2.destroyAllWindows()