import h5py
import matplotlib.pyplot as plt
import numpy as np

path = "/home/hisham246/uwaterloo/robosuite_datasets/table_wiping/1772929779_6655116/demo.hdf5"

with h5py.File(path, "r") as f:
    img = f["data"]["demo_1"]["obs"]["robot0_eye_in_hand_image"][0]

img_fixed = np.flipud(img)

plt.imshow(img_fixed)
plt.title("demo_1 - robot0_eye_in_hand_image[0] flipped")
plt.axis("off")
plt.show()