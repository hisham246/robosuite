import h5py
import matplotlib.pyplot as plt

path = "/home/hisham246/uwaterloo/robosuite_datasets/table_wiping/1772920271_630833/demo.hdf5"

with h5py.File(path, "r") as f:
    img = f["data"]["demo_1"]["obs"]["agentview_image"][0]

plt.imshow(img)
plt.title("demo_1 - agentview_image[0]")
plt.axis("off")
plt.show()