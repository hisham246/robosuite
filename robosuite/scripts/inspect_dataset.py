import h5py
import numpy as np
import sys
import os

path = "/home/hisham246/uwaterloo/wipe_demos/1772749431_4007533/demo.hdf5"
f = h5py.File(path, "r")

print("Top-level keys:", list(f.keys()))
print("data keys (episodes):", list(f["data"].keys())[:10], "...")
print("data attrs:", dict(f["data"].attrs))

# pick first demo group
demo_keys = sorted([k for k in f["data"].keys() if k.startswith("demo_")])
if not demo_keys:
    print("No demo_* groups found.")
    sys.exit(0)

ep = demo_keys[0]
g = f["data"][ep]
print("\nEpisode:", ep)
print("Episode attrs:", dict(g.attrs))
print("Datasets:", list(g.keys()))

# shapes / dtypes
for name in g.keys():
    d = g[name]
    if hasattr(d, "shape"):
        print(f"  - {name}: shape={d.shape}, dtype={d.dtype}")

# peek at states/actions
states = g["states"][()]
actions = g["actions"][()]
print("\nstates[0] first 10:", states[0][:10])
print("actions[0] first 10:", actions[0][:10])

# if you stored these attrs in your modified saver
if "state_dim_original" in g.attrs:
    S = int(g.attrs["state_dim_original"])
    print("\nstate_dim_original =", S)
    if states.shape[1] > S:
        ft = states[:, S:]
        print("appended dims =", ft.shape[1])
        print("ft[0] =", ft[0])
        print("ft min/max =", float(ft.min()), float(ft.max()))
else:
    print("\nNo state_dim_original attr found (states may be unmodified).")

ft = states[:, 15:21]
F = ft[:, :3]
Fmag = np.linalg.norm(F, axis=1)
print("Force magnitude stats:", Fmag.min(), Fmag.max(), Fmag[:10])

f.close()