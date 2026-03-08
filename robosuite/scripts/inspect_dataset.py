import h5py
import numpy as np
import os

path = "/home/hisham246/uwaterloo/robosuite_datasets/table_wiping/1772928862_7047362/demo.hdf5"

def summarize_array(name, arr, max_preview=8):
    print(f"    {name}: shape={arr.shape}, dtype={arr.dtype}")
    if arr.ndim >= 2 and arr.shape[0] > 0:
        print(f"      first row (first {max_preview} vals): {arr[0, :max_preview]}")
        print(f"      last  row (first {max_preview} vals): {arr[-1, :max_preview]}")
        print(f"      min={arr.min():.6f}, max={arr.max():.6f}, mean={arr.mean():.6f}")
    elif arr.ndim == 1 and arr.shape[0] > 0:
        print(f"      first {max_preview} vals: {arr[:max_preview]}")
        print(f"      min={arr.min():.6f}, max={arr.max():.6f}, mean={arr.mean():.6f}")

with h5py.File(path, "r") as f:
    print(f"Opened: {path}")
    print("=" * 80)

    print("Top-level keys:", list(f.keys()))
    if "data" not in f:
        raise RuntimeError("No 'data' group found in this HDF5 file.")

    data_grp = f["data"]

    print("\n[DATA GROUP ATTRIBUTES]")
    for k, v in data_grp.attrs.items():
        print(f"  {k}: {v}")

    demo_keys = sorted([k for k in data_grp.keys() if k.startswith("demo_")])
    print(f"\nFound {len(demo_keys)} demo episodes")
    print("Demo keys:", demo_keys[:10], "..." if len(demo_keys) > 10 else "")

    if not demo_keys:
        raise RuntimeError("No demo_* groups found.")

    print("\n" + "=" * 80)
    print("[PER-DEMO INSPECTION]")
    print("=" * 80)

    for ep in demo_keys:
        g = data_grp[ep]
        print(f"\nEpisode: {ep}")
        print("-" * 80)

        # print("  Attributes:")
        # for k, v in g.attrs.items():
        #     print(f"    {k}: {v}")

        print("  Datasets:")
        for name in g.keys():
            d = g[name]
            if hasattr(d, "shape"):
                print(f"    {name}: shape={d.shape}, dtype={d.dtype}")

        if "states" not in g or "actions" not in g:
            print("  Missing 'states' or 'actions' dataset, skipping detailed inspection.")
            continue

        states = g["states"][()]
        actions = g["actions"][()]

        print("\n  [RAW DATA SUMMARY]")
        summarize_array("states", states)
        summarize_array("actions", actions)

        T = states.shape[0]
        print(f"    Number of timesteps: {T}")

        # Split original state vs appended FT if attrs exist
        if "state_dim_original" in g.attrs:
            S = int(g.attrs["state_dim_original"])
            print(f"\n  state_dim_original = {S}")

            state_orig = states[:, :S]
            state_extra = states[:, S:]

            summarize_array("state_orig", state_orig)

            if state_extra.shape[1] > 0:
                summarize_array("state_extra", state_extra)

                # If this is standard single-arm FT appended as [Fx,Fy,Fz,Tx,Ty,Tz]
                if state_extra.shape[1] >= 6:
                    F = state_extra[:, :3]
                    Tau = state_extra[:, 3:6]
                    Fmag = np.linalg.norm(F, axis=1)
                    Tmag = np.linalg.norm(Tau, axis=1)

                    print("\n  [APPENDED FT SUMMARY]")
                    print(f"    F first row:   {F[0]}")
                    print(f"    Tau first row: {Tau[0]}")
                    print(f"    |F| min/max/mean: {Fmag.min():.6f} / {Fmag.max():.6f} / {Fmag.mean():.6f}")
                    print(f"    |T| min/max/mean: {Tmag.min():.6f} / {Tmag.max():.6f} / {Tmag.mean():.6f}")

                    # Show a few example rows
                    print("    First 5 |F|:", Fmag[:5])
                    print("    First 5 |T|:", Tmag[:5])
            else:
                print("  No appended dimensions after original state.")
        else:
            print("\n  No 'state_dim_original' attr found, cannot reliably split appended data.")

        # Check consistency
        if states.shape[0] != actions.shape[0]:
            print(f"\n  WARNING: states and actions length mismatch: {states.shape[0]} vs {actions.shape[0]}")
        else:
            print(f"\n  states/actions length match: {states.shape[0]}")

    print("\n" + "=" * 80)
    print("Done.")