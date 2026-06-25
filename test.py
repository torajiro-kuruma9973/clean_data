import json, nibabel as nib, numpy as np

n = "0"
seg = np.asanyarray(nib.load(f"seg/{n}.nii.gz").dataobj)
fg_frames = {k for k in range(seg.shape[2]) if np.any(seg[:, :, k] != 0)}

split = json.load(open("split.json"))
covered = set()
for s, e in zip(split[n]["starters"], split[n]["ends"]):
    covered.update(range(s, e))

missing = sorted(fg_frames - covered)
print("前景帧:", sorted(fg_frames))
print("区间未覆盖的前景帧:", missing)