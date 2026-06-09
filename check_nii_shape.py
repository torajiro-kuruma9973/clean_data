from pathlib import Path
from typing import Union

import nibabel as nib


def print_nii_shape(nii_path: Union[str, Path]) -> None:
    nii_path = Path(nii_path)
    img = nib.load(str(nii_path))
    print(f"{nii_path}: shape = {img.shape}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Print NIfTI file shape.")
    parser.add_argument("nii_path", help="Path to .nii or .nii.gz file")
    args = parser.parse_args()

    print_nii_shape(args.nii_path)


if __name__ == "__main__":
    main()