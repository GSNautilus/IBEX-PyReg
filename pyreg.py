# pyreg.py 20260305_01
# Garrick Salois
# Script to register multiplexed immunofluorescence image datasets by DAPI channel
# Uses phase correlation for coarse translation and ANTsPyx SyN for deformable refinement
#
# Requirements: numpy, tifffile, antspyx, scikit-image, scipy
#
# Command examples:
#   python pyreg.py "D:\\202504 Ibex\\APP30"
#   python pyreg.py "D:\\202504 Ibex\\APP30" "D:\\202504 Ibex\\APP32"
#   python pyreg.py data --crop_size 1024 --threads 8
#   python pyreg.py data --axes ZCYX --dapi_channel 0
#
# CLI arguments:
#   input_dirs   One or more directories containing per-cycle TIFF stacks.
#
# CLI options:
#   --crop_size N      Center crop size for phase correlation (default: 2048)
#   --threads N        ITK/ANTs thread count via env var (default: 4)
#   --axes MODE        Input stack axis interpretation: auto, ZCYX, or CZYX (default: auto)
#   --dapi_channel N   Channel index used as DAPI registration target (default: 0)

from __future__ import annotations

import argparse
import os
import time
from glob import glob
from pathlib import Path

import ants
import numpy as np
import tifffile
from scipy.ndimage import shift as nd_shift
from skimage.registration import phase_cross_correlation


def configure_threads(num_threads: int) -> None:
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(num_threads)


def normalize_to_u12(stack: np.ndarray) -> np.ndarray:
    stack = stack.astype(np.float32, copy=False)
    min_val = float(stack.min())
    max_val = float(stack.max())
    if max_val > min_val:
        stack = (stack - min_val) / (max_val - min_val)
        stack = np.clip(stack, 0.0, 1.0) * 4095.0
    else:
        stack = np.zeros_like(stack, dtype=np.float32)
    return stack.astype(np.uint16)


def infer_and_convert_to_zcyx(img: np.ndarray, axes_mode: str) -> np.ndarray:
    if img.ndim != 4:
        raise ValueError(f"Expected 4D stack, got shape {img.shape}.")

    if axes_mode == "ZCYX":
        return img
    if axes_mode == "CZYX":
        return np.transpose(img, (1, 0, 2, 3))

    # Auto mode: infer whether first or second axis is channels.
    dim0, dim1 = img.shape[0], img.shape[1]
    likely_chan0 = dim0 <= 8
    likely_chan1 = dim1 <= 8

    if likely_chan0 and not likely_chan1:
        return np.transpose(img, (1, 0, 2, 3))
    if likely_chan1 and not likely_chan0:
        return img

    if likely_chan0 and likely_chan1:
        # Ambiguous case (both small): prefer whichever axis is smaller as channels.
        if dim0 < dim1:
            return np.transpose(img, (1, 0, 2, 3))
        return img

    raise ValueError(
        "Could not infer axes in auto mode. Use --axes ZCYX or --axes CZYX explicitly."
    )


def load_stack(path: str, axes_mode: str, dapi_channel: int) -> tuple[np.ndarray, np.ndarray]:
    img = tifffile.imread(path)
    img = infer_and_convert_to_zcyx(img, axes_mode)
    img = normalize_to_u12(img)

    if dapi_channel < 0 or dapi_channel >= img.shape[1]:
        raise IndexError(
            f"DAPI channel {dapi_channel} out of range for stack with {img.shape[1]} channels: {path}"
        )

    dapi = img[:, dapi_channel, :, :]
    return img, dapi


def max_project_channels(stack_zcyx: np.ndarray) -> np.ndarray:
    # Max over Z to obtain one YX image per channel.
    return np.max(stack_zcyx, axis=0)


def center_crop(img: np.ndarray, crop_size: int) -> np.ndarray:
    h, w = img.shape
    crop_h = min(crop_size, h)
    crop_w = min(crop_size, w)
    cy, cx = h // 2, w // 2
    half_h = crop_h // 2
    half_w = crop_w // 2
    y0 = max(0, cy - half_h)
    x0 = max(0, cx - half_w)
    y1 = y0 + crop_h
    x1 = x0 + crop_w
    return img[y0:y1, x0:x1]


def register_phase_corr(ref: np.ndarray, moving: np.ndarray, crop_size: int) -> np.ndarray:
    ref_crop = center_crop(ref, crop_size)
    moving_crop = center_crop(moving, crop_size)
    shift_estimate, _, _ = phase_cross_correlation(ref_crop, moving_crop, upsample_factor=100)
    return shift_estimate


def apply_shift_stack(stack: np.ndarray, shift: np.ndarray) -> np.ndarray:
    shifted_stack = np.zeros_like(stack)
    for z in range(stack.shape[0]):
        for c in range(stack.shape[1]):
            shifted_stack[z, c] = nd_shift(
                stack[z, c], shift=shift, order=1, mode="constant", cval=0.0
            )
    return shifted_stack


def register_ants(fixed: np.ndarray, moving: np.ndarray) -> dict:
    fixed_img = ants.from_numpy(fixed.astype(np.float32))
    moving_img = ants.from_numpy(moving.astype(np.float32))
    start = time.time()

    tx = ants.registration(
        fixed=fixed_img,
        moving=moving_img,
        type_of_transform="SyN",
        regIterations=(40, 20, 10),
        verbose=False,
    )

    duration = time.time() - start
    print(f"ANTs registration completed in {duration:.2f} seconds")

    try:
        field = ants.image_read(tx["fwdtransforms"][0])
        warp = field.numpy()

        if warp.shape[0] == 2:
            dx, dy = warp[0], warp[1]
        elif warp.shape[-1] == 2:
            dx, dy = warp[..., 0], warp[..., 1]
        else:
            raise ValueError("Unrecognized warp field shape.")

        print(f"Warp field shape: {warp.shape}")
        print(f"Displacement X: min={dx.min():.4f}, max={dx.max():.4f}")
        print(f"Displacement Y: min={dy.min():.4f}, max={dy.max():.4f}")
    except Exception as exc:
        print(f"Could not summarize warp field: {exc}")

    return tx


def apply_ants_transform(stack_cyx: np.ndarray, tx: dict, ref_shape_yx: tuple[int, int]) -> np.ndarray:
    channels, _, _ = stack_cyx.shape
    aligned = np.zeros((channels, ref_shape_yx[0], ref_shape_yx[1]), dtype=stack_cyx.dtype)
    ref = ants.from_numpy(np.zeros(ref_shape_yx, dtype=np.float32))

    for i in range(channels):
        moving = ants.from_numpy(stack_cyx[i].astype(np.float32))
        warped = ants.apply_transforms(fixed=ref, moving=moving, transformlist=tx["fwdtransforms"])
        aligned[i] = warped.numpy()

    return aligned


def extract_output_name(path: str) -> str:
    stem = Path(path).stem
    parts = stem.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}_Reg.tif"
    return f"{stem}_Reg.tif"


def process_directory(input_dir: str, crop_size: int, axes_mode: str, dapi_channel: int) -> None:
    output_dir = os.path.join(input_dir, "reg_output")
    os.makedirs(output_dir, exist_ok=True)

    tiff_files = sorted(glob(os.path.join(input_dir, "*.tif")))
    if len(tiff_files) < 2:
        raise ValueError(f"Need at least two TIFF stacks in {input_dir}. Found {len(tiff_files)}.")

    output_path = os.path.join(output_dir, extract_output_name(tiff_files[0]))

    ref_stack, ref_dapi = load_stack(tiff_files[0], axes_mode=axes_mode, dapi_channel=dapi_channel)
    ref_proj_dapi = np.max(ref_dapi, axis=0)
    ref_proj_all = max_project_channels(ref_stack)
    aligned_channels = [ref_proj_all]

    for path in tiff_files[1:]:
        cycle_start = time.time()
        print(f"\nRegistering {os.path.basename(path)}...")

        stack, dapi = load_stack(path, axes_mode=axes_mode, dapi_channel=dapi_channel)
        proj_dapi = np.max(dapi, axis=0)

        shift_estimate = register_phase_corr(ref_proj_dapi, proj_dapi, crop_size=crop_size)
        print(f"Phase correlation shift: {shift_estimate}")

        stack_shifted = apply_shift_stack(stack, shift_estimate)
        proj_all_shifted = max_project_channels(stack_shifted)
        proj_dapi_shifted = np.max(stack_shifted[:, dapi_channel, :, :], axis=0)

        tx = register_ants(ref_proj_dapi, proj_dapi_shifted)
        aligned_proj = apply_ants_transform(proj_all_shifted, tx, ref_proj_dapi.shape)
        aligned_channels.append(aligned_proj)

        cycle_duration = time.time() - cycle_start
        print(f"Cycle completed in {cycle_duration:.2f} seconds")

    final = np.concatenate(aligned_channels, axis=0)
    final_5d = final[np.newaxis, np.newaxis, ...]

    tifffile.imwrite(
        output_path,
        final_5d.astype(np.uint16),
        imagej=True,
        metadata={
            "axes": "TZCYX",
            "ImageJ": {"Ranges": [(0, 4095)] * final_5d.shape[2]},
        },
    )

    print(f"\nAligned stacks saved to:\n{output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register IBEX cycles by DAPI using phase correlation + ANTs SyN"
    )
    parser.add_argument("input_dirs", nargs="+", help="One or more directories containing TIFF stacks")
    parser.add_argument("--crop_size", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--axes", choices=["auto", "ZCYX", "CZYX"], default="auto")
    parser.add_argument("--dapi_channel", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_threads(args.threads)

    total_start = time.time()
    for input_dir in args.input_dirs:
        print(f"\nProcessing directory: {input_dir}")
        process_directory(
            input_dir,
            crop_size=args.crop_size,
            axes_mode=args.axes,
            dapi_channel=args.dapi_channel,
        )

    total_end = time.time()
    print(f"\nTotal runtime for all directories: {total_end - total_start:.2f} seconds")


if __name__ == "__main__":
    main()
