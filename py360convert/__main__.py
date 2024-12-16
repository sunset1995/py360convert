import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import py360convert


def _assert_height_width(args):
    if args.height is None:
        print("Error: --height required.")
        sys.exit(1)
    if args.width is None:
        print("Error: --width required.")
        sys.exit(1)


def _size_to_dims(args):
    if args.size is not None:
        if args.format == "horizon":
            args.height = args.size
            args.width = args.size * 6
        elif args.format == "dice":
            args.height = args.size * 3
            args.width = args.size * 4
        else:
            raise NotImplementedError("")
    _assert_height_width(args)


def main():
    parser = argparse.ArgumentParser(
        description="Conversion between cubemap and equirectangular or equirectangular to perspective.", add_help=False
    )
    parser.add_argument("--help", action="help", help="Show this help message and exit")

    subparsers = parser.add_subparsers(dest="command", help="Convert command.")

    c2e_parser = subparsers.add_parser("c2e", help="Convert cubemap to equirectangular.", add_help=False)
    c2e_parser.add_argument("--help", action="help", help="Show this help message and exit")
    c2e_parser.add_argument("input", type=Path, help="Path to input image.")
    c2e_parser.add_argument("output", type=Path, help="Path to output image.")
    c2e_parser.add_argument("--height", "-h", type=int, required=True, help="Output image height in pixels.")
    c2e_parser.add_argument("--width", "-w", type=int, required=True, help="Output image width in pixels.")
    c2e_parser.add_argument(
        "--mode", "-m", default="bilinear", choices=["bilinear", "nearest"], help="Resampling method."
    )

    e2c_parser = subparsers.add_parser("e2c", help="Convert equirectangular to cubemap.", add_help=False)
    e2c_parser.add_argument("--help", action="help", help="Show this help message and exit")
    e2c_parser.add_argument("input", type=Path, help="Path to input image.")
    e2c_parser.add_argument("output", type=Path, help="Path to output image.")
    e2c_parser.add_argument("--format", "-f", choices=["horizon", "dice"], default="dice", help="Output image layout.")
    e2c_parser.add_argument("--height", "-h", type=int, help="Output image height in pixels.")
    e2c_parser.add_argument("--width", "-w", type=int, help="Output image width in pixels.")
    e2c_parser.add_argument("--size", "-s", type=int, help="Side length of each cube face. Overrides height/width.")
    e2c_parser.add_argument(
        "--mode", "-m", default="bilinear", choices=["bilinear", "nearest"], help="Resampling method."
    )

    e2p_parser = subparsers.add_parser("e2p", help="Convert equirectangular to perspective.", add_help=False)
    e2p_parser.add_argument("--help", action="help", help="Show this help message and exit")
    e2p_parser.add_argument("input", type=Path, help="Path to input image.")
    e2p_parser.add_argument("output", type=Path, help="Path to output image.")
    e2p_parser.add_argument("--height", "-h", type=int, required=True, help="Output image height in pixels.")
    e2p_parser.add_argument("--width", "-w", type=int, required=True, help="Output image width in pixels.")
    e2p_parser.add_argument("--h-fov", type=float, default=60, help="Horizontal FoV in degrees.")
    e2p_parser.add_argument("--v-fov", type=float, default=60, help="Vertical FoV in degrees.")
    e2p_parser.add_argument("--h-view", type=float, default=0, help="Horizontal viewing angle in degrees.")
    e2p_parser.add_argument("--v-view", type=float, default=0, help="Vertical viewing angle in degrees.")
    e2p_parser.add_argument(
        "--mode", "-m", default="bilinear", choices=["bilinear", "nearest"], help="Resampling method."
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)
    elif args.command == "c2e":
        _assert_height_width(args)
        img = np.array(Image.open(args.input))
        out = py360convert.c2e(img, h=args.height, w=args.width, mode=args.mode)  # pyright: ignore[reportCallIssue]
        Image.fromarray(out).save(args.output)
    elif args.command == "e2c":
        _size_to_dims(args)
        img = np.array(Image.open(args.input))
        out = py360convert.e2c(
            img,
            face_w=args.width,
            mode=args.mode,
            cube_format=args.format,
        )  # pyright: ignore[reportCallIssue]
        Image.fromarray(out).save(args.output)
    elif args.command == "e2p":
        _assert_height_width(args)
        img = np.array(Image.open(args.input))
        out = py360convert.e2p(
            img,
            fov_deg=(args.h_fov, args.v_fov),
            u_deg=args.h_view,
            v_deg=args.v_view,
            out_hw=(args.h, args.w),
            in_rot_deg=0.0,
            mode=args.mode,
        )
        Image.fromarray(out).save(args.output)
    else:
        raise NotImplementedError(f'Command "{args.command}" not yet implemented.')


main()
