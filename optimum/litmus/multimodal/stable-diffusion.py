import argparse
from pathlib import Path

from optimum.litmus import utils
from optimum.litmus.multimodal import (
    BATCH_SIZE,
    INPUT_LENGTH,
    LATENT_HEIGHT,
    LATENT_WIDTH,
    compile,
    export,
    simplify,
)


def main():
    parser = argparse.ArgumentParser("FuriosaAI litmus Stable Diffusion using HF Optimum API.")
    parser.add_argument("output_dir", type=Path, help="path to directory to save outputs")
    parser.add_argument(
        "--version",
        "-v",
        choices=["1.5", "2.1"],
        required=True,
        help="available model versions",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        default=BATCH_SIZE,
        type=utils.check_non_negative,
        help="batch size for latent and prompt inputs",
    )
    parser.add_argument(
        "--latent_shape",
        default=(LATENT_HEIGHT, LATENT_WIDTH),
        nargs=2,
        type=int,
        metavar=("latent_height", "latent_width"),
        help="shape of latent input. Note it is 1/8 of output image sizes",
    )
    parser.add_argument(
        "--input-len",
        default=INPUT_LENGTH,
        type=utils.check_non_negative,
        help="length of input prompt",
    )
    args = parser.parse_args()

    version = args.version
    if version == "1.5":
        model_tag = "runwayml/stable-diffusion-v1-5"
    elif version == "2.1":
        model_tag = "stabilityai/stable-diffusion-2-1"
    else:
        raise ValueError(f"Unsupported version: {version}")

    task = "stable-diffusion"
    output_dir = args.output_dir

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    print("Exporting ONNX Model...")
    export(model_tag=model_tag, output_dir=output_dir, task=task, framework="pt")

    print("Simplifying ONNX Model...")
    latent_shape = tuple(args.latent_shape)
    simplify(
        output_dir,
        args.batch_size,
        latent_shape,
        args.input_len,
        task,
    )

    compile(output_dir)


if __name__ == "__main__":
    main()
