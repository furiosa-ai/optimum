import argparse
from pathlib import Path

from optimum.litmus import compile_onnx, export_onnx, utils
from optimum.litmus.nlp import BATCH_SIZE, INPUT_LENGTH, TASKS, simplify


def main():
    parser = argparse.ArgumentParser("FuriosaAI litmus Stable Diffusion using HF Optimum API.")
    parser.add_argument("output_dir", type=Path, help="path to directory to save outputs")
    parser.add_argument(
        "--version", "-v", choices=["xl-base", "xl-refiner", "1.5", "2.1"], required=True, help="available model sizes"
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        default=BATCH_SIZE,
        type=utils.check_non_negative,
        help="Batch size for model inputs",
    )
    parser.add_argument(
        "--input-len",
        default=INPUT_LENGTH,
        type=utils.check_non_negative,
        help="Length of input prommpt",
    )
    parser.add_argument(
        "--gen-step",
        default=0,
        type=utils.check_non_negative,
        help="Generation step to simplify onnx graph",
    )
    parser.add_argument(
        "--task",
        default="image-to-text",
        type=str,
        choices=TASKS,
        help="Task to export model for",
    )
    args = parser.parse_args()

    version = args.version
    if version in ["xl-base", "xl-refiner"]:
        model_tag = f"stabilityai/stable-diffusion-{version}-0.9"
    elif version == "1.5":
        model_tag = "runwayml/stable-diffusion-v1-5"
    elif version == "2.1":
        model_tag = "stabilityai/stable-diffusion-2-1"
    else:
        raise ValueError(f"Unsupported version: {version}")

    output_dir = args.output_dir

    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    
    print("Exporting ONNX Model...")
    print(model_tag)
    export_onnx(model_name_or_path=model_tag, output=output_dir, task=args.task, framework="pt")

    # print("Simplifying ONNX Model...")
    # simplify(
    #     output_dir / "decoder_model_merged.onnx",
    #     output_dir,
    #     args.batch_size,
    #     args.input_len,
    #     args.gen_step,
    #     args.task,
    # )

    # compile_onnx(
    #     output_dir / f"decoder_model-opt_gen_step={args.gen_step}.onnx",
    #     output_dir / f"decoder_model-opt_gen_step={args.gen_step}.dfg",
    #     output_dir / f"decoder_model-opt_gen_step={args.gen_step}.dot",
    # )


if __name__ == "__main__":
    main()
