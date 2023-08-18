import copy
from pathlib import Path
from typing import Optional, Tuple, Union

import onnx
from optimum.litmus import compile_onnx, export_onnx, simplify_onnx, utils

TASKS = ["stable-diffusion"]
BATCH_SIZE = 1
INPUT_LENGTH = 77
LATENT_HEIGHT = LATENT_WIDTH = 64


def export(model_tag: str, output_dir: Union[Path, str], task: str, framework: str = "pt") -> None:
    if task != "stable-diffusion":
        raise Exception(f"Unsupported model task: {task}")
    export_stable_diffusion_onnx(model_tag, output_dir, framework)


def simplify(
    input_model: Union[Path, onnx.ModelProto],
    batch_size: int,
    latent_shape: Tuple[int, int],
    input_len: int,
    time_step: int,
    task: str = "stable-diffusion",
) -> onnx.ModelProto:
    if task != "stable-diffusion":
        raise Exception(f"Unsupported model task: {task}")
    model_opt = simplify_stable_diffusion_onnx(
        input_model, batch_size, latent_shape, input_len, time_step
    )
    return model_opt


def compile(model_dir: Path, task: str = "stable-diffusion") -> None:
    if task != "stable-diffusion":
        raise Exception(f"Unsupported model task: {task}")
    compile_stable_diffusion_onnx(model_dir)


def export_stable_diffusion_onnx(
    model_tag: str, output_dir: Union[Path, str], framework: str = "pt"
):
    output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
    for module in ["text_encoder", "unet", "vae_decoder"]:
        if not (output_dir / module / "model.onnx").exists():
            print(f"Exporting {module} module...")
            export_onnx(
                model_name_or_path=model_tag,
                output=output_dir,
                task="stable-diffusion",
                framework="pt",
            )
        else:
            print("All ONNX modules have already been exproted.")


def simplify_stable_diffusion_onnx(
    input_model: Path,
    batch_size: int,
    latent_shape: Tuple[int, int],
    input_len: int,
    time_step: int,
):
    input_shapes = {
        "text_encoder": {"input_ids": [batch_size, input_len]},
        "unet": {
            "sample": [batch_size, 4, *latent_shape],
            "timestep": [batch_size],
            "encoder_hidden_states": [batch_size, input_len, 768],
        },
        "vae_decoder": {"latent_sample": [1, 4, *latent_shape]},
    }

    for module, fixed_input_shapes in input_shapes.items():
        print(f"Simplifying {module} module...")
        module_dir = input_model / module
        if module == "unet":
            onnx_model = utils.load_onnx(module_dir / "model.onnx")
            hidden_state_input = onnx_model.graph.input[2]
            assert hidden_state_input.name == "encoder_hidden_states"
            hidden_state_dim = hidden_state_input.type.tensor_type.shape.dim[2].dim_value
            fixed_input_shapes["encoder_hidden_states"][-1] = hidden_state_dim
            simplify_onnx(onnx_model, module_dir / "model-opt.onnx", fixed_input_shapes)
        else:
            simplify_onnx(
                module_dir / "model.onnx", module_dir / "model-opt.onnx", fixed_input_shapes
            )


def compile_stable_diffusion_onnx(model_dir: Path) -> None:
    for module in ["text_encoder", "unet", "vae_decoder"]:
        print(f"Compiling {module} module...")
        module_dir = model_dir / module
        compile_onnx(
            module_dir / "model-opt.onnx",
            module_dir / "model-opt.dfg",
            module_dir / "model-opt.dot",
        )
