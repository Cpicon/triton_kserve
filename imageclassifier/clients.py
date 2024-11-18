import json
import os
from pathlib import Path
from typing import Any, Dict

import click
import timm
import torch


def create_model_repository(
    model_name: str,
    version: int,
    backend: str,
    config: Dict[str, Any],
    base_path: str = "model_repository",
) -> None:
    """
    Create the model repository structure and config file for a Triton model.

    Args:
        model_name: Name of the model
        version: Model version number
        backend: Backend to use (e.g. "pytorch", "onnx", etc)
        config: Dictionary containing input and output tensor configurations
        base_path: Base path for model repository
    """
    # Create directory structure, check if the version already exists, if then increment the version +1 of the last version available
    model_path = Path(base_path) / model_name / str(version)
    while model_path.exists():
        version += 1
        model_path = Path(base_path) / model_name / str(version)
    model_path.mkdir(parents=True, exist_ok=True)

    # TODO: input and output are a list of dictionaries, so we need to iterate over them
    # Create config.pbtxt content
    config_content = f"""name: "{model_name}"
backend: "{backend}"
input [
  {{
    name: "{config['input']['name']}"
    data_type: {config['input']['data_type']}
    dims: {config['input']['dims']}
  }}
]
output [
  {{
    name: "{config['output']['name']}"
    data_type: {config['output']['data_type']}
    dims: {config['output']['dims']}
  }}
]
"""

    # Write config file
    config_path = Path(base_path) / model_name / "config.pbtxt"
    with open(config_path, "w") as f:
        f.write(config_content)


@click.group()
def pbtxt_generator():
    """CLI tool for generating Triton model repository structure."""
    pass


@pbtxt_generator.command()
@click.argument("model_name")
@click.argument("version", type=int)
@click.argument("backend")
@click.argument("config_path", type=click.Path(exists=True))
@click.option(
    "--base-path",
    default="model_repository",
    help="Base path for model repository",
)
def create_repository(
    model_name: str,
    version: int,
    backend: str,
    config_path: str,
    base_path: str = "model_repository",
):
    """
    Create a model repository structure for Triton Inference Server.

    Example usage:
    python imageclassifier/clients.py create-repository vit_base_patch16_384 1 pytorch config.json

    Example config.json content:
    {
        "input": {
            "name": "input_images:0",
            "data_type": "TYPE_FP32",
            "dims": [-1, 3, 384, 384]
        },
        "output": {
            "name": "feature_fusion/Conv_7/Sigmoid:0",
            "data_type": "TYPE_FP32",
            "dims": [-1, 7]
        }
    }
    """
    try:
        with open(config_path) as f:
            config = json.load(f)
        create_model_repository(
            model_name, version, backend, config, base_path
        )
        click.echo(f"Successfully created model repository for {model_name}")
    except json.JSONDecodeError:
        click.echo("Error: Config file must be a valid JSON file")
    except Exception as e:
        click.echo(f"Error creating repository: {str(e)}")


@pbtxt_generator.command()
@click.argument("model_name")
@click.option(
    "--base-path",
    default="model_repository",
    help="Base path of model repository",
)
def download_model(model_name: str, base_path: str = "model_repository"):
    """
    Download a pretrained model from timm and save it to the model repository.
    The model will be saved in the highest version number directory found.
    If no version exists, it will create version 1.

    Example usage:
    python imageclassifier/clients.py download-model vit_base_patch16_384
    """
    try:
        # Find the model base directory
        model_base_path = Path(base_path) / model_name
        if not model_base_path.exists():
            click.echo(
                f"Error: Model repository {model_base_path} does not exist. Please run create-repository first."
            )
            return

        # Find the highest version number
        versions = [
            int(p.name)
            for p in model_base_path.iterdir()
            if p.is_dir() and p.name.isdigit()
        ]
        if not versions:
            path = os.path.join(base_path, model_name)
            click.echo(
                f"Error: No version directories found in {path}. Please run create-repository first."
            )
            return

        latest_version = max(versions)
        model_path = model_base_path / str(latest_version)

        # Check if directory is empty
        if any(model_path.iterdir()):
            path = os.path.join(base_path, model_name, str(latest_version))
            click.echo(f"Error: Version {path} directory is not empty")
            return

        # Create and configure the model
        model = timm.create_model(model_name, pretrained=True, num_classes=7)
        # Save model state dict
        model_file = model_path / "model.pt"
        traced_model = torch.jit.trace(model, torch.randn(1, 3, 384, 384))
        torch.jit.save(traced_model, model_file)

        click.echo(
            f"Successfully downloaded model {model_name} to {model_file}"
        )
    except Exception as e:
        click.echo(f"Error downloading model: {str(e)}")


if __name__ == "__main__":
    pbtxt_generator()
