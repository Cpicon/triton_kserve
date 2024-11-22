import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from click.testing import CliRunner

# Import the functions and CLI
from imageclassifier import create_model_repository, pbtxt_generator
from imageclassifier.model_repository_cli import generate_ensemble_config


# Define a fixture for the CliRunner
@pytest.fixture
def runner():
    return CliRunner()


def generate_ensemble_config(
    ensemble_steps: list[dict[str, any]],
) -> str:
    """
    Generates a config.pbtxt file for an ensemble model.

    Args:
        ensemble_steps: List of ensemble step dictionaries with model configuration.
    """
    config_lines = []

    # Add ensemble scheduling
    config_lines.append('platform: "ensemble"')
    config_lines.append("ensemble_scheduling {")
    config_lines.append("  step [")

    for idx, step in enumerate(ensemble_steps):
        config_lines.append("    {")
        config_lines.append(f'      model_name: "{step["model_name"]}"')
        config_lines.append(f'      model_version: {step["model_version"]}')

        # Add input maps
        if "input_map" in step:
            for key, value in step["input_map"].items():
                config_lines.append("      input_map {")
                config_lines.append(f'        key: "{key}"')
                config_lines.append(f'        value: "{value}"')
                config_lines.append("      }")

        # Add output maps
        if "output_map" in step:
            for key, value in step["output_map"].items():
                config_lines.append("      output_map {")
                config_lines.append(f'        key: "{key}"')
                config_lines.append(f'        value: "{value}"')
                config_lines.append("      }")

        config_lines.append("    }")

    config_lines.append("  ]")
    config_lines.append("}")

    return "\n".join(config_lines)


def test_create_model_repository(tmp_path: Path):
    # Arrange
    model_name = "test_model"
    version = 1
    backend = "pytorch"
    config = {
        "input": {
            "name": "input_images:0",
            "data_type": "TYPE_FP32",
            "dims": [-1, 3, 384, 384],
        },
        "output": {
            "name": "feature_fusion/Sigmoid:0",
            "data_type": "TYPE_FP32",
            "dims": [-1, 7],
        },
    }
    base_path = tmp_path / "model_repository"

    # Act
    create_model_repository(
        model_name, version, backend, config, base_path=str(base_path)
    )

    # Assert
    model_path = base_path / model_name / str(version)
    config_path = base_path / model_name / "config.pbtxt"
    assert model_path.exists()
    assert config_path.exists()

    # Check contents of config.pbtxt
    expected_config_content = f"""name: "{model_name}"
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
    with open(config_path, "r") as f:
        actual_config_content = f.read()
    assert actual_config_content == expected_config_content


def test_create_repository_command(runner, tmp_path: Path):
    # GIVEN
    model_name = "test_model"
    version = 1
    backend = "pytorch"
    config = {
        "input": {
            "name": "input_images:0",
            "data_type": "TYPE_FP32",
            "dims": [-1, 3, 384, 384],
        },
        "output": {
            "name": "feature_fusion/Sigmoid:0",
            "data_type": "TYPE_FP32",
            "dims": [-1, 7],
        },
    }
    config_json = json.dumps(config)
    config_path = tmp_path / "config.json"
    config_path.write_text(config_json)

    # Verify that the model repository was created
    model_path = tmp_path / model_name / str(version)
    config_pbtxt = tmp_path / model_name / "config.pbtxt"

    # WITH
    result = runner.invoke(
        pbtxt_generator,
        [
            "create-repository",
            model_name,
            str(version),
            backend,
            str(config_path),
            "--base-path",
            str(tmp_path),
        ],
    )
    # THEN
    assert result.exit_code == 0
    assert (
        f"Successfully created model repository for {model_name}"
        in result.output
    )
    assert model_path.exists()
    assert config_pbtxt.exists()


def test_create_repository_command_invalid_json(runner, tmp_path: Path):
    # GIVEN
    model_name = "test_model"
    version = 1
    backend = "pytorch"
    invalid_config_content = "{invalid_json: true}"
    config_path = tmp_path / "config.json"
    config_path.write_text(invalid_config_content)

    # WITH
    result = runner.invoke(
        pbtxt_generator,
        [
            "create-repository",
            model_name,
            str(version),
            backend,
            str(config_path),
        ],
    )

    # THEN
    assert result.exit_code == 0
    assert "Error: Config file must be a valid JSON file" in result.output


def test_download_model_command_success(
    monkeypatch,
    runner,
    tmp_path: Path,
):

    # GIVEN
    model_name = "model_name"
    version = 1
    model_repository_path = (
        tmp_path / "model_repository" / model_name / str(version)
    )
    model_file = model_repository_path / "model.pt"
    # WITH
    model_repository_path.mkdir(parents=True)

    # Mock timm.create_model in the imageclassifier.model_repository_cli module
    mock_model = Mock()
    monkeypatch.setattr(
        "imageclassifier.model_repository_cli.timm.create_model",
        Mock(return_value=mock_model),
    )

    # Mock torch.jit.trace in the imageclassifier.model_repository_cli module
    mock_traced_model = Mock()
    monkeypatch.setattr(
        "imageclassifier.model_repository_cli.torch.jit.trace",
        Mock(return_value=mock_traced_model),
    )

    # Mock torch.jit.save in the imageclassifier.model_repository_cli module
    mock_torch_save = Mock()
    monkeypatch.setattr(
        "imageclassifier.model_repository_cli.torch.jit.save", mock_torch_save
    )
    result = runner.invoke(
        pbtxt_generator,
        [
            "download-model",
            model_name,
            "--base-path",
            str(tmp_path / "model_repository"),
        ],
    )

    # THEN
    assert result.exit_code == 0
    assert (
        f"Successfully downloaded model {model_name} to {model_file}"
        in result.output
    )
    mock_torch_save.assert_called_once()
