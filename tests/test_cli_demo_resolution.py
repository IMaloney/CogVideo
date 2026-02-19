"""
Tests for RESOLUTION_MAP fallback behavior in cli_demo.py

This module tests the graceful fallback mechanism when model names
are not found in the RESOLUTION_MAP dictionary.
"""

import logging
import sys
from unittest.mock import MagicMock, patch, Mock
import pytest


# Mock heavy dependencies before importing cli_demo
@pytest.fixture(scope="module", autouse=True)
def mock_heavy_imports():
    """Mock heavy ML dependencies to allow testing without installation."""
    mock_torch = MagicMock()
    mock_torch.dtype = MagicMock()
    mock_torch.bfloat16 = MagicMock()
    mock_torch.float16 = MagicMock()

    sys.modules['torch'] = mock_torch
    sys.modules['diffusers'] = MagicMock()
    sys.modules['diffusers.utils'] = MagicMock()

    yield

    # Cleanup
    for module in ['torch', 'diffusers', 'diffusers.utils']:
        if module in sys.modules:
            del sys.modules[module]


@pytest.fixture
def resolution_map():
    """Fixture providing the RESOLUTION_MAP from cli_demo."""
    # Import after mocking dependencies
    from inference.cli_demo import RESOLUTION_MAP

    return RESOLUTION_MAP


@pytest.fixture
def cli_demo_module():
    """Fixture providing the cli_demo module."""
    import inference.cli_demo as cli_demo

    return cli_demo


class TestResolutionMapFallback:
    """Test suite for RESOLUTION_MAP lookup and fallback behavior."""

    def test_valid_model_name_returns_correct_resolution(
        self, resolution_map, cli_demo_module, caplog
    ):
        """Test that valid model names return their correct resolutions."""

        # Test each valid model in RESOLUTION_MAP
        test_cases = [
            ("THUDM/CogVideoX-5b", "cogvideox-5b", (480, 720)),
            ("THUDM/CogVideoX-2b", "cogvideox-2b", (480, 720)),
            ("THUDM/CogVideoX-5b-I2V", "cogvideox-5b-i2v", (480, 720)),
            ("THUDM/CogVideoX1.5-5b", "cogvideox1.5-5b", (768, 1360)),
            ("THUDM/CogVideoX1.5-5b-I2V", "cogvideox1.5-5b-i2v", (768, 1360)),
        ]

        for model_path, expected_key, expected_resolution in test_cases:
            with caplog.at_level(logging.WARNING):
                caplog.clear()

                # Extract model name using the same logic as cli_demo
                model_name = model_path.split("/")[-1].lower()

                # Verify the model name is in the map
                assert (
                    expected_key in resolution_map
                ), f"Expected key '{expected_key}' not in RESOLUTION_MAP"
                assert resolution_map[expected_key] == expected_resolution

                # Test that accessing the resolution works without KeyError
                try:
                    actual_resolution = resolution_map[model_name]
                    assert actual_resolution == expected_resolution
                except KeyError:
                    pytest.fail(f"Valid model '{model_name}' raised KeyError")

    def test_invalid_model_name_raises_key_error(self, resolution_map):
        """Test that invalid model names raise KeyError (which is then caught by try/except)."""

        invalid_models = [
            "invalidmodel-xyz",
            "unknown-5b",
            "cogvideox-5b-custom",  # Not in map
        ]

        for invalid_model in invalid_models:
            with pytest.raises(KeyError):
                _ = resolution_map[invalid_model]

    def test_local_path_model_name_extraction(self):
        """Test that local paths are parsed correctly to extract model names."""

        local_paths = [
            ("./local_models/CogVideoX-5B", "cogvideox-5b"),
            ("/path/to/custom_model", "custom_model"),
            ("~/models/my-cogvideo-model", "my-cogvideo-model"),
            ("./CogVideoX-Custom", "cogvideox-custom"),
        ]

        for local_path, expected_model_name in local_paths:
            # Extract model name using the same logic as cli_demo
            model_name = local_path.split("/")[-1].lower()
            assert model_name == expected_model_name

    def test_fallback_code_structure(self, cli_demo_module):
        """Test that the fallback code structure is present in generate_video function."""
        import inspect

        # Get the source code of generate_video
        source = inspect.getsource(cli_demo_module.generate_video)

        # Verify try/except block exists
        assert "try:" in source, "generate_video should have try/except block"
        assert "except KeyError:" in source, "Should catch KeyError specifically"
        assert (
            "RESOLUTION_MAP[model_name]" in source
        ), "Should access RESOLUTION_MAP with model_name"

        # Verify fallback logic exists
        assert (
            "default_resolution" in source or "(480, 720)" in source
        ), "Should define default resolution"
        assert "logging.warning" in source, "Should log warning on fallback"
        assert "valid models" in source.lower(), "Warning should mention valid models"

    def test_resolution_map_completeness(self, resolution_map):
        """Test that RESOLUTION_MAP contains expected models."""
        expected_models = {
            "cogvideox1.5-5b-i2v",
            "cogvideox1.5-5b",
            "cogvideox-5b-i2v",
            "cogvideox-5b",
            "cogvideox-2b",
        }

        assert (
            set(resolution_map.keys()) == expected_models
        ), "RESOLUTION_MAP should contain all expected models"

        # Verify all resolutions are tuples of two integers
        for model_name, resolution in resolution_map.items():
            assert isinstance(resolution, tuple), f"Resolution for '{model_name}' should be a tuple"
            assert (
                len(resolution) == 2
            ), f"Resolution for '{model_name}' should have 2 elements (height, width)"
            assert all(
                isinstance(dim, int) for dim in resolution
            ), f"Resolution for '{model_name}' should contain integers"

    def test_fallback_warning_message_format(self, cli_demo_module, caplog):
        """Test the format and content of the fallback warning message."""

        # Mock the generate_video call to test just the resolution lookup part
        with (
            patch.object(cli_demo_module, 'CogVideoXPipeline') as mock_pipeline,
            patch.object(cli_demo_module, 'CogVideoXImageToVideoPipeline'),
            patch.object(cli_demo_module, 'CogVideoXVideoToVideoPipeline'),
            patch.object(cli_demo_module, 'export_to_video'),
            caplog.at_level(logging.WARNING),
        ):
            # Setup mock to avoid actual model loading
            mock_instance = MagicMock()
            mock_pipeline.from_pretrained.return_value = mock_instance
            mock_instance.return_value = MagicMock(frames=[])

            caplog.clear()

            try:
                cli_demo_module.generate_video(
                    prompt="Test prompt",
                    model_path="invalid/unknown-model",
                    generate_type="t2v",
                    num_inference_steps=1,
                )
            except Exception:
                # We're only interested in the warning, not successful execution
                pass

            # Check that a warning was logged
            warning_records = [r for r in caplog.records if r.levelname == "WARNING"]
            assert len(warning_records) > 0, "Should log at least one warning"

            # Get the first warning message
            warning_msg = warning_records[0].message

            # Verify warning message contains key information
            assert "unknown-model" in warning_msg.lower(), "Should mention the invalid model name"
            assert "not found in resolution map" in warning_msg.lower(), "Should explain the issue"
            assert (
                "480" in warning_msg and "720" in warning_msg
            ), "Should mention default resolution"
            assert "valid models:" in warning_msg.lower(), "Should list valid models"

            # Verify all valid model names are in the warning
            from inference.cli_demo import RESOLUTION_MAP

            for model_name in RESOLUTION_MAP.keys():
                assert model_name in warning_msg, f"Warning should list '{model_name}'"

    def test_case_insensitive_model_name_matching(self, resolution_map):
        """Test that model names are converted to lowercase for matching."""

        # All keys in RESOLUTION_MAP should be lowercase
        for key in resolution_map.keys():
            assert key == key.lower(), f"RESOLUTION_MAP key '{key}' should be lowercase"

        # Test that various case inputs would work after .lower() conversion
        test_paths = [
            ("THUDM/COGVIDEOX-5B", "cogvideox-5b"),
            ("thudm/cogvideox-5b", "cogvideox-5b"),
            ("ThUdM/CoGvIdEoX-5b", "cogvideox-5b"),
        ]

        for model_path, expected_key in test_paths:
            model_name = model_path.split("/")[-1].lower()
            assert model_name == expected_key
            assert model_name in resolution_map

    def test_default_resolution_value(self, cli_demo_module):
        """Test that the default fallback resolution is (480, 720)."""
        import inspect

        source = inspect.getsource(cli_demo_module.generate_video)

        # The default should be defined as (480, 720)
        assert "(480, 720)" in source, "Default resolution should be (480, 720)"

    def test_resolution_dimensions_order(self, resolution_map):
        """Test that resolutions are in (height, width) format."""

        # Based on the code comments and typical usage
        for model_name, (dim1, dim2) in resolution_map.items():
            # Height typically comes first
            # Verify dimensions are reasonable (not negative or zero)
            assert dim1 > 0, f"First dimension for {model_name} should be positive"
            assert dim2 > 0, f"Second dimension for {model_name} should be positive"

            # CogVideoX1.5 models should have larger resolutions
            if "1.5" in model_name:
                assert dim1 >= 768, f"CogVideoX1.5 model {model_name} should have height >= 768"
                assert dim2 >= 1360, f"CogVideoX1.5 model {model_name} should have width >= 1360"
            else:
                # Standard CogVideoX models
                assert dim1 == 480, f"Standard CogVideoX model {model_name} should have height 480"
                assert dim2 == 720, f"Standard CogVideoX model {model_name} should have width 720"


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
