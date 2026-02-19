"""
Tests for multi-GPU device_map support in cli_demo.py

These tests verify:
- device_map=None → CPU offload enabled, no device_map passed to from_pretrained
- device_map="auto" → No CPU offload, device_map="auto" passed to from_pretrained
- device_map="balanced" → No CPU offload, device_map="balanced" passed
- device_map="sequential" → No CPU offload, device_map="sequential" passed
- All three pipeline types (t2v, i2v, v2v) work correctly
- Backward compatibility (default behavior unchanged)
"""

import sys
import os
from unittest.mock import MagicMock
import importlib.util
import pytest


def create_mocked_cli_demo():
    """
    Load cli_demo.py with mocked heavy dependencies.
    Returns the module and the mock objects for assertions.
    """
    # Create mock for torch
    mock_torch = MagicMock()
    mock_torch.bfloat16 = "bfloat16"
    mock_torch.float16 = "float16"
    mock_torch.Generator.return_value.manual_seed.return_value = MagicMock()

    # Create pipeline mocks
    mock_t2v_pipe = MagicMock()
    mock_t2v_pipe.scheduler = MagicMock()
    mock_t2v_pipe.scheduler.config = {}
    mock_t2v_pipe.vae = MagicMock()
    mock_t2v_pipe.return_value = MagicMock(frames=[[MagicMock()]])

    mock_i2v_pipe = MagicMock()
    mock_i2v_pipe.scheduler = MagicMock()
    mock_i2v_pipe.scheduler.config = {}
    mock_i2v_pipe.vae = MagicMock()
    mock_i2v_pipe.return_value = MagicMock(frames=[[MagicMock()]])

    mock_v2v_pipe = MagicMock()
    mock_v2v_pipe.scheduler = MagicMock()
    mock_v2v_pipe.scheduler.config = {}
    mock_v2v_pipe.vae = MagicMock()
    mock_v2v_pipe.return_value = MagicMock(frames=[[MagicMock()]])

    # Create pipeline class mocks
    mock_CogVideoXPipeline = MagicMock()
    mock_CogVideoXPipeline.from_pretrained.return_value = mock_t2v_pipe

    mock_CogVideoXImageToVideoPipeline = MagicMock()
    mock_CogVideoXImageToVideoPipeline.from_pretrained.return_value = mock_i2v_pipe

    mock_CogVideoXVideoToVideoPipeline = MagicMock()
    mock_CogVideoXVideoToVideoPipeline.from_pretrained.return_value = mock_v2v_pipe

    mock_CogVideoXDPMScheduler = MagicMock()
    mock_CogVideoXDPMScheduler.from_config.return_value = MagicMock()

    # Create mock diffusers module
    mock_diffusers = MagicMock()
    mock_diffusers.CogVideoXPipeline = mock_CogVideoXPipeline
    mock_diffusers.CogVideoXImageToVideoPipeline = mock_CogVideoXImageToVideoPipeline
    mock_diffusers.CogVideoXVideoToVideoPipeline = mock_CogVideoXVideoToVideoPipeline
    mock_diffusers.CogVideoXDPMScheduler = mock_CogVideoXDPMScheduler

    mock_diffusers_utils = MagicMock()
    mock_diffusers_utils.export_to_video = MagicMock()
    mock_diffusers_utils.load_image = MagicMock(return_value=MagicMock())
    mock_diffusers_utils.load_video = MagicMock(return_value=MagicMock())

    # Save original modules
    original_modules = {}
    for mod_name in ['torch', 'diffusers', 'diffusers.utils']:
        if mod_name in sys.modules:
            original_modules[mod_name] = sys.modules[mod_name]

    # Remove cli_demo if cached
    if 'cli_demo' in sys.modules:
        del sys.modules['cli_demo']

    # Install mocks
    sys.modules['torch'] = mock_torch
    sys.modules['diffusers'] = mock_diffusers
    sys.modules['diffusers.utils'] = mock_diffusers_utils

    try:
        # Load cli_demo.py using importlib
        cli_demo_path = os.path.join(os.path.dirname(__file__), '..', 'inference', 'cli_demo.py')
        spec = importlib.util.spec_from_file_location("cli_demo", cli_demo_path)
        cli_demo = importlib.util.module_from_spec(spec)
        sys.modules['cli_demo'] = cli_demo
        spec.loader.exec_module(cli_demo)

        return {
            'cli_demo': cli_demo,
            'mock_torch': mock_torch,
            'mock_CogVideoXPipeline': mock_CogVideoXPipeline,
            'mock_CogVideoXImageToVideoPipeline': mock_CogVideoXImageToVideoPipeline,
            'mock_CogVideoXVideoToVideoPipeline': mock_CogVideoXVideoToVideoPipeline,
            'mock_CogVideoXDPMScheduler': mock_CogVideoXDPMScheduler,
            'mock_t2v_pipe': mock_t2v_pipe,
            'mock_i2v_pipe': mock_i2v_pipe,
            'mock_v2v_pipe': mock_v2v_pipe,
        }
    finally:
        # Restore original modules
        for mod_name, mod in original_modules.items():
            sys.modules[mod_name] = mod


class TestDeviceMapSupport:
    """Test suite for multi-GPU device_map functionality."""

    def _get_common_args(self, generate_type="t2v", device_map=None):
        """Get common arguments for generate_video function."""
        args = {
            "prompt": "A test video",
            "model_path": "THUDM/CogVideoX-5b",
            "generate_type": generate_type,
            "output_path": "./test_output.mp4",
            "num_inference_steps": 1,
            "num_frames": 9,
            "seed": 42,
            "device_map": device_map,
        }
        if generate_type in ("i2v", "v2v"):
            args["image_or_video_path"] = "/fake/path/image.png"
        return args

    # =========================================================================
    # Tests for device_map=None (default behavior - CPU offload)
    # =========================================================================

    def test_device_map_none_t2v_enables_cpu_offload(self):
        """Test that device_map=None enables CPU offload for t2v pipeline."""
        mocks = create_mocked_cli_demo()
        cli_demo = mocks['cli_demo']

        cli_demo.generate_video(**self._get_common_args(generate_type="t2v", device_map=None))

        # Verify from_pretrained was called WITHOUT device_map
        mocks['mock_CogVideoXPipeline'].from_pretrained.assert_called_once()
        call_kwargs = mocks['mock_CogVideoXPipeline'].from_pretrained.call_args[1]
        assert 'device_map' not in call_kwargs, "device_map should NOT be passed when None"

        # Verify CPU offload was enabled
        mocks['mock_t2v_pipe'].enable_sequential_cpu_offload.assert_called_once()

    def test_device_map_none_i2v_enables_cpu_offload(self):
        """Test that device_map=None enables CPU offload for i2v pipeline."""
        mocks = create_mocked_cli_demo()
        cli_demo = mocks['cli_demo']

        cli_demo.generate_video(**self._get_common_args(generate_type="i2v", device_map=None))

        # Verify from_pretrained was called WITHOUT device_map
        mocks['mock_CogVideoXImageToVideoPipeline'].from_pretrained.assert_called_once()
        call_kwargs = mocks['mock_CogVideoXImageToVideoPipeline'].from_pretrained.call_args[1]
        assert 'device_map' not in call_kwargs, "device_map should NOT be passed when None"

        # Verify CPU offload was enabled
        mocks['mock_i2v_pipe'].enable_sequential_cpu_offload.assert_called_once()

    def test_device_map_none_v2v_enables_cpu_offload(self):
        """Test that device_map=None enables CPU offload for v2v pipeline."""
        mocks = create_mocked_cli_demo()
        cli_demo = mocks['cli_demo']

        cli_demo.generate_video(**self._get_common_args(generate_type="v2v", device_map=None))

        # Verify from_pretrained was called WITHOUT device_map
        mocks['mock_CogVideoXVideoToVideoPipeline'].from_pretrained.assert_called_once()
        call_kwargs = mocks['mock_CogVideoXVideoToVideoPipeline'].from_pretrained.call_args[1]
        assert 'device_map' not in call_kwargs, "device_map should NOT be passed when None"

        # Verify CPU offload was enabled
        mocks['mock_v2v_pipe'].enable_sequential_cpu_offload.assert_called_once()

    # =========================================================================
    # Tests for device_map="auto"
    # =========================================================================

    def test_device_map_auto_t2v_no_cpu_offload(self):
        """Test that device_map='auto' passes device_map and skips CPU offload for t2v."""
        mocks = create_mocked_cli_demo()
        cli_demo = mocks['cli_demo']

        cli_demo.generate_video(**self._get_common_args(generate_type="t2v", device_map="auto"))

        # Verify from_pretrained was called WITH device_map="auto"
        mocks['mock_CogVideoXPipeline'].from_pretrained.assert_called_once()
        call_kwargs = mocks['mock_CogVideoXPipeline'].from_pretrained.call_args[1]
        assert call_kwargs['device_map'] == "auto"

        # Verify CPU offload was NOT enabled
        mocks['mock_t2v_pipe'].enable_sequential_cpu_offload.assert_not_called()

    def test_device_map_auto_i2v_no_cpu_offload(self):
        """Test that device_map='auto' passes device_map and skips CPU offload for i2v."""
        mocks = create_mocked_cli_demo()
        cli_demo = mocks['cli_demo']

        cli_demo.generate_video(**self._get_common_args(generate_type="i2v", device_map="auto"))

        # Verify from_pretrained was called WITH device_map="auto"
        mocks['mock_CogVideoXImageToVideoPipeline'].from_pretrained.assert_called_once()
        call_kwargs = mocks['mock_CogVideoXImageToVideoPipeline'].from_pretrained.call_args[1]
        assert call_kwargs['device_map'] == "auto"

        # Verify CPU offload was NOT enabled
        mocks['mock_i2v_pipe'].enable_sequential_cpu_offload.assert_not_called()

    def test_device_map_auto_v2v_no_cpu_offload(self):
        """Test that device_map='auto' passes device_map and skips CPU offload for v2v."""
        mocks = create_mocked_cli_demo()
        cli_demo = mocks['cli_demo']

        cli_demo.generate_video(**self._get_common_args(generate_type="v2v", device_map="auto"))

        # Verify from_pretrained was called WITH device_map="auto"
        mocks['mock_CogVideoXVideoToVideoPipeline'].from_pretrained.assert_called_once()
        call_kwargs = mocks['mock_CogVideoXVideoToVideoPipeline'].from_pretrained.call_args[1]
        assert call_kwargs['device_map'] == "auto"

        # Verify CPU offload was NOT enabled
        mocks['mock_v2v_pipe'].enable_sequential_cpu_offload.assert_not_called()

    # =========================================================================
    # Tests for device_map="balanced"
    # =========================================================================

    def test_device_map_balanced_t2v_no_cpu_offload(self):
        """Test that device_map='balanced' passes device_map and skips CPU offload for t2v."""
        mocks = create_mocked_cli_demo()
        cli_demo = mocks['cli_demo']

        cli_demo.generate_video(**self._get_common_args(generate_type="t2v", device_map="balanced"))

        # Verify from_pretrained was called WITH device_map="balanced"
        mocks['mock_CogVideoXPipeline'].from_pretrained.assert_called_once()
        call_kwargs = mocks['mock_CogVideoXPipeline'].from_pretrained.call_args[1]
        assert call_kwargs['device_map'] == "balanced"

        # Verify CPU offload was NOT enabled
        mocks['mock_t2v_pipe'].enable_sequential_cpu_offload.assert_not_called()

    def test_device_map_balanced_i2v_no_cpu_offload(self):
        """Test that device_map='balanced' passes device_map and skips CPU offload for i2v."""
        mocks = create_mocked_cli_demo()
        cli_demo = mocks['cli_demo']

        cli_demo.generate_video(**self._get_common_args(generate_type="i2v", device_map="balanced"))

        # Verify from_pretrained was called WITH device_map="balanced"
        mocks['mock_CogVideoXImageToVideoPipeline'].from_pretrained.assert_called_once()
        call_kwargs = mocks['mock_CogVideoXImageToVideoPipeline'].from_pretrained.call_args[1]
        assert call_kwargs['device_map'] == "balanced"

        # Verify CPU offload was NOT enabled
        mocks['mock_i2v_pipe'].enable_sequential_cpu_offload.assert_not_called()

    def test_device_map_balanced_v2v_no_cpu_offload(self):
        """Test that device_map='balanced' passes device_map and skips CPU offload for v2v."""
        mocks = create_mocked_cli_demo()
        cli_demo = mocks['cli_demo']

        cli_demo.generate_video(**self._get_common_args(generate_type="v2v", device_map="balanced"))

        # Verify from_pretrained was called WITH device_map="balanced"
        mocks['mock_CogVideoXVideoToVideoPipeline'].from_pretrained.assert_called_once()
        call_kwargs = mocks['mock_CogVideoXVideoToVideoPipeline'].from_pretrained.call_args[1]
        assert call_kwargs['device_map'] == "balanced"

        # Verify CPU offload was NOT enabled
        mocks['mock_v2v_pipe'].enable_sequential_cpu_offload.assert_not_called()

    # =========================================================================
    # Tests for device_map="sequential"
    # =========================================================================

    def test_device_map_sequential_t2v_no_cpu_offload(self):
        """Test that device_map='sequential' passes device_map and skips CPU offload for t2v."""
        mocks = create_mocked_cli_demo()
        cli_demo = mocks['cli_demo']

        cli_demo.generate_video(
            **self._get_common_args(generate_type="t2v", device_map="sequential")
        )

        # Verify from_pretrained was called WITH device_map="sequential"
        mocks['mock_CogVideoXPipeline'].from_pretrained.assert_called_once()
        call_kwargs = mocks['mock_CogVideoXPipeline'].from_pretrained.call_args[1]
        assert call_kwargs['device_map'] == "sequential"

        # Verify CPU offload was NOT enabled
        mocks['mock_t2v_pipe'].enable_sequential_cpu_offload.assert_not_called()

    def test_device_map_sequential_i2v_no_cpu_offload(self):
        """Test that device_map='sequential' passes device_map and skips CPU offload for i2v."""
        mocks = create_mocked_cli_demo()
        cli_demo = mocks['cli_demo']

        cli_demo.generate_video(
            **self._get_common_args(generate_type="i2v", device_map="sequential")
        )

        # Verify from_pretrained was called WITH device_map="sequential"
        mocks['mock_CogVideoXImageToVideoPipeline'].from_pretrained.assert_called_once()
        call_kwargs = mocks['mock_CogVideoXImageToVideoPipeline'].from_pretrained.call_args[1]
        assert call_kwargs['device_map'] == "sequential"

        # Verify CPU offload was NOT enabled
        mocks['mock_i2v_pipe'].enable_sequential_cpu_offload.assert_not_called()

    def test_device_map_sequential_v2v_no_cpu_offload(self):
        """Test that device_map='sequential' passes device_map and skips CPU offload for v2v."""
        mocks = create_mocked_cli_demo()
        cli_demo = mocks['cli_demo']

        cli_demo.generate_video(
            **self._get_common_args(generate_type="v2v", device_map="sequential")
        )

        # Verify from_pretrained was called WITH device_map="sequential"
        mocks['mock_CogVideoXVideoToVideoPipeline'].from_pretrained.assert_called_once()
        call_kwargs = mocks['mock_CogVideoXVideoToVideoPipeline'].from_pretrained.call_args[1]
        assert call_kwargs['device_map'] == "sequential"

        # Verify CPU offload was NOT enabled
        mocks['mock_v2v_pipe'].enable_sequential_cpu_offload.assert_not_called()

    # =========================================================================
    # VAE optimizations (should always be enabled)
    # =========================================================================

    @pytest.mark.parametrize("device_map", [None, "auto", "balanced", "sequential"])
    def test_vae_optimizations_always_enabled_t2v(self, device_map):
        """Test that VAE slicing and tiling are always enabled for t2v regardless of device_map."""
        mocks = create_mocked_cli_demo()
        cli_demo = mocks['cli_demo']

        cli_demo.generate_video(**self._get_common_args(generate_type="t2v", device_map=device_map))

        # VAE optimizations should always be called
        mocks['mock_t2v_pipe'].vae.enable_slicing.assert_called_once()
        mocks['mock_t2v_pipe'].vae.enable_tiling.assert_called_once()

    @pytest.mark.parametrize("device_map", [None, "auto", "balanced", "sequential"])
    def test_vae_optimizations_always_enabled_i2v(self, device_map):
        """Test that VAE slicing and tiling are always enabled for i2v regardless of device_map."""
        mocks = create_mocked_cli_demo()
        cli_demo = mocks['cli_demo']

        cli_demo.generate_video(**self._get_common_args(generate_type="i2v", device_map=device_map))

        # VAE optimizations should always be called
        mocks['mock_i2v_pipe'].vae.enable_slicing.assert_called_once()
        mocks['mock_i2v_pipe'].vae.enable_tiling.assert_called_once()

    @pytest.mark.parametrize("device_map", [None, "auto", "balanced", "sequential"])
    def test_vae_optimizations_always_enabled_v2v(self, device_map):
        """Test that VAE slicing and tiling are always enabled for v2v regardless of device_map."""
        mocks = create_mocked_cli_demo()
        cli_demo = mocks['cli_demo']

        cli_demo.generate_video(**self._get_common_args(generate_type="v2v", device_map=device_map))

        # VAE optimizations should always be called
        mocks['mock_v2v_pipe'].vae.enable_slicing.assert_called_once()
        mocks['mock_v2v_pipe'].vae.enable_tiling.assert_called_once()

    # =========================================================================
    # Invalid device_map values
    # =========================================================================

    def test_invalid_device_map_raises_error(self):
        """Test that invalid device_map values raise ValueError."""
        mocks = create_mocked_cli_demo()
        cli_demo = mocks['cli_demo']

        with pytest.raises(ValueError) as exc_info:
            cli_demo.generate_video(
                **self._get_common_args(generate_type="t2v", device_map="invalid")
            )

        assert "Invalid device_map" in str(exc_info.value)
        assert "invalid" in str(exc_info.value)

    @pytest.mark.parametrize("invalid_value", ["cuda:0", "cpu", "GPU", "multi", "distributed"])
    def test_various_invalid_device_map_values(self, invalid_value):
        """Test that various invalid device_map values all raise ValueError."""
        mocks = create_mocked_cli_demo()
        cli_demo = mocks['cli_demo']

        with pytest.raises(ValueError) as exc_info:
            cli_demo.generate_video(
                **self._get_common_args(generate_type="t2v", device_map=invalid_value)
            )

        assert "Invalid device_map" in str(exc_info.value)

    # =========================================================================
    # Backward compatibility tests
    # =========================================================================

    def test_default_behavior_unchanged_no_device_map_arg(self):
        """Test that default behavior (no device_map argument) remains unchanged."""
        mocks = create_mocked_cli_demo()
        cli_demo = mocks['cli_demo']

        # Call without device_map argument at all (relies on default)
        args = {
            "prompt": "A test video",
            "model_path": "THUDM/CogVideoX-5b",
            "generate_type": "t2v",
            "output_path": "./test_output.mp4",
            "num_inference_steps": 1,
            "num_frames": 9,
            "seed": 42,
            # Note: device_map is intentionally NOT included to test default
        }
        cli_demo.generate_video(**args)

        # Verify from_pretrained was called WITHOUT device_map
        mocks['mock_CogVideoXPipeline'].from_pretrained.assert_called_once()
        call_kwargs = mocks['mock_CogVideoXPipeline'].from_pretrained.call_args[1]
        assert 'device_map' not in call_kwargs

        # Verify CPU offload WAS enabled (default single-GPU behavior)
        mocks['mock_t2v_pipe'].enable_sequential_cpu_offload.assert_called_once()

    # =========================================================================
    # Parametrized comprehensive tests
    # =========================================================================

    @pytest.mark.parametrize(
        "device_map,should_have_device_map,should_cpu_offload",
        [
            (None, False, True),
            ("auto", True, False),
            ("balanced", True, False),
            ("sequential", True, False),
        ],
    )
    def test_device_map_behavior_t2v(self, device_map, should_have_device_map, should_cpu_offload):
        """Comprehensive test for device_map behavior with t2v pipeline."""
        mocks = create_mocked_cli_demo()
        cli_demo = mocks['cli_demo']

        cli_demo.generate_video(**self._get_common_args(generate_type="t2v", device_map=device_map))

        # Check from_pretrained call
        mocks['mock_CogVideoXPipeline'].from_pretrained.assert_called_once()
        call_kwargs = mocks['mock_CogVideoXPipeline'].from_pretrained.call_args[1]

        if should_have_device_map:
            assert 'device_map' in call_kwargs
            assert call_kwargs['device_map'] == device_map
        else:
            assert 'device_map' not in call_kwargs

        # Check CPU offload
        if should_cpu_offload:
            mocks['mock_t2v_pipe'].enable_sequential_cpu_offload.assert_called_once()
        else:
            mocks['mock_t2v_pipe'].enable_sequential_cpu_offload.assert_not_called()

    @pytest.mark.parametrize(
        "device_map,should_have_device_map,should_cpu_offload",
        [
            (None, False, True),
            ("auto", True, False),
            ("balanced", True, False),
            ("sequential", True, False),
        ],
    )
    def test_device_map_behavior_i2v(self, device_map, should_have_device_map, should_cpu_offload):
        """Comprehensive test for device_map behavior with i2v pipeline."""
        mocks = create_mocked_cli_demo()
        cli_demo = mocks['cli_demo']

        cli_demo.generate_video(**self._get_common_args(generate_type="i2v", device_map=device_map))

        # Check from_pretrained call
        mocks['mock_CogVideoXImageToVideoPipeline'].from_pretrained.assert_called_once()
        call_kwargs = mocks['mock_CogVideoXImageToVideoPipeline'].from_pretrained.call_args[1]

        if should_have_device_map:
            assert 'device_map' in call_kwargs
            assert call_kwargs['device_map'] == device_map
        else:
            assert 'device_map' not in call_kwargs

        # Check CPU offload
        if should_cpu_offload:
            mocks['mock_i2v_pipe'].enable_sequential_cpu_offload.assert_called_once()
        else:
            mocks['mock_i2v_pipe'].enable_sequential_cpu_offload.assert_not_called()

    @pytest.mark.parametrize(
        "device_map,should_have_device_map,should_cpu_offload",
        [
            (None, False, True),
            ("auto", True, False),
            ("balanced", True, False),
            ("sequential", True, False),
        ],
    )
    def test_device_map_behavior_v2v(self, device_map, should_have_device_map, should_cpu_offload):
        """Comprehensive test for device_map behavior with v2v pipeline."""
        mocks = create_mocked_cli_demo()
        cli_demo = mocks['cli_demo']

        cli_demo.generate_video(**self._get_common_args(generate_type="v2v", device_map=device_map))

        # Check from_pretrained call
        mocks['mock_CogVideoXVideoToVideoPipeline'].from_pretrained.assert_called_once()
        call_kwargs = mocks['mock_CogVideoXVideoToVideoPipeline'].from_pretrained.call_args[1]

        if should_have_device_map:
            assert 'device_map' in call_kwargs
            assert call_kwargs['device_map'] == device_map
        else:
            assert 'device_map' not in call_kwargs

        # Check CPU offload
        if should_cpu_offload:
            mocks['mock_v2v_pipe'].enable_sequential_cpu_offload.assert_called_once()
        else:
            mocks['mock_v2v_pipe'].enable_sequential_cpu_offload.assert_not_called()


class TestValidDeviceMapsConstant:
    """Test the VALID_DEVICE_MAPS constant."""

    def test_valid_device_maps_contains_expected_values(self):
        """Verify VALID_DEVICE_MAPS contains the expected device map options."""
        mocks = create_mocked_cli_demo()
        cli_demo = mocks['cli_demo']

        assert "auto" in cli_demo.VALID_DEVICE_MAPS
        assert "balanced" in cli_demo.VALID_DEVICE_MAPS
        assert "sequential" in cli_demo.VALID_DEVICE_MAPS
        assert len(cli_demo.VALID_DEVICE_MAPS) == 3

    def test_valid_device_maps_is_set(self):
        """Verify VALID_DEVICE_MAPS is a set for O(1) lookup."""
        mocks = create_mocked_cli_demo()
        cli_demo = mocks['cli_demo']

        assert isinstance(cli_demo.VALID_DEVICE_MAPS, set)


class TestCLIArgumentParsing:
    """Test CLI argument parsing for device_map."""

    def test_cli_device_map_choices(self):
        """Test that CLI parser accepts valid device_map choices."""
        import argparse

        # Create a minimal parser matching the CLI
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--device_map",
            type=str,
            default=None,
            choices=["auto", "balanced", "sequential"],
        )

        # Test valid values
        for valid_value in ["auto", "balanced", "sequential"]:
            args = parser.parse_args(["--device_map", valid_value])
            assert args.device_map == valid_value

        # Test default (no argument)
        args = parser.parse_args([])
        assert args.device_map is None

    def test_cli_device_map_invalid_choice(self):
        """Test that CLI parser rejects invalid device_map choices."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--device_map",
            type=str,
            default=None,
            choices=["auto", "balanced", "sequential"],
        )

        with pytest.raises(SystemExit):
            parser.parse_args(["--device_map", "invalid"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
