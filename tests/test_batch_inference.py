"""
Comprehensive tests for the batch inference pipeline.

Tests cover:
- JSONL parsing (valid and invalid input)
- Batch processing logic (iteration, state tracking)
- Resume capability (read/write .batch_state.json)
- Error handling (skip failed items, continue processing)
- Multi-GPU job distribution (--gpu_id, --num_gpus)
- Output path creation
- Progress tracking
- All generation types: t2v, i2v, v2v
- Edge cases: empty batch, corrupted state, missing files, invalid JSONL
"""

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest


# =============================================================================
# Mock heavy dependencies before importing batch_inference
# =============================================================================

# Create mock torch module
mock_torch = MagicMock()
mock_torch.bfloat16 = "bfloat16"
mock_torch.float16 = "float16"
mock_torch.Generator = MagicMock

# Create mock diffusers module
mock_diffusers = MagicMock()

# Create mock tqdm that works as a context manager
class MockTqdm:
    """Mock tqdm that works as a context manager and returns a dummy progress bar."""
    def __init__(self, iterable=None, total=None, **kwargs):
        self.iterable = iterable
        self.n = 0
        
    def __iter__(self):
        if self.iterable is not None:
            for item in self.iterable:
                yield item
                self.n += 1
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
    
    def update(self, n=1):
        self.n += n
        
    def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
        pass


mock_tqdm = MagicMock()
mock_tqdm.tqdm = MockTqdm

# Install mocks before importing
sys.modules["torch"] = mock_torch
sys.modules["diffusers"] = mock_diffusers
sys.modules["diffusers.utils"] = mock_diffusers.utils
sys.modules["tqdm"] = mock_tqdm


def _load_batch_inference_module():
    """Load the batch_inference module from tools directory."""
    module_path = Path(__file__).parent.parent / "tools" / "batch_inference.py"
    spec = importlib.util.spec_from_file_location("batch_inference", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["batch_inference"] = module
    spec.loader.exec_module(module)
    return module


batch_inference = _load_batch_inference_module()

# Import the symbols we need
BatchJob = batch_inference.BatchJob
BatchState = batch_inference.BatchState
load_batch_file = batch_inference.load_batch_file
load_pipeline = batch_inference.load_pipeline
generate_single_video = batch_inference.generate_single_video
run_batch = batch_inference.run_batch
RESOLUTION_MAP = batch_inference.RESOLUTION_MAP


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_job_dict():
    """A valid job dictionary."""
    return {
        "prompt": "A cat playing piano",
        "output_name": "cat_piano.mp4",
        "num_frames": 49,
        "guidance_scale": 7.0,
        "seed": 123,
    }


@pytest.fixture
def sample_i2v_job_dict():
    """A valid i2v job dictionary."""
    return {
        "prompt": "Transform this image",
        "output_name": "i2v_output.mp4",
        "image_path": "/path/to/image.jpg",
    }


@pytest.fixture
def sample_v2v_job_dict():
    """A valid v2v job dictionary."""
    return {
        "prompt": "Enhance this video",
        "output_name": "v2v_output.mp4",
        "video_path": "/path/to/video.mp4",
    }


@pytest.fixture
def valid_jsonl_content():
    """Valid JSONL content with multiple jobs."""
    jobs = [
        {"prompt": "A cat playing piano", "output_name": "cat_piano.mp4"},
        {"prompt": "Waves crashing on beach", "output_name": "beach.mp4", "num_frames": 49},
        {"prompt": "A dog running", "output_name": "dog.mp4", "seed": 999},
    ]
    return "\n".join(json.dumps(j) for j in jobs)


@pytest.fixture
def batch_file(tmp_path, valid_jsonl_content):
    """Create a temporary batch file with valid JSONL content."""
    batch_path = tmp_path / "batch.jsonl"
    batch_path.write_text(valid_jsonl_content)
    return batch_path


@pytest.fixture
def mock_pipeline():
    """Create a mock pipeline that returns fake video frames."""
    mock = MagicMock()
    # Mock the pipeline call to return a fake frames result
    mock_result = MagicMock()
    mock_result.frames = [[MagicMock()]]  # Nested list like real output
    mock.return_value = mock_result
    mock.scheduler = MagicMock()
    mock.vae = MagicMock()
    return mock


# =============================================================================
# BatchJob Tests
# =============================================================================


class TestBatchJob:
    """Tests for the BatchJob dataclass."""

    def test_from_dict_basic(self, sample_job_dict):
        """Test creating a BatchJob from a dictionary."""
        job = BatchJob.from_dict(sample_job_dict, line_number=1)
        
        assert job.prompt == "A cat playing piano"
        assert job.output_name == "cat_piano.mp4"
        assert job.num_frames == 49
        assert job.guidance_scale == 7.0
        assert job.seed == 123
        assert job.line_number == 1
        assert job.status == "pending"

    def test_from_dict_minimal(self):
        """Test creating a BatchJob with only required fields."""
        data = {"prompt": "Test prompt", "output_name": "test.mp4"}
        job = BatchJob.from_dict(data)
        
        assert job.prompt == "Test prompt"
        assert job.output_name == "test.mp4"
        assert job.num_frames is None
        assert job.image_path is None
        assert job.video_path is None

    def test_from_dict_with_image_path(self, sample_i2v_job_dict):
        """Test creating a BatchJob with image_path for i2v."""
        job = BatchJob.from_dict(sample_i2v_job_dict)
        
        assert job.image_path == "/path/to/image.jpg"
        assert job.video_path is None

    def test_from_dict_with_video_path(self, sample_v2v_job_dict):
        """Test creating a BatchJob with video_path for v2v."""
        job = BatchJob.from_dict(sample_v2v_job_dict)
        
        assert job.video_path == "/path/to/video.mp4"
        assert job.image_path is None

    def test_validate_valid_job(self, sample_job_dict):
        """Test validation of a valid job returns no errors."""
        job = BatchJob.from_dict(sample_job_dict)
        errors = job.validate()
        
        assert errors == []

    def test_validate_missing_prompt(self):
        """Test validation catches missing prompt."""
        job = BatchJob.from_dict({"output_name": "test.mp4"})
        errors = job.validate()
        
        assert "Missing required field: prompt" in errors

    def test_validate_missing_output_name(self):
        """Test validation catches missing output_name."""
        job = BatchJob.from_dict({"prompt": "Test"})
        errors = job.validate()
        
        assert "Missing required field: output_name" in errors

    def test_validate_missing_both_required(self):
        """Test validation catches both missing required fields."""
        job = BatchJob.from_dict({})
        errors = job.validate()
        
        assert len(errors) == 2
        assert "Missing required field: prompt" in errors
        assert "Missing required field: output_name" in errors


# =============================================================================
# BatchState Tests
# =============================================================================


class TestBatchState:
    """Tests for the BatchState dataclass."""

    def test_create_new_state(self):
        """Test creating a new BatchState."""
        state = BatchState(
            batch_file="batch.jsonl",
            output_dir="./output",
            model_path="THUDM/CogVideoX1.5-5B",
            generate_type="t2v",
        )
        
        assert state.batch_file == "batch.jsonl"
        assert state.completed == []
        assert state.failed == []

    def test_mark_completed(self):
        """Test marking a job as completed."""
        state = BatchState(
            batch_file="batch.jsonl",
            output_dir="./output",
            model_path="model",
            generate_type="t2v",
        )
        
        state.mark_completed("video1.mp4")
        state.mark_completed("video2.mp4")
        
        assert "video1.mp4" in state.completed
        assert "video2.mp4" in state.completed

    def test_mark_completed_no_duplicates(self):
        """Test that marking the same job twice doesn't duplicate."""
        state = BatchState(
            batch_file="batch.jsonl",
            output_dir="./output",
            model_path="model",
            generate_type="t2v",
        )
        
        state.mark_completed("video1.mp4")
        state.mark_completed("video1.mp4")
        
        assert state.completed.count("video1.mp4") == 1

    def test_mark_failed(self):
        """Test marking a job as failed."""
        state = BatchState(
            batch_file="batch.jsonl",
            output_dir="./output",
            model_path="model",
            generate_type="t2v",
        )
        
        state.mark_failed("video1.mp4", "CUDA out of memory")
        
        assert len(state.failed) == 1
        assert state.failed[0]["output_name"] == "video1.mp4"
        assert state.failed[0]["error"] == "CUDA out of memory"

    def test_is_completed(self):
        """Test checking if a job is completed."""
        state = BatchState(
            batch_file="batch.jsonl",
            output_dir="./output",
            model_path="model",
            generate_type="t2v",
        )
        
        state.mark_completed("video1.mp4")
        
        assert state.is_completed("video1.mp4") is True
        assert state.is_completed("video2.mp4") is False

    def test_save_and_load(self, tmp_path):
        """Test saving and loading state."""
        state_file = tmp_path / ".batch_state.json"
        
        # Create and save state
        state = BatchState(
            batch_file="batch.jsonl",
            output_dir="./output",
            model_path="model",
            generate_type="t2v",
            started_at="2024-01-01T00:00:00",
        )
        state.mark_completed("video1.mp4")
        state.mark_failed("video2.mp4", "Error")
        state.save(state_file)
        
        # Load state
        loaded = BatchState.load(state_file)
        
        assert loaded is not None
        assert loaded.batch_file == "batch.jsonl"
        assert "video1.mp4" in loaded.completed
        assert len(loaded.failed) == 1
        assert loaded.failed[0]["output_name"] == "video2.mp4"

    def test_load_nonexistent_file(self, tmp_path):
        """Test loading from a nonexistent file returns None."""
        state_file = tmp_path / "nonexistent.json"
        
        loaded = BatchState.load(state_file)
        
        assert loaded is None

    def test_load_corrupted_file(self, tmp_path):
        """Test loading a corrupted state file returns None."""
        state_file = tmp_path / ".batch_state.json"
        state_file.write_text("{ invalid json }")
        
        loaded = BatchState.load(state_file)
        
        assert loaded is None

    def test_load_invalid_json_structure(self, tmp_path):
        """Test loading a file with valid JSON but invalid structure."""
        state_file = tmp_path / ".batch_state.json"
        state_file.write_text('{"wrong": "structure"}')
        
        loaded = BatchState.load(state_file)
        
        # Should return None due to missing required fields
        assert loaded is None


# =============================================================================
# JSONL Parsing Tests
# =============================================================================


class TestLoadBatchFile:
    """Tests for JSONL parsing."""

    def test_load_valid_jsonl(self, batch_file):
        """Test loading a valid JSONL file."""
        jobs = load_batch_file(batch_file)
        
        assert len(jobs) == 3
        assert jobs[0].prompt == "A cat playing piano"
        assert jobs[1].output_name == "beach.mp4"
        assert jobs[2].seed == 999

    def test_load_jsonl_with_comments(self, tmp_path):
        """Test that comment lines are skipped."""
        content = """# This is a comment
{"prompt": "Test", "output_name": "test.mp4"}
# Another comment
{"prompt": "Test2", "output_name": "test2.mp4"}"""
        batch_file = tmp_path / "batch.jsonl"
        batch_file.write_text(content)
        
        jobs = load_batch_file(batch_file)
        
        assert len(jobs) == 2

    def test_load_jsonl_with_empty_lines(self, tmp_path):
        """Test that empty lines are skipped."""
        content = """{"prompt": "Test", "output_name": "test.mp4"}

{"prompt": "Test2", "output_name": "test2.mp4"}

"""
        batch_file = tmp_path / "batch.jsonl"
        batch_file.write_text(content)
        
        jobs = load_batch_file(batch_file)
        
        assert len(jobs) == 2

    def test_load_jsonl_invalid_json_line(self, tmp_path):
        """Test that invalid JSON lines are skipped with warning."""
        content = """{"prompt": "Test", "output_name": "test.mp4"}
{invalid json here}
{"prompt": "Test2", "output_name": "test2.mp4"}"""
        batch_file = tmp_path / "batch.jsonl"
        batch_file.write_text(content)
        
        jobs = load_batch_file(batch_file)
        
        # Should skip the invalid line
        assert len(jobs) == 2

    def test_load_jsonl_missing_required_fields(self, tmp_path):
        """Test that jobs with missing required fields are skipped."""
        content = """{"prompt": "Test", "output_name": "test.mp4"}
{"prompt": "No output name"}
{"output_name": "no_prompt.mp4"}
{"prompt": "Test2", "output_name": "test2.mp4"}"""
        batch_file = tmp_path / "batch.jsonl"
        batch_file.write_text(content)
        
        jobs = load_batch_file(batch_file)
        
        # Should skip the two invalid jobs
        assert len(jobs) == 2
        assert jobs[0].output_name == "test.mp4"
        assert jobs[1].output_name == "test2.mp4"

    def test_load_empty_batch_file(self, tmp_path):
        """Test loading an empty batch file."""
        batch_file = tmp_path / "empty.jsonl"
        batch_file.write_text("")
        
        jobs = load_batch_file(batch_file)
        
        assert len(jobs) == 0

    def test_load_batch_file_all_comments(self, tmp_path):
        """Test loading a file with only comments."""
        content = """# Comment 1
# Comment 2
# Comment 3"""
        batch_file = tmp_path / "comments.jsonl"
        batch_file.write_text(content)
        
        jobs = load_batch_file(batch_file)
        
        assert len(jobs) == 0

    def test_load_batch_file_tracks_line_numbers(self, tmp_path):
        """Test that line numbers are tracked correctly."""
        content = """# Comment on line 1
{"prompt": "Line 2", "output_name": "line2.mp4"}

{"prompt": "Line 4", "output_name": "line4.mp4"}"""
        batch_file = tmp_path / "batch.jsonl"
        batch_file.write_text(content)
        
        jobs = load_batch_file(batch_file)
        
        assert jobs[0].line_number == 2
        assert jobs[1].line_number == 4


# =============================================================================
# Multi-GPU Distribution Tests
# =============================================================================


class TestMultiGPUDistribution:
    """Tests for multi-GPU job distribution logic."""

    def test_single_gpu_gets_all_jobs(self):
        """Test that a single GPU processes all jobs."""
        all_jobs = [
            BatchJob(prompt=f"Job {i}", output_name=f"video{i}.mp4")
            for i in range(10)
        ]
        
        # Simulate distribution for single GPU (gpu_id=0, num_gpus=1)
        gpu_id = 0
        num_gpus = 1
        jobs = [j for i, j in enumerate(all_jobs) if i % num_gpus == gpu_id]
        
        assert len(jobs) == 10

    def test_two_gpu_distribution(self):
        """Test job distribution across 2 GPUs."""
        all_jobs = [
            BatchJob(prompt=f"Job {i}", output_name=f"video{i}.mp4")
            for i in range(10)
        ]
        
        # GPU 0 gets jobs 0, 2, 4, 6, 8
        gpu0_jobs = [j for i, j in enumerate(all_jobs) if i % 2 == 0]
        # GPU 1 gets jobs 1, 3, 5, 7, 9
        gpu1_jobs = [j for i, j in enumerate(all_jobs) if i % 2 == 1]
        
        assert len(gpu0_jobs) == 5
        assert len(gpu1_jobs) == 5
        assert gpu0_jobs[0].output_name == "video0.mp4"
        assert gpu1_jobs[0].output_name == "video1.mp4"

    def test_four_gpu_distribution(self):
        """Test job distribution across 4 GPUs."""
        all_jobs = [
            BatchJob(prompt=f"Job {i}", output_name=f"video{i}.mp4")
            for i in range(12)
        ]
        
        num_gpus = 4
        for gpu_id in range(num_gpus):
            jobs = [j for i, j in enumerate(all_jobs) if i % num_gpus == gpu_id]
            assert len(jobs) == 3  # 12 jobs / 4 GPUs = 3 each

    def test_uneven_distribution(self):
        """Test distribution when jobs don't divide evenly."""
        all_jobs = [
            BatchJob(prompt=f"Job {i}", output_name=f"video{i}.mp4")
            for i in range(10)
        ]
        
        num_gpus = 3
        gpu0_jobs = [j for i, j in enumerate(all_jobs) if i % num_gpus == 0]
        gpu1_jobs = [j for i, j in enumerate(all_jobs) if i % num_gpus == 1]
        gpu2_jobs = [j for i, j in enumerate(all_jobs) if i % num_gpus == 2]
        
        # 10 jobs / 3 GPUs: GPU0 gets 4, GPU1 gets 3, GPU2 gets 3
        assert len(gpu0_jobs) == 4  # indices 0, 3, 6, 9
        assert len(gpu1_jobs) == 3  # indices 1, 4, 7
        assert len(gpu2_jobs) == 3  # indices 2, 5, 8


# =============================================================================
# Pipeline Loading Tests
# =============================================================================


class TestLoadPipeline:
    """Tests for pipeline loading."""

    @patch.object(batch_inference, "CogVideoXPipeline")
    def test_load_t2v_pipeline(self, mock_cogvideo_pipeline):
        """Test loading a text-to-video pipeline."""
        mock_pipe = MagicMock()
        mock_pipe.scheduler = MagicMock()
        mock_pipe.vae = MagicMock()
        mock_cogvideo_pipeline.from_pretrained.return_value = mock_pipe
        
        with patch.object(batch_inference, "CogVideoXDPMScheduler"):
            pipe = load_pipeline(
                model_path="THUDM/CogVideoX1.5-5B",
                generate_type="t2v",
                enable_cpu_offload=False,
            )
        
        mock_cogvideo_pipeline.from_pretrained.assert_called_once()
        assert pipe is mock_pipe

    @patch.object(batch_inference, "CogVideoXImageToVideoPipeline")
    def test_load_i2v_pipeline(self, mock_i2v_pipeline):
        """Test loading an image-to-video pipeline."""
        mock_pipe = MagicMock()
        mock_pipe.scheduler = MagicMock()
        mock_pipe.vae = MagicMock()
        mock_i2v_pipeline.from_pretrained.return_value = mock_pipe
        
        with patch.object(batch_inference, "CogVideoXDPMScheduler"):
            pipe = load_pipeline(
                model_path="THUDM/CogVideoX1.5-5B-I2V",
                generate_type="i2v",
                enable_cpu_offload=False,
            )
        
        mock_i2v_pipeline.from_pretrained.assert_called_once()
        assert pipe is mock_pipe

    @patch.object(batch_inference, "CogVideoXVideoToVideoPipeline")
    def test_load_v2v_pipeline(self, mock_v2v_pipeline):
        """Test loading a video-to-video pipeline."""
        mock_pipe = MagicMock()
        mock_pipe.scheduler = MagicMock()
        mock_pipe.vae = MagicMock()
        mock_v2v_pipeline.from_pretrained.return_value = mock_pipe
        
        with patch.object(batch_inference, "CogVideoXDPMScheduler"):
            pipe = load_pipeline(
                model_path="THUDM/CogVideoX1.5-5B",
                generate_type="v2v",
                enable_cpu_offload=False,
            )
        
        mock_v2v_pipeline.from_pretrained.assert_called_once()
        assert pipe is mock_pipe

    @patch.object(batch_inference, "CogVideoXPipeline")
    def test_load_pipeline_with_cpu_offload(self, mock_cogvideo_pipeline):
        """Test that CPU offload is enabled when requested."""
        mock_pipe = MagicMock()
        mock_pipe.scheduler = MagicMock()
        mock_pipe.vae = MagicMock()
        mock_cogvideo_pipeline.from_pretrained.return_value = mock_pipe
        
        with patch.object(batch_inference, "CogVideoXDPMScheduler"):
            load_pipeline(
                model_path="model",
                generate_type="t2v",
                enable_cpu_offload=True,
            )
        
        mock_pipe.enable_sequential_cpu_offload.assert_called_once()
        mock_pipe.to.assert_not_called()

    @patch.object(batch_inference, "CogVideoXPipeline")
    def test_load_pipeline_without_cpu_offload(self, mock_cogvideo_pipeline):
        """Test that pipeline moves to CUDA when CPU offload is disabled."""
        mock_pipe = MagicMock()
        mock_pipe.scheduler = MagicMock()
        mock_pipe.vae = MagicMock()
        mock_cogvideo_pipeline.from_pretrained.return_value = mock_pipe
        
        with patch.object(batch_inference, "CogVideoXDPMScheduler"):
            load_pipeline(
                model_path="model",
                generate_type="t2v",
                enable_cpu_offload=False,
            )
        
        mock_pipe.to.assert_called_once_with("cuda")
        mock_pipe.enable_sequential_cpu_offload.assert_not_called()

    @patch.object(batch_inference, "CogVideoXPipeline")
    def test_load_pipeline_with_lora(self, mock_cogvideo_pipeline):
        """Test loading pipeline with LoRA weights."""
        mock_pipe = MagicMock()
        mock_pipe.scheduler = MagicMock()
        mock_pipe.vae = MagicMock()
        mock_cogvideo_pipeline.from_pretrained.return_value = mock_pipe
        
        with patch.object(batch_inference, "CogVideoXDPMScheduler"):
            load_pipeline(
                model_path="model",
                generate_type="t2v",
                lora_path="/path/to/lora",
                enable_cpu_offload=False,
            )
        
        mock_pipe.load_lora_weights.assert_called_once()
        mock_pipe.fuse_lora.assert_called_once()


# =============================================================================
# Video Generation Tests
# =============================================================================


class TestGenerateSingleVideo:
    """Tests for single video generation."""

    @patch.object(batch_inference, "export_to_video")
    def test_generate_t2v_video(self, mock_export, mock_pipeline, tmp_path):
        """Test generating a text-to-video."""
        job = BatchJob(
            prompt="A cat playing piano",
            output_name="cat.mp4",
            num_frames=49,
        )
        output_path = tmp_path / "cat.mp4"
        
        generate_single_video(
            pipe=mock_pipeline,
            job=job,
            generate_type="t2v",
            model_name="cogvideox-5b",
            output_path=output_path,
        )
        
        mock_pipeline.assert_called_once()
        mock_export.assert_called_once()

    @patch.object(batch_inference, "export_to_video")
    @patch.object(batch_inference, "load_image")
    def test_generate_i2v_video(self, mock_load_image, mock_export, mock_pipeline, tmp_path):
        """Test generating an image-to-video."""
        mock_load_image.return_value = MagicMock()
        
        job = BatchJob(
            prompt="Transform this",
            output_name="i2v.mp4",
            image_path="/path/to/image.jpg",
        )
        output_path = tmp_path / "i2v.mp4"
        
        generate_single_video(
            pipe=mock_pipeline,
            job=job,
            generate_type="i2v",
            model_name="cogvideox-5b-i2v",
            output_path=output_path,
        )
        
        mock_load_image.assert_called_once_with(image="/path/to/image.jpg")
        mock_pipeline.assert_called_once()

    @patch.object(batch_inference, "export_to_video")
    @patch.object(batch_inference, "load_video")
    def test_generate_v2v_video(self, mock_load_video, mock_export, mock_pipeline, tmp_path):
        """Test generating a video-to-video."""
        mock_load_video.return_value = MagicMock()
        
        job = BatchJob(
            prompt="Enhance this",
            output_name="v2v.mp4",
            video_path="/path/to/video.mp4",
        )
        output_path = tmp_path / "v2v.mp4"
        
        generate_single_video(
            pipe=mock_pipeline,
            job=job,
            generate_type="v2v",
            model_name="cogvideox-5b",
            output_path=output_path,
        )
        
        mock_load_video.assert_called_once_with("/path/to/video.mp4")
        mock_pipeline.assert_called_once()

    def test_generate_i2v_missing_image_path(self, mock_pipeline, tmp_path):
        """Test that i2v generation fails without image_path."""
        job = BatchJob(
            prompt="Transform this",
            output_name="i2v.mp4",
            # Missing image_path
        )
        output_path = tmp_path / "i2v.mp4"
        
        with pytest.raises(ValueError, match="image_path is required"):
            generate_single_video(
                pipe=mock_pipeline,
                job=job,
                generate_type="i2v",
                model_name="cogvideox-5b-i2v",
                output_path=output_path,
            )

    def test_generate_v2v_missing_video_path(self, mock_pipeline, tmp_path):
        """Test that v2v generation fails without video_path."""
        job = BatchJob(
            prompt="Enhance this",
            output_name="v2v.mp4",
            # Missing video_path
        )
        output_path = tmp_path / "v2v.mp4"
        
        with pytest.raises(ValueError, match="video_path is required"):
            generate_single_video(
                pipe=mock_pipeline,
                job=job,
                generate_type="v2v",
                model_name="cogvideox-5b",
                output_path=output_path,
            )

    @patch.object(batch_inference, "export_to_video")
    def test_generate_uses_job_specific_params(self, mock_export, mock_pipeline, tmp_path):
        """Test that job-specific parameters override defaults."""
        job = BatchJob(
            prompt="Test",
            output_name="test.mp4",
            num_frames=33,
            guidance_scale=9.0,
            num_inference_steps=25,
            seed=456,
        )
        output_path = tmp_path / "test.mp4"
        
        generate_single_video(
            pipe=mock_pipeline,
            job=job,
            generate_type="t2v",
            model_name="cogvideox-5b",
            output_path=output_path,
            default_num_frames=81,
            default_guidance_scale=6.0,
            default_num_inference_steps=50,
            default_seed=42,
        )
        
        # Check the pipeline was called with job-specific values
        call_kwargs = mock_pipeline.call_args[1]
        assert call_kwargs["num_frames"] == 33
        assert call_kwargs["guidance_scale"] == 9.0
        assert call_kwargs["num_inference_steps"] == 25


# =============================================================================
# Batch Processing Tests
# =============================================================================


class TestRunBatch:
    """Tests for the main batch processing function."""

    @patch.object(batch_inference, "load_pipeline")
    @patch.object(batch_inference, "generate_single_video")
    def test_run_batch_basic(self, mock_generate, mock_load_pipeline, batch_file, tmp_path):
        """Test basic batch processing."""
        mock_pipe = MagicMock()
        mock_load_pipeline.return_value = mock_pipe
        
        output_dir = tmp_path / "output"
        
        summary = run_batch(
            batch_file=batch_file,
            model_path="THUDM/CogVideoX1.5-5B",
            output_dir=output_dir,
            generate_type="t2v",
            resume=False,
        )
        
        assert summary["total"] == 3
        assert summary["completed"] == 3
        assert summary["failed"] == 0
        assert mock_generate.call_count == 3

    @patch.object(batch_inference, "load_pipeline")
    @patch.object(batch_inference, "generate_single_video")
    def test_run_batch_creates_output_dir(self, mock_generate, mock_load_pipeline, batch_file, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        mock_pipe = MagicMock()
        mock_load_pipeline.return_value = mock_pipe
        
        output_dir = tmp_path / "nested" / "output" / "dir"
        assert not output_dir.exists()
        
        run_batch(
            batch_file=batch_file,
            model_path="model",
            output_dir=output_dir,
            resume=False,
        )
        
        assert output_dir.exists()

    @patch.object(batch_inference, "load_pipeline")
    @patch.object(batch_inference, "generate_single_video")
    def test_run_batch_saves_state(self, mock_generate, mock_load_pipeline, batch_file, tmp_path):
        """Test that state is saved after processing."""
        mock_pipe = MagicMock()
        mock_load_pipeline.return_value = mock_pipe
        
        output_dir = tmp_path / "output"
        
        run_batch(
            batch_file=batch_file,
            model_path="model",
            output_dir=output_dir,
            resume=True,
        )
        
        state_file = output_dir / ".batch_state.json"
        assert state_file.exists()
        
        with open(state_file) as f:
            state_data = json.load(f)
        
        assert len(state_data["completed"]) == 3

    @patch.object(batch_inference, "load_pipeline")
    @patch.object(batch_inference, "generate_single_video")
    def test_run_batch_handles_errors(self, mock_generate, mock_load_pipeline, batch_file, tmp_path):
        """Test that batch continues after individual job failures."""
        mock_pipe = MagicMock()
        mock_load_pipeline.return_value = mock_pipe
        
        # Make the second call raise an exception
        mock_generate.side_effect = [None, RuntimeError("CUDA OOM"), None]
        
        output_dir = tmp_path / "output"
        
        summary = run_batch(
            batch_file=batch_file,
            model_path="model",
            output_dir=output_dir,
            resume=False,
        )
        
        assert summary["completed"] == 2
        assert summary["failed"] == 1
        assert mock_generate.call_count == 3

    @patch.object(batch_inference, "load_pipeline")
    @patch.object(batch_inference, "generate_single_video")
    def test_run_batch_logs_errors_to_file(self, mock_generate, mock_load_pipeline, batch_file, tmp_path):
        """Test that errors are logged to errors.log."""
        mock_pipe = MagicMock()
        mock_load_pipeline.return_value = mock_pipe
        mock_generate.side_effect = RuntimeError("Test error")
        
        output_dir = tmp_path / "output"
        
        run_batch(
            batch_file=batch_file,
            model_path="model",
            output_dir=output_dir,
            resume=False,
        )
        
        error_log = output_dir / "errors.log"
        assert error_log.exists()
        content = error_log.read_text()
        assert "Test error" in content

    @patch.object(batch_inference, "load_pipeline")
    @patch.object(batch_inference, "generate_single_video")
    def test_run_batch_resume_skips_completed(self, mock_generate, mock_load_pipeline, batch_file, tmp_path):
        """Test that resume skips already-completed jobs."""
        mock_pipe = MagicMock()
        mock_load_pipeline.return_value = mock_pipe
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        # Create a state file showing first job completed
        state = BatchState(
            batch_file=str(batch_file),
            output_dir=str(output_dir),
            model_path="model",
            generate_type="t2v",
            completed=["cat_piano.mp4"],
        )
        state.save(output_dir / ".batch_state.json")
        
        summary = run_batch(
            batch_file=batch_file,
            model_path="model",
            output_dir=output_dir,
            resume=True,
        )
        
        # Should only process 2 jobs (skipping the completed one)
        assert summary["total"] == 2
        assert summary["skipped"] == 1
        assert mock_generate.call_count == 2

    @patch.object(batch_inference, "load_pipeline")
    def test_run_batch_empty_file(self, mock_load_pipeline, tmp_path):
        """Test running batch with empty file."""
        batch_file = tmp_path / "empty.jsonl"
        batch_file.write_text("")
        
        output_dir = tmp_path / "output"
        
        summary = run_batch(
            batch_file=batch_file,
            model_path="model",
            output_dir=output_dir,
            resume=False,
        )
        
        assert summary["total"] == 0
        # Pipeline should not be loaded for empty batch
        mock_load_pipeline.assert_not_called()

    @patch.object(batch_inference, "load_pipeline")
    @patch.object(batch_inference, "generate_single_video")
    def test_run_batch_multi_gpu_distribution(self, mock_generate, mock_load_pipeline, tmp_path):
        """Test multi-GPU job distribution in run_batch."""
        # Create batch file with 6 jobs
        jobs = [
            {"prompt": f"Job {i}", "output_name": f"video{i}.mp4"}
            for i in range(6)
        ]
        batch_file = tmp_path / "batch.jsonl"
        batch_file.write_text("\n".join(json.dumps(j) for j in jobs))
        
        mock_pipe = MagicMock()
        mock_load_pipeline.return_value = mock_pipe
        
        output_dir = tmp_path / "output"
        
        # GPU 0 of 3 should get jobs 0, 3 (indices 0, 3)
        summary = run_batch(
            batch_file=batch_file,
            model_path="model",
            output_dir=output_dir,
            gpu_id=0,
            num_gpus=3,
            resume=False,
        )
        
        assert summary["total"] == 2  # GPU 0 gets 2 jobs
        assert mock_generate.call_count == 2

    @patch.object(batch_inference, "load_pipeline")
    @patch.object(batch_inference, "generate_single_video")
    def test_run_batch_state_persists_after_each_job(self, mock_generate, mock_load_pipeline, batch_file, tmp_path):
        """Test that state is saved after each job, not just at the end."""
        mock_pipe = MagicMock()
        mock_load_pipeline.return_value = mock_pipe
        
        # Track how many times state file is written
        state_writes = []
        original_save = BatchState.save
        
        def tracking_save(self, state_file):
            original_save(self, state_file)
            state_writes.append(len(self.completed))
        
        with patch.object(BatchState, 'save', tracking_save):
            output_dir = tmp_path / "output"
            run_batch(
                batch_file=batch_file,
                model_path="model",
                output_dir=output_dir,
                resume=True,
            )
        
        # State should be saved 3 times (once per job)
        assert len(state_writes) == 3
        assert state_writes == [1, 2, 3]  # Progressively more completed jobs


# =============================================================================
# Resolution Map Tests
# =============================================================================


class TestResolutionMap:
    """Tests for resolution mapping."""

    def test_resolution_map_contains_expected_models(self):
        """Test that RESOLUTION_MAP contains expected model keys."""
        assert "cogvideox1.5-5b-i2v" in RESOLUTION_MAP
        assert "cogvideox1.5-5b" in RESOLUTION_MAP
        assert "cogvideox-5b-i2v" in RESOLUTION_MAP
        assert "cogvideox-5b" in RESOLUTION_MAP
        assert "cogvideox-2b" in RESOLUTION_MAP

    def test_resolution_values_are_valid(self):
        """Test that all resolution values are valid (height, width) tuples."""
        for model, resolution in RESOLUTION_MAP.items():
            assert isinstance(resolution, tuple)
            assert len(resolution) == 2
            assert resolution[0] > 0  # height
            assert resolution[1] > 0  # width


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_batch_job_with_all_optional_fields(self):
        """Test BatchJob with all optional fields set."""
        data = {
            "prompt": "Test",
            "output_name": "test.mp4",
            "image_path": "/img.jpg",
            "video_path": "/vid.mp4",
            "num_frames": 100,
            "guidance_scale": 10.0,
            "num_inference_steps": 100,
            "seed": 9999,
            "width": 1920,
            "height": 1080,
        }
        job = BatchJob.from_dict(data)
        
        assert job.width == 1920
        assert job.height == 1080
        assert job.num_frames == 100

    def test_batch_job_preserves_extra_fields(self):
        """Test that extra fields in JSON don't cause errors."""
        data = {
            "prompt": "Test",
            "output_name": "test.mp4",
            "extra_field": "ignored",
            "another_field": 123,
        }
        job = BatchJob.from_dict(data)
        
        assert job.prompt == "Test"
        assert job.output_name == "test.mp4"
        # Extra fields should be ignored without error

    @patch.object(batch_inference, "load_pipeline")
    @patch.object(batch_inference, "generate_single_video")
    def test_run_batch_corrupted_state_starts_fresh(self, mock_generate, mock_load_pipeline, batch_file, tmp_path):
        """Test that corrupted state file doesn't prevent batch from running."""
        mock_pipe = MagicMock()
        mock_load_pipeline.return_value = mock_pipe
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        # Create corrupted state file
        state_file = output_dir / ".batch_state.json"
        state_file.write_text("{corrupted: json")
        
        summary = run_batch(
            batch_file=batch_file,
            model_path="model",
            output_dir=output_dir,
            resume=True,
        )
        
        # Should start fresh and process all jobs
        assert summary["total"] == 3
        assert mock_generate.call_count == 3

    def test_batch_state_updated_at_is_set_on_save(self, tmp_path):
        """Test that updated_at is set when saving state."""
        state = BatchState(
            batch_file="batch.jsonl",
            output_dir="./output",
            model_path="model",
            generate_type="t2v",
        )
        
        assert state.updated_at == ""
        
        state_file = tmp_path / "state.json"
        state.save(state_file)
        
        # updated_at should be set after save
        assert state.updated_at != ""

    @patch.object(batch_inference, "load_pipeline")
    @patch.object(batch_inference, "generate_single_video")
    def test_run_batch_all_jobs_fail(self, mock_generate, mock_load_pipeline, batch_file, tmp_path):
        """Test batch where all jobs fail."""
        mock_pipe = MagicMock()
        mock_load_pipeline.return_value = mock_pipe
        mock_generate.side_effect = RuntimeError("All fail")
        
        output_dir = tmp_path / "output"
        
        summary = run_batch(
            batch_file=batch_file,
            model_path="model",
            output_dir=output_dir,
            resume=False,
        )
        
        assert summary["completed"] == 0
        assert summary["failed"] == 3

    def test_jsonl_with_unicode(self, tmp_path):
        """Test JSONL parsing with unicode characters."""
        content = """{"prompt": "猫がピアノを弾いている", "output_name": "unicode1.mp4"}
{"prompt": "海辺の波 🌊", "output_name": "unicode2.mp4"}
{"prompt": "Café résumé naïve", "output_name": "unicode3.mp4"}"""
        batch_file = tmp_path / "unicode.jsonl"
        batch_file.write_text(content, encoding="utf-8")
        
        jobs = load_batch_file(batch_file)
        
        assert len(jobs) == 3
        assert jobs[0].prompt == "猫がピアノを弾いている"
        assert "🌊" in jobs[1].prompt

    def test_jsonl_with_very_long_prompt(self, tmp_path):
        """Test JSONL with very long prompts."""
        long_prompt = "A " + "very " * 1000 + "long prompt"
        content = json.dumps({"prompt": long_prompt, "output_name": "long.mp4"})
        batch_file = tmp_path / "long.jsonl"
        batch_file.write_text(content)
        
        jobs = load_batch_file(batch_file)
        
        assert len(jobs) == 1
        assert len(jobs[0].prompt) > 4000
