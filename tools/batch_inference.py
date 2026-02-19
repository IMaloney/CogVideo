#!/usr/bin/env python3
"""
Batch Inference Pipeline for CogVideo

A production-grade tool for generating multiple videos from a batch input file.
Supports text-to-video (t2v), image-to-video (i2v), and video-to-video (v2v) generation.

Features:
- JSONL batch input format (one job per line)
- Resume capability (skips already-completed jobs)
- Progress tracking with ETA
- Robust error handling (logs errors, continues batch)
- Memory-efficient (loads model once, generates sequentially)
- Multi-GPU support via job-level parallelism

Input Format (JSONL):
Each line is a JSON object with the following fields:
    - prompt (str, required): Text description for video generation
    - output_name (str, required): Output filename (without path, e.g., "my_video.mp4")
    - image_path (str, optional): Path to input image for i2v generation
    - video_path (str, optional): Path to input video for v2v generation
    - num_frames (int, optional): Number of frames (default: 81)
    - guidance_scale (float, optional): CFG scale (default: 6.0)
    - num_inference_steps (int, optional): Inference steps (default: 50)
    - seed (int, optional): Random seed (default: 42)
    - width (int, optional): Video width
    - height (int, optional): Video height

Example JSONL (resources/example_batch.jsonl):
    {"prompt": "A cat playing piano", "output_name": "cat_piano.mp4"}
    {"prompt": "Waves crashing on beach", "output_name": "beach.mp4", "num_frames": 49}
    {"prompt": "Transform this image", "output_name": "i2v_output.mp4", "image_path": "./input.jpg"}

Usage:
    # Basic usage (text-to-video)
    python tools/batch_inference.py \\
        --batch_file resources/example_batch.jsonl \\
        --model_path THUDM/CogVideoX1.5-5B \\
        --output_dir ./batch_output

    # Image-to-video batch
    python tools/batch_inference.py \\
        --batch_file resources/i2v_batch.jsonl \\
        --model_path THUDM/CogVideoX1.5-5B-I2V \\
        --generate_type i2v \\
        --output_dir ./batch_output

    # Resume interrupted batch
    python tools/batch_inference.py \\
        --batch_file resources/example_batch.jsonl \\
        --model_path THUDM/CogVideoX1.5-5B \\
        --output_dir ./batch_output \\
        --resume

    # Multi-GPU: distribute jobs across GPUs
    CUDA_VISIBLE_DEVICES=0 python tools/batch_inference.py --batch_file batch.jsonl --gpu_id 0 --num_gpus 4 &
    CUDA_VISIBLE_DEVICES=1 python tools/batch_inference.py --batch_file batch.jsonl --gpu_id 1 --num_gpus 4 &
    CUDA_VISIBLE_DEVICES=2 python tools/batch_inference.py --batch_file batch.jsonl --gpu_id 2 --num_gpus 4 &
    CUDA_VISIBLE_DEVICES=3 python tools/batch_inference.py --batch_file batch.jsonl --gpu_id 3 --num_gpus 4 &

Author: CogVideo Contributors
License: Apache 2.0
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import torch
from tqdm import tqdm

from diffusers import (
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXPipeline,
    CogVideoXVideoToVideoPipeline,
)
from diffusers.utils import export_to_video, load_image, load_video


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Recommended resolution for each model (height, width)
RESOLUTION_MAP = {
    "cogvideox1.5-5b-i2v": (768, 1360),
    "cogvideox1.5-5b": (768, 1360),
    "cogvideox-5b-i2v": (480, 720),
    "cogvideox-5b": (480, 720),
    "cogvideox-2b": (480, 720),
}


@dataclass
class BatchJob:
    """Represents a single job in the batch."""

    prompt: str
    output_name: str
    image_path: Optional[str] = None
    video_path: Optional[str] = None
    num_frames: Optional[int] = None
    guidance_scale: Optional[float] = None
    num_inference_steps: Optional[int] = None
    seed: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None

    # Internal fields
    line_number: int = 0
    status: str = "pending"
    error: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any], line_number: int = 0) -> "BatchJob":
        """Create a BatchJob from a dictionary."""
        return cls(
            prompt=data.get("prompt", ""),
            output_name=data.get("output_name", ""),
            image_path=data.get("image_path"),
            video_path=data.get("video_path"),
            num_frames=data.get("num_frames"),
            guidance_scale=data.get("guidance_scale"),
            num_inference_steps=data.get("num_inference_steps"),
            seed=data.get("seed"),
            width=data.get("width"),
            height=data.get("height"),
            line_number=line_number,
        )

    def validate(self) -> List[str]:
        """Validate the job and return list of errors."""
        errors = []
        if not self.prompt:
            errors.append("Missing required field: prompt")
        if not self.output_name:
            errors.append("Missing required field: output_name")
        return errors


@dataclass
class BatchState:
    """Tracks batch progress for resume capability."""

    batch_file: str
    output_dir: str
    model_path: str
    generate_type: str
    completed: List[str] = field(default_factory=list)
    failed: List[Dict[str, str]] = field(default_factory=list)
    started_at: str = ""
    updated_at: str = ""

    @classmethod
    def load(cls, state_file: Path) -> Optional["BatchState"]:
        """Load state from file."""
        if not state_file.exists():
            return None
        try:
            with open(state_file, "r") as f:
                data = json.load(f)
            return cls(**data)
        except Exception as e:
            logger.warning(f"Failed to load state file: {e}")
            return None

    def save(self, state_file: Path):
        """Save state to file."""
        self.updated_at = datetime.now().isoformat()
        with open(state_file, "w") as f:
            json.dump(self.__dict__, f, indent=2)

    def mark_completed(self, output_name: str):
        """Mark a job as completed."""
        if output_name not in self.completed:
            self.completed.append(output_name)

    def mark_failed(self, output_name: str, error: str):
        """Mark a job as failed."""
        self.failed.append({"output_name": output_name, "error": error})

    def is_completed(self, output_name: str) -> bool:
        """Check if a job was already completed."""
        return output_name in self.completed


def load_batch_file(batch_file: Path) -> List[BatchJob]:
    """
    Load and parse a JSONL batch file.

    Args:
        batch_file: Path to the JSONL file

    Returns:
        List of BatchJob objects
    """
    jobs = []

    with open(batch_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            try:
                data = json.loads(line)
                job = BatchJob.from_dict(data, line_number=line_num)

                # Validate job
                errors = job.validate()
                if errors:
                    logger.warning(f"Line {line_num}: Invalid job - {', '.join(errors)}")
                    continue

                jobs.append(job)

            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: Invalid JSON - {e}")
                continue

    return jobs


def load_pipeline(
    model_path: str,
    generate_type: str,
    dtype: torch.dtype = torch.bfloat16,
    lora_path: Optional[str] = None,
    enable_cpu_offload: bool = True,
):
    """
    Load the appropriate pipeline for the generation type.

    Args:
        model_path: Path to the model
        generate_type: Type of generation (t2v, i2v, v2v)
        dtype: Data type for computation
        lora_path: Optional path to LoRA weights
        enable_cpu_offload: Whether to enable CPU offloading

    Returns:
        The loaded pipeline
    """
    logger.info(f"Loading pipeline for {generate_type} from {model_path}")

    if generate_type == "i2v":
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
    elif generate_type == "t2v":
        pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)
    else:  # v2v
        pipe = CogVideoXVideoToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)

    # Load LoRA weights if provided
    if lora_path:
        logger.info(f"Loading LoRA weights from {lora_path}")
        pipe.load_lora_weights(
            lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="batch_lora"
        )
        pipe.fuse_lora(components=["transformer"], lora_scale=1.0)

    # Set scheduler
    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )

    # Enable memory optimizations
    if enable_cpu_offload:
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to("cuda")

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    logger.info("Pipeline loaded successfully")
    return pipe


def generate_single_video(
    pipe,
    job: BatchJob,
    generate_type: str,
    model_name: str,
    output_path: Path,
    default_num_frames: int = 81,
    default_guidance_scale: float = 6.0,
    default_num_inference_steps: int = 50,
    default_seed: int = 42,
    fps: int = 16,
):
    """
    Generate a single video from a job.

    Args:
        pipe: The loaded pipeline
        job: The batch job to process
        generate_type: Type of generation
        model_name: Name of the model (for resolution lookup)
        output_path: Full path for output video
        default_*: Default values for optional parameters
        fps: Frames per second for output video
    """
    # Determine resolution
    desired_resolution = RESOLUTION_MAP.get(model_name.lower(), (480, 720))
    height = job.height if job.height else desired_resolution[0]
    width = job.width if job.width else desired_resolution[1]

    # Use job-specific or default values
    num_frames = job.num_frames or default_num_frames
    guidance_scale = job.guidance_scale or default_guidance_scale
    num_inference_steps = job.num_inference_steps or default_num_inference_steps
    seed = job.seed or default_seed

    # Load image/video if needed
    image = None
    video = None

    if generate_type == "i2v":
        if not job.image_path:
            raise ValueError("image_path is required for i2v generation")
        image = load_image(image=job.image_path)
    elif generate_type == "v2v":
        if not job.video_path:
            raise ValueError("video_path is required for v2v generation")
        video = load_video(job.video_path)

    # Generate video
    generator = torch.Generator().manual_seed(seed)

    if generate_type == "i2v":
        video_frames = pipe(
            height=height,
            width=width,
            prompt=job.prompt,
            image=image,
            num_videos_per_prompt=1,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            generator=generator,
        ).frames[0]
    elif generate_type == "t2v":
        video_frames = pipe(
            height=height,
            width=width,
            prompt=job.prompt,
            num_videos_per_prompt=1,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            generator=generator,
        ).frames[0]
    else:  # v2v
        video_frames = pipe(
            height=height,
            width=width,
            prompt=job.prompt,
            video=video,
            num_videos_per_prompt=1,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            generator=generator,
        ).frames[0]

    # Export video
    export_to_video(video_frames, str(output_path), fps=fps)


def run_batch(
    batch_file: Path,
    model_path: str,
    output_dir: Path,
    generate_type: str = "t2v",
    dtype: torch.dtype = torch.bfloat16,
    lora_path: Optional[str] = None,
    enable_cpu_offload: bool = True,
    resume: bool = True,
    gpu_id: int = 0,
    num_gpus: int = 1,
    default_num_frames: int = 81,
    default_guidance_scale: float = 6.0,
    default_num_inference_steps: int = 50,
    default_seed: int = 42,
    fps: int = 16,
) -> Dict[str, Any]:
    """
    Run batch inference on a JSONL file.

    Args:
        batch_file: Path to the JSONL batch file
        model_path: Path to the model
        output_dir: Directory for output videos
        generate_type: Type of generation (t2v, i2v, v2v)
        dtype: Data type for computation
        lora_path: Optional path to LoRA weights
        enable_cpu_offload: Whether to enable CPU offloading
        resume: Whether to resume from previous state
        gpu_id: GPU ID for multi-GPU distribution
        num_gpus: Total number of GPUs for distribution
        default_*: Default values for optional parameters
        fps: Frames per second for output videos

    Returns:
        Summary dictionary with statistics
    """
    # Setup paths
    output_dir.mkdir(parents=True, exist_ok=True)
    state_file = output_dir / ".batch_state.json"
    error_log = output_dir / "errors.log"

    # Load jobs
    logger.info(f"Loading batch file: {batch_file}")
    all_jobs = load_batch_file(batch_file)
    logger.info(f"Found {len(all_jobs)} valid jobs in batch file")

    # Distribute jobs across GPUs if using multi-GPU
    if num_gpus > 1:
        jobs = [j for i, j in enumerate(all_jobs) if i % num_gpus == gpu_id]
        logger.info(f"GPU {gpu_id}/{num_gpus}: Processing {len(jobs)} jobs")
    else:
        jobs = all_jobs

    # Load or create state
    state = None
    if resume:
        state = BatchState.load(state_file)
        if state:
            logger.info(f"Resuming batch: {len(state.completed)} already completed")

    if state is None:
        state = BatchState(
            batch_file=str(batch_file),
            output_dir=str(output_dir),
            model_path=model_path,
            generate_type=generate_type,
            started_at=datetime.now().isoformat(),
        )

    # Filter out completed jobs
    if resume:
        pending_jobs = [j for j in jobs if not state.is_completed(j.output_name)]
        skipped = len(jobs) - len(pending_jobs)
        if skipped > 0:
            logger.info(f"Skipping {skipped} already-completed jobs")
        jobs = pending_jobs

    if not jobs:
        logger.info("No jobs to process")
        return {"total": 0, "completed": 0, "failed": 0, "skipped": len(all_jobs)}

    # Load pipeline
    model_name = model_path.split("/")[-1]
    pipe = load_pipeline(
        model_path=model_path,
        generate_type=generate_type,
        dtype=dtype,
        lora_path=lora_path,
        enable_cpu_offload=enable_cpu_offload,
    )

    # Process jobs
    completed = 0
    failed = 0
    start_time = time.time()

    with tqdm(total=len(jobs), desc="Generating videos", unit="video") as pbar:
        for job in jobs:
            output_path = output_dir / job.output_name

            try:
                logger.info(f"Processing: {job.output_name} - \"{job.prompt[:50]}...\"")

                generate_single_video(
                    pipe=pipe,
                    job=job,
                    generate_type=generate_type,
                    model_name=model_name,
                    output_path=output_path,
                    default_num_frames=default_num_frames,
                    default_guidance_scale=default_guidance_scale,
                    default_num_inference_steps=default_num_inference_steps,
                    default_seed=default_seed,
                    fps=fps,
                )

                state.mark_completed(job.output_name)
                completed += 1
                logger.info(f"Completed: {job.output_name}")

            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                logger.error(f"Failed: {job.output_name} - {error_msg}")

                # Log full traceback to error log
                with open(error_log, "a") as f:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"Job: {job.output_name}\n")
                    f.write(f"Prompt: {job.prompt}\n")
                    f.write(f"Time: {datetime.now().isoformat()}\n")
                    f.write(f"Error: {error_msg}\n")
                    f.write(traceback.format_exc())

                state.mark_failed(job.output_name, error_msg)
                failed += 1

            # Save state after each job (for resume)
            state.save(state_file)
            pbar.update(1)

            # Update ETA in progress bar
            elapsed = time.time() - start_time
            if completed + failed > 0:
                avg_time = elapsed / (completed + failed)
                remaining = len(jobs) - (completed + failed)
                eta_seconds = avg_time * remaining
                pbar.set_postfix(
                    {"done": completed, "failed": failed, "ETA": f"{eta_seconds/60:.1f}m"}
                )

    # Final summary
    elapsed_total = time.time() - start_time
    summary = {
        "total": len(jobs),
        "completed": completed,
        "failed": failed,
        "skipped": len(all_jobs) - len(jobs),
        "elapsed_seconds": elapsed_total,
        "avg_seconds_per_video": elapsed_total / max(completed + failed, 1),
    }

    logger.info("=" * 60)
    logger.info("BATCH COMPLETE")
    logger.info(f"  Total jobs: {summary['total']}")
    logger.info(f"  Completed: {summary['completed']}")
    logger.info(f"  Failed: {summary['failed']}")
    logger.info(f"  Elapsed: {elapsed_total/60:.1f} minutes")
    if summary['failed'] > 0:
        logger.info(f"  See errors in: {error_log}")
    logger.info("=" * 60)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Batch inference for CogVideo - generate multiple videos from a JSONL file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic text-to-video batch
  python batch_inference.py --batch_file prompts.jsonl --model_path THUDM/CogVideoX1.5-5B

  # Image-to-video batch with custom output directory
  python batch_inference.py --batch_file i2v.jsonl --model_path THUDM/CogVideoX1.5-5B-I2V \\
      --generate_type i2v --output_dir ./my_videos

  # Multi-GPU: run on 4 GPUs (one process per GPU)
  for i in {0..3}; do
      CUDA_VISIBLE_DEVICES=$i python batch_inference.py --batch_file batch.jsonl \\
          --gpu_id $i --num_gpus 4 &
  done

JSONL Format:
  Each line is a JSON object with: prompt (required), output_name (required),
  and optional: image_path, video_path, num_frames, guidance_scale,
  num_inference_steps, seed, width, height
        """,
    )

    # Required arguments
    parser.add_argument("--batch_file", type=str, required=True, help="Path to JSONL batch file")
    parser.add_argument(
        "--model_path",
        type=str,
        default="THUDM/CogVideoX1.5-5B",
        help="Path to the model (default: THUDM/CogVideoX1.5-5B)",
    )

    # Output settings
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./batch_output",
        help="Directory for output videos (default: ./batch_output)",
    )
    parser.add_argument(
        "--generate_type",
        type=str,
        choices=["t2v", "i2v", "v2v"],
        default="t2v",
        help="Generation type (default: t2v)",
    )

    # Model settings
    parser.add_argument(
        "--lora_path", type=str, default=None, help="Path to LoRA weights (optional)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "bfloat16"],
        default="bfloat16",
        help="Data type for computation (default: bfloat16)",
    )
    parser.add_argument(
        "--disable_cpu_offload",
        action="store_true",
        help="Disable CPU offloading (uses more VRAM but faster)",
    )

    # Default generation parameters
    parser.add_argument(
        "--num_frames", type=int, default=81, help="Default number of frames (default: 81)"
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=6.0, help="Default guidance scale (default: 6.0)"
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="Default inference steps (default: 50)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Default random seed (default: 42)")
    parser.add_argument("--fps", type=int, default=16, help="Output video FPS (default: 16)")

    # Resume and multi-GPU
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from previous state (default: True)",
    )
    parser.add_argument("--no_resume", action="store_true", help="Don't resume, start fresh")
    parser.add_argument(
        "--gpu_id", type=int, default=0, help="GPU ID for multi-GPU distribution (default: 0)"
    )
    parser.add_argument(
        "--num_gpus", type=int, default=1, help="Total number of GPUs for distribution (default: 1)"
    )

    args = parser.parse_args()

    # Validate batch file exists
    batch_file = Path(args.batch_file)
    if not batch_file.exists():
        logger.error(f"Batch file not found: {batch_file}")
        sys.exit(1)

    # Parse dtype
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    # Run batch
    try:
        summary = run_batch(
            batch_file=batch_file,
            model_path=args.model_path,
            output_dir=Path(args.output_dir),
            generate_type=args.generate_type,
            dtype=dtype,
            lora_path=args.lora_path,
            enable_cpu_offload=not args.disable_cpu_offload,
            resume=not args.no_resume,
            gpu_id=args.gpu_id,
            num_gpus=args.num_gpus,
            default_num_frames=args.num_frames,
            default_guidance_scale=args.guidance_scale,
            default_num_inference_steps=args.num_inference_steps,
            default_seed=args.seed,
            fps=args.fps,
        )

        # Exit with error code if any failures
        if summary["failed"] > 0:
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("\nBatch interrupted by user. Progress saved for resume.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Batch failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
