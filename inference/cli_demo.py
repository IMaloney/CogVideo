"""
This script demonstrates how to generate a video using the CogVideoX model with the Hugging Face `diffusers` pipeline.
The script supports different types of video generation, including text-to-video (t2v), image-to-video (i2v),
and video-to-video (v2v), depending on the input data and different weight.

- text-to-video: THUDM/CogVideoX-5b, THUDM/CogVideoX-2b or THUDM/CogVideoX1.5-5b
- video-to-video: THUDM/CogVideoX-5b, THUDM/CogVideoX-2b or THUDM/CogVideoX1.5-5b
- image-to-video: THUDM/CogVideoX-5b-I2V or THUDM/CogVideoX1.5-5b-I2V

Running the Script:
To run the script, use the following command with appropriate arguments:

```bash
# Single GPU (default behavior, uses CPU offload):
$ python cli_demo.py --prompt "A girl riding a bike." --model_path THUDM/CogVideoX1.5-5b --generate_type "t2v"

# Multi-GPU with balanced device mapping:
$ python cli_demo.py --prompt "A girl riding a bike." --model_path THUDM/CogVideoX1.5-5b --generate_type "t2v" --device_map balanced

# Multi-GPU with auto device mapping:
$ python cli_demo.py --prompt "A girl riding a bike." --model_path THUDM/CogVideoX1.5-5b --generate_type "t2v" --device_map auto
```

You can change `pipe.enable_sequential_cpu_offload()` to `pipe.enable_model_cpu_offload()` to speed up inference, but this will use more GPU memory

Multi-GPU Support:
- Use `--device_map balanced` to distribute the model evenly across available GPUs (recommended for inference)
- Use `--device_map auto` for automatic device placement by accelerate
- Use `--device_map sequential` to fill GPUs sequentially (useful for uneven memory)
- Default behavior (no --device_map) uses CPU offload for single-GPU setups

Additional options are available to specify the model path, guidance scale, number of inference steps, video generation type, and output paths.

"""

import argparse
import logging
from typing import Literal, Optional

import torch

from diffusers import (
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXPipeline,
    CogVideoXVideoToVideoPipeline,
)
from diffusers.utils import export_to_video, load_image, load_video


logging.basicConfig(level=logging.INFO)

# Recommended resolution for each model (width, height)
RESOLUTION_MAP = {
    # cogvideox1.5-*
    "cogvideox1.5-5b-i2v": (768, 1360),
    "cogvideox1.5-5b": (768, 1360),
    # cogvideox-*
    "cogvideox-5b-i2v": (480, 720),
    "cogvideox-5b": (480, 720),
    "cogvideox-2b": (480, 720),
}

# Valid device_map options for multi-GPU support
VALID_DEVICE_MAPS = {"auto", "balanced", "sequential"}


def generate_video(
    prompt: str,
    model_path: str,
    lora_path: str = None,
    lora_rank: int = 128,
    num_frames: int = 81,
    width: Optional[int] = None,
    height: Optional[int] = None,
    output_path: str = "./output.mp4",
    image_or_video_path: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    generate_type: str = Literal["t2v", "i2v", "v2v"],  # i2v: image to video, v2v: video to video
    seed: int = 42,
    fps: int = 16,
    device_map: Optional[str] = None,
):
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - model_path (str): The path of the pre-trained model to be used.
    - lora_path (str): The path of the LoRA weights to be used.
    - lora_rank (int): The rank of the LoRA weights.
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - num_frames (int): Number of frames to generate. CogVideoX1.0 generates 49 frames for 6 seconds at 8 fps, while CogVideoX1.5 produces either 81 or 161 frames, corresponding to 5 seconds or 10 seconds at 16 fps.
    - width (int): The width of the generated video, applicable only for CogVideoX1.5-5B-I2V
    - height (int): The height of the generated video, applicable only for CogVideoX1.5-5B-I2V
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - num_videos_per_prompt (int): Number of videos to generate per prompt.
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - generate_type (str): The type of video generation (e.g., 't2v', 'i2v', 'v2v').·
    - seed (int): The seed for reproducibility.
    - fps (int): The frames per second for the generated video.
    - device_map (str): Device placement strategy for multi-GPU support. Options:
        - None (default): Uses sequential CPU offload for single-GPU setups
        - "balanced": Distributes model layers evenly across available GPUs (recommended)
        - "auto": Automatic device placement by accelerate library
        - "sequential": Fills GPUs one by one in order

    Multi-GPU Usage Examples:
        # Balanced distribution across GPUs (recommended):
        generate_video(prompt="...", model_path="...", device_map="balanced")

        # Automatic device placement:
        generate_video(prompt="...", model_path="...", device_map="auto")
    """

    # 1.  Load the pre-trained CogVideoX pipeline with the specified precision (bfloat16).
    # When device_map is specified, the model is distributed across multiple GPUs.
    # When device_map is None (default), CPU offload is enabled for single-GPU setups.

    image = None
    video = None

    model_name = model_path.split("/")[-1].lower()
    desired_resolution = RESOLUTION_MAP[model_name]
    if width is None or height is None:
        height, width = desired_resolution
        logging.info(
            f"\033[1mUsing default resolution {desired_resolution} for {model_name}\033[0m"
        )
    elif (height, width) != desired_resolution:
        if generate_type == "i2v":
            # For i2v models, use user-defined width and height
            logging.warning(
                f"\033[1;31mThe width({width}) and height({height}) are not recommended for {model_name}. The best resolution is {desired_resolution}.\033[0m"
            )
        else:
            # Otherwise, use the recommended width and height
            logging.warning(
                f"\033[1;31m{model_name} is not supported for custom resolution. Setting back to default resolution {desired_resolution}.\033[0m"
            )
            height, width = desired_resolution

    # Validate device_map if provided
    if device_map is not None and device_map not in VALID_DEVICE_MAPS:
        raise ValueError(
            f"Invalid device_map '{device_map}'. Must be one of: {', '.join(sorted(VALID_DEVICE_MAPS))} or None"
        )

    # Build kwargs for from_pretrained - add device_map only when specified
    load_kwargs = {"torch_dtype": dtype}
    if device_map is not None:
        load_kwargs["device_map"] = device_map
        logging.info(f"\033[1mUsing device_map='{device_map}' for multi-GPU inference\033[0m")

    if generate_type == "i2v":
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_path, **load_kwargs)
        image = load_image(image=image_or_video_path)
    elif generate_type == "t2v":
        pipe = CogVideoXPipeline.from_pretrained(model_path, **load_kwargs)
    else:
        pipe = CogVideoXVideoToVideoPipeline.from_pretrained(model_path, **load_kwargs)
        video = load_video(image_or_video_path)

    # If you're using with lora, add this code
    if lora_path:
        pipe.load_lora_weights(
            lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1"
        )
        pipe.fuse_lora(components=["transformer"], lora_scale=1.0)

    # 2. Set Scheduler.
    # Can be changed to `CogVideoXDPMScheduler` or `CogVideoXDDIMScheduler`.
    # We recommend using `CogVideoXDDIMScheduler` for CogVideoX-2B.
    # using `CogVideoXDPMScheduler` for CogVideoX-5B / CogVideoX-5B-I2V.

    # pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )

    # 3. Enable CPU offload for the model (only when not using multi-GPU device_map).
    # When device_map is specified, the model is already distributed across GPUs,
    # so CPU offload is not needed and would conflict with device placement.
    if device_map is None:
        # Single-GPU mode: use CPU offload to manage memory
        # Turn off if you have multiple GPUs or enough GPU memory (such as H100)
        # pipe.enable_model_cpu_offload()
        pipe.enable_sequential_cpu_offload()

    # VAE optimizations work in both single and multi-GPU modes
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    # 4. Generate the video frames based on the prompt.
    # `num_frames` is the Number of frames to generate.
    if generate_type == "i2v":
        video_generate = pipe(
            height=height,
            width=width,
            prompt=prompt,
            image=image,
            # The path of the image, the resolution of video will be the same as the image for CogVideoX1.5-5B-I2V, otherwise it will be 720 * 480
            num_videos_per_prompt=num_videos_per_prompt,  # Number of videos to generate per prompt
            num_inference_steps=num_inference_steps,  # Number of inference steps
            num_frames=num_frames,  # Number of frames to generate
            use_dynamic_cfg=True,  # This id used for DPM scheduler, for DDIM scheduler, it should be False
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),  # Set the seed for reproducibility
        ).frames[0]
    elif generate_type == "t2v":
        video_generate = pipe(
            height=height,
            width=width,
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),
        ).frames[0]
    else:
        video_generate = pipe(
            height=height,
            width=width,
            prompt=prompt,
            video=video,  # The path of the video to be used as the background of the video
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),  # Set the seed for reproducibility
        ).frames[0]
    export_to_video(video_generate, output_path, fps=fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a video from a text prompt using CogVideoX"
    )
    parser.add_argument(
        "--prompt", type=str, required=True, help="The description of the video to be generated"
    )
    parser.add_argument(
        "--image_or_video_path",
        type=str,
        default=None,
        help="The path of the image to be used as the background of the video",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="THUDM/CogVideoX1.5-5B",
        help="Path of the pre-trained model use",
    )
    parser.add_argument(
        "--lora_path", type=str, default=None, help="The path of the LoRA weights to be used"
    )
    parser.add_argument("--lora_rank", type=int, default=128, help="The rank of the LoRA weights")
    parser.add_argument(
        "--output_path", type=str, default="./output.mp4", help="The path save generated video"
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance"
    )
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Inference steps")
    parser.add_argument(
        "--num_frames", type=int, default=81, help="Number of steps for the inference process"
    )
    parser.add_argument("--width", type=int, default=None, help="The width of the generated video")
    parser.add_argument(
        "--height", type=int, default=None, help="The height of the generated video"
    )
    parser.add_argument(
        "--fps", type=int, default=16, help="The frames per second for the generated video"
    )
    parser.add_argument(
        "--num_videos_per_prompt",
        type=int,
        default=1,
        help="Number of videos to generate per prompt",
    )
    parser.add_argument(
        "--generate_type", type=str, default="t2v", help="The type of video generation"
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", help="The data type for computation"
    )
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")
    parser.add_argument(
        "--device_map",
        type=str,
        default=None,
        choices=["auto", "balanced", "sequential"],
        help="Device placement strategy for multi-GPU inference. Options: 'balanced' (recommended, distributes evenly), 'auto' (automatic placement), 'sequential' (fills GPUs in order). Default: None (uses CPU offload for single-GPU)",
    )

    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    generate_video(
        prompt=args.prompt,
        model_path=args.model_path,
        lora_path=args.lora_path,
        lora_rank=args.lora_rank,
        output_path=args.output_path,
        num_frames=args.num_frames,
        width=args.width,
        height=args.height,
        image_or_video_path=args.image_or_video_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        dtype=dtype,
        generate_type=args.generate_type,
        seed=args.seed,
        fps=args.fps,
        device_map=args.device_map,
    )
