#!/usr/bin/env python3
"""
HY-World 1.5 (WorldPlay) - Gradio Web UI
Windows + Blackwell GPU Compatible Version
"""

import os
import sys
import argparse
import tempfile
import time
from datetime import datetime
from pathlib import Path

# Set environment variables before importing torch
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gradio as gr
import torch
from PIL import Image
from loguru import logger

# Check if models are available
MODEL_BASE = os.environ.get("MODEL_BASE", os.path.join(os.path.dirname(__file__), "ckpts"))

# Global Qwen2-VL model instance (lazy loading)
_qwen_vl_model = None
_qwen_vl_processor = None


def get_qwen_vl_model():
    """Get or create the Qwen2-VL model for image analysis."""
    global _qwen_vl_model, _qwen_vl_processor
    
    if _qwen_vl_model is not None:
        return _qwen_vl_model, _qwen_vl_processor
    
    logger.info("Loading Qwen2-VL model for image analysis...")
    
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    
    model_name = "Qwen/Qwen2-VL-7B-Instruct"
    
    _qwen_vl_model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    _qwen_vl_processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    
    logger.info("Qwen2-VL model loaded successfully!")
    
    return _qwen_vl_model, _qwen_vl_processor


def analyze_image_for_prompt(image_path, progress=gr.Progress()):
    """Analyze an image and generate a descriptive prompt for video generation."""
    
    if image_path is None:
        return "Please upload an image first."
    
    try:
        progress(0.1, desc="Loading Qwen2-VL model...")
        model, processor = get_qwen_vl_model()
        
        progress(0.3, desc="Processing image...")
        
        # Load and prepare the image
        image = Image.open(image_path).convert("RGB")
        
        # Create the message for Qwen2-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe this image in detail for a video generation prompt. Focus on: the scene, lighting, atmosphere, objects, colors, and any notable features. Write in English, in 2-3 sentences, suitable for an AI video generation model. Do not include any preamble, just output the description directly."},
                ],
            }
        ]
        
        # Prepare inputs using the chat template
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        
        progress(0.5, desc="Generating description...")
        
        # Generate the description
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        
        # Decode the output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        
        progress(1.0, desc="Done!")
        
        logger.info(f"Generated prompt: {output_text}")
        return output_text.strip()
        
    except Exception as e:
        import traceback
        error_msg = f"Error analyzing image: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return error_msg


def check_environment():
    """Check if the environment is properly configured."""
    info = []
    
    # PyTorch version
    info.append(f"PyTorch: {torch.__version__}")
    
    # CUDA availability
    if torch.cuda.is_available():
        info.append(f"CUDA: {torch.version.cuda}")
        info.append(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        info.append(f"VRAM: {props.total_memory / 1024**3:.1f} GB")
    else:
        info.append("CUDA: Not available")
    
    return "\n".join(info)


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for Windows compatibility."""
    # Remove or replace characters that are invalid on Windows
    invalid_chars = '<>:"/\\|?*\n\r'
    for char in invalid_chars:
        filename = filename.replace(char, '')
    # Remove trailing spaces and dots
    filename = filename.strip('. ')
    return filename


# Global pipeline instance (lazy loading)
_pipeline = None
_pipeline_config = None


def get_pipeline(model_path, model_type, action_ckpt, offloading=True, dtype="bf16"):
    """Get or create the inference pipeline."""
    global _pipeline, _pipeline_config
    
    config_key = (model_path, model_type, action_ckpt, offloading, dtype)
    
    if _pipeline is not None and _pipeline_config == config_key:
        return _pipeline
    
    # Import here to avoid loading at startup
    from hyvideo.pipelines.worldplay_video_pipeline import HunyuanVideo_1_5_Pipeline
    from hyvideo.commons.parallel_states import initialize_parallel_state
    
    # Initialize parallel state for single GPU
    initialize_parallel_state(sp=1)
    
    transformer_dtype = torch.bfloat16 if dtype == "bf16" else torch.float32
    
    logger.info(f"Loading pipeline from {model_path}...")
    logger.info(f"Action checkpoint: {action_ckpt}")
    logger.info(f"Model type: {model_type}")
    
    _pipeline = HunyuanVideo_1_5_Pipeline.create_pipeline(
        pretrained_model_name_or_path=model_path,
        transformer_version="480p_i2v",
        enable_offloading=offloading,
        enable_group_offloading=offloading,
        create_sr_pipeline=False,  # Disable SR for now
        force_sparse_attn=False,
        transformer_dtype=transformer_dtype,
        action_ckpt=action_ckpt,
    )
    
    _pipeline_config = config_key
    logger.info("Pipeline loaded successfully!")
    
    return _pipeline


def generate_video(
    image_path,
    prompt,
    pose_string,
    model_type,
    num_inference_steps,
    video_length,
    seed,
    offloading,
    dtype,
    progress=gr.Progress(track_tqdm=True),
):
    """Generate video from image and pose."""
    
    if image_path is None:
        return None, "Please upload an input image."
    
    if not prompt.strip():
        return None, "Please enter a prompt."
    
    # Check model paths
    model_path = os.environ.get("MODEL_PATH", os.path.join(MODEL_BASE, "HunyuanVideo-1.5"))
    
    if model_type == "ar_distilled":
        action_ckpt = os.environ.get("AR_DISTILL_ACTION_MODEL_PATH", 
                                      os.path.join(MODEL_BASE, "HY-WorldPlay", "ar_distilled_action_model", "diffusion_pytorch_model.safetensors"))
        actual_model_type = "ar"
        few_step = True
    elif model_type == "ar":
        action_ckpt = os.environ.get("AR_ACTION_MODEL_PATH",
                                     os.path.join(MODEL_BASE, "HY-WorldPlay", "ar_model", "diffusion_pytorch_model.safetensors"))
        actual_model_type = "ar"
        few_step = False
    else:  # bi
        action_ckpt = os.environ.get("BI_ACTION_MODEL_PATH",
                                     os.path.join(MODEL_BASE, "HY-WorldPlay", "bidirectional_model", "diffusion_pytorch_model.safetensors"))
        actual_model_type = "bi"
        few_step = False
    
    # Verify paths exist
    if not os.path.exists(model_path):
        return None, f"Model path not found: {model_path}\nPlease run download_models.py first."
    
    if not os.path.exists(action_ckpt):
        return None, f"Action model not found: {action_ckpt}\nPlease run download_models.py first."
    
    try:
        # Import required modules
        from hyvideo.generate import pose_to_input, save_video, pose_string_to_json
        from hyvideo.commons.infer_state import initialize_infer_state
        import json
        
        # Calculate video_length from pose data
        if pose_string.endswith(".json"):
            pose_json = json.load(open(pose_string, "r"))
        else:
            pose_json = pose_string_to_json(pose_string)
        
        pose_keys = list(pose_json.keys())
        latent_num_from_pose = len(pose_keys)
        required_video_length = latent_num_from_pose * 4 - 3
        
        if video_length != required_video_length:
            logger.warning(f"Pose data requires {required_video_length} frames, adjusting from {video_length}")
            video_length = required_video_length
        
        # Create args-like object for infer_state
        class Args:
            pass
        
        args = Args()
        args.use_sageattn = False
        args.sage_blocks_range = "0-53"
        args.use_vae_parallel = False
        args.use_fp8_gemm = False
        args.quant_type = "fp8-per-block"
        args.include_patterns = "double_blocks"
        args.enable_torch_compile = False  # Disable torch.compile for Windows compatibility
        
        initialize_infer_state(args)
        
        # Get pipeline
        pipe = get_pipeline(model_path, actual_model_type, action_ckpt, offloading, dtype)
        
        # Convert pose string to inputs
        latent_num = (video_length - 1) // 4 + 1
        viewmats, Ks, action = pose_to_input(pose_string, latent_num)
        
        # Generate video
        logger.info(f"Generating video with {video_length} frames...")
        
        chunk_latent_frames = 4 if actual_model_type == "ar" else 16
        
        out = pipe(
            enable_sr=False,
            prompt=prompt,
            aspect_ratio="16:9",
            num_inference_steps=num_inference_steps if not few_step else 4,
            sr_num_inference_steps=None,
            video_length=video_length,
            negative_prompt="",
            seed=seed,
            output_type="pt",
            prompt_rewrite=False,
            return_pre_sr_video=False,
            viewmats=viewmats.unsqueeze(0),
            Ks=Ks.unsqueeze(0),
            action=action.unsqueeze(0),
            few_step=few_step,
            chunk_latent_frames=chunk_latent_frames,
            model_type=actual_model_type,
            user_height=480,
            user_width=832,
            reference_image=image_path,
        )
        
        # Save video
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        safe_prompt = sanitize_filename(prompt[:50])
        output_dir = os.path.join(os.path.dirname(__file__), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f"{timestamp}_{safe_prompt}.mp4")
        save_video(out.videos, output_path)
        
        logger.info(f"Video saved to: {output_path}")
        
        return output_path, f"Video generated successfully!\nSaved to: {output_path}"
        
    except Exception as e:
        import traceback
        error_msg = f"Error during generation:\n{str(e)}\n\n{traceback.format_exc()}"
        logger.error(error_msg)
        return None, error_msg


def create_ui():
    """Create the Gradio UI."""
    
    with gr.Blocks(title="HY-World 1.5 (WorldPlay)") as demo:
        gr.Markdown("""
        # HY-World 1.5 (WorldPlay)
        
        **Interactive World Model with Real-Time Latency and Geometric Consistency**
        
        Upload an image, enter a prompt, and control the camera trajectory using WASD-style pose strings.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.Markdown("### Input")
                
                input_image = gr.Image(
                    label="Input Image",
                    type="filepath",
                    sources=["upload", "clipboard"],
                )
                
                analyze_btn = gr.Button(
                    "Analyze Image (Auto-generate Prompt)",
                    variant="secondary",
                    size="sm",
                )
                
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the scene...",
                    lines=3,
                    value="A paved pathway leads towards a stone arch bridge spanning a calm body of water. Lush green trees and foliage line the path.",
                )
                
                gr.Markdown("### Camera Control")
                
                pose_string = gr.Textbox(
                    label="Pose String",
                    placeholder="w-31 (forward 31 latents)",
                    value="w-31",
                    info="Format: action-duration. Actions: w (forward), s (backward), a (left), d (right), up, down, left, right",
                )
                
                with gr.Accordion("Advanced Settings", open=False):
                    model_type = gr.Radio(
                        label="Model Type",
                        choices=[
                            ("AR Distilled (Fast, 4 steps)", "ar_distilled"),
                            ("AR (50 steps)", "ar"),
                            ("Bidirectional (50 steps)", "bi"),
                        ],
                        value="ar_distilled",
                    )
                    
                    num_inference_steps = gr.Slider(
                        label="Inference Steps (for non-distilled models)",
                        minimum=10,
                        maximum=100,
                        value=50,
                        step=5,
                    )
                    
                    video_length = gr.Slider(
                        label="Video Length (frames)",
                        minimum=17,
                        maximum=125,
                        value=125,
                        step=4,
                        info="Must satisfy (length-1) % 4 == 0",
                    )
                    
                    seed = gr.Number(
                        label="Seed",
                        value=123,
                        precision=0,
                    )
                    
                    offloading = gr.Checkbox(
                        label="Enable CPU Offloading",
                        value=True,
                        info="Reduces VRAM usage but slower",
                    )
                    
                    dtype = gr.Radio(
                        label="Data Type",
                        choices=[("BFloat16 (Faster)", "bf16"), ("Float32 (Higher quality)", "fp32")],
                        value="bf16",
                    )
                
                generate_btn = gr.Button("Generate Video", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                # Output section
                gr.Markdown("### Output")
                
                output_video = gr.Video(
                    label="Generated Video",
                    autoplay=True,
                )
                
                status_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=5,
                )
                
                # Environment info
                with gr.Accordion("Environment Info", open=False):
                    env_info = gr.Textbox(
                        label="System Information",
                        value=check_environment(),
                        interactive=False,
                        lines=5,
                    )
        
        # Pose string examples
        gr.Markdown("""
        ### Pose String Examples
        
        | Example | Description |
        |---------|-------------|
        | `w-31` | Move forward for 31 latents (125 frames) |
        | `w-15,d-16` | Forward 15, then right 16 |
        | `w-10,right-5,d-16` | Forward, turn right, then move right |
        | `a-15,w-16` | Left 15, then forward 16 |
        """)
        
        # Event handlers
        analyze_btn.click(
            fn=analyze_image_for_prompt,
            inputs=[input_image],
            outputs=[prompt],
        )
        
        generate_btn.click(
            fn=generate_video,
            inputs=[
                input_image,
                prompt,
                pose_string,
                model_type,
                num_inference_steps,
                video_length,
                seed,
                offloading,
                dtype,
            ],
            outputs=[output_video, status_text],
            concurrency_limit=1,  # Only one generation at a time
        )
    
    # Enable queue for long-running tasks (no timeout)
    demo.queue(
        default_concurrency_limit=1,
    )
    
    return demo


def main():
    parser = argparse.ArgumentParser(description="HY-World 1.5 (WorldPlay) - Gradio Web UI")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    parser.add_argument("--share", action="store_true", help="Create a public link")
    parser.add_argument("--model-path", type=str, default=None, help="Path to HunyuanVideo-1.5 model")
    parser.add_argument("--action-ckpt-dir", type=str, default=None, help="Directory containing action checkpoints")
    
    args = parser.parse_args()
    
    # Set model paths from arguments
    if args.model_path:
        os.environ["MODEL_PATH"] = args.model_path
    
    if args.action_ckpt_dir:
        os.environ["AR_DISTILL_ACTION_MODEL_PATH"] = os.path.join(
            args.action_ckpt_dir, "ar_distilled_action_model", "diffusion_pytorch_model.safetensors"
        )
        os.environ["AR_ACTION_MODEL_PATH"] = os.path.join(
            args.action_ckpt_dir, "ar_model", "diffusion_pytorch_model.safetensors"
        )
        os.environ["BI_ACTION_MODEL_PATH"] = os.path.join(
            args.action_ckpt_dir, "bidirectional_model", "diffusion_pytorch_model.safetensors"
        )
    
    # Create and launch UI
    demo = create_ui()
    
    logger.info(f"Starting HY-World 1.5 (WorldPlay) Web UI on {args.host}:{args.port}")
    
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=False,  # We open browser manually with delay
    )


if __name__ == "__main__":
    main()
