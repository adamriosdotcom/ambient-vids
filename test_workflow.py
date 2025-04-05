import os
import cv2
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import replicate
import ffmpeg
from datetime import datetime
import time
import torch
import shutil

# Fix for torch.classes.__path__ error
os.environ['PYTORCH_JIT'] = '0'

# Load environment variables
load_dotenv()
REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN')

# Configure minimal test parameters
scene = "a cozy cabin in a snowy forest"
subjects = "a crackling fireplace with soft embers"
clip_duration = 5  # 5 seconds per clip
number_of_clips = 2  # Only 2 clips for minimal test
fade_duration = 1.0  # Crossfade duration
num_loops = 2  # Number of times to loop the final video

# Initialize run ID
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"🚀 Starting test workflow with run_id: {run_id}")

# Step 1: Generate optimized prompt
print("\n1️⃣ Generating optimized prompt...")
user_image_prompt = f"""
Give me a highly detailed prompt to provide to an image generator based on the scene and subjects below. The image should be highly detailed and textured, like a heavily stylized and realistic illustration. The scene is magical, colorful, awe-inspiring. emphasize architecture, subject placement, and details that resonate deeply. Be imaginative and descriptive. IMPORTANT: try your best to incorporate elements that have subtle movement because the image is ultimately going to be used to create a looping video which will serve as background ambience, so if there is water, we will want that flowing, if there is tall grass, we want that blowing in the breeze, if there is smoke, we want to see it, if there are animals, we want them grazing or walking, etc.

Ensure logical consistency - walking paths and streams should lead somewhere and not stop randomly, gates should but connected to a wall or fence and not standing by themselves, etc. Add thoughtful details that give a rich backstory to the image.

The beauty should be fairytale-like. Perfect lighting, one in a million compositions, surreal colors. This image should be the ideal and perfect example of the scene.

scene: {scene}
subjects: {subjects}
"""

try:
    output = replicate.run(
        "anthropic/claude-3.7-sonnet", 
        input={
            "prompt": user_image_prompt,
            "temperature": 0.7,
            "max_tokens": 1024  # Fixed: minimum 1024 tokens
        }
    )
    optimized_image_prompt = "".join(output)
    print("✅ Optimized prompt generated!")
    print("\nOptimized prompt:")
    print(optimized_image_prompt)
except Exception as e:
    print(f"❌ Error generating optimized prompt: {e}")
    exit(1)  # Exit if we can't generate a prompt

# Step 2: Generate initial image
print("\n2️⃣ Generating initial image...")
try:
    image_output = replicate.run(
        "google/imagen-3",
        input={
            "prompt": optimized_image_prompt,
            "aspect_ratio": "16:9",
            "negative_prompt": "fast movement",
            "safety_filter_level": "block_medium_and_above"
        }
    )
    initial_image_path = f"{run_id}_image_output.png"
    with open(initial_image_path, "wb") as img_file:
        img_file.write(image_output.read())
    print(f"✅ Initial image saved to: {initial_image_path}")
except Exception as e:
    print(f"❌ Error generating image: {e}")
    exit(1)  # Exit if we can't generate an image

# Step 3: Sharpen/upscale the image
print("\n3️⃣ Upscaling the image...")
def sharpen_frame(frame):
    """Sharpens/upscales a frame using the Real-ESRGAN model."""
    temp_input = "temp_frame.png"
    temp_output = "temp_frame_sharpened.png"
    cv2.imwrite(temp_input, frame)
    input_data = {
        "image": open(temp_input, "rb")
    }
    try:
        output = replicate.run("recraft-ai/recraft-crisp-upscale", input=input_data)
        with open(temp_output, "wb") as f:
            f.write(output.read())
        sharpened = cv2.imread(temp_output)
        if os.path.exists(temp_input):
            os.remove(temp_input)
        if os.path.exists(temp_output):
            os.remove(temp_output)
        torch.cuda.empty_cache()
        return sharpened
    except Exception as e:
        print(f"❌ Error upscaling image: {e}")
        return frame  # Return original frame if upscaling fails

upscaled_image_path = f"{run_id}_upscaled_image.png"
try:
    init_img = cv2.imread(initial_image_path)
    if init_img is not None:
        upscaled_img = sharpen_frame(init_img)
        cv2.imwrite(upscaled_image_path, upscaled_img)
        print(f"✅ Upscaled image saved to: {upscaled_image_path}")
        current_start_image_path = upscaled_image_path
    else:
        print("❌ Error reading initial image for upscaling")
        exit(1)  # Exit if we can't read the image for upscaling
except Exception as e:
    print(f"❌ Error during upscaling: {e}")
    exit(1)  # Exit if upscaling fails

# Utility functions for video processing
def extract_frames(video_path):
    """Extracts all frames from a video file and returns (frames, fps)."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps

def write_video(frames, fps, output_path):
    """Writes a list of frames to a video file."""
    if not frames:
        return
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

def register_clipB(clipB_frames, ref_frame):
    """Aligns clipB's frames to the reference frame using ORB feature matching."""
    orb = cv2.ORB_create(500)
    kp1, des1 = orb.detectAndCompute(clipB_frames[0], None)
    kp2, des2 = orb.detectAndCompute(ref_frame, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) < 4:
        print("⚠️ Not enough matches; skipping registration for this clip.")
        return clipB_frames
    matches = sorted(matches, key=lambda m: m.distance)[:50]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    if H is None:
        print("⚠️ Homography computation failed; skipping registration.")
        return clipB_frames
    
    h, w = ref_frame.shape[:2]
    registered = []
    for frame in clipB_frames:
        warped = cv2.warpPerspective(frame, H, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        registered.append(warped)
    return registered

def chain_crossfade(clip_paths, output_path, fade_duration):
    """Chains multiple video clips with crossfade transitions."""
    if len(clip_paths) < 2:
        print("⚠️ Need at least 2 clips to crossfade.")
        if len(clip_paths) == 1:
            shutil.copy(clip_paths[0], output_path)
            return
        return

    inputs = []
    durations = []
    target_width, target_height = 1920, 1080

    for path in clip_paths:
        input_stream = ffmpeg.input(path)
        scaled_stream = ffmpeg.filter(input_stream['v'], 'scale', target_width, target_height)
        inputs.append(scaled_stream)

        info = ffmpeg.probe(path)
        durations.append(float(info['format']['duration']))

    out_stream = inputs[0]
    current_duration = durations[0]

    for i in range(1, len(inputs)):
        out_stream = ffmpeg.filter(
            [out_stream, inputs[i]],
            'xfade',
            transition='fade',
            duration=fade_duration,
            offset=current_duration - fade_duration
        )
        current_duration = current_duration + durations[i] - fade_duration

    (
        ffmpeg
        .output(out_stream, output_path, vcodec='libx264', acodec='aac', pix_fmt='yuv420p')
        .overwrite_output()
        .run()
    )
    print(f"✅ Created crossfaded video: {output_path}")

# Step 4: Generate video clips
print("\n4️⃣ Generating video clips...")
clip_paths = []
user_video_prompt = "camera-tilt:0, camera-zoom:0, camera-pan:0, camera-rotate:0, camera-fixed-position:True"

for clip_number in range(1, number_of_clips + 1):
    time.sleep(1)
    video_path = f"{run_id}_{clip_number}_video_output.mp4"
    print(f"🎬 Creating clip {clip_number}/{number_of_clips}...")

    # For the last clip, force looping by using the initial image as end_image
    if clip_number == number_of_clips:
        input_dict = {
            "prompt": user_video_prompt,
            "duration": clip_duration,
            "cfg_scale": 0,
            "start_image": open(current_start_image_path, "rb"),
            "aspect_ratio": "16:9",
            "negative_prompt": "",
            "end_image": open(upscaled_image_path, "rb")
        }
    else:
        input_dict = {
            "prompt": user_video_prompt,
            "duration": clip_duration,
            "cfg_scale": 0,
            "start_image": open(current_start_image_path, "rb"),
            "aspect_ratio": "16:9",
            "negative_prompt": ""
        }

    try:
        # Request the video clip from the replicate model
        video_output = replicate.run("kwaivgi/kling-v1.6-pro", input=input_dict)
        with open(video_path, "wb") as vid_file:
            vid_file.write(video_output.read())
        print(f"✅ Clip {clip_number} created: {video_path}")

        # If this is not the first clip, apply registration to align it
        if clip_number > 1:
            frames, fps = extract_frames(video_path)
            ref_img = cv2.imread(current_start_image_path)
            registered_frames = register_clipB(frames, ref_img)
            write_video(registered_frames, fps, video_path)
            print(f"✅ Clip {clip_number} registered to previous clip's last frame.")

        clip_paths.append(video_path)

        # Extract the last frame from this clip and sharpen it for the next clip's start image
        if clip_number < number_of_clips:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    sharpened_last = sharpen_frame(frame)
                    last_frame_path = f"{run_id}_last_frame_{clip_number}.png"
                    cv2.imwrite(last_frame_path, sharpened_last)
                    current_start_image_path = last_frame_path
                    print(f"✅ Sharpened last frame of clip {clip_number}, used for next clip.")
                else:
                    print("⚠️ Could not read the last frame.")
                    exit(1)
            else:
                print("⚠️ No frames found in clip.")
                exit(1)
    except Exception as e:
        print(f"❌ Error generating video clip {clip_number}: {e}")
        exit(1)  # Exit if video generation fails

# Step 5: Combine all clips with crossfade transitions
print("\n5️⃣ Combining clips with crossfade transitions...")
if len(clip_paths) == 1:
    full_video_path = f"{run_id}_video_output.mp4"
    shutil.copy(clip_paths[0], full_video_path)
    print(f"✅ Only 1 clip. Copied {clip_paths[0]} to {full_video_path}")
else:
    full_video_path = f"{run_id}_video_output.mp4"
    chain_crossfade(clip_paths, full_video_path, fade_duration)
    print(f"✅ Full video created: {full_video_path}")

# Step 6: Loop the video
print("\n6️⃣ Creating looped video...")
loop_paths = []
for i in range(num_loops):
    copy_name = f"loop_copy_{i}.mp4"
    shutil.copy(full_video_path, copy_name)
    loop_paths.append(copy_name)

final_looped_path = f"{run_id}_looped_final.mp4"
try:
    chain_crossfade(loop_paths, final_looped_path, fade_duration)
    print(f"✅ Final looped video created: {final_looped_path}")
except Exception as e:
    print(f"❌ Error creating looped video: {e}")
    exit(1)

# Clean up temporary loop files
for p in loop_paths:
    if os.path.exists(p):
        os.remove(p)

# Final summary
print("\n🎉 Workflow complete!")
print(f"Initial image: {initial_image_path}")
print(f"Upscaled image: {upscaled_image_path}")
print(f"Final video: {final_looped_path}")
print(f"Duration: {clip_duration * number_of_clips * num_loops} seconds") 