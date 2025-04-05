import streamlit as st
import os
# Fix for torch.classes.__path__ error
os.environ['PYTORCH_JIT'] = '0'  # Disable JIT to avoid the path error
import cv2
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import replicate
import ffmpeg
from datetime import datetime
import time
import torch
import tempfile
from PIL import Image
import uuid
import base64
import io
import glob
import requests
import shutil

# Load environment variables
load_dotenv()
REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN')

# Configuration parameters
CLIP_DURATION = 10  # 10 seconds per clip (changed from 5)
NUM_LOOPS = 2      # Loop final video twice

# Constants for audio resources
AUDIO_DIR = "audio_resources"
CLASSICAL_DIR = "100ClassicalMusicMasterpieces"

# Initialize available classical music once at app startup
def find_matching_files():
    available_classical = {}
    
    # Check if classical directory exists
    classical_dir = os.path.abspath(CLASSICAL_DIR)
    if not os.path.exists(classical_dir):
        print(f"Classical music directory {classical_dir} not found")
        return available_classical
    
    # Define categories for classical music based on mood/style
    CLASSICAL_CATEGORIES = {
        "Calm": [
            "1825 Schubert - Ave Maria.mp3",
            "1875 Faure - Pavane.mp3",
            "1890 Debussy - Clair de Lune.mp3",
            "1888 Satie - Gymnopédie No.1.mp3",
            "1877 Saint-Saens - The Swan.mp3"
        ],
        "Melancholic": [
            "1827 Beethoven - Moonlight Sonata.mp3",
            "1838 Chopin - Nocturne Op. 9 No. 2.mp3",
            "1903 Sibelius - Valse Triste.mp3",
            "1822 Schubert - Symphony No.8 in B minor, 'Unfinished'.mp3"
        ],
        "Uplifting": [
            "1723 Vivaldi - The Four Seasons - Spring.mp3",
            "1785 Mozart - Eine Kleine Nachtmusik.mp3",
            "1823 Beethoven - Symphony No. 9, 'Choral' - Ode to Joy.mp3",
            "1778 Rondo Alla Turca, from Piano Sonata in A.mp3"
        ],
        "Dramatic": [
            "1870 Wagner- Ride of the Valkyries; from 'The Valkyrie'.mp3",
            "1871 Grieg - In the Hall of the Mountain King.mp3",
            "1798 Beethoven - Symphony No.5 in C minor - 1st movement.mp3"
        ]
    }
    
    # Go through each category and find files that match or contain the titles
    for category, titles in CLASSICAL_CATEGORIES.items():
        available_classical[category] = {}
        
        for title in titles:
            # Get year and composer from the filename
            parts = title.split(' ', 1)
            if len(parts) > 1:
                year = parts[0]
                composer_title = parts[1]
                
                # Look for files matching this pattern
                matches = []
                for file in os.listdir(classical_dir):
                    if file.endswith('.mp3') and (
                        file == title or 
                        composer_title in file or 
                        any(composer.lower() in file.lower() for composer in composer_title.lower().split(' - ', 1))
                    ):
                        matches.append(file)
                
                if matches:
                    # Use the best match (exact match preferred)
                    best_match = matches[0]
                    if title in matches:
                        best_match = title
                    
                    # Extract just the composer and title for display
                    display_name = composer_title
                    file_path = os.path.join(classical_dir, best_match)
                    if os.path.exists(file_path):
                        available_classical[category][display_name] = file_path
    
    # If any category is empty, fill with some default files
    for category in CLASSICAL_CATEGORIES.keys():
        if not available_classical.get(category, {}):
            available_classical[category] = {}
            # Just take some mp3 files we can find
            mp3_files = [os.path.join(classical_dir, f) for f in os.listdir(classical_dir) 
                        if f.endswith('.mp3') and os.path.isfile(os.path.join(classical_dir, f))]
            
            for i, file in enumerate(mp3_files[:3]):
                if os.path.exists(file):
                    filename = os.path.basename(file)
                    # Try to extract composer/title
                    if " - " in filename:
                        display_name = filename.split(" ", 1)[1]
                    else:
                        display_name = filename
                    available_classical[category][display_name] = file
    
    return available_classical

# Function to add audio to video
def add_audio_to_video(video_path, audio_path, output_path, loop_audio=True):
    """Adds audio to a video file, optionally looping the audio to match video length"""
    try:
        # Convert paths to absolute paths
        video_path = os.path.abspath(video_path)
        audio_path = os.path.abspath(audio_path)
        output_path = os.path.abspath(output_path)
        
        # Get video duration
        video_info = ffmpeg.probe(video_path)
        video_duration = float(video_info['format']['duration'])
        
        # Get audio duration
        audio_info = ffmpeg.probe(audio_path)
        audio_duration = float(audio_info['format']['duration'])
        
        if loop_audio and audio_duration < video_duration:
            # Create a direct ffmpeg command that uses stream_loop option
            cmd = f'ffmpeg -i "{video_path}" -stream_loop -1 -i "{audio_path}" -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 -shortest "{output_path}" -y'
            result = os.system(cmd)
        else:
            # Add audio directly (without looping)
            cmd = f'ffmpeg -i "{video_path}" -i "{audio_path}" -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 -shortest "{output_path}" -y'
            result = os.system(cmd)
        
        # Check if the output file was created successfully
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
        else:
            st.error(f"Error: Output file was not created properly")
            return None
            
    except Exception as e:
        st.error(f"Error adding audio to video: {e}")
        return None

# Initialize the available classical music files
CLASSICAL_MUSIC = find_matching_files()

# Initialize session state if needed
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'run_id' not in st.session_state:
    st.session_state.run_id = str(uuid.uuid4())[:8]
if 'scene_result' not in st.session_state:
    st.session_state.scene_result = None
if 'all_prompts' not in st.session_state:
    st.session_state.all_prompts = []
if 'optimized_prompts' not in st.session_state:
    st.session_state.optimized_prompts = []
if 'prompt' not in st.session_state:
    st.session_state.prompt = ""
if 'selected_prompt' not in st.session_state:
    st.session_state.selected_prompt = ""
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = []
if 'selected_image' not in st.session_state:
    st.session_state.selected_image = None
if 'selected_image_path' not in st.session_state:
    st.session_state.selected_image_path = ""
if 'upscaled_image_path' not in st.session_state:
    st.session_state.upscaled_image_path = ""
if 'generated_videos' not in st.session_state:
    st.session_state.generated_videos = []
if 'selected_video' not in st.session_state:
    st.session_state.selected_video = None
if 'clip_paths' not in st.session_state:
    st.session_state.clip_paths = []
if 'current_start_image_path' not in st.session_state:
    st.session_state.current_start_image_path = ""
if 'loop_count' not in st.session_state:
    st.session_state.loop_count = 1
if 'final_video_path' not in st.session_state:
    st.session_state.final_video_path = ""
if 'target_duration' not in st.session_state:
    st.session_state.target_duration = 30
if 'audio_added' not in st.session_state:
    st.session_state.audio_added = False
if 'selected_audio_path' not in st.session_state:
    st.session_state.selected_audio_path = None

# Utility functions
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
        print("Not enough matches; skipping registration for this clip.")
        return clipB_frames
    matches = sorted(matches, key=lambda m: m.distance)[:50]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    if H is None:
        print("Homography computation failed; skipping registration.")
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
        raise ValueError("Need at least 2 clips to crossfade.")

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

def loop_video(input_path, output_path, num_loops=2):
    """Loop a video a specified number of times."""
    # Create temporary copies for looping
    loop_paths = []
    for i in range(num_loops):
        copy_name = f"temp_loop_copy_{i}.mp4"
        os.system(f"cp {input_path} {copy_name}")
        loop_paths.append(copy_name)
    
    # Crossfade loop copies together
    chain_crossfade(loop_paths, output_path, 1.0)
    
    # Clean up temp files
    for p in loop_paths:
        if os.path.exists(p):
            os.remove(p)
    
    return output_path

def sharpen_frame(frame):
    """Sharpens/upscales a frame using the Real-ESRGAN model."""
    temp_input = "temp_frame.png"
    temp_output = "temp_frame_sharpened.png"
    cv2.imwrite(temp_input, frame)
    input_data = {
        "image": open(temp_input, "rb")
    }
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

def get_video_thumbnail(video_path):
    """Extract the middle frame from a video for thumbnail display."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None

def image_to_base64(image):
    """Convert a PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def get_video_html(video_path):
    """Generate HTML for video display."""
    video_file = open(video_path, 'rb')
    video_bytes = video_file.read()
    b64 = base64.b64encode(video_bytes).decode()
    return f"""
    <video width="100%" controls>
        <source src="data:video/mp4;base64,{b64}" type="video/mp4">
    </video>
    """

def check_files(files, details=False):
    """Check if files exist and print their details."""
    results = []
    for file in files:
        exists = os.path.exists(file)
        if exists:
            size = os.path.getsize(file)
            if details:
                # Try to get more file info
                try:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img = Image.open(file)
                        results.append((file, exists, size, f"{img.width}x{img.height} pixels, {img.mode} mode"))
                    elif file.lower().endswith(('.mp4', '.mov')):
                        cap = cv2.VideoCapture(file)
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        cap.release()
                        results.append((file, exists, size, f"{width}x{height} pixels, {frames} frames, {fps} fps"))
                    else:
                        results.append((file, exists, size, ""))
                except Exception as e:
                    results.append((file, exists, size, f"Error: {e}"))
            else:
                results.append((file, exists, size, ""))
        else:
            results.append((file, exists, 0, ""))
    return results

def generate_optimized_prompt(prompt):
    """Generate a single optimized prompt from the combined prompt."""
    # Base prompt template
    user_image_prompt = f"""
    Give me a highly detailed prompt to provide to an image generator based on the scene description below. The image should be highly detailed and textured, like a heavily stylized and realistic illustration. The scene is magical, colorful, awe-inspiring. emphasize architecture, subject placement, and details that resonate deeply. Be imaginative and descriptive. IMPORTANT: try your best to incorporate elements that have subtle movement because the image is ultimately going to be used to create a looping video which will serve as background ambience, so if there is water, we will want that flowing, if there is tall grass, we want that blowing in the breeze, if there is smoke, we want to see it, if there are animals, we want them grazing or walking, etc.

    Ensure logical consistency - walking paths and streams should lead somewhere and not stop randomly, gates should but connected to a wall or fence and not standing by themselves, etc. Add thoughtful details that give a rich backstory to the image.

    The beauty should be fairytale-like. Perfect lighting, one in a million compositions, surreal colors. This image should be the ideal and perfect example of the scene.
    
    avoid chaotic, busy scenes and prefer beautiful, more minimal, well-balanced scenes.

    description: {prompt}
    """
    
    # Generate a single optimized prompt
    try:
        output = replicate.run(
            "anthropic/claude-3.7-sonnet", 
            input={
                "prompt": user_image_prompt,
                "temperature": 0.7,
                "max_tokens": 1024
            }
        )
        optimized = "".join(output)
        
        # Return the optimized prompt if successful
        if optimized:
            return optimized
        else:
            st.error("Failed to generate optimized prompt")
            return None
    except Exception as e:
        st.error(f"Error generating optimized prompt: {e}")
        return None

# Main UI layout
st.title("Ambience Video Creator")

# Step 1: Initial prompt input
if st.session_state.step == 1:
    st.header("Step 1: Enter Your Scene Description")
    
    prompt = st.text_area("Description (e.g., 'a cozy library at night with a cat curled up on a chair')", 
                         value=st.session_state.prompt or "a cozy cabin in a snowy forest with a crackling fireplace with soft embers",
                         height=100)
    
    if st.button("Generate Images"):
        if prompt:
            with st.spinner("Generating optimized prompt..."):
                st.session_state.prompt = prompt
                
                # Generate single optimized prompt
                optimized_prompt = generate_optimized_prompt(prompt)
                
                if optimized_prompt:
                    # Store the optimized prompt
                    st.session_state.selected_prompt = optimized_prompt
                    
                    # Display the generated prompt
                    st.success("Optimized prompt generated!")
                    st.text_area("Generated Prompt", optimized_prompt, height=200)
                    
                    # Generate images from this prompt
                    with st.spinner("Generating images from the optimized prompt..."):
                        try:
                            image_paths = []
                            
                            # Create placeholders for image generation progress
                            st.write("Generating images...")
                            progress_bar = st.progress(0)
                            
                            # Only generate 2 images now
                            for j in range(2):
                                progress_bar.progress((j) / 2)
                                st.write(f"Generating image {j+1}/2...")
                                
                                image_output = replicate.run(
                                    "google/imagen-3",
                                    input={
                                        "prompt": optimized_prompt,
                                        "aspect_ratio": "16:9",
                                        "negative_prompt": "fast movement",
                                        "safety_filter_level": "block_medium_and_above"
                                    }
                                )
                                
                                # Save the image
                                image_path = f"{st.session_state.run_id}_image_{j}.png"
                                with open(image_path, "wb") as img_file:
                                    img_file.write(image_output.read())
                                
                                # Verify the image was saved correctly
                                if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
                                    st.success(f"Image {j+1} generated successfully!")
                                    # Display a thumbnail of the image
                                    try:
                                        img = Image.open(image_path)
                                        st.image(img, caption=f"Image {j+1}", width=300)
                                    except Exception as e:
                                        st.error(f"Error displaying thumbnail: {e}")
                                else:
                                    st.error(f"Failed to save image {j+1}")
                                
                                image_paths.append(image_path)
                            
                            progress_bar.progress(1.0)
                            
                            # Update session state with generated images
                            st.session_state.generated_images = image_paths
                            
                            # Count valid images
                            valid_images = [p for p in image_paths if os.path.exists(p) and os.path.getsize(p) > 0]
                            if len(valid_images) > 0:
                                st.success(f"Generated {len(valid_images)} images successfully!")
                                # Skip step 2 and go directly to step 3 (image selection)
                                st.session_state.step = 3
                                st.button("Proceed to Image Selection", on_click=lambda: st.experimental_rerun())
                            else:
                                st.error("No valid images were generated. Please try again.")
                        except Exception as e:
                            st.error(f"Error generating images: {e}")
                            st.error(f"Exception details: {str(e)}")
                else:
                    st.error("Failed to generate prompt. Please try again.")

# Step 3: Image selection
elif st.session_state.step == 3:
    st.header("Step 3: Select an Image")
    
    # If we have the optimized prompt, display it
    if st.session_state.selected_prompt:
        with st.expander("Show Optimized Prompt"):
            st.text_area("Prompt Used", st.session_state.selected_prompt, height=150)
    
    # Add debug info
    st.write(f"Found {len(st.session_state.generated_images)} generated images")
    
    # Check if files exist first
    file_checks = check_files(st.session_state.generated_images, details=True)
    with st.expander("Show File Check Results"):
        st.write("File check results:")
        for file, exists, size, details in file_checks:
            if exists:
                st.success(f"✅ {os.path.basename(file)}: {size} bytes - {details}")
            else:
                st.error(f"❌ {os.path.basename(file)}: Not found")
    
    # Display valid images
    valid_images = [f for f, exists, _, _ in file_checks if exists]
    
    if not valid_images:
        st.error("No valid images found. Please try generating images again.")
        if st.button("Return to Step 1"):
            st.session_state.step = 1
            st.experimental_rerun()
    else:
        # Display the valid images side by side
        cols = st.columns(len(valid_images))
        for i, image_path in enumerate(valid_images):
            with cols[i]:
                try:
                    # Try to load with PIL first to verify the image is valid
                    try:
                        img = Image.open(image_path)
                        st.image(img, caption=f"Option {i+1}")
                    except Exception as e:
                        st.error(f"Error loading image with PIL: {e}")
                        # Fallback to direct file path
                        st.image(image_path, caption=f"Option {i+1}")
                    
                    if st.button(f"Select Image {i+1}"):
                        st.session_state.selected_image_path = image_path
                        
                        # Upscale the selected image
                        with st.spinner("Upscaling your selected image..."):
                            img = cv2.imread(image_path)
                            if img is None:
                                st.error(f"OpenCV could not read image {image_path}")
                                # Try with PIL and convert to CV2
                                pil_img = Image.open(image_path)
                                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                            
                            upscaled_img = sharpen_frame(img)
                            upscaled_path = f"{st.session_state.run_id}_upscaled_image.png"
                            cv2.imwrite(upscaled_path, upscaled_img)
                            st.session_state.upscaled_image_path = upscaled_path
                            st.session_state.current_start_image_path = upscaled_path
                            
                            # Show upscaled image
                            st.success(f"Image upscaled successfully!")
                            st.image(upscaled_path, caption="Upscaled Image")
                        
                        # Generate 2 video options with shorter duration (changed from 3 to 2)
                        with st.spinner("Generating initial video options..."):
                            st.session_state.generated_videos = []
                            user_video_prompt = "camera-tilt:0, camera-zoom:0, camera-pan:0, camera-rotate:0, camera-fixed-position:True"
                            
                            try:
                                # Generate 2 videos instead of 3
                                for j in range(2):
                                    video_path = f"{st.session_state.run_id}_video_{j}.mp4"
                                    
                                    input_dict = {
                                        "prompt": user_video_prompt,
                                        "duration": CLIP_DURATION,  # Use global variable
                                        "cfg_scale": 0,
                                        "start_image": open(upscaled_path, "rb"),
                                        "aspect_ratio": "16:9",
                                        "negative_prompt": ""
                                    }
                                    
                                    video_output = replicate.run("kwaivgi/kling-v1.6-pro", input=input_dict)
                                    with open(video_path, "wb") as vid_file:
                                        vid_file.write(video_output.read())
                                    
                                    st.session_state.generated_videos.append(video_path)
                                
                                st.success(f"Generated {len(st.session_state.generated_videos)} videos")
                                st.session_state.step = 4
                                st.experimental_rerun()
                            except Exception as e:
                                st.error(f"Error generating videos: {e}")
                except Exception as e:
                    st.error(f"Error processing image {image_path}: {e}")

# Step 4: Initial Video Selection
elif st.session_state.step == 4:
    st.header("Step 4: Select Initial Video")
    
    # If we have the selected prompt, display it
    if st.session_state.selected_prompt:
        with st.expander("Show Optimized Prompt"):
            st.text_area("Prompt Used", st.session_state.selected_prompt, height=150)
    
    # If we have the selected image, display it
    if st.session_state.selected_image_path and os.path.exists(st.session_state.selected_image_path):
        with st.expander("Show Selected Image"):
            try:
                img = Image.open(st.session_state.selected_image_path)
                st.image(img, caption="Selected Image", width=300)
            except Exception as e:
                st.error(f"Error displaying selected image: {e}")
    
    # Add debug info
    st.write(f"Found {len(st.session_state.generated_videos)} generated videos")
    
    # Check if files exist first
    file_checks = check_files(st.session_state.generated_videos, details=True)
    with st.expander("Show Video File Check Results"):
        st.write("Video file check results:")
        for file, exists, size, details in file_checks:
            if exists:
                st.success(f"✅ {os.path.basename(file)}: {size} bytes - {details}")
            else:
                st.error(f"❌ {os.path.basename(file)}: Not found")
    
    # Display valid videos
    valid_videos = [f for f, exists, _, _ in file_checks if exists]
    
    if not valid_videos:
        st.error("No valid videos found. Please try generating videos again.")
        if st.button("Return to Step 3"):
            st.session_state.step = 3
            st.experimental_rerun()
    else:
        st.write("Please select one of the videos below:")
        
        # Display the valid videos side by side
        cols = st.columns(len(valid_videos))
        for i, video_path in enumerate(valid_videos):
            with cols[i]:
                try:
                    # Display thumbnails first
                    try:
                        thumbnail = get_video_thumbnail(video_path)
                        if thumbnail is not None:
                            st.image(thumbnail, caption=f"Video {i+1} Thumbnail")
                    except Exception as e:
                        st.warning(f"Could not generate thumbnail: {e}")
                    
                    # Display video
                    st.video(video_path)
                    
                    if st.button(f"Select Video {i+1}", key=f"select_video_{i}"):
                        st.session_state.clip_paths.append(video_path)
                        st.success(f"Selected video {i+1}")
                        
                        # Extract the last frame, sharpen it for the next clip's start image
                        with st.spinner("Processing last frame..."):
                            cap = cv2.VideoCapture(video_path)
                            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
                            ret, frame = cap.read()
                            cap.release()
                            
                            if ret:
                                # Show the last frame
                                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Last Frame")
                                
                                # Sharpen the frame
                                st.info("Upscaling last frame...")
                                sharpened_last = sharpen_frame(frame)
                                last_frame_path = f"{st.session_state.run_id}_last_frame.png"
                                cv2.imwrite(last_frame_path, sharpened_last)
                                st.session_state.current_start_image_path = last_frame_path
                                
                                # Show the sharpened frame
                                st.image(cv2.cvtColor(sharpened_last, cv2.COLOR_BGR2RGB), caption="Sharpened Last Frame")
                                st.success("Last frame processed successfully!")
                            else:
                                st.error("Could not read the last frame.")
                        
                        # For minimal test, go directly to step 5
                        st.info("Moving to next step...")
                        st.session_state.step = 5
                        st.button("Continue", on_click=lambda: st.experimental_rerun())
                except Exception as e:
                    st.error(f"Error processing video {video_path}: {e}")

# Step 5: Continue or Complete
elif st.session_state.step == 5:
    st.header(f"Step 5: Continue or Complete (Current Loop: {st.session_state.loop_count})")
    
    # Show video path info in expandable section
    with st.expander("Current Video Information"):
        st.write(f"Videos in sequence so far: {len(st.session_state.clip_paths)}")
        for i, path in enumerate(st.session_state.clip_paths):
            st.write(f"Video {i+1}: {path}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Add Another Video Segment"):
            # Generate 2 new video options using the last frame
            with st.spinner("Generating more video options..."):
                st.session_state.generated_videos = []
                user_video_prompt = "camera-tilt:0, camera-zoom:0, camera-pan:0, camera-rotate:0, camera-fixed-position:True"
                
                try:
                    # Generate 2 videos
                    for j in range(2):
                        video_path = f"{st.session_state.run_id}_video_{st.session_state.loop_count}_{j}.mp4"
                        
                        input_dict = {
                            "prompt": user_video_prompt,
                            "duration": CLIP_DURATION,
                            "cfg_scale": 0,
                            "start_image": open(st.session_state.current_start_image_path, "rb"),
                            "aspect_ratio": "16:9",
                            "negative_prompt": ""
                        }
                        
                        video_output = replicate.run("kwaivgi/kling-v1.6-pro", input=input_dict)
                        with open(video_path, "wb") as vid_file:
                            vid_file.write(video_output.read())
                        
                        # Register frames
                        frames, fps = extract_frames(video_path)
                        ref_img = cv2.imread(st.session_state.current_start_image_path)
                        registered_frames = register_clipB(frames, ref_img)
                        write_video(registered_frames, fps, video_path)
                        
                        st.session_state.generated_videos.append(video_path)
                    
                    st.session_state.step = 6
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error generating videos: {e}")
    
    with col2:
        if st.button("Complete the Loop"):
            # Generate final video using the initial image as end point
            with st.spinner("Generating final loop closure video..."):
                final_clip_path = f"{st.session_state.run_id}_final_clip.mp4"
                
                input_dict = {
                    "prompt": "camera-tilt:0, camera-zoom:0, camera-pan:0, camera-rotate:0, camera-fixed-position:True",
                    "duration": CLIP_DURATION,
                    "cfg_scale": 0,
                    "start_image": open(st.session_state.current_start_image_path, "rb"),
                    "aspect_ratio": "16:9",
                    "negative_prompt": "",
                    "end_image": open(st.session_state.upscaled_image_path, "rb")
                }
                
                try:
                    video_output = replicate.run("kwaivgi/kling-v1.6-pro", input=input_dict)
                    with open(final_clip_path, "wb") as vid_file:
                        vid_file.write(video_output.read())
                    
                    # Register frames
                    frames, fps = extract_frames(final_clip_path)
                    ref_img = cv2.imread(st.session_state.current_start_image_path)
                    registered_frames = register_clipB(frames, ref_img)
                    write_video(registered_frames, fps, final_clip_path)
                    
                    st.session_state.clip_paths.append(final_clip_path)
                    
                    # Combine all clips with crossfades
                    output_path = f"{st.session_state.run_id}_complete_video.mp4"
                    chain_crossfade(st.session_state.clip_paths, output_path, 1.0)
                    st.session_state.final_video_path = output_path
                    
                    st.session_state.step = 7
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error generating final video: {e}")

# Step 6: Additional Video Selection - update to display only 2 videos
elif st.session_state.step == 6:
    st.header(f"Step 6: Select Video for Segment {st.session_state.loop_count + 1}")
    
    # Check if files exist first
    file_checks = check_files(st.session_state.generated_videos, details=True)
    with st.expander("Show Video File Check Results"):
        st.write("Video file check results:")
        for file, exists, size, details in file_checks:
            if exists:
                st.success(f"✅ {os.path.basename(file)}: {size} bytes - {details}")
            else:
                st.error(f"❌ {os.path.basename(file)}: Not found")
    
    # Display valid videos
    valid_videos = [f for f, exists, _, _ in file_checks if exists]
    
    if not valid_videos:
        st.error("No valid videos found. Please try generating videos again.")
        if st.button("Return to Step 5"):
            st.session_state.step = 5
            st.experimental_rerun()
    else:
        # Display the 2 generated videos side by side
        cols = st.columns(len(valid_videos))
        for i, video_path in enumerate(valid_videos):
            with cols[i]:
                try:
                    # Display thumbnails first
                    try:
                        thumbnail = get_video_thumbnail(video_path)
                        if thumbnail is not None:
                            st.image(thumbnail, caption=f"Video {i+1} Thumbnail")
                    except Exception as e:
                        st.warning(f"Could not generate thumbnail: {e}")
                    
                    # Display video
                    st.video(video_path)
                    
                    if st.button(f"Select Video {i+1}", key=f"select_video_segment_{i}"):
                        st.session_state.clip_paths.append(video_path)
                        st.success(f"Selected video {i+1}")
                        
                        # Extract the last frame, sharpen it for the next clip's start image
                        with st.spinner("Processing last frame..."):
                            cap = cv2.VideoCapture(video_path)
                            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
                            ret, frame = cap.read()
                            cap.release()
                            
                            if ret:
                                # Show the last frame
                                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Last Frame")
                                
                                # Sharpen the frame
                                st.info("Upscaling last frame...")
                                sharpened_last = sharpen_frame(frame)
                                last_frame_path = f"{st.session_state.run_id}_last_frame_{st.session_state.loop_count}.png"
                                cv2.imwrite(last_frame_path, sharpened_last)
                                st.session_state.current_start_image_path = last_frame_path
                                
                                # Show the sharpened frame
                                st.image(cv2.cvtColor(sharpened_last, cv2.COLOR_BGR2RGB), caption="Sharpened Last Frame")
                                st.success("Last frame processed successfully!")
                            else:
                                st.error("Could not read the last frame.")
                        
                        st.session_state.loop_count += 1
                        st.session_state.step = 5  # Go back to continue/complete step
                        st.button("Continue", on_click=lambda: st.experimental_rerun())
                except Exception as e:
                    st.error(f"Error processing video {video_path}: {e}")

# Step 7: Final Video Duration and Download
elif st.session_state.step == 7:
    st.header("Step 7: Final Video Settings")
    
    if os.path.exists(st.session_state.final_video_path):
        # Show original video
        st.subheader("Original Combined Video")
        st.video(st.session_state.final_video_path)
        
        # Calculate original duration
        try:
            info = ffmpeg.probe(st.session_state.final_video_path)
            orig_duration = float(info['format']['duration'])
            st.info(f"Original video duration: {orig_duration:.1f} seconds")
            
            # Allow user to customize final video duration
            st.subheader("Customize Final Video")
            
            # Option to specify target duration in minutes
            col1, col2 = st.columns(2)
            with col1:
                target_minutes = st.number_input("Target Duration (minutes)", 
                                            min_value=1, 
                                            max_value=60, 
                                            value=5)
            with col2:
                loop_method = st.selectbox("Looping Method", 
                                     options=["Auto-loop to reach duration", "Specify number of loops"])
            
            if loop_method == "Auto-loop to reach duration":
                # Convert minutes to seconds
                target_seconds = target_minutes * 60
                # Calculate number of loops needed
                num_loops_needed = max(1, int(np.ceil(target_seconds / orig_duration)))
                st.write(f"Will loop video {num_loops_needed} times to reach approximately {target_minutes} minutes")
                loops_to_use = num_loops_needed
            else:
                # Let user specify exact number of loops
                loops_to_use = st.number_input("Number of loops", min_value=1, max_value=10, value=2)
                estimated_duration = orig_duration * loops_to_use
                st.write(f"Estimated final duration: {estimated_duration/60:.1f} minutes")
            
            # Audio options section
            st.subheader("Add Audio to Your Video")
            
            # Check if classical music directory exists
            if os.path.exists(CLASSICAL_DIR) and sum(len(cat) for cat in CLASSICAL_MUSIC.values()) > 0:
                st.success(f"Found {sum(len(cat) for cat in CLASSICAL_MUSIC.values())} classical music tracks")
                audio_type = st.selectbox("Audio Type", ["None", "Classical Music", "Upload Your Own"])
            else:
                st.warning("Classical music directory not found. You can still upload your own audio.")
                audio_type = st.selectbox("Audio Type", ["None", "Upload Your Own"])
            
            audio_file_path = None
            
            if audio_type == "Classical Music":
                # Display music categories
                music_category = st.selectbox("Music Mood", list(CLASSICAL_MUSIC.keys()))
                
                # Display music options based on mood
                if CLASSICAL_MUSIC[music_category]:
                    music_options = CLASSICAL_MUSIC[music_category]
                    selected_music = st.selectbox("Choose Classical Track", list(music_options.keys()))
                    
                    # Display selected track for playback
                    music_file = music_options[selected_music]
                    if music_file and os.path.exists(music_file):
                        st.success(f"Selected: {selected_music}")
                        st.audio(music_file)
                        audio_file_path = music_file
                    else:
                        st.error(f"File not found: {music_file}")
                else:
                    st.warning(f"No tracks available for {music_category}")
                    
            elif audio_type == "Upload Your Own":
                custom_audio = st.file_uploader("Upload Audio File (MP3, WAV)", type=["mp3", "wav"])
                if custom_audio is not None:
                    # Save uploaded file to disk
                    temp_path = f"uploaded_{custom_audio.name}"
                    with open(temp_path, "wb") as f:
                        f.write(custom_audio.read())
                    st.success("Audio uploaded successfully!")
                    st.audio(temp_path)
                    audio_file_path = temp_path
            
            if audio_type != "None":
                # Show audio volume control
                audio_volume = st.slider("Audio Volume", 0.1, 1.0, 0.5, 0.1)
            
            # Create final video button
            if st.button("Create Final Video"):
                video_with_audio = None
                
                with st.spinner(f"Creating final looped video..."):
                    # Loop the video the specified number of times
                    looped_output_path = f"{st.session_state.run_id}_looped_final.mp4"
                    loop_video(st.session_state.final_video_path, looped_output_path, loops_to_use)
                    
                    # Check if audio should be added
                    if audio_type != "None" and audio_file_path and os.path.exists(audio_file_path):
                        with st.spinner("Adding audio to video..."):
                            st.session_state.selected_audio_path = audio_file_path
                            video_with_audio = f"{st.session_state.run_id}_final_with_audio.mp4"
                            
                            # Adjust audio volume if needed
                            if audio_volume != 1.0:
                                temp_audio = f"temp_adjusted_audio.mp3"
                                os.system(f'ffmpeg -i "{audio_file_path}" -filter:a "volume={audio_volume}" -y "{temp_audio}"')
                                audio_to_use = temp_audio
                            else:
                                audio_to_use = audio_file_path
                            
                            # Add audio to the looped video
                            result = add_audio_to_video(looped_output_path, audio_to_use, video_with_audio, loop_audio=True)
                            
                            # Clean up temp file
                            if audio_volume != 1.0 and os.path.exists(temp_audio):
                                os.remove(temp_audio)
                            
                            if result:
                                st.session_state.final_video_path = video_with_audio
                                st.session_state.audio_added = True
                            else:
                                st.error("Failed to add audio to video. Using video without audio.")
                                st.session_state.final_video_path = looped_output_path
                    else:
                        st.session_state.final_video_path = looped_output_path
                    
                    st.success(f"🎉 Your ambience video is ready! It's been looped {loops_to_use} times.")
                    
                    # Display final video
                    st.subheader("Final Video")
                    st.video(st.session_state.final_video_path)
                    
                    # Create download link
                    with open(st.session_state.final_video_path, "rb") as file:
                        btn = st.download_button(
                            label="Download Video",
                            data=file,
                            file_name=f"ambience_{st.session_state.run_id}{'.with_audio' if st.session_state.audio_added else ''}.mp4",
                            mime="video/mp4"
                        )
        except Exception as e:
            st.error(f"Error processing video: {e}")
            # Fallback to simple looping
            if st.button("Create Final Video (Simple Mode)"):
                with st.spinner(f"Creating looped video..."):
                    looped_output_path = f"{st.session_state.run_id}_looped_final.mp4"
                    loop_video(st.session_state.final_video_path, looped_output_path, NUM_LOOPS)
                    st.session_state.final_video_path = looped_output_path
                    
                    st.success(f"🎉 Your ambience video is ready!")
                    st.video(looped_output_path)
                    
                    with open(looped_output_path, "rb") as file:
                        btn = st.download_button(
                            label="Download Video",
                            data=file,
                            file_name=f"ambience_{st.session_state.run_id}.mp4",
                            mime="video/mp4"
                        )
    else:
        st.error(f"Final video file {st.session_state.final_video_path} not found")
    
    if st.button("Start Over"):
        # Reset all session state
        for key in list(st.session_state.keys()):
            if key != 'step':
                st.session_state[key] = None if key not in ['run_id', 'prompt', 'optimized_prompts', 
                                                           'generated_images', 'generated_videos', 
                                                           'clip_paths', 'loop_count', 'target_duration'] else (
                    str(uuid.uuid4())[:8] if key == 'run_id' else (
                        "" if key == 'prompt' else (
                            [] if key in ['optimized_prompts', 'generated_images', 'generated_videos', 'clip_paths'] else (
                                1 if key == 'loop_count' else 30))))
        st.session_state.step = 1
        st.experimental_rerun()

st.sidebar.markdown("## Ambience Video Creator")
st.sidebar.markdown(f"Current step: {st.session_state.step if st.session_state.step < 3 else st.session_state.step-1}/6")
progress = st.sidebar.progress((st.session_state.step if st.session_state.step < 3 else st.session_state.step-1) / 6)

# Display current workflow stage in the sidebar
workflow_stages = [
    "Enter Scene Description & Generate Images",
    "Select Generated Image",
    "Select Initial Video",
    "Continue or Complete Loop",
    "Select Additional Videos",
    "Final Video Settings & Audio"
]

st.sidebar.markdown("### Workflow Stages")
for i, stage in enumerate(workflow_stages):
    display_step = i + 1
    current_step = st.session_state.step if st.session_state.step < 3 else st.session_state.step - 1
    
    if display_step < current_step:
        st.sidebar.markdown(f"✅ {display_step}. {stage}")
    elif display_step == current_step:
        st.sidebar.markdown(f"🔄 {display_step}. {stage}")
    else:
        st.sidebar.markdown(f"⏳ {display_step}. {stage}")

# Display audio information if it's added
if 'audio_added' in st.session_state and st.session_state.audio_added:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎵 Audio Added")
    if 'selected_audio_path' in st.session_state and st.session_state.selected_audio_path:
        audio_name = os.path.basename(st.session_state.selected_audio_path)
        st.sidebar.write(f"Track: {audio_name}")

# Define step formatting constants
completed_step = "✅ "
current_step = "🔄 "

# Sidebar for workflow tracking
with st.sidebar:
    st.header("Workflow Steps")
    
    # Check if any steps are completed
    if st.session_state.step >= 1:
        st.markdown(completed_step if st.session_state.step > 1 else current_step + " Enter Scene Description & Generate Images")
    if st.session_state.step >= 3:
        st.markdown(completed_step if st.session_state.step > 3 else current_step + " Select an Image")
    if st.session_state.step >= 4:
        st.markdown(completed_step if st.session_state.step > 4 else current_step + " Select Initial Video")
    if st.session_state.step >= 5:
        st.markdown(completed_step if st.session_state.step > 5 else current_step + " Select Final Video")
    if st.session_state.step >= 6:
        st.markdown(completed_step if st.session_state.step > 6 else current_step + " Choose Video Length")
    if st.session_state.step >= 7:
        st.markdown(completed_step if st.session_state.step > 7 else current_step + " Final Video Settings & Audio")

# Reset session state function
def reset_session_state():
    for key in list(st.session_state.keys()):
        if key != 'step':
            st.session_state[key] = None if key not in ['run_id', 'prompt', 'optimized_prompts', 
                                                       'generated_images', 'generated_videos', 
                                                       'clip_paths', 'loop_count', 'target_duration'] else (
                str(uuid.uuid4())[:8] if key == 'run_id' else (
                    "" if key == 'prompt' else (
                        [] if key in ['optimized_prompts', 'generated_images', 'generated_videos', 'clip_paths'] else (
                            1 if key == 'loop_count' else 30))))
    st.session_state.step = 1
    st.experimental_rerun() 