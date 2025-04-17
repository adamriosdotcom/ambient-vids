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
import json
import re # Add re import

# Load environment variables
load_dotenv()
REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN')

# Configuration parameters
# CLIP_DURATION = 10  # 10 seconds per clip (changed from 5)
NUM_LOOPS = 2      # Loop final video twice

# Constants for audio resources
AUDIO_DIR = "audio_resources"
CLASSICAL_DIR = "100ClassicalMusicMasterpieces"

# Project folder management
def create_project_folder(run_id):
    """Create a project folder structure for a given run ID"""
    # Create main project directory
    project_dir = f"projects/{run_id}"
    os.makedirs(project_dir, exist_ok=True)
    
    # Create subdirectories for different assets
    os.makedirs(f"{project_dir}/images", exist_ok=True)
    os.makedirs(f"{project_dir}/videos", exist_ok=True)
    os.makedirs(f"{project_dir}/final", exist_ok=True)
    
    return project_dir

def get_project_path(run_id, asset_type, filename):
    """Get the full path for a project asset"""
    project_dir = f"projects/{run_id}"
    
    if asset_type == "image":
        return os.path.abspath(f"{project_dir}/images/{filename}") # Use absolute paths
    elif asset_type == "video":
        return os.path.abspath(f"{project_dir}/videos/{filename}") # Use absolute paths
    elif asset_type == "final":
        return os.path.abspath(f"{project_dir}/final/{filename}") # Use absolute paths
    elif asset_type == "state":
        return os.path.abspath(f"{project_dir}/session_state.json") # State file path
    else:
        return os.path.abspath(f"{project_dir}/{filename}") # Use absolute paths

# Function to generate a project name from the prompt
def generate_project_name(prompt):
    """Generates a safe directory name from a prompt."""
    if not prompt:
        prompt = "untitled"
    
    # Keep alphanumeric and spaces, convert to lowercase
    sanitized = re.sub(r'[^a-z0-9\s]+', '', prompt.lower())
    
    # Replace whitespace with underscores
    sanitized = re.sub(r'\s+', '_', sanitized).strip('_')
    
    # Truncate
    max_len = 40
    if len(sanitized) > max_len:
        # Try to truncate at the last underscore before max_len
        last_underscore = sanitized.rfind('_', 0, max_len)
        if last_underscore > 0:
            sanitized = sanitized[:last_underscore]
        else:
            sanitized = sanitized[:max_len]
            
    # Handle empty string after sanitization
    if not sanitized:
        sanitized = "project"
        
    # Append timestamp for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return f"{sanitized}_{timestamp}"

# Persistence functions
SAVABLE_STATE_KEYS = [
    'step', 'run_id', 'history', 'prompt', 'selected_prompt', 
    'optimized_prompts', 'generated_images', 'selected_image_path', 
    'upscaled_image_path', 'generated_videos', 'video_attempts', 
    'selected_video', 'clip_paths', 'current_start_image_path', 
    'loop_count', 'final_video_path', 'target_duration', 
    'audio_added', 'selected_audio_path', 'clip_duration'
]

def save_session_to_file(run_id):
    """Saves the current savable session state to a JSON file."""
    state_file = get_project_path(run_id, "state", "session_state.json")
    state_to_save = {key: st.session_state.get(key) for key in SAVABLE_STATE_KEYS if key in st.session_state}
    
    try:
        with open(state_file, 'w') as f:
            json.dump(state_to_save, f, indent=4)
        print(f"Session state saved to {state_file}")
    except Exception as e:
        st.error(f"Error saving session state: {e}")

def load_session_from_file(run_id):
    """Loads session state from a JSON file and updates the current session."""
    state_file = get_project_path(run_id, "state", "session_state.json")
    if not os.path.exists(state_file):
        st.error(f"State file not found for project {run_id}")
        return False
        
    try:
        with open(state_file, 'r') as f:
            loaded_state = json.load(f)
        
        # Update current session state
        for key, value in loaded_state.items():
            if key in SAVABLE_STATE_KEYS:
                st.session_state[key] = value
            else:
                print(f"Warning: Skipping unknown key '{key}' from state file.")
        
        st.success(f"Project {run_id} loaded successfully.")
        # Ensure essential defaults if missing
        if 'step' not in st.session_state: st.session_state.step = 1
        if 'clip_duration' not in st.session_state: st.session_state.clip_duration = 10
        if 'history' not in st.session_state: st.session_state.history = {}
        
        return True
    except Exception as e:
        st.error(f"Error loading session state: {e}")
        return False

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

# Initial check: Load existing project or start new?
if 'run_id' not in st.session_state:
    st.title("Load or Start New Project")
    
    # List existing projects
    project_base_dir = "projects"
    os.makedirs(project_base_dir, exist_ok=True)
    existing_projects = [d for d in os.listdir(project_base_dir) 
                         if os.path.isdir(os.path.join(project_base_dir, d))] 
    
    if existing_projects:
        st.subheader("Load Existing Project")
        selected_project_id = st.selectbox("Select Project ID", options=[""] + existing_projects)
        if selected_project_id:
            if st.button(f"Load Project: {selected_project_id}"):
                if load_session_from_file(selected_project_id):
                    go_to_step(st.session_state.step)
                else:
                    # Error handled in load_session_from_file
                    pass # Stay on loading page
    else:
        st.info("No existing projects found.")

    st.subheader("Start New Project")
    if st.button("Start New Ambience Video"):
        # Initialize new session
        # Generate NEW ID using default name + timestamp
        st.session_state.run_id = generate_project_name("new_project") 
        create_project_folder(st.session_state.run_id)
        st.session_state.step = 1
        st.session_state.history = {}
        st.session_state.clip_duration = 10 # Default for new projects
        # Initialize other keys to defaults or empty
        for key in SAVABLE_STATE_KEYS:
            if key not in ['run_id', 'step', 'history', 'clip_duration']:
                 if key in ['optimized_prompts', 'generated_images', 'generated_videos', 'clip_paths']:
                     st.session_state[key] = []
                 elif key == 'video_attempts':
                     st.session_state[key] = 0
                 elif key == 'loop_count':
                     st.session_state[key] = 1
                 elif key == 'target_duration':
                     st.session_state[key] = 30
                 elif key == 'audio_added':
                     st.session_state[key] = False
                 else:
                     st.session_state[key] = None
                     
        save_session_to_file(st.session_state.run_id) # Save initial state
        st.experimental_rerun() # Rerun to start the workflow

# --- Main Workflow Logic (Only runs if run_id is set) --- #
elif 'run_id' in st.session_state:
    st.title(f"Ambience Video Creator (Project: {st.session_state.run_id})")

    # Navigation sidebar (add this to show the workflow and allow jumping to steps)
    with st.sidebar:
        st.header("Navigation")
        st.write("Click a step to navigate:")
        
        # Only show steps we've visited or are at currently
        max_step = max(st.session_state.step, 1)
        
        if st.button("🏠 Step 1: Enter Description", disabled=False):
            go_to_step(1)
        
        if max_step >= 3 and st.button("🖼️ Step 3: Select Image", disabled=False):
            go_to_step(3)
        
        if max_step >= 4 and st.button("🎬 Step 4: Select Initial Video", disabled=False):
            go_to_step(4)
        
        if max_step >= 5 and st.button("➕ Step 5: Continue or Complete", disabled=False):
            go_to_step(5)
        
        if max_step >= 6 and st.button("�� Step 6: Additional Videos", disabled=False):
            go_to_step(6)
        
        if max_step >= 7 and st.button("⚙️ Step 7: Final Settings", disabled=False):
            go_to_step(7)

    # Step 1: Initial prompt input
    if st.session_state.step == 1:
        st.header("Step 1: Enter Your Scene Description & Settings")
        
        prompt_input = st.text_area("Description (e.g., 'a cozy library at night with a cat curled up on a chair')", 
                                  value=st.session_state.prompt or "a cozy cabin in a snowy forest with a crackling fireplace with soft embers",
                                  height=100)

        # Add clip duration selection
        st.session_state.clip_duration = st.radio(
            "Select Video Clip Duration (seconds)",
            (5, 10),
            index=1 if st.session_state.clip_duration == 10 else 0, # Default selection based on state
            horizontal=True
        )
        st.info(f"Each generated video segment will be {st.session_state.clip_duration} seconds long.")
        
        if st.button("Generate Images"):
            if prompt_input:
                st.session_state.prompt = prompt_input # Set the actual prompt here
                
                # Check if we need to rename the project based on the first real prompt
                if st.session_state.run_id.startswith("new_project_"):
                    old_run_id = st.session_state.run_id
                    new_run_id = generate_project_name(st.session_state.prompt)
                    st.session_state.run_id = new_run_id
                    # Rename the folder
                    old_project_dir = f"projects/{old_run_id}"
                    new_project_dir = f"projects/{new_run_id}"
                    if os.path.exists(old_project_dir):
                        try:
                            os.rename(old_project_dir, new_project_dir)
                            print(f"Renamed project folder from {old_run_id} to {new_run_id}")
                            # Update the title immediately
                            st.title(f"Ambience Video Creator (Project: {st.session_state.run_id})")
                        except OSError as e:
                            st.error(f"Error renaming project folder: {e}")
                            st.session_state.run_id = old_run_id # Revert if rename fails
                    else:
                         # If old folder doesn't exist, just create the new one
                         create_project_folder(st.session_state.run_id)

                # Create placeholders for image generation progress
                st.write("Generating optimized prompts and images...")
                progress_bar = st.progress(0)
                
                # Store all generated prompts
                all_prompts = []
                image_paths = []
                
                # Generate 4 different optimized prompts and corresponding images
                try:
                    for j in range(4):
                        # Update progress
                        progress_bar.progress(j / 4 * 0.5)  # First half of progress is for prompts
                        st.write(f"Generating optimized prompt {j+1}/4...")
                        
                        # Generate a unique optimized prompt each time
                        with st.spinner(f"Creating optimized prompt {j+1}..."):
                            optimized_prompt = generate_optimized_prompt(prompt_input)
                            
                            if optimized_prompt:
                                all_prompts.append(optimized_prompt)
                                
                                # Show the generated prompt in an expander
                                with st.expander(f"Optimized Prompt {j+1}"):
                                    st.text_area(f"Prompt {j+1}", optimized_prompt, height=150)
                                
                                # Generate an image from this specific prompt
                                st.write(f"Generating image {j+1} from prompt {j+1}...")
                                progress_bar.progress(j / 4 * 0.5 + 0.125)  # Update progress
                                
                                image_output = replicate.run(
                                    "google/imagen-3",
                                    input={
                                        "prompt": optimized_prompt,
                                        "aspect_ratio": "16:9",
                                        "negative_prompt": "fast movement",
                                        "safety_filter_level": "block_medium_and_above"
                                    }
                                )
                                
                                # Save the image to project folder
                                image_filename = f"image_{j}.png"
                                image_path = get_project_path(st.session_state.run_id, "image", image_filename)
                                
                                with open(image_path, "wb") as img_file:
                                    img_file.write(image_output.read())
                                
                                # Verify the image was saved correctly
                                if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
                                    st.success(f"Image {j+1} generated successfully!")
                                    # Display a thumbnail of the image
                                    try:
                                        img = Image.open(image_path)
                                        st.image(img, caption=f"Image {j+1} (from Prompt {j+1})", width=300)
                                    except Exception as e:
                                        st.error(f"Error displaying thumbnail: {e}")
                                else:
                                    st.error(f"Failed to save image {j+1}")
                                
                                # Store the image path
                                image_paths.append(image_path)
                            else:
                                st.error(f"Failed to generate prompt {j+1}. Skipping this iteration.")
                    
                    progress_bar.progress(1.0)
                    
                    # Store all generated prompts and images in session state
                    st.session_state.optimized_prompts = all_prompts
                    st.session_state.generated_images = image_paths
                    
                    # Save state after generating images
                    save_step_state(1)
                    save_step_state(3)
                    save_session_to_file(st.session_state.run_id)
                    
                    # Count valid images
                    valid_images = [p for p in image_paths if os.path.exists(p) and os.path.getsize(p) > 0]
                    if len(valid_images) > 0:
                        st.success(f"Generated {len(valid_images)} images from {len(all_prompts)} unique prompts!")
                        # Skip step 2 and go directly to step 3 (image selection)
                        st.session_state.step = 3
                        st.button("Proceed to Image Selection", on_click=lambda: st.experimental_rerun())
                    else:
                        st.error("No valid images were generated. Please try again.")
                except Exception as e:
                    st.error(f"Error in prompt generation process: {e}")
                    st.error(f"Exception details: {str(e)}")

    # Step 3: Image selection
    elif st.session_state.step == 3:
        st.header("Step 3: Select an Image")
        
        # Add "Regenerate Images" button
        if st.button("🔄 Regenerate Images", help="Generate new optimized prompts and images"):
            if st.session_state.prompt:
                st.write("Generating new optimized prompts and images...")
                progress_bar = st.progress(0)
                
                try:
                    # Store all generated prompts
                    all_prompts = []
                    image_paths = []
                    attempt = len(st.session_state.history.get(3, [])) + 1
                    
                    # Generate 4 different optimized prompts and corresponding images
                    for j in range(4):
                        # Update progress
                        progress_bar.progress(j / 4 * 0.5)  # First half of progress is for prompts
                        
                        # Generate a unique optimized prompt each time
                        optimized_prompt = generate_optimized_prompt(st.session_state.prompt)
                        
                        if optimized_prompt:
                            all_prompts.append(optimized_prompt)
                            
                            # Generate an image from this specific prompt
                            progress_bar.progress(j / 4 * 0.5 + 0.125)  # Update progress
                            
                            image_output = replicate.run(
                                "google/imagen-3",
                                input={
                                    "prompt": optimized_prompt,
                                    "aspect_ratio": "16:9",
                                    "negative_prompt": "fast movement",
                                    "safety_filter_level": "block_medium_and_above"
                                }
                            )
                            
                            # Save the image to project folder with attempt number
                            image_filename = f"image_{attempt}_{j}.png"
                            image_path = get_project_path(st.session_state.run_id, "image", image_filename)
                            
                            with open(image_path, "wb") as img_file:
                                img_file.write(image_output.read())
                            
                            # Store the image path
                            image_paths.append(image_path)
                    
                    progress_bar.progress(1.0)
                    
                    # Store all generated prompts and images in session state
                    st.session_state.optimized_prompts = all_prompts
                    st.session_state.generated_images = image_paths
                    
                    # Save state after generating images
                    save_step_state(3)
                    save_session_to_file(st.session_state.run_id)
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error regenerating images: {e}")
        
        # If we have the optimized prompts, display them
        if st.session_state.optimized_prompts:
            with st.expander("Show Optimized Prompts"):
                for i, opt_prompt in enumerate(st.session_state.optimized_prompts):
                    if i < len(st.session_state.optimized_prompts):
                        st.text_area(f"Prompt {i+1}", opt_prompt, height=100)
        
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
                            prompt_index = i if i < len(st.session_state.optimized_prompts) else 0
                            st.image(img, caption=f"Option {i+1} (Prompt {prompt_index+1})")
                        except Exception as e:
                            st.error(f"Error loading image with PIL: {e}")
                            # Fallback to direct file path
                            st.image(image_path, caption=f"Option {i+1}")
                        
                        if st.button(f"Select Image {i+1}"):
                            st.session_state.selected_image_path = image_path
                            # Store the corresponding prompt
                            prompt_index = i if i < len(st.session_state.optimized_prompts) else 0
                            st.session_state.selected_prompt = st.session_state.optimized_prompts[prompt_index]
                            
                            try:
                                # Upscale the selected image
                                with st.spinner("Upscaling your selected image..."):
                                    img = cv2.imread(image_path)
                                    if img is None:
                                        st.error(f"OpenCV could not read image {image_path}")
                                        # Try with PIL and convert to CV2
                                        pil_img = Image.open(image_path)
                                        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                                    
                                    upscaled_img = sharpen_frame(img)
                                    upscaled_filename = "upscaled_image.png"
                                    upscaled_path = get_project_path(st.session_state.run_id, "image", upscaled_filename)
                                    cv2.imwrite(upscaled_path, upscaled_img)
                                    st.session_state.upscaled_image_path = upscaled_path
                                    st.session_state.current_start_image_path = upscaled_path
                                    
                                    # Show upscaled image
                                    st.success(f"Image upscaled successfully!")
                                    st.image(upscaled_path, caption="Upscaled Image")
                                
                                # Generate 2 video options with selected duration
                                with st.spinner(f"Generating initial {st.session_state.clip_duration}s video options..."):
                                    st.session_state.generated_videos = []
                                    # Updated video prompts for fixed camera + focus
                                    user_video_prompt = "Locked camera shot, fixed perspective, absolutely no camera movement. Maintain focus on the primary subject and keep a constant distance. camera-tilt:0, camera-zoom:0, camera-pan:0, camera-rotate:0"
                                    negative_prompt_text = "camera movement, pan, tilt, zoom, rotation, camera shake, unsteady camera, focus shift, distance change"
                                    
                                    # Generate 2 videos (removed nested try)
                                    for j in range(2):
                                        video_filename = f"video_0_{j}.mp4"
                                        video_path = get_project_path(st.session_state.run_id, "video", video_filename)
                                        
                                        input_dict = {
                                            "prompt": user_video_prompt,
                                            "duration": st.session_state.clip_duration, # Use selected duration
                                            "cfg_scale": 0,
                                            "start_image": open(upscaled_path, "rb"),
                                            "aspect_ratio": "16:9",
                                            "negative_prompt": negative_prompt_text
                                        }
                                        
                                        video_output = replicate.run("kwaivgi/kling-v1.6-pro", input=input_dict)
                                        with open(video_path, "wb") as vid_file:
                                            vid_file.write(video_output.read())
                                        
                                        st.session_state.generated_videos.append(video_path)
                                    
                                    st.success(f"Generated {len(st.session_state.generated_videos)} videos")
                                    # Save state before advancing
                                    save_step_state(4)
                                    save_session_to_file(st.session_state.run_id)
                                    # Reset video attempt counter
                                    st.session_state.video_attempts = 0
                                    st.session_state.step = 4
                                    st.experimental_rerun()
                                
                            except Exception as e:
                                st.error(f"Error in image selection/video generation process: {e}")
                    except Exception as e:
                        st.error(f"Error processing image {image_path}: {e}")

    # Step 4: Initial Video Selection
    elif st.session_state.step == 4:
        st.header("Step 4: Select Initial Video")
        
        # Add "Regenerate Videos" button
        if st.button("🔄 Regenerate Videos", help="Generate new video options from the selected image"):
            if st.session_state.upscaled_image_path and os.path.exists(st.session_state.upscaled_image_path):
                with st.spinner(f"Generating new {st.session_state.clip_duration}s video options..."):
                    try:
                        st.session_state.video_attempts += 1
                        attempt = st.session_state.video_attempts
                        st.session_state.generated_videos = []
                        # Updated video prompts for fixed camera + focus
                        user_video_prompt = "Locked camera shot, fixed perspective, absolutely no camera movement. Maintain focus on the primary subject and keep a constant distance. camera-tilt:0, camera-zoom:0, camera-pan:0, camera-rotate:0"
                        negative_prompt_text = "camera movement, pan, tilt, zoom, rotation, camera shake, unsteady camera, focus shift, distance change"
                        
                        # Generate 2 videos
                        for j in range(2):
                            video_filename = f"video_{attempt}_{j}.mp4"
                            video_path = get_project_path(st.session_state.run_id, "video", video_filename)
                            
                            input_dict = {
                                "prompt": user_video_prompt,
                                "duration": st.session_state.clip_duration, # Use selected duration
                                "cfg_scale": 0,
                                "start_image": open(st.session_state.upscaled_image_path, "rb"),
                                "aspect_ratio": "16:9",
                                "negative_prompt": negative_prompt_text
                            }
                            
                            video_output = replicate.run("kwaivgi/kling-v1.6-pro", input=input_dict)
                            with open(video_path, "wb") as vid_file:
                                vid_file.write(video_output.read())
                            
                            st.session_state.generated_videos.append(video_path)
                        
                        save_step_state(4)
                        save_session_to_file(st.session_state.run_id)
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error regenerating videos: {e}")
        
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
        if st.session_state.video_attempts > 0:
            st.write(f"Video generation attempt: {st.session_state.video_attempts + 1}")
        
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
                                    last_frame_filename = "last_frame.png"
                                    last_frame_path = get_project_path(st.session_state.run_id, "image", last_frame_filename)
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
                            save_session_to_file(st.session_state.run_id) # Save state before moving
                            st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error processing video {video_path}: {e}")

    # Step 5: Continue or Complete
    elif st.session_state.step == 5:
        st.header("Step 5: Continue or Complete")
        
        # Add "Regenerate Videos" button
        if st.button("🔄 Regenerate Videos", help="Generate new video options from the selected image"):
            if st.session_state.upscaled_image_path and os.path.exists(st.session_state.upscaled_image_path):
                with st.spinner(f"Generating new {st.session_state.clip_duration}s video options..."):
                    try:
                        st.session_state.video_attempts += 1
                        attempt = st.session_state.video_attempts
                        st.session_state.generated_videos = []
                        # Updated video prompts for fixed camera + focus
                        user_video_prompt = "Locked camera shot, fixed perspective, absolutely no camera movement. Maintain focus on the primary subject and keep a constant distance. camera-tilt:0, camera-zoom:0, camera-pan:0, camera-rotate:0"
                        negative_prompt_text = "camera movement, pan, tilt, zoom, rotation, camera shake, unsteady camera, focus shift, distance change"
                        
                        # Generate 2 videos
                        for j in range(2):
                            video_filename = f"video_{attempt}_{j}.mp4"
                            video_path = get_project_path(st.session_state.run_id, "video", video_filename)
                            
                            input_dict = {
                                "prompt": user_video_prompt,
                                "duration": st.session_state.clip_duration, # Use selected duration
                                "cfg_scale": 0,
                                "start_image": open(st.session_state.upscaled_image_path, "rb"),
                                "aspect_ratio": "16:9",
                                "negative_prompt": negative_prompt_text
                            }
                            
                            video_output = replicate.run("kwaivgi/kling-v1.6-pro", input=input_dict)
                            with open(video_path, "wb") as vid_file:
                                vid_file.write(video_output.read())
                            
                            st.session_state.generated_videos.append(video_path)
                        
                        save_step_state(5)
                        save_session_to_file(st.session_state.run_id)
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error regenerating videos: {e}")
        
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
        if st.session_state.video_attempts > 0:
            st.write(f"Video generation attempt: {st.session_state.video_attempts + 1}")
        
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
            if st.button("Return to Step 4"):
                st.session_state.step = 4
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
                                    last_frame_filename = "last_frame.png"
                                    last_frame_path = get_project_path(st.session_state.run_id, "image", last_frame_filename)
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
                            save_session_to_file(st.session_state.run_id) # Save state before moving
                            st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error processing video {video_path}: {e}")

    # Step 6: Additional Videos
    elif st.session_state.step == 6:
        st.header("Step 6: Additional Videos")
        
        # Add "Regenerate Videos" button
        if st.button("🔄 Regenerate Videos", help="Generate new video options from the selected image"):
            if st.session_state.upscaled_image_path and os.path.exists(st.session_state.upscaled_image_path):
                with st.spinner(f"Generating new {st.session_state.clip_duration}s video options..."):
                    try:
                        st.session_state.video_attempts += 1
                        attempt = st.session_state.video_attempts
                        st.session_state.generated_videos = []
                        # Updated video prompts for fixed camera + focus
                        user_video_prompt = "Locked camera shot, fixed perspective, absolutely no camera movement. Maintain focus on the primary subject and keep a constant distance. camera-tilt:0, camera-zoom:0, camera-pan:0, camera-rotate:0"
                        negative_prompt_text = "camera movement, pan, tilt, zoom, rotation, camera shake, unsteady camera, focus shift, distance change"
                        
                        # Generate 2 videos
                        for j in range(2):
                            video_filename = f"video_{attempt}_{j}.mp4"
                            video_path = get_project_path(st.session_state.run_id, "video", video_filename)
                            
                            input_dict = {
                                "prompt": user_video_prompt,
                                "duration": st.session_state.clip_duration, # Use selected duration
                                "cfg_scale": 0,
                                "start_image": open(st.session_state.upscaled_image_path, "rb"),
                                "aspect_ratio": "16:9",
                                "negative_prompt": negative_prompt_text
                            }
                            
                            video_output = replicate.run("kwaivgi/kling-v1.6-pro", input=input_dict)
                            with open(video_path, "wb") as vid_file:
                                vid_file.write(video_output.read())
                            
                            st.session_state.generated_videos.append(video_path)
                        
                        save_step_state(6)
                        save_session_to_file(st.session_state.run_id)
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error regenerating videos: {e}")
        
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
        if st.session_state.video_attempts > 0:
            st.write(f"Video generation attempt: {st.session_state.video_attempts + 1}")
        
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
                                    last_frame_filename = "last_frame.png"
                                    last_frame_path = get_project_path(st.session_state.run_id, "image", last_frame_filename)
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
                            save_session_to_file(st.session_state.run_id) # Save state before moving
                            st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error processing video {video_path}: {e}")

    # Step 7: Final Settings
    elif st.session_state.step == 7:
        st.header("Step 7: Final Settings")
        
        # Add "Regenerate Videos" button
        if st.button("🔄 Regenerate Videos", help="Generate new video options from the selected image"):
            if st.session_state.upscaled_image_path and os.path.exists(st.session_state.upscaled_image_path):
                with st.spinner(f"Generating new {st.session_state.clip_duration}s video options..."):
                    try:
                        st.session_state.video_attempts += 1
                        attempt = st.session_state.video_attempts
                        st.session_state.generated_videos = []
                        # Updated video prompts for fixed camera + focus
                        user_video_prompt = "Locked camera shot, fixed perspective, absolutely no camera movement. Maintain focus on the primary subject and keep a constant distance. camera-tilt:0, camera-zoom:0, camera-pan:0, camera-rotate:0"
                        negative_prompt_text = "camera movement, pan, tilt, zoom, rotation, camera shake, unsteady camera, focus shift, distance change"
                        
                        # Generate 2 videos
                        for j in range(2):
                            video_filename = f"video_{attempt}_{j}.mp4"
                            video_path = get_project_path(st.session_state.run_id, "video", video_filename)
                            
                            input_dict = {
                                "prompt": user_video_prompt,
                                "duration": st.session_state.clip_duration, # Use selected duration
                                "cfg_scale": 0,
                                "start_image": open(st.session_state.upscaled_image_path, "rb"),
                                "aspect_ratio": "16:9",
                                "negative_prompt": negative_prompt_text
                            }
                            
                            video_output = replicate.run("kwaivgi/kling-v1.6-pro", input=input_dict)
                            with open(video_path, "wb") as vid_file:
                                vid_file.write(video_output.read())
                            
                            st.session_state.generated_videos.append(video_path)
                        
                        save_step_state(7)
                        save_session_to_file(st.session_state.run_id)
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error regenerating videos: {e}")
        
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
        if st.session_state.video_attempts > 0:
            st.write(f"Video generation attempt: {st.session_state.video_attempts + 1}")
        
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
            if st.button("Return to Step 6"):
                st.session_state.step = 6
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
                                    last_frame_filename = "last_frame.png"
                                    last_frame_path = get_project_path(st.session_state.run_id, "image", last_frame_filename)
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
                            save_session_to_file(st.session_state.run_id) # Save state before moving
                            st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error processing video {video_path}: {e}")

# Reset session state function (Should clear IN-MEMORY state for starting NEW)
def reset_session_state():
    # Store run_id before clearing
    current_run_id = st.session_state.get('run_id', None)
    
    # Clear all keys except maybe preserved ones if needed later
    keys_to_clear = list(st.session_state.keys())
    for key in keys_to_clear:
        del st.session_state[key]
    
    # Re-initialize essential keys for a new session
    # Generate NEW ID using default name + timestamp (will be renamed in step 1)
    st.session_state.run_id = generate_project_name("new_project") 
    create_project_folder(st.session_state.run_id)
    st.session_state.step = 1
    st.session_state.history = {}
    st.session_state.clip_duration = 10 # Default
    # Initialize other savable keys to defaults/empty
    for key in SAVABLE_STATE_KEYS:
        if key not in ['run_id', 'step', 'history', 'clip_duration']:
            if key in ['optimized_prompts', 'generated_images', 'generated_videos', 'clip_paths']:
                st.session_state[key] = []
            elif key == 'video_attempts':
                st.session_state[key] = 0
            elif key == 'loop_count':
                st.session_state[key] = 1
            elif key == 'target_duration':
                st.session_state[key] = 30
            elif key == 'audio_added':
                st.session_state[key] = False
            else:
                st.session_state[key] = None
                
    save_session_to_file(st.session_state.run_id) # Save initial state of NEW project
    # Don't rerun here, let the main flow handle it after reset