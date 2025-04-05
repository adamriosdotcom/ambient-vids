import os
import tempfile
import streamlit as st
import numpy as np
import pandas as pd
import ffmpeg
import requests
from io import BytesIO
from pathlib import Path

# Constants for audio resources
AUDIO_DIR = "audio_resources"

# Define dictionary of free audio resources
AMBIENT_SOUNDS = {
    "Fireplace": {
        "Crackling Fire": "https://ia800901.us.archive.org/23/items/FireplaceCracklingSoundEffect/Fireplace%20Crackling%20Sound%20Effect.mp3",
        "Fireplace with Wind": "https://ia601508.us.archive.org/24/items/soundscrate-fireplace-crackling-sound-effect/Soundscrate-fireplace-crackling-sound-effect.mp3", 
        "Cozy Evening Fire": "https://ia600101.us.archive.org/7/items/fireplace-sound-hd/fireplace-sound-hd.mp3"
    },
    "Rain": {
        "Gentle Rain": "https://ia801002.us.archive.org/27/items/MediumRain/Medium%20Rain.mp3",
        "Thunderstorm": "https://ia800503.us.archive.org/8/items/ThunderstormRainSoundEffectsHighQuality/Thunderstorm%20Rain%20Sound%20Effects%20-%20High%20Quality.mp3",
        "Rain on Window": "https://ia801002.us.archive.org/14/items/RainOnRoofWindows/Rain%20On%20Roof%20%26%20Windows.mp3"
    },
    "Forest": {
        "Forest Ambience": "https://ia800701.us.archive.org/23/items/forestsoundeffects/Forest%20Sound%20Effects.mp3",
        "Bird Chirping": "https://ia800505.us.archive.org/5/items/BirdSoundsForestInMorning/Bird%20Sounds%20-%20Forest%20in%20Morning.mp3",
        "Woodland Stream": "https://ia800204.us.archive.org/10/items/ForestStreamAndWaterBirdSoundEffectsForRelaxationAndSleep/Forest,%20Stream%20and%20Water%20Bird%20Sound%20Effects%20For%20Relaxation%20and%20Sleep.mp3"
    },
    "Ocean": {
        "Ocean Waves": "https://ia800501.us.archive.org/22/items/OceanWavesSoundEffect/Ocean%20Waves%20Sound%20Effect.mp3",
        "Calm Sea": "https://ia800501.us.archive.org/5/items/SeaShoreWavesSoundEffect/Sea%20Shore%20Waves%20Sound%20Effect.mp3", 
        "Beach Ambience": "https://ia800502.us.archive.org/6/items/RelaxingBeachSound/Relaxing%20Beach%20Sound.mp3"
    }
}

CLASSICAL_MUSIC = {
    "Calm": {
        "Gymnopédie No.1 (Erik Satie)": "https://ia800507.us.archive.org/14/items/18gymnopediens1/01_gymnopedies_1.mp3",
        "Clair de Lune (Debussy)": "https://ia801009.us.archive.org/11/items/ClairDeLune_653/Debussy-ClairDeLune.mp3",
        "Canon in D (Pachelbel)": "https://ia800201.us.archive.org/12/items/PachelbelCanonInD_595/PachelbelCanonInD.mp3"
    },
    "Melancholic": {
        "Moonlight Sonata (Beethoven)": "https://ia800906.us.archive.org/14/items/BeethovenMoonlightSonata1stMovement/Beethoven-MoonlightSonata1stMovement.mp3",
        "Prelude in E-Minor (Chopin)": "https://ia800307.us.archive.org/34/items/Chopin-PreludeInE-minorOp.28No.4/Chopin-PreludeInE-minorOp.28No.4.mp3",
        "Adagio for Strings (Barber)": "https://ia600303.us.archive.org/21/items/SamuelBarberAdagioForStrings/SamuelBarberAdagioForStrings.mp3"
    },
    "Uplifting": {
        "Spring (Vivaldi)": "https://ia802605.us.archive.org/8/items/Antonio_Vivaldi_-_The_Four_Seasons_-_Spring/Antonio_Vivaldi_-_Spring_-_01_-_allegro.mp3",
        "Ode to Joy (Beethoven)": "https://ia800201.us.archive.org/14/items/OdeToJoy_393/BeethovenOdeToJoy.mp3",
        "Morning Mood (Grieg)": "https://ia903208.us.archive.org/19/items/GriegMorningMoodPeerGyntSuite/Grieg-MorningMoodPeerGyntSuite.mp3"
    }
}

def ensure_audio_dir():
    """Creates the audio resources directory if it doesn't exist"""
    os.makedirs(AUDIO_DIR, exist_ok=True)

def download_audio(url, category, name):
    """Downloads audio file from URL and saves to audio directory"""
    try:
        ensure_audio_dir()
        # Create a clean filename
        clean_name = name.replace(" ", "_").lower()
        # Create subdirectory for category
        category_dir = os.path.join(AUDIO_DIR, category.lower())
        os.makedirs(category_dir, exist_ok=True)
        
        # Define the file path
        file_path = os.path.join(category_dir, f"{clean_name}.mp3")
        
        # Check if file already exists
        if os.path.exists(file_path):
            return file_path
        
        # Print debug information
        print(f"Downloading audio from: {url}")
        
        # Download the file
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"Audio downloaded successfully to: {file_path}")
            return file_path
        else:
            error_msg = f"Failed to download audio: {response.status_code}"
            print(error_msg)
            st.error(error_msg)
            return None
    except Exception as e:
        error_msg = f"Error downloading audio: {str(e)}"
        print(error_msg)
        st.error(error_msg)
        return None

def get_audio_file(audio_type, category, name):
    """Gets the path to an audio file, downloading it if necessary"""
    if audio_type == "Ambient Sounds":
        url = AMBIENT_SOUNDS[category][name]
        return download_audio(url, category, name)
    elif audio_type == "Classical Music":
        url = CLASSICAL_MUSIC[category][name]
        return download_audio(url, category, name)
    return None

def mix_audio(ambient_path, music_path, output_path, ambient_volume=0.7, music_volume=0.4):
    """Mixes ambient sound with music at the specified volumes"""
    try:
        (
            ffmpeg
            .input(ambient_path)
            .filter('volume', volume=ambient_volume)
            .output('temp_ambient.mp3')
            .overwrite_output()
            .run(quiet=True)
        )
        
        (
            ffmpeg
            .input(music_path)
            .filter('volume', volume=music_volume)
            .output('temp_music.mp3')
            .overwrite_output()
            .run(quiet=True)
        )
        
        (
            ffmpeg
            .input('temp_ambient.mp3')
            .input('temp_music.mp3')
            .filter_complex('amix=inputs=2:duration=longest')
            .output(output_path)
            .overwrite_output()
            .run(quiet=True)
        )
        
        # Clean up temp files
        os.remove('temp_ambient.mp3')
        os.remove('temp_music.mp3')
        
        return output_path
    except Exception as e:
        st.error(f"Error mixing audio: {e}")
        return None

def add_audio_to_video(video_path, audio_path, output_path, loop_audio=True):
    """Adds audio to a video file, optionally looping the audio to match video length"""
    try:
        # Get video duration
        video_info = ffmpeg.probe(video_path)
        video_duration = float(video_info['format']['duration'])
        
        # Get audio duration
        audio_info = ffmpeg.probe(audio_path)
        audio_duration = float(audio_info['format']['duration'])
        
        if loop_audio and audio_duration < video_duration:
            # Create a temporary file for the looped audio
            temp_audio = "temp_looped_audio.mp3"
            loop_count = int(np.ceil(video_duration / audio_duration))
            
            input_list = []
            for i in range(loop_count):
                input_list.append(ffmpeg.input(audio_path))
            
            (
                ffmpeg.concat(*input_list, v=0, a=1)
                .output(temp_audio)
                .overwrite_output()
                .run(quiet=True)
            )
            
            # Now add the looped audio to the video
            (
                ffmpeg
                .input(video_path)
                .input(temp_audio)
                .output(output_path, vcodec='copy', acodec='aac', map=['0:v', '1:a'], shortest=None)
                .overwrite_output()
                .run(quiet=True)
            )
            
            # Clean up temp file
            os.remove(temp_audio)
        else:
            # Add audio directly (without looping)
            (
                ffmpeg
                .input(video_path)
                .input(audio_path)
                .output(output_path, vcodec='copy', acodec='aac', map=['0:v', '1:a'], shortest=None)
                .overwrite_output()
                .run(quiet=True)
            )
        
        return output_path
    except Exception as e:
        st.error(f"Error adding audio to video: {e}")
        return None

# Download initial sample files on module load
def download_sample_files():
    """Pre-downloads sample audio files for testing"""
    sample_files = {}
    
    # Download one sample from each category
    try:
        # Ambient sample
        ambient_category = list(AMBIENT_SOUNDS.keys())[0]
        ambient_name = list(AMBIENT_SOUNDS[ambient_category].keys())[0]
        ambient_url = AMBIENT_SOUNDS[ambient_category][ambient_name]
        ambient_file = download_audio(ambient_url, ambient_category, ambient_name)
        if ambient_file:
            sample_files['ambient'] = ambient_file
        
        # Classical sample
        music_category = list(CLASSICAL_MUSIC.keys())[0]
        music_name = list(CLASSICAL_MUSIC[music_category].keys())[0]
        music_url = CLASSICAL_MUSIC[music_category][music_name]
        music_file = download_audio(music_url, music_category, music_name)
        if music_file:
            sample_files['classical'] = music_file
            
        print(f"Sample files downloaded: {sample_files}")
        return sample_files
    except Exception as e:
        print(f"Error downloading sample files: {e}")
        return {}

# Streamlit interface for testing
def audio_ui_test():
    st.title("Ambience Audio Selector")
    
    # Display debug information
    st.write("This is a testing interface for the audio feature. If you encounter errors, please check the logs.")
    
    # Check if ffmpeg is available
    try:
        import shutil
        ffmpeg_path = shutil.which('ffmpeg')
        if ffmpeg_path:
            st.success(f"FFmpeg found at: {ffmpeg_path}")
        else:
            st.warning("FFmpeg executable not found in PATH. Some features may not work.")
    except Exception as e:
        st.warning(f"Could not check for FFmpeg: {e}")
    
    # Pre-download sample files
    if 'sample_files' not in st.session_state:
        st.session_state.sample_files = download_sample_files()
        
    if st.session_state.sample_files:
        st.success("Sample audio files downloaded successfully. You can play them below.")
        
        # Display sample ambient sound
        if 'ambient' in st.session_state.sample_files:
            st.subheader("Sample Ambient Sound")
            st.audio(st.session_state.sample_files['ambient'])
            
        # Display sample classical music
        if 'classical' in st.session_state.sample_files:
            st.subheader("Sample Classical Music")
            st.audio(st.session_state.sample_files['classical'])
    
    # Audio type selection
    audio_type = st.selectbox("Audio Type", 
                             ["Ambient Sounds", "Classical Music", "Combine Both", "Upload Your Own"])
    
    # Initialize session state variables for audio
    if 'ambient_category' not in st.session_state:
        st.session_state.ambient_category = None
    if 'ambient_sound' not in st.session_state:
        st.session_state.ambient_sound = None
    if 'music_category' not in st.session_state:
        st.session_state.music_category = None
    if 'music_track' not in st.session_state:
        st.session_state.music_track = None
    if 'audio_files' not in st.session_state:
        st.session_state.audio_files = []
    
    # Ambient sound options
    if audio_type in ["Ambient Sounds", "Combine Both"]:
        ambient_category = st.selectbox("Ambient Sound Category", 
                                      list(AMBIENT_SOUNDS.keys()))
        st.session_state.ambient_category = ambient_category
        
        # Display ambient sound options based on category
        ambient_options = AMBIENT_SOUNDS[ambient_category]
        selected_ambient = st.selectbox("Choose Ambient Sound", 
                                       list(ambient_options.keys()))
        st.session_state.ambient_sound = selected_ambient
        
        # Download the audio file
        ambient_url = ambient_options[selected_ambient]
        try:
            audio_file = get_audio_file("Ambient Sounds", ambient_category, selected_ambient)
            if audio_file and os.path.exists(audio_file):
                st.success(f"Audio file downloaded to: {audio_file}")
                st.audio(audio_file)
            else:
                st.warning("Could not download audio file. Trying direct URL...")
                st.audio(ambient_url)
        except Exception as e:
            st.error(f"Error playing audio: {e}")
            
    # Classical music options
    if audio_type in ["Classical Music", "Combine Both"]:
        music_category = st.selectbox("Music Mood", 
                                    list(CLASSICAL_MUSIC.keys()))
        st.session_state.music_category = music_category
        
        # Display music options based on mood
        music_options = CLASSICAL_MUSIC[music_category]
        selected_music = st.selectbox("Choose Classical Track", 
                                     list(music_options.keys()))
        st.session_state.music_track = selected_music
        
        # Download the audio file
        music_url = music_options[selected_music]
        try:
            audio_file = get_audio_file("Classical Music", music_category, selected_music)
            if audio_file and os.path.exists(audio_file):
                st.success(f"Audio file downloaded to: {audio_file}")
                st.audio(audio_file)
            else:
                st.warning("Could not download audio file. Trying direct URL...")
                st.audio(music_url)
        except Exception as e:
            st.error(f"Error playing audio: {e}")
    
    # Volume mixing if combining both
    if audio_type == "Combine Both":
        ambient_volume = st.slider("Ambient Sound Volume", 0.0, 1.0, 0.7, 0.1)
        music_volume = st.slider("Music Volume", 0.0, 1.0, 0.4, 0.1)
        
        if st.button("Mix Audio for Preview"):
            with st.spinner("Mixing audio..."):
                # Download audio files
                ambient_file = get_audio_file("Ambient Sounds", ambient_category, selected_ambient)
                music_file = get_audio_file("Classical Music", music_category, selected_music)
                
                if ambient_file and music_file:
                    # Mix the audio
                    mixed_file = os.path.join(AUDIO_DIR, "mixed_preview.mp3")
                    mix_audio(ambient_file, music_file, mixed_file, ambient_volume, music_volume)
                    
                    # Display mixed audio
                    st.success("Audio mixed successfully!")
                    st.audio(mixed_file)
    
    # Upload option
    if audio_type == "Upload Your Own":
        custom_audio = st.file_uploader("Upload Audio File (MP3, WAV)", type=["mp3", "wav"])
        if custom_audio is not None:
            st.audio(custom_audio)
            
    # Video testing section
    st.subheader("Test with a Video")
    
    # Upload a test video
    test_video = st.file_uploader("Upload a Test Video", type=["mp4"])
    if test_video is not None:
        # Save the uploaded video to a temporary file
        temp_video = "temp_test_video.mp4"
        with open(temp_video, "wb") as f:
            f.write(test_video.read())
        
        # Display the video
        st.video(temp_video)
        
        # Button to add audio to video
        if st.button("Add Audio to Video"):
            with st.spinner("Processing video with audio..."):
                try:
                    # Get the selected audio based on the type
                    if audio_type == "Ambient Sounds":
                        audio_file = get_audio_file("Ambient Sounds", st.session_state.ambient_category, st.session_state.ambient_sound)
                    elif audio_type == "Classical Music":
                        audio_file = get_audio_file("Classical Music", st.session_state.music_category, st.session_state.music_track)
                    elif audio_type == "Combine Both":
                        # Mix the audio first
                        ambient_file = get_audio_file("Ambient Sounds", st.session_state.ambient_category, st.session_state.ambient_sound)
                        music_file = get_audio_file("Classical Music", st.session_state.music_category, st.session_state.music_track)
                        audio_file = os.path.join(AUDIO_DIR, "mixed_audio.mp3")
                        mix_audio(ambient_file, music_file, audio_file, ambient_volume, music_volume)
                    elif audio_type == "Upload Your Own":
                        # Save the uploaded audio to a temporary file
                        audio_file = "temp_uploaded_audio.mp3"
                        with open(audio_file, "wb") as f:
                            f.write(custom_audio.read())
                    
                    # Add the audio to the video
                    output_path = "test_video_with_audio.mp4"
                    add_audio_to_video(temp_video, audio_file, output_path, loop_audio=True)
                    
                    # Display the result
                    st.success("Audio added to video successfully!")
                    st.video(output_path)
                except Exception as e:
                    st.error(f"Error processing video: {e}")

if __name__ == "__main__":
    audio_ui_test() 