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
        "Crackling Fire": "https://freesound.org/data/previews/347/347172_5121236-lq.mp3",
        "Fireplace with Wind": "https://freesound.org/data/previews/362/362042_5833740-lq.mp3",
        "Cozy Evening Fire": "https://freesound.org/data/previews/195/195243_1038806-lq.mp3"
    },
    "Rain": {
        "Gentle Rain": "https://freesound.org/data/previews/169/169255_2963332-lq.mp3",
        "Thunderstorm": "https://freesound.org/data/previews/435/435752_6142149-lq.mp3",
        "Rain on Window": "https://freesound.org/data/previews/258/258557_4631955-lq.mp3"
    },
    "Forest": {
        "Forest Ambience": "https://freesound.org/data/previews/459/459997_4766846-lq.mp3",
        "Bird Chirping": "https://freesound.org/data/previews/501/501948_7587775-lq.mp3",
        "Woodland Stream": "https://freesound.org/data/previews/163/163578_2858355-lq.mp3"
    },
    "Ocean": {
        "Ocean Waves": "https://freesound.org/data/previews/328/328298_5629541-lq.mp3",
        "Calm Sea": "https://freesound.org/data/previews/187/187404_1979597-lq.mp3", 
        "Beach Ambience": "https://freesound.org/data/previews/178/178655_1648170-lq.mp3"
    }
}

CLASSICAL_MUSIC = {
    "Calm": {
        "Gymnopédie No.1 (Erik Satie)": "https://www.dropbox.com/scl/fi/g4h8k7pji2e7jytdhvcm7/erik-satie-gymnopedie-no-1.mp3?rlkey=lzzgxpjc6aahxm87w3qjbxl0o&dl=1",
        "Clair de Lune (Debussy)": "https://www.dropbox.com/scl/fi/fmxgjpgdxyww5iazklxri/debussy-clair-de-lune.mp3?rlkey=vl9kxjijlb5btfaxbz4y12jzj&dl=1",
        "Canon in D (Pachelbel)": "https://www.dropbox.com/scl/fi/2bnjc1gysb8y2f2zk0i66/pachelbel-canon-in-d.mp3?rlkey=y3a0nzh3e6vb6oxupdqr2m3ga&dl=1"
    },
    "Melancholic": {
        "Moonlight Sonata (Beethoven)": "https://www.dropbox.com/scl/fi/s96jjrptgr5t14hdl4rzg/beethoven-moonlight-sonata-1st-mvt.mp3?rlkey=jnk7vt1c6uxn4zzwbx49cqfgn&dl=1",
        "Prelude in E-Minor (Chopin)": "https://www.dropbox.com/scl/fi/8i4wr7xrjw0a04ijsbnhc/chopin-prelude-e-minor.mp3?rlkey=n12sehcyiupp8hj47cjyxzlsm&dl=1",
        "Adagio for Strings (Barber)": "https://www.dropbox.com/scl/fi/j0vij5m31nhhd1d7pfnre/barber-adagio-for-strings.mp3?rlkey=rg6k3yhcwuty55a3t8i8vynfd&dl=1"
    },
    "Uplifting": {
        "Spring (Vivaldi)": "https://www.dropbox.com/scl/fi/o5i02rzmzawp2q3gqclud/vivaldi-spring-allegro.mp3?rlkey=ihu5mf1jy6cvwnzgx0r1e5wd4&dl=1",
        "Ode to Joy (Beethoven)": "https://www.dropbox.com/scl/fi/qgm8eelgjuv7y5laxk2c5/beethoven-ode-to-joy.mp3?rlkey=0dhtgj9l5hnnfkx80sgj7w5r0&dl=1",
        "Morning Mood (Grieg)": "https://www.dropbox.com/scl/fi/vthaqwnj1t9ogltokwl6d/grieg-morning-mood.mp3?rlkey=q9hrrjzg9t5prmaxfmyxvl5p1&dl=1"
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
        
        # Download the file
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            return file_path
        else:
            raise Exception(f"Failed to download audio: {response.status_code}")
    except Exception as e:
        st.error(f"Error downloading audio: {e}")
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

# Streamlit interface for testing
def audio_ui_test():
    st.title("Ambience Audio Selector")
    
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
        
        # Download and display audio for preview
        ambient_url = ambient_options[selected_ambient]
        st.audio(ambient_url)
    
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
        
        # Download and display music for preview
        music_url = music_options[selected_music]
        st.audio(music_url)
    
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