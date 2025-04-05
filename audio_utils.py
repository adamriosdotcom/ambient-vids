import os
import tempfile
import streamlit as st
import numpy as np
import pandas as pd
import ffmpeg
import requests
import glob
from io import BytesIO
from pathlib import Path

# Constants for audio resources
AUDIO_DIR = "audio_resources"
CLASSICAL_DIR = "100ClassicalMusicMasterpieces"

# Define categories for classical music based on mood/style
CLASSICAL_CATEGORIES = {
    "Calm": [
        "1825 Schubert - Ave Maria.mp3",
        "1875 Faure - Pavane.mp3",
        "1890 Debussy - Clair de Lune.mp3",
        "1888 Satie - Gymnopédie No.1.mp3",
        "1877 Saint-Saens - The Swan.mp3",
        "1894 Massenet - Meditation from Thais.mp3"
    ],
    "Melancholic": [
        "1827 Beethoven - Moonlight Sonata.mp3",
        "1838 Chopin - Nocturne Op. 9 No. 2.mp3",
        "1849 Chopin - Funeral March, Sonata No. 2.mp3",
        "1903 Sibelius - Valse Triste.mp3",
        "1899 Elgar - Nimrod from Enigma Variations.mp3",
        "1822 Schubert - Symphony No.8 in B minor, 'Unfinished'.mp3"
    ],
    "Uplifting": [
        "1723 Vivaldi - The Four Seasons - Spring.mp3",
        "1785 Mozart - Eine Kleine Nachtmusik.mp3",
        "1741 Handel - Water Music Suite No.2 in D.mp3",
        "1823 Beethoven - Symphony No. 9, 'Choral' - Ode to Joy.mp3",
        "1874 Johann Strauss II - The Blue Danube Waltz.mp3",
        "1778 Rondo Alla Turca, from Piano Sonata in A.mp3"
    ],
    "Dramatic": [
        "1870 Wagner- Ride of the Valkyries; from 'The Valkyrie'.mp3",
        "1916 Holst - Mars, from 'The Planets'.mp3",
        "1874 Mussorgsky - Night on a Bare Mountain.mp3",
        "1871 Grieg - In the Hall of the Mountain King.mp3",
        "1882 Tchaikovsky - 1812 Overture.mp3",
        "1798 Beethoven - Symphony No.5 in C minor - 1st movement.mp3"
    ]
}

# Find actual matching files in the classical directory
def find_matching_files():
    available_classical = {}
    
    # Check if classical directory exists
    if not os.path.exists(CLASSICAL_DIR):
        print(f"Warning: Classical music directory {CLASSICAL_DIR} not found")
        return available_classical
    
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
                for file in os.listdir(CLASSICAL_DIR):
                    if file.endswith('.mp3') and (
                        file == title or 
                        composer_title in file or 
                        all(word in file.lower() for word in composer_title.lower().split(' - ')[0:1])
                    ):
                        matches.append(file)
                
                if matches:
                    # Use the best match (exact match preferred)
                    best_match = matches[0]
                    if title in matches:
                        best_match = title
                    
                    # Extract just the composer and title for display
                    display_name = composer_title
                    file_path = os.path.join(CLASSICAL_DIR, best_match)
                    available_classical[category][display_name] = file_path
    
    # If any category is empty, fill with some default files
    for category in CLASSICAL_CATEGORIES.keys():
        if not available_classical.get(category, {}):
            available_classical[category] = {}
            # Just take the first 5 mp3 files we can find
            mp3_files = glob.glob(os.path.join(CLASSICAL_DIR, "*.mp3"))
            for i, file in enumerate(mp3_files[:5]):
                filename = os.path.basename(file)
                # Try to extract composer/title
                if " - " in filename:
                    display_name = filename.split(" ", 1)[1]
                else:
                    display_name = filename
                available_classical[category][display_name] = file
    
    return available_classical

# Placeholder ambient sounds - these will need to be replaced with actual files
AMBIENT_SOUNDS = {
    "Fireplace": {
        "Crackling Fire": None,
        "Fireplace with Wind": None,
        "Cozy Evening Fire": None
    },
    "Rain": {
        "Gentle Rain": None,
        "Thunderstorm": None,
        "Rain on Window": None
    },
    "Forest": {
        "Forest Ambience": None,
        "Bird Chirping": None,
        "Woodland Stream": None
    },
    "Ocean": {
        "Ocean Waves": None,
        "Calm Sea": None,
        "Beach Ambience": None
    }
}

# Initialize the available classical music files
CLASSICAL_MUSIC = find_matching_files()

def ensure_audio_dir():
    """Creates the audio resources directory if it doesn't exist"""
    os.makedirs(AUDIO_DIR, exist_ok=True)

def get_audio_file(audio_type, category, name):
    """Gets the path to an audio file"""
    if audio_type == "Ambient Sounds":
        # For ambient sounds, we just return a placeholder for now
        # In a real implementation, you would have local ambient sound files
        st.warning("Ambient sounds are not yet available.")
        return None
    elif audio_type == "Classical Music":
        # Get the path to the selected classical music file
        if category in CLASSICAL_MUSIC and name in CLASSICAL_MUSIC[category]:
            return CLASSICAL_MUSIC[category][name]
        st.warning(f"Selected music {name} not found.")
        return None
    return None

def mix_audio(ambient_path, music_path, output_path, ambient_volume=0.7, music_volume=0.4):
    """Mixes ambient sound with music at the specified volumes"""
    try:
        if not ambient_path or not os.path.exists(ambient_path):
            st.error("Ambient sound file not found.")
            return None
            
        if not music_path or not os.path.exists(music_path):
            st.error("Music file not found.")
            return None
            
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
            
            # Use filter_complex to loop audio instead of concat
            loop_filter = f"aloop=loop={loop_count-1}:size=2e+09"
            
            (
                ffmpeg
                .input(audio_path)
                .filter(loop_filter)
                .output(temp_audio, to=str(video_duration))
                .overwrite_output()
                .run(quiet=True)
            )
            
            # Now add the looped audio to the video
            (
                ffmpeg
                .input(video_path)
                .input(temp_audio)
                .output(output_path, vcodec='copy', acodec='aac', shortest=None)
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
                .output(output_path, vcodec='copy', acodec='aac', shortest=None)
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
    
    # Display debug information
    st.write("This is a testing interface for the audio feature.")
    
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
    
    # Check if we have classical music files
    if not any(CLASSICAL_MUSIC.values()):
        st.error(f"No classical music files found in {CLASSICAL_DIR}")
    else:
        st.success(f"Found {sum(len(cat) for cat in CLASSICAL_MUSIC.values())} classical music files")
        
        # Display some sample info
        for category, tracks in CLASSICAL_MUSIC.items():
            if tracks:
                st.subheader(f"{category} Music")
                for title, path in list(tracks.items())[:3]:  # Show the first 3 tracks in each category
                    st.write(f"- {title}")
    
    # Audio type selection
    audio_type = st.selectbox("Audio Type", 
                             ["Classical Music", "Ambient Sounds (Coming Soon)", "Upload Your Own"])
    
    # Initialize session state variables for audio
    if 'music_category' not in st.session_state:
        st.session_state.music_category = None
    if 'music_track' not in st.session_state:
        st.session_state.music_track = None
    if 'audio_files' not in st.session_state:
        st.session_state.audio_files = []
    
    # Classical music options
    if audio_type == "Classical Music":
        music_category = st.selectbox("Music Mood", 
                                    list(CLASSICAL_MUSIC.keys()))
        st.session_state.music_category = music_category
        
        # Display music options based on mood
        if CLASSICAL_MUSIC[music_category]:
            music_options = CLASSICAL_MUSIC[music_category]
            selected_music = st.selectbox("Choose Classical Track", 
                                      list(music_options.keys()))
            st.session_state.music_track = selected_music
            
            # Display selected track for playback
            music_file = music_options[selected_music]
            if music_file and os.path.exists(music_file):
                st.success(f"Playing: {selected_music}")
                st.audio(music_file)
            else:
                st.error(f"File not found: {music_file}")
        else:
            st.warning(f"No tracks available for {music_category}")
    
    # Ambient sounds options (coming soon)
    elif audio_type == "Ambient Sounds (Coming Soon)":
        st.info("Ambient sounds are coming soon! For now, you can use classical music or upload your own sounds.")
    
    # Upload option
    elif audio_type == "Upload Your Own":
        custom_audio = st.file_uploader("Upload Audio File (MP3, WAV)", type=["mp3", "wav"])
        if custom_audio is not None:
            # Save uploaded file to disk
            temp_path = f"uploaded_{custom_audio.name}"
            with open(temp_path, "wb") as f:
                f.write(custom_audio.read())
            st.session_state.custom_audio_path = temp_path
            st.success("Audio uploaded successfully!")
            st.audio(temp_path)
    
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
                    if audio_type == "Classical Music":
                        audio_file = get_audio_file("Classical Music", st.session_state.music_category, st.session_state.music_track)
                    elif audio_type == "Upload Your Own":
                        audio_file = st.session_state.custom_audio_path
                    else:
                        st.warning("Please select Classical Music or Upload Your Own audio first.")
                        audio_file = None
                    
                    if audio_file and os.path.exists(audio_file):
                        # Add the audio to the video
                        output_path = "test_video_with_audio.mp4"
                        add_audio_to_video(temp_video, audio_file, output_path, loop_audio=True)
                        
                        # Display the result
                        st.success("Audio added to video successfully!")
                        st.video(output_path)
                    else:
                        st.error("Audio file not available or could not be loaded.")
                except Exception as e:
                    st.error(f"Error processing video: {e}")

if __name__ == "__main__":
    audio_ui_test() 