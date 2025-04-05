# Ambience Video Creator

A user-friendly application for creating looping ambience videos. This tool leverages advanced AI models to generate beautiful, seamless ambience videos from simple text prompts.

## Features

- Text-to-video generation with optimized prompts
- Multiple options at each stage of creation
- Seamless video transitions with crossfades
- Custom loop duration settings
- Simple, intuitive interface

## Requirements

- Python 3.8+
- Required packages (install with `pip install -r requirements.txt`):
  - streamlit
  - opencv-python
  - numpy
  - pandas
  - python-dotenv
  - replicate
  - ffmpeg-python
  - torch
  - pillow

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with your Replicate API token:
   ```
   REPLICATE_API_TOKEN=your_token_here
   ```
4. Run the application: `streamlit run app.py`

## Usage

1. Enter a scene description prompt
2. Follow the step-by-step interface to:
   - Select your preferred optimized prompt
   - Choose your favorite generated image
   - Select from video options
   - Continue adding video segments as desired
   - Specify final video duration
3. Download your completed ambience video

## How It Works

This application orchestrates several AI models through the Replicate API:
- Claude 3.7 Sonnet for prompt optimization
- Google Imagen 3 for image generation
- Kling v1.6 Pro for video generation
- Recraft Crisp for image upscaling

The underlying functionality is implemented in the workflow-v7.py script, with this UI providing a user-friendly interface to the process. 