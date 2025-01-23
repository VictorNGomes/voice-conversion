# MaskCycleGAN Voice Conversion App

## Overview
Voice conversion Streamlit app using MaskCycleGAN for transforming audio between different speakers.

## Prerequisites
- Python 3.8+
- PyTorch
- Streamlit
- torchaudio

## Installation
```bash
pip install streamlit torch torchaudio
```

## Dataset and Speakers
- Speakers: `famous`, `mozilla`
- Conversion directions: `mozilla` → `famous` and vice versa

## Audio Examples

### Example 1: Mozilla to Famous
- Source (Mozilla): (any voice)[https://github.com/VictorNGomes/voice-conversion/blob/main/temp_input.mp3]
- Converted (Famous): [/home/victor/voice-conversion-romario/converted_any_voice_to_romario.wav]


## Running the App
```bash
streamlit run voice_conversion_app.py
```

## Note
Replace `[...]` with actual audio file paths in your repository.
