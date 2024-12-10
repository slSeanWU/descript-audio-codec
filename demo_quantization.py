import streamlit as st
from pathlib import Path

# Configuration
st.set_page_config(page_title="Audio Model Comparison", layout="centered")

# Audio Model Data
model_data = {
    "Original Model": {
        "size": "306.52 MB",
        "speech_audio": "audio_demo/baseline_speech_0009.wav",
        "music_audio": "audio_demo/baseline_music_0000.wav",
    },
    "Quantized Model (Entire Enc + downsample) + (1st & 2nd DecBlocks)": {
        "size": "195.96 MB",
        "speech_audio": "audio_demo/quantized_speech_0009.wav",
        "music_audio": "audio_demo/quantized_music_0000.wav",
    },
}

# Title
st.title("Audio Model Comparison")
st.write("Compare the original and quantized audio models in terms of size and audio quality.")

# Audio Clips Directory
audio_dir = Path("audio_demo")

for model_name, details in model_data.items():
    st.subheader(model_name)
    st.write(f"**Model Size:** {details['size']}")

    # Speech Audio Button
    st.write("Speech Clip:")
    speech_audio_path = details["speech_audio"]  # Directly use the path
    if Path(speech_audio_path).exists():
        st.audio(speech_audio_path, format="audio/wav")
    else:
        st.warning(f"Speech audio file missing for {model_name}!")

    # Music Audio Button
    st.write("Music Clip:")
    music_audio_path = details["music_audio"]  # Directly use the path
    if Path(music_audio_path).exists():
        st.audio(music_audio_path, format="audio/wav")
    else:
        st.warning(f"Music audio file missing for {model_name}!")

st.write("---")
st.write("This demo showcases how model quantization can significantly reduce size while maintaining audio quality.")
