import streamlit as st
import torch
import numpy as np
import noisereduce as nr
from TTS.api import TTS
from indictrans.inference.engine import Model
from pydub import AudioSegment
import io
import os
import gdown
import soundfile as sf

# --- Configuration ---
MODEL_CACHE = "./model_cache"
os.makedirs(MODEL_CACHE, exist_ok=True)

# Download translation model files
@st.cache_resource
def download_models():
    if not os.path.exists("indic-en"):
        url = "https://drive.google.com/drive/folders/1D9LZ7qGV8qsdPMPV5q2NSXYMhqSjl1xW?usp=sharing"
        gdown.download_folder(url, output="indic-en")

# --- Initialize Models ---
@st.cache_resource
def load_models():
    download_models()
    
    # Translation model
    trans_model = Model(expdir="indic-en", device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Voice cloning model (XTTS v2)
    tts = TTS(
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        progress_bar=False,
        gpu=torch.cuda.is_available()
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    return trans_model, tts

trans_model, tts = load_models()

# --- Audio Processing ---
def process_audio(uploaded_file, reduce_noise=True):
    # Load audio
    audio = AudioSegment.from_file(uploaded_file)
    
    # Convert to numpy array
    samples = np.array(audio.get_array_of_samples())
    sample_rate = audio.frame_rate
    
    # Noise reduction
    if reduce_noise and len(samples) > 0:
        try:
            samples_float = samples.astype(np.float32) / (2**15)
            noise_samples = samples_float[:int(0.5 * sample_rate)]
            
            cleaned = nr.reduce_noise(
                y=samples_float,
                y_noise=noise_samples,
                sr=sample_rate,
                prop_decrease=0.85
            )
            
            samples = (cleaned * (2**15)).astype(np.int16)
        except Exception as e:
            st.warning(f"Noise reduction failed: {str(e)}")
    
    # Create cleaned audio
    cleaned_audio = AudioSegment(
        samples.tobytes(),
        frame_rate=sample_rate,
        sample_width=audio.sample_width,
        channels=1
    ).set_frame_rate(16000)
    
    buffer = io.BytesIO()
    cleaned_audio.export(buffer, format="wav")
    return buffer

# --- Translation & Synthesis ---
def generate_voice(text, audio_ref, target_lang):
    try:
        # Translate to Hindi if needed
        if target_lang == "Hindi":
            translated = trans_model.translate_paragraph([text], "en", "hi")[0]
        else:
            translated = text
        
        # Generate speech
        temp_path = os.path.join(MODEL_CACHE, "output.wav")
        tts.tts_to_file(
            text=translated,
            speaker_wav=audio_ref,
            language="hi" if target_lang == "Hindi" else "en",
            file_path=temp_path
        )
        
        # Convert to MP3
        audio = AudioSegment.from_wav(temp_path)
        buffer = io.BytesIO()
        audio.export(buffer, format="mp3")
        return buffer.getvalue(), translated
    
    except Exception as e:
        st.error(f"Generation failed: {str(e)}")
        return None, None

# --- Streamlit UI ---
st.set_page_config(page_title="VoiceClone Pro", layout="wide")
st.title("🎙️ VoiceClone Pro")
st.markdown("⚠️ **Ethical Use**: Only clone voices with explicit permission")

# Sidebar Controls
with st.sidebar:
    st.header("Settings")
    target_lang = st.selectbox("Output Language", ["English", "Hindi"])
    reduce_noise = st.checkbox("Enable Noise Reduction", True)
    reference_audio = st.file_uploader("Upload Reference Voice", type=["wav", "mp3"])

# Main Interface
col1, col2 = st.columns([2, 1])
with col1:
    input_text = st.text_area("Input Text", height=200,
                            placeholder="Enter text to convert to speech...")

with col2:
    st.subheader("Audio Preview")
    if reference_audio:
        st.audio(reference_audio, format="audio/wav")

# Processing
if st.button("Generate Voice") and reference_audio and input_text:
    with st.spinner("Processing..."):
        try:
            processed_audio = process_audio(reference_audio, reduce_noise)
            audio_bytes, translated = generate_voice(input_text, processed_audio, target_lang)
            
            if audio_bytes:
                st.subheader("Output Audio")
                st.audio(audio_bytes, format="audio/mp3")
                
                st.download_button(
                    "Download MP3",
                    data=audio_bytes,
                    file_name="cloned_voice.mp3",
                    mime="audio/mp3"
                )
                
                if target_lang == "Hindi":
                    st.subheader("Hindi Translation")
                    st.write(translated)
        except Exception as e:
            st.error(f"Processing error: {str(e)}")
else:
    st.warning("Please upload reference audio and enter text")

# Technical Info
with st.expander("Technical Specifications"):
    st.write("""
    **System Requirements**:
    - Python 3.11+
    - CUDA 12.1 (recommended)
    - RAM: 4GB+ 
    
    **Processing Pipeline**:
    1. Audio preprocessing (noise reduction + resampling)
    2. Text translation (English ↔ Hindi)
    3. Voice cloning with XTTS v2
    4. MP3 conversion
    """)
