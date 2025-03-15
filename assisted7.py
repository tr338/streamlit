import streamlit as st
from PIL import Image
import speech_recognition as sr
import time
import cv2
import numpy as np
import os
import threading
from gtts import gTTS

st.title("Visual Question Answering (VQA) App")

# Function to speak text using gTTS instead of pyttsx3
def speak_text(text):
    """Convert text to speech using gTTS instead of pyttsx3."""
    try:
        # Create a temporary audio file
        tts = gTTS(text=text, lang='en')
        temp_file = "temp_speech.mp3"
        tts.save(temp_file)
        
        # Play the audio file
        os.system(f"start {temp_file}")  # For Windows
        # For Linux: os.system(f"mpg123 {temp_file}")
        # For Mac: os.system(f"afplay {temp_file}")
        
        time.sleep(len(text.split()) * 0.3)  # Approximate time to speak the text
        
    except Exception as e:
        st.warning(f"TTS error: {e}. Continuing without speech.")

# Detect ANY speech to trigger image capture
def wait_for_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Say anything to start capturing the image...")
        speak_text("Say anything to start capturing the image.")
        recognizer.adjust_for_ambient_noise(source, duration=2)
        try:
            st.info("Listening... Speak now!")
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=3)
            text = recognizer.recognize_google(audio).lower()
            st.write(f"Detected Speech: {text}")  # Debugging: Show detected speech
            return True  # Move to image capture if any speech is detected
        except sr.WaitTimeoutError:
            st.error("Timeout: No speech detected.")
        except sr.UnknownValueError:
            st.error("Could not understand audio.")
        except sr.RequestError:
            st.error("Speech recognition service error.")
    return False  # Do not proceed if no speech is detected

# Capture image using OpenCV
def capture_image():
    st.success("Speech detected! Capturing image in 3 seconds...")
    speak_text("Speech detected. Capturing image in three seconds.")
    time.sleep(3)
    cap = cv2.VideoCapture(0)  # Open the default camera (0)
    if not cap.isOpened():
        st.error("Could not open the camera. Please check your webcam.")
        return None
    ret, frame = cap.read()
    cap.release()  # Release the camera
    if not ret:
        st.error("Failed to capture image. Please try again.")
        return None
    # Convert OpenCV image (BGR) to PIL Image (RGB)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)
    st.image(pil_image, caption="Captured Image", use_column_width=True)
    speak_text("Image captured successfully.")
    return pil_image

# Function to capture speech input for question
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Adjusting for ambient noise... Please wait")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        st.info("Listening... Speak now!")
        speak_text("Listening, please speak now.")
        try:
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=5)
            text = recognizer.recognize_google(audio)
            return text
        except sr.WaitTimeoutError:
            return "Listening timed out. Please try again."
        except sr.UnknownValueError:
            return "Could not understand audio."
        except sr.RequestError:
            return "Speech recognition service unavailable."

# Initialize session state for app flow management
if 'app_stage' not in st.session_state:
    st.session_state.app_stage = 'start'
    st.session_state.captured_image = None
    st.session_state.question = None

# Main app flow logic
if st.session_state.app_stage == 'start':
    if st.button("Start VQA App"):
        st.session_state.app_stage = 'listen'
        st.rerun()

elif st.session_state.app_stage == 'listen':
    if wait_for_speech():
        img = capture_image()
        if img is not None:
            st.session_state.captured_image = img
            st.session_state.app_stage = 'question'
            st.rerun()
        else:
            st.error("Failed to capture image. Please try again.")
            if st.button("Try Again"):
                st.rerun()

elif st.session_state.app_stage == 'question':
    st.image(st.session_state.captured_image, caption="Captured Image", use_column_width=True)
    st.session_state.question = recognize_speech()
    st.text_input("Recognized Question:", value=st.session_state.question, disabled=True)
    
    speak_text("Processing your request...")
    time.sleep(2)
    answer = "This is a dummy answer. In a real implementation, you would process the image and question with a VQA model here."
    st.write("### Answer:", answer)
    speak_text(answer)
    
    st.session_state.app_stage = 'result'
    st.rerun()

elif st.session_state.app_stage == 'result':
    st.image(st.session_state.captured_image, caption="Captured Image", use_column_width=True)
    st.text_input("Your Question:", value=st.session_state.question, disabled=True)
    st.write("### Answer:", "This is a dummy answer. In a real implementation, you would process the image and question with a VQA model here.")
    
    if st.button("Ask Another Question"):
        st.session_state.app_stage = 'start'
        st.session_state.captured_image = None
        st.session_state.question = None
        st.rerun()