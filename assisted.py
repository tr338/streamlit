import streamlit as st
from PIL import Image
import speech_recognition as sr
import pyttsx3
import time
import cv2
import numpy as np

st.title("Visual Question Answering (VQA) App")

# Initialize text-to-speech engine
engine = pyttsx3.init()

def speak_text(text):
    """Convert text to speech."""
    engine.say(text)
    engine.runAndWait()
    time.sleep(1)  # Pause before listening

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

# Start process
if wait_for_speech():
    captured_image = capture_image()
    if not captured_image:
        st.warning("Retry capturing the image.")

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

# Speech input after capturing image
st.info("Please ask your question after the beep...")
speak_text("Please ask your question after the beep.")
question = recognize_speech()
st.text_input("Recognized Question:", value=question, disabled=True)

# Dummy Output for Testing
speak_text("Processing your request...")
time.sleep(2)
answer = "This is a dummy answer."  # Placeholder answer for testing
st.write("### Answer:", answer)
speak_text(answer)
