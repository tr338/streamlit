import streamlit as st
from PIL import Image
import speech_recognition as sr
import time
import cv2
import numpy as np
import pygame
from gtts import gTTS
import os
import tempfile

st.title("Visual Question Answering (VQA) App")
st.subheader("Fully Voice-Controlled for Accessibility")

# Initialize pygame mixer for audio playback
if 'audio_initialized' not in st.session_state:
    pygame.mixer.init()
    st.session_state.audio_initialized = True

# Improved function to speak text using gTTS and pygame
def speak_text(text):
    """Convert text to speech using gTTS and pygame for better control."""
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            temp_filename = temp_file.name
        
        # Generate speech
        tts = gTTS(text=text, lang='en')
        tts.save(temp_filename)
        
        # Wait for any current playback to finish
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        
        # Play the audio file
        pygame.mixer.music.load(temp_filename)
        pygame.mixer.music.play()
        
        # Wait for playback to complete
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        
        # Clean up the temporary file
        try:
            os.remove(temp_filename)
        except:
            pass  # Ignore errors during cleanup
            
    except Exception as e:
        st.warning(f"TTS error: {e}. Continuing without speech.")

# Detect speech to trigger actions
def listen_for_speech(prompt_text, timeout=10, phrase_time=5):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info(prompt_text)
        speak_text(prompt_text)
        recognizer.adjust_for_ambient_noise(source, duration=1)
        try:
            st.info("Listening... Speak now!")
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time)
            text = recognizer.recognize_google(audio).lower()
            st.write(f"Detected Speech: {text}")
            return text
        except sr.WaitTimeoutError:
            st.error("Timeout: No speech detected.")
            speak_text("No speech detected. Let's try again.")
            return None
        except sr.UnknownValueError:
            st.error("Could not understand audio.")
            speak_text("I couldn't understand that. Let's try again.")
            return None
        except sr.RequestError:
            st.error("Speech recognition service error.")
            speak_text("There was a problem with the speech recognition service. Let's try again.")
            return None

# Capture image using OpenCV
def capture_image():
    st.success("Capturing image in 3 seconds...")
    speak_text("Capturing image in three, two, one.")
    time.sleep(3)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open the camera.")
        speak_text("Could not open the camera. Please check your webcam.")
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret:
        st.error("Failed to capture image.")
        speak_text("Failed to capture image. Let's try again.")
        return None
    
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)
    st.image(pil_image, caption="Captured Image", use_column_width=True)
    speak_text("Image captured successfully.")
    return pil_image

# Process the answer (dummy function, replace with actual VQA model)
def process_image_question(image, question):
    # In a real implementation, this would call your VQA model
    time.sleep(2)  # Simulate processing time
    return "This is a dummy answer. In a real implementation, you would process the image and question with a VQA model here."

# Initialize session state
if 'app_stage' not in st.session_state:
    st.session_state.app_stage = 'welcome'
    st.session_state.captured_image = None
    st.session_state.last_answer = None

# Welcome message only on first load
if st.session_state.app_stage == 'welcome':
    welcome_msg = "Welcome to the Visual Question Answering App. This app is fully voice-controlled to assist blind users. Say anything when you're ready to begin."
    st.info(welcome_msg)
    speak_text(welcome_msg)
    st.session_state.app_stage = 'start'

# Main app flow
if st.session_state.app_stage == 'start':
    # Listen for any speech to begin
    speech = listen_for_speech("Say anything to start capturing an image.")
    if speech is not None:
        # Capture image
        img = capture_image()
        if img is not None:
            st.session_state.captured_image = img
            st.session_state.app_stage = 'question'
            st.rerun()
        else:
            speak_text("Let's try capturing the image again.")
            time.sleep(1)
            st.rerun()

elif st.session_state.app_stage == 'question':
    # Display captured image
    if st.session_state.captured_image:
        st.image(st.session_state.captured_image, caption="Captured Image", use_column_width=True)
    
    # Get question from speech
    question = listen_for_speech("Please ask your question about the image.")
    if question is not None:
        st.text_input("Your Question:", value=question, disabled=True)
        
        # Process and speak the answer
        speak_text("Processing your question. Please wait.")
        answer = process_image_question(st.session_state.captured_image, question)
        st.session_state.last_answer = answer
        st.write("### Answer:", answer)
        
        # Make sure to speak the answer clearly
        speak_text(f"Here is the answer to your question: {answer}")
        
        # Add a short delay to ensure the answer is fully spoken
        time.sleep(1)
        
        # After answering, move to next action
        st.session_state.app_stage = 'next_action'
        st.rerun()

elif st.session_state.app_stage == 'next_action':
    # Display the captured image and last answer for continuity
    if st.session_state.captured_image:
        st.image(st.session_state.captured_image, caption="Captured Image", use_column_width=True)
    
    if st.session_state.last_answer:
        st.write("### Last Answer:", st.session_state.last_answer)
    
    # If this is the first time in this stage, speak the answer again to ensure it's heard
    if 'answer_repeated' not in st.session_state or not st.session_state.answer_repeated:
        speak_text(f"To repeat the answer: {st.session_state.last_answer}")
        st.session_state.answer_repeated = True
    
    # Ask what to do next
    next_prompt = "What would you like to do next? Say 'new image' to capture a new image, 'new question' to ask another question about this image, or 'exit' to finish."
    next_action = listen_for_speech(next_prompt)
    
    if next_action is not None:
        if "new image" in next_action or "capture" in next_action:
            speak_text("Starting a new session with a new image.")
            st.session_state.app_stage = 'start'
            st.session_state.captured_image = None
            st.session_state.last_answer = None
            st.session_state.answer_repeated = False
            st.rerun()
        elif "new question" in next_action or "another question" in next_action or "ask again" in next_action:
            speak_text("OK, let's ask another question about this image.")
            st.session_state.app_stage = 'question'
            st.session_state.answer_repeated = False
            st.rerun()
        elif "exit" in next_action or "quit" in next_action or "finish" in next_action:
            speak_text("Thank you for using the Visual Question Answering App. Goodbye!")
            st.success("Session ended. Refresh the page to start again.")
            # No rerun here to end the session
        else:
            speak_text("I didn't understand that command. Let's try again.")
            st.session_state.answer_repeated = True  # Don't repeat the answer again
            st.rerun()