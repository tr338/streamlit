# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import json
import pickle
import time
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint

from tqdm import tqdm
from PIL import Image
import cv2

from datasets import load_dataset
from torchvision import transforms, models

from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer, util

import kagglehub
import gdown

import streamlit as st
import speech_recognition as sr
import pygame
from gtts import gTTS



import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from transformers import BertModel, BertTokenizer

class VQAModel_trained(nn.Module):
    def __init__(self, num_answers):
        super(VQAModel_trained, self).__init__()
        self.cnn = models.resnet50(pretrained=True)  # Pre-trained ResNet-50
        self.cnn.fc = nn.Identity()  # Remove classification layer

        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.fc1 = nn.Linear(2048 + 768, 1024)
        self.fc2 = nn.Linear(1024, num_answers)
        self.dropout = nn.Dropout(0.5)

    def forward(self, images, input_ids, attention_mask):
        image_features = self.cnn(images)
        outputs = self.bert(input_ids, attention_mask)
        question_features = outputs.last_hidden_state[:, 0, :]
        combined_features = torch.cat((image_features, question_features), dim=1)
        x = self.fc1(combined_features)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load the model for inference
file_id = "1gcl7ugGFazUQX4THBekpUEcnlMCnBsug"
output_path = "best_model.pth"
device = torch.device('cpu')
gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)

# Load only the model weights
checkpoint = torch.load(output_path, map_location=device)
model_weights = checkpoint["model_state_dict"]  # Extract model weights

loaded_model = VQAModel_trained(num_answers=582)  
loaded_model.load_state_dict(model_weights)
loaded_model.to(device)

print("Model loaded successfully for inference! ✅")

# Load tokenizer and image transformations
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



st.title("Visual Question Answering (VQA) App")
st.subheader("Fully Voice-Controlled for Accessibility")

os.environ["SDL_AUDIODRIVER"] = "dsp"

if 'audio_initialized' not in st.session_state:
    try:
        pygame.mixer.init()
        st.session_state.audio_initialized = True
    except pygame.error as e:
        st.session_state.audio_initialized = False
        st.warning(f"⚠️ No audio device found. Audio playback may not work. ({e})")




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


def vqa_prediction(question, image_path):

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    

    inputs = tokenizer(question, return_tensors="pt", padding="max_length", truncation=True, max_length=30)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    with torch.no_grad():
        output = loaded_model(image, input_ids, attention_mask)
    
    predicted_idx = torch.argmax(output, dim=1).item()

    file_path = Path("answer_space.txt") 
    with file_path.open() as f:
        answer_space = f.read().splitlines()
    
    predicted_answer = answer_space[predicted_idx]

    print(f"Question: {question}")
    return predicted_answer


# # Process the answer (dummy function, replace with actual VQA model)
# def process_image_question(image, question):
#     # In a real implementation, this would call your VQA model
#     time.sleep(2)  # Simulate processing time
#     return vqa_prediction(question, image)

from io import BytesIO

def process_image_question(image, question):
    time.sleep(2)  # Simulate processing time

    # Convert the uploaded image to a file-like object
    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")  # Save image as PNG in memory
    image_bytes.seek(0)  # Move to the beginning of the file

    return vqa_prediction(question, image_bytes)  # Pass the file-like object





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
