# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import kagglehub
import torch
import numpy as np
import pandas as pd
import transformers
import os
from pathlib import Path
import gdown
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from torch import nn
from tqdm import tqdm
from datasets import load_dataset
from torchvision import transforms, models
from transformers import BertTokenizer
import kagglehub
import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer, util
import pickle
import streamlit as st


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



device = torch.device('cpu')





#trial

file_id = "1r5yrPh0oHsIxzEbdbq_y4ebJwzFPX68J"
output_path = "trained_model.pth"  # The file will be saved with this name


gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)

# Load the model
device = torch.device('cpu')
loaded_model = VQAModel_trained(num_answers=582)  # Make sure class is defined
loaded_model.load_state_dict(torch.load(output_path, map_location=device))
loaded_model.to(device)

#trial ends





#actulayy works


#loaded_model = VQAModel_trained(num_answers=582)  # Ensure the class is defined before loading
#loaded_model.load_state_dict(torch.load('C:/Users/Lenovo/Documents/DeployMLModel/trained_model.pth', map_location=torch.device('cpu')))
#loaded_model.to(torch.device("cpu"))  # Move model to CPU

##till here



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define image transformation (same as used in training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 🔹 **Sample Image and Question**

def vqa_prediction(question, image_path):
#image_path = "C:/Users/Lenovo/Downloads/image1004.png"  # Replace with an actual image path
#question = "What is shown in the image?"

# Preprocess the image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    
    # Tokenize the question
    inputs = tokenizer(question, return_tensors="pt", padding="max_length", truncation=True, max_length=30)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # 🔹 **Make the prediction**
    with torch.no_grad():
        output = loaded_model(image, input_ids, attention_mask)
    
    # Get predicted answer index
    predicted_idx = torch.argmax(output, dim=1).item()
    
    # 🔹 **Load Answer Space (answer labels)**
    file_path = Path("answer_space.txt") 
    with file_path.open() as f:
        answer_space = f.read().splitlines()
    
    #with open("C:/Users/Lenovo/Downloads/answer_space.txt") as f:
     #   answer_space = f.read().splitlines()
    
    predicted_answer = answer_space[predicted_idx]
    
    # 🔹 **Display the Result**
    print(f"Question: {question}")
    return predicted_answer
    


def main():
    st.title('VQA app ')
    que = st.text_input('Question')
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Open the image
        image = Image.open(uploaded_file)
    
        # Display the image
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
        # Define a save path
        save_path = os.path.join("uploads", uploaded_file.name)
    
        # Ensure the "uploads" directory exists
        os.makedirs("uploads", exist_ok=True)
    
        # Save the image
        image.save(save_path)

    preds = ''
    
    if st.button('vqa answer '):
        preds = vqa_prediction(que, save_path)
    
    st.success(preds)
    
if __name__ == '__main__':
    main()
