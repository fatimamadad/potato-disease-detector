# 🥔 PotatoDetect - AI-Powered Potato Disease Detection

A complete, production-ready web application that uses AI to detect potato diseases from leaf photos and provides expert treatment recommendations.

---

## 📋 Table of Contents

- [About the Project](#about-the-project)
- [Live Demo](#live-demo)
- [Features](#features)
- [AI Feature Details](#ai-feature-details)
- [Tech Stack](#tech-stack)
- [Screenshots](#screenshots)
- [How to Run Locally](#how-to-run-locally)
- [Environment Variables](#environment-variables)
- [Project Structure](#project-structure)
- [License](#license)

---

## 🎯 About the Project

### The Problem
Potato farmers face significant crop losses due to undetected diseases like Early Blight and Late Blight. Traditional diagnosis requires agricultural experts, which are often unavailable in rural areas. By the time farmers notice symptoms, the disease has already spread.

### Our Solution
**PotatoDetect** is an AI-powered web application that allows farmers to instantly identify potato diseases by uploading a photo of the affected leaf. The app provides 98% accurate detection with detailed treatment recommendations.

### Who It's For
- Small-scale potato farmers
- Commercial growers
- Agricultural students
- Home gardeners
- Agricultural extension officers

---

## 🌐 Live Demo

**URL:** [https://potato-disease-detector-c1gb.vercel.app/](https://potato-disease-detector-c1gb.vercel.app/)

---

## ✨ Features

### Disease Detection
- Upload potato leaf photos (PNG, JPG, JPEG)
- AI-powered diagnosis with 98% accuracy
- Real-time results with confidence scores
- Detects: Early Blight, Late Blight, Healthy
- Detailed treatment recommendations
- Drag and drop support

### AI Chat Assistant
- 24/7 expert guidance on potato diseases
- Natural language conversations
- Context-aware responses
- Direct image upload within chat
- Treatment planning assistance

### User Experience
- Mobile-responsive design
- Modern, clean interface
- Real-time feedback with loading states
- Comprehensive disease information
- Testimonials from real farmers

---

## 🤖 AI Feature Details

### Disease Detection Model
- **Model:** TensorFlow Lite (ResNet50-based)
- **Training Data:** 3,000+ potato leaf images
- **Accuracy:** 98%
- **Inference Time:** < 500ms
- **Model Source:** Hugging Face Hub

**How it works:**
1. User uploads a potato leaf image
2. Image is preprocessed (resized to 224x224)
3. Model runs inference using TensorFlow Lite
4. Results returned with disease class and confidence
5. Custom treatment recommendations shown

### Groq AI Chat Assistant
- **Model:** Llama 3.1-8B-Instruct
- **Provider:** Groq API

---

## 🛠️ Tech Stack

### Frontend
- HTML5, CSS3, JavaScript (ES6)
- Font Awesome Icons
- Google Fonts (Inter)
- Responsive Design

### Backend
- Python 3.9+
- Flask Framework
- TensorFlow Lite
- Pillow, NumPy

### APIs & Services
- Groq API (Llama 3.1)
- Hugging Face Hub
- REST API

### Deployment
- Vercel 
- GitHub

---


---

