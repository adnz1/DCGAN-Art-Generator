# DCGAN-Art-Generator
A Python-based Deep Convolutional Generative Adversarial Network (DCGAN) that generates unique, AI-crafted images from random noise. Built with PyTorch, this project allows us to train on our datasets or experiment with pretrained weights to produce surreal and abstract digital art.
---

## Overview

This project enables users to:
- Select from different AI-generated art styles
- View results instantly in a browser
- Explore unique artworks generated from random noise

---

## Models Used

### **Abstract GAN** (`abstract_generator.pth`)
- **Dataset:** Abstract art images
- **Styles:** Produces bold, chaotic, and glitch-inspired visuals

### **Cherry Blossom GAN** (`cherry_generator.pth`)
- **Dataset:** Cherry blossom photographs
- **Styles:** Produces soft, dreamy, pastel-style visuals

---

## How to Run

```bash
# 1. Install Dependencies
pip install streamlit torch torchvision pillow matplotlib

# 2. Ensure File Setup
# Place all `.pth` and `.py` files in the same directory as `app.py`.

# 3. Run the App
streamlit run app.py

# 4. View in Browser
# Open your browser and go to:
# http://localhost:8501

   
