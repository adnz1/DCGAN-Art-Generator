import streamlit as st
import torch
import torchvision.utils as vutils
import numpy as np
import os
import random
from PIL import Image, ImageFilter, ImageEnhance
from abstract_generator_model import Generator as AbstractGenerator
from cherry_generator_model import Generator as CherryGenerator

# Constants
nz = 100
ngpu = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to load a generator from a .pth file
def load_generator(path, generator_class):
    model = generator_class(nz=nz, ngpu=ngpu).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

# Load models
abstract_G = load_generator("abstract_generator.pth", AbstractGenerator)
cherry_G = load_generator("cherry_generator.pth", CherryGenerator)

# Streamlit config
st.set_page_config(layout="wide", page_title="AI Art Generator")

# Sidebar Info
with st.sidebar:
    st.title("About the Project")
    st.write("""
    This AI-powered web app generates original art using two trained GAN models:

    **Abstract GAN**
    - Trained on abstract artwork
    - Produces bold, chaotic, and glitch-inspired visuals

    **Cherry Blossom GAN**
    - Trained on curated cherry blossom imagery
    - Produces soft, dreamy, pastel-style visuals

    Built using PyTorch + Streamlit.
    """)

# Title
st.title("AI Art Generator")
st.markdown("Select a style and generate original AI artwork.")

# Tabs
tab1, tab2 = st.tabs(["🎨 Generate", "🖼️ Gallery"])

if "gallery" not in st.session_state:
    st.session_state.gallery = []

with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        style = st.selectbox("Choose Style", ["Cherry Blossom", "Abstract"])
    with col2:
        st.markdown("###")
        generate = st.button("Generate Artwork")

    if generate:
        if style == "Cherry Blossom":
            generator = cherry_G
            noise = torch.randn(1, nz, 1, 1, device=device)  # soft noise
        else:
            generator = abstract_G
            noise = torch.randn(1, nz, 1, 1, device=device) * 1.5 + 1  # bold, glitchy

        with torch.no_grad():
            fake = generator(noise).detach().cpu()

        grid = vutils.make_grid(fake, padding=2, normalize=True)
        npimg = grid.mul(255).clamp(0, 255).byte().numpy()
        img = np.transpose(npimg, (1, 2, 0))
        img = Image.fromarray(img).resize((192, 192), Image.NEAREST)

        st.session_state.gallery.append((style, img))

with tab2:
    if st.session_state.gallery:
        st.markdown("### Gallery")
        cols = st.columns(3)
        for i, (label, image) in enumerate(st.session_state.gallery[::-1]):
            with cols[i % 3]:
                st.image(image, caption=label, width=192)
    else:
        st.info("No artwork generated yet. Go to the Generate tab to start.")

if st.button("Clear Gallery"):
    st.session_state.gallery.clear()
