"""Streamlit App
"""
import streamlit as st

import config
from deploy.inference import Inference
from models import models

st.set_page_config("VAE_GAN")
st.title("Variational AutoEncoder (VAE) and Generative Adversarial Network (GAN)")

mnist = str(config.DEPLOY_PATH / "images" / "mnist.png")
fmnist = str(config.DEPLOY_PATH / "images" / "fmnist.png")
cifar = str(config.DEPLOY_PATH / "images" / "cifar10.png")

st.image(
    [mnist, fmnist, cifar], width=220, caption=["Mnist", "Fashion Mnist", "CIFAR-10"]
)

inferer = Inference()

dataset = st.selectbox("Dataset", ["mnist", "fmnist", "cifar10"])
hidden_size = st.radio("Hidden Size", [2, 32, 64, 128])
model_type = st.radio("Model", models.keys())

class_no = st.selectbox("Digit", list(range(10)))

target_img, recon_img = inferer.reconstruction(
    class_no, model_type, hidden_size, dataset
)

st.image(
    [
        target_img.permute(1, 2, 0).detach().numpy(),
        recon_img.permute(1, 2, 0).detach().numpy(),
    ],
    caption=["Original Image", "Reconstructed Image"],
)

# streamlit run /src/deploy/streamlit_app.py
