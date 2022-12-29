"""Streamlit App
"""
import streamlit as st
from torchvision.utils import make_grid

import config
from dataloaders import dataloaders
from deploy.inference import InferVAE

st.set_page_config("VAE_GAN")


@st.cache(allow_output_mutation=True)
def initialize_inferer():
    """Initialize inference module

    Returns:
        Callable: Inferencen Module
    """
    return InferVAE()


inferer = initialize_inferer()

data_imgs = {
    "mnist": str(config.DEPLOY_PATH / "images" / "mnist.png"),
    "fmnist": str(config.DEPLOY_PATH / "images" / "fmnist.png"),
    "cifar10": str(config.DEPLOY_PATH / "images" / "cifar10.png"),
}

st.title("VAE and GAN on MNIST, Fashion MNIST and CIFAR-10")

with st.sidebar:
    st.header("Settings")
    dataset = st.selectbox("Dataset", ["mnist", "fmnist", "cifar10"])
    _, input_size = dataloaders[dataset]
    with st.expander("See sample images"):
        st.image(data_imgs[dataset])
    hidden_size = st.radio("Hidden Size", [2, 16, 64])

tab1, tab2, tab3 = st.tabs(
    ["Image Generation", "Image Reconstruction", "Image Interpolation"]
)

### Generation
with tab1:
    st.header("Image Generation")
    st.markdown(
        "Generate a random sample from Standard Gaussian Distribution from the Latent Space"
    )
    tab1_1, tab1_2, tab1_3 = st.tabs(
        ["Random Generation", "Class Generation", "Batch Generation"]
    )

    with tab1_1:  # Random Generation
        model1_1 = st.radio(
            "Random model",
            ["BaseVAE", "DeepVAE", "ConvVAE", "BaseCVAE", "DeepCVAE", "ConvCVAE"],
            horizontal=True,
        )
        with st.form("Image Gen Form"):
            gen_random_1 = st.form_submit_button("Generate Image")
            if gen_random_1:
                recon_img, z = inferer.generate_image(model1_1, hidden_size, dataset)
                st.image(
                    recon_img.permute(1, 2, 0).detach().numpy(),
                    width=400,
                    caption="Reconstructed Image",
                )
                st.write(z)

    with tab1_2:  # Conditional Generation
        col_1_2_1, col_1_2_2 = st.columns([4, 6], gap="large")
        with col_1_2_1:
            class_name_1_2 = st.selectbox(
                "Cond Gen Class", config.CLASSES[dataset].keys()
            )
            class_idx_1_2 = config.CLASSES[dataset][class_name_1_2]
        with col_1_2_2:
            model1_2 = st.radio(
                "Conditional model",
                ["BaseCVAE", "DeepCVAE", "ConvCVAE"],
                horizontal=True,
            )
        with st.form("Conditional Image Gen Form"):
            gen_random_2 = st.form_submit_button("Generate Image")
            if gen_random_2:
                imgs, _ = inferer.generate_image(
                    model1_2, hidden_size, dataset, [class_idx_1_2] * 16
                )
                cond_img_grid = make_grid(imgs, nrow=4)
                st.image(
                    cond_img_grid.permute(1, 2, 0).detach().numpy(),
                    width=650,
                    caption=f"Batch of {class_name_1_2} Images",
                )

    with tab1_3:  # Batch Generation
        model1_3 = st.radio(
            "Batch Generation model",
            ["BaseCVAE", "DeepCVAE", "ConvCVAE"],
            horizontal=True,
        )
        with st.form("Batch Image Gen Form"):
            gen_random_3 = st.form_submit_button("Generate Batch Image")
            if gen_random_3:
                batch_img_grid = inferer.generate_batch(
                    model1_3, hidden_size, dataset, num_images=8
                )
                st.image(
                    batch_img_grid.permute(1, 2, 0).detach().numpy(),
                    width=650,
                    caption="Batch of images for all classes",
                )

### Reconstruction
with tab2:
    st.header("Image Reconstruction")
    recon_class_name = st.selectbox("Class", config.CLASSES[dataset].keys())
    recon_class_idx = config.CLASSES[dataset][recon_class_name]
    model2 = st.radio(
        "Recon model",
        ["BaseVAE", "DeepVAE", "ConvVAE", "BaseCVAE", "DeepCVAE", "ConvCVAE"],
        horizontal=True,
    )
    target_col, recon_col = st.columns(2)

    with st.form("Image Recon Form"):
        sample_img_recon = st.form_submit_button("Reconstruct Image")
        if sample_img_recon:
            with target_col:
                target_img = inferer.sample_image(recon_class_idx, dataset)
                st.image(
                    target_img.permute(1, 2, 0).detach().numpy(),
                    width=300,
                    caption="Original Image",
                )

            with recon_col:
                recon_img = inferer.reconstruction(
                    target_img, model2, hidden_size, dataset, recon_class_idx
                )
                st.image(
                    recon_img.permute(1, 2, 0).detach().numpy(),
                    width=300,
                    caption="Reconstructed Image",
                )

with tab3:
    st.header("Image Interpolation")
    model3 = st.radio(
        "Interpolate model", ["BaseVAE", "DeepVAE", "ConvVAE"], horizontal=True
    )
    img1, img2 = st.columns(2)

    with img1:
        digit_class1 = st.selectbox("First Class", config.CLASSES[dataset].keys())
        first_img = inferer.sample_image(config.CLASSES[dataset][digit_class1], dataset)
        st.image(first_img.permute(1, 2, 0).detach().numpy(), width=300)

    with img2:
        digit_class2 = st.selectbox("Second Class", config.CLASSES[dataset].keys())
        second_img = inferer.sample_image(
            config.CLASSES[dataset][digit_class2], dataset
        )
        st.image(second_img.permute(1, 2, 0).detach().numpy(), width=300)

    with st.form("Image Interpolate Form"):
        interpolate_submit = st.form_submit_button("Interpolate Image")
        if interpolate_submit:
            intermediate_images = inferer.interpolate(
                first_img, second_img, model3, hidden_size, dataset
            )
            st.image(intermediate_images.permute(1, 2, 0).detach().numpy(), width=650)

# streamlit run ./src/streamlit_app.py
