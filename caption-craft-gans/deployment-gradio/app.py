import streamlit as st
import os
import PIL.Image
import skimage.io as io


from predictor import Predictor
# Your Predictor and other classes/functions here

# Dynamic WEIGHTS_PATHS
models_folder = "./models"
WEIGHTS_PATHS = {file_name: os.path.join(models_folder, file_name) 
                 for file_name in os.listdir(models_folder) 
                 if file_name.endswith(".pt")}

# Instantiate the Predictor
predictor = Predictor()

st.title("Image Captioning with CLIP and GPT-2")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])
if uploaded_image is not None:
    
    # image = io.imread(image)
    # pil_image = PIL.Image.fromarray(image)
    
    image = PIL.Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    model_choice = st.selectbox("Choose a model", list(WEIGHTS_PATHS.keys()))
    use_beam_search = st.checkbox("Use Beam Search")
    
    if st.button("Generate Caption"):
        prediction = predictor.predict(image, model_choice, use_beam_search)
        st.write("Prediction:", prediction)
