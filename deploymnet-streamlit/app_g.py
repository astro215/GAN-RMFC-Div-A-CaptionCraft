import gradio as gr
import os
import PIL.Image
import skimage.io as io
from predictor import Predictor

# Dynamic WEIGHTS_PATHS
models_folder = "./models"
WEIGHTS_PATHS = {file_name: os.path.join(models_folder, file_name) 
                 for file_name in os.listdir(models_folder) 
                 if file_name.endswith(".pt")}

# Instantiate the Predictor
predictor = Predictor()

def predict_image(image, model_choice, use_beam_search):
    # Load the uploaded image
    pil_image = PIL.Image.open(image)
    
    # Make prediction
    prediction = predictor.predict(pil_image, model_choice, use_beam_search)
    
    return prediction

# Define Gradio interface
image_input = gr.Image(label="Upload an image...", type="filepath")
model_choice = gr.Dropdown(label="Choose a model", choices=list(WEIGHTS_PATHS.keys()))
use_beam_search = gr.Checkbox(label="Use Beam Search")

output_text = gr.Textbox(label="Prediction")

title = "Image Captioning with CLIP and GPT-2"
description = "Upload an image, choose a model, and generate a caption."

examples = [["download.jpeg", "model-epoch4 - best.pt", True]]

demo = gr.Interface(fn=predict_image, inputs=[image_input, model_choice, use_beam_search], outputs=output_text, 
             title=title, description=description, examples=examples)

demo.launch(share=True)