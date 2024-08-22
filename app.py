import gradio as gr
from fastai.vision.all import *

def is_cat(x): return x[0].isupper()

learn = load_learner('model.pkl')

def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return f"Prediction: {pred}. Probability: {probs[idx]:.04f}"

iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Cat Classifier",
    description="Upload an image to check if it's a cat!"
)

iface.launch(inline=False)