# python3 -m venv venv
# . venv/bin/activate
# pip install streamlit
# pip install torch torchvision
# streamlit run main.py
import streamlit as st

from PIL import Image
import style
import os
import utils
from torchvision import transforms
import torch

model_path = "neural_style/saved_models/" + 'candy' + ".pth"
model = style.load_model(model_path)
style_img = "neural_style/images/style-images/candy.jpg"
img = "neural_style/test.jpg"
content_image = utils.load_image(img)
content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
content_image = content_transform(content_image)
content_image = content_image.unsqueeze(0).to('cpu')
st.write(content_image)

with torch.no_grad():
    output = model(content_image).cpu()

st.image(output[0])
'''

st.title('PyTorch Style Transfer')
st.write(style.stylize)
st.write(os.listdir())

img = st.sidebar.selectbox(
    'Select Image',
    ('amber.jpg', 'cat.png')
)

style_name = st.sidebar.selectbox(
    'Select Style',
    ('candy', 'mosaic', 'rain_princess', 'udnie')
)


model= "neural_style/saved_models/" + style_name + ".pth"
input_image = "neural_style/images/content-images/" + img
# input_image = "neural_style/test.jpg"
output_image = "neural_style/images/output-images/" + style_name + "-" + img

st.write('### Source image:')

image = Image.open(input_image)
st.image(image, width=400) # image: numpy array

clicked = st.button('Stylize')

model = style.load_model(model)
style.stylize(model, input_image, output_image)


if clicked:
    model = style.load_model(model)
    style.stylize(model, input_image, output_image)

    st.write('### Output image:')
    image = Image.open(output_image)
    st.image(image, width=400)
'''