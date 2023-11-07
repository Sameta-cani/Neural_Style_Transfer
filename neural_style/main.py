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

test = "neural_style"

st.title('PyTorch Style Transfer')

img = st.sidebar.selectbox(
    'Select Image',
    ('amber.jpg', 'cat.png', 'Image upload')
)

style_name = st.sidebar.selectbox(
    'Select Style',
    ('candy', 'mosaic', 'rain_princess', 'udnie')
)
st.write('### Source image:')

if img == 'Image upload':
	img_file = st.file_uploader('이미지를 업로드 하세요.', type=['png', 'jpg', 'jpeg'])
	if img_file is not None:
		utils.save_uploaded_file(test + "/images/content-images/", img_file)
		img = img_file.name
# if img_file is not None:

# local test: "neural_style" -> "."
# git test: "." -> "neural_style"

model = test + "/saved_models/" + style_name + ".pth"
input_image = test + "/images/content-images/" + img
output_image = test + "/images/output-images/" + style_name + "-" + img

if os.path.exists(input_image):
    image = Image.open(input_image)
    st.image(image, width=400) # image: numpy array

clicked = st.button('Stylize')
if clicked:
    st.write('start')
    model = style.load_model(model)
    st.write(output_image)
    content_image = utils.load_image(input_image)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to('cpu')
    with torch.no_grad():
        output = model(content_image).cpu()

    st.image(output[0], width=400)
'''      
if clicked:
    model = style.load_model(model)
    style.stylize(model, input_image, output_image)

    st.write('### Output image:')
    image = Image.open(output_image)
    st.image(image, width=400)
'''