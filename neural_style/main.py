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
import base64

import streamlit as st
import base64

def download_file(file_content, file_name):
    b64 = base64.b64encode(file_content).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{file_name}">Download {file_name}</a>'
    st.markdown(href, unsafe_allow_html=True)



test = "neural_style" # "."

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
    model = style.load_model(model)
    data = style.stylize(model, input_image, output_image)
    data_img = data[0].clone().clamp(0, 255).numpy()
    data_img = data_img.transpose(1, 2, 0).astype("uint8")
    data_img = Image.fromarray(data_img)
    st.write('### Output image:')
    # image = Image.open(output_image)
    st.image(data_img, width=400)
    tmp = style_name + "-" + img
    data_img.save(tmp)
    with open(tmp, "rb") as file:
        btn = st.download_button(
            label="Download image",
            data=file,
            file_name=tmp,
            mime="image/png"
          )

'''    if save_button:
        download_file(data[0].numpy(), style_name + "-" + img)
        st.write('save!')'''