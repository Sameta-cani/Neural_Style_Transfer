# python3 -m venv venv
# . venv/bin/activate
# pip install streamlit
# pip install torch torchvision
# streamlit run main.py
import streamlit as st

from PIL import Image
import style



st.title('PyTorch Style Transfer')
st.write(style.stylize)
'''
img = st.sidebar.selectbox(
    'Select Image',
    ('amber.jpg', 'cat.png')
)

style_name = st.sidebar.selectbox(
    'Select Style',
    ('candy', 'mosaic', 'rain_princess', 'udnie')
)


model= "saved_models/" + style_name + ".pth"
# input_image = "images/content-images/" + img
input_image = "test.jpg"
output_image = "images/output-images/" + style_name + "-" + img

st.write('### Source image3:')
image = Image.open('C:\\Users\\Sangjin\\OneDrive\\바탕 화면\\기타\\문서\\STREAMLIT-STYLE-TRANSFER\\neural_style\\test.jpg')
# st.image(image)

# image = Image.open(input_image)
st.image(input_image, width=400) # image: numpy array

clicked = st.button('Stylize')

if clicked:
    model = style.load_model(model)
    style.stylize(model, input_image, output_image)

    st.write('### Output image:')
    image = Image.open(output_image)
    st.image(image, width=400)
'''