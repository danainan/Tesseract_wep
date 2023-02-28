import cv2
import numpy as np
from tesserocr import PyTessBaseAPI, RIL, get_languages, PSM , iterate_level , OEM
from PIL import Image , ImageDraw , ImageFont
import streamlit as st
import streamlit.components.v1 as components
from pythainlp import word_tokenize 




components.html(
    """
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
    <style>
        .container {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }
        .card {
            width: 100%;
            max-width: 500px;
        }
    </style>

    <div class="container">
        
        <img src="https://user-images.githubusercontent.com/71175110/220308869-f596631e-cd64-4a05-acf5-ca3a59f22966.jpg" class="img-fluid" alt="Responsive image">
    </div>
    </div>


    """,
    height=100,
)

# st.title("OCR WITH TESSERACT-OCR")
# st.write("This is a simple OCR web app")



image = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

if image is not None:

    image = Image.open(image)
    # image_arr = np.array(image)
    # st.image(image, caption='Uploaded Image.', width=300)
    im_gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    # kernel = np.ones((3, 3), np.uint8)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    # hsv = cv2.cvtColor(image_arr, cv2.COLOR_BGR2HSV)
    # kernel = np.array([[0, -1, 0],
    #                 [-1, 5,-1],
    #                [0, -1, 0]])


   

    #threshold
    ret, thres = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    
    

    im_processed = cv2.dilate(thres, kernel, iterations=1)
    # im_processed = cv2.morphologyEx(im_processed, cv2.MORPH_OPEN, kernel, iterations=1)
    # im_processed = cv2.erode(thres, kernel, iterations=1)
    # im_processed = cv2.morphologyEx(im_processed, cv2.MORPH_CLOSE, kernel, iterations=1)

    

    # new_img = cv2.morphologyEx(thres, cv2.MORPH_CLOSE, kernel, iterations=1)

    #noise removal
    # new_img = cv2.fastNlMeansDenoising(thres, None, 10, 7, 21)


    #sharpening
    # im_processed = cv2.filter2D(thres, -1, kernel)

    # #brightness
    # im_processed = cv2.addWeighted(thres, 1.5, np.zeros(thres.shape, thres.dtype), 0, 0)

    #brightness
    #new_img = cv2.addWeighted(image_arr, 1.5, np.zeros(image_arr.shape, image_arr.dtype), 0, 0)

    images = [image,im_processed]

    image_on_row = st.columns(len(images))
    for i in range(len(images)):
        image_on_row[i].image(images[i], use_column_width=True)

    # st.image(image, caption='Uploaded Image.', use_column_width=True)
    # st.image(new_img, caption='Processed Image.', use_column_width=True)

    # st.image(new_img, caption='Processed Image.', width=300)
    # # # #tesserocr
    with PyTessBaseAPI(path='C:/Users/User/anaconda3/share/tessdata_best-main', lang="tha+eng" , oem=OEM.LSTM_ONLY, psm=PSM.SPARSE_TEXT_OSD) as api:
        # api.SetImageFile("image/imageProcessed/image{}.jpg".format(i))

        #use lstm
        # api.SetVariable("lstm_choice_mode", "2")
        im_ocr = Image.fromarray(im_processed)
        api.SetImage(im_ocr)
        # api.SetPageSegMode(PSM.SINGLE_BLOCK)
        # api.SetPageSegMode(PSM.SPARSE_TEXT)
        # api.SetPageSegMode(PSM.SPARSE_TEXT_OSD)
        # api.SetPageSegMode(PSM.AUTO_OSD)
        # api.SetPageSegMode(PSM.AUTO_ONLY)
        # api.SetPageSegMode(PSM.AUTO)
        # api.SetPageSegMode(PSM.SINGLE_COLUMN)
        
        # api.SetVariable("textord_min_linesize", "2")
        # api.SetVariable('preserve_interword_spaces', '1')
        text = api.GetUTF8Text()
        conf = api.MeanTextConf()
        st.write("Confidence: {}".format(conf))
        # st.write(text)
        
        #append text to array
        text_array = []
        text_array.append(text.replace("\n", " "))
        

        st.write(text_array)

















        
            

       

        

        


        


       






