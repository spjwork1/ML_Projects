import streamlit as st
import pytesseract
from PIL import Image
import cv2
import numpy as np
import os

# Set the path to Tesseract executable (Windows users need this)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to preprocess the image
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # Apply dilation to connect text components
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    gray = cv2.dilate(gray, kernel, iterations=1)
    
    return gray

def perform_ocr(image):
    # Perform OCR on the preprocessed image
    try:
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        return f"Error performing OCR: {str(e)}"

def main():
    # Move all Streamlit sidebar operations before any other Streamlit commands
    st.sidebar.header("Preprocessing Options")
    apply_preprocessing = st.sidebar.checkbox("Apply Preprocessing", value=True)
    
    st.title("OCR Application")
    st.write("Upload an image and extract text from it!")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Convert PIL Image to numpy array for OpenCV processing
        image_np = np.array(image)
        
        if st.button("Extract Text"):
            with st.spinner("Extracting text..."):
                if apply_preprocessing:
                    processed_image = preprocess_image(image_np)
                    st.image(processed_image, caption='Preprocessed Image', use_column_width=True)
                    text = perform_ocr(processed_image)
                else:
                    text = perform_ocr(image_np)
                
                # Display extracted text
                st.header("Extracted Text")
                st.write(text)
                
                # Add download button for extracted text
                st.download_button(
                    label="Download extracted text",
                    data=text,
                    file_name="extracted_text.txt",
                    mime="text/plain"
                )

if __name__ == "__main__":
    main()