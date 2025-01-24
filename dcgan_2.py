import streamlit as st
import requests
from PIL import Image, ImageFilter, ImageEnhance
from io import BytesIO

# Mapping nationalities to their corresponding codes used by the API
nationality_codes = {
    "USA": "us",
    "UK": "gb",
    "France": "fr",
    "Denmark": "dk",
    # "India": "in",  # Note: Random User API might not have data for some countries like India
    "Australia": "au",
    "Canada": "ca"
}

# Function to fetch a human face with specific attributes
def fetch_face_with_attributes(gender, nationality):
    # Convert to lowercase for API compatibility
    gender = gender.lower()
    nationality = nationality_codes.get(nationality, "").lower()

    # Construct the API request URL
    url = f"https://randomuser.me/api/?gender={gender}&nat={nationality}&inc=picture,gender,nat"
    
    response = requests.get(url).json()
    user_info = response['results'][0]

    # Ensure the returned user matches the requested gender and nationality
    if user_info['gender'] == gender and user_info['nat'].lower() == nationality:
        image_url = user_info['picture']['large']
        img_response = requests.get(image_url)
        img = Image.open(BytesIO(img_response.content))
        return img
    else:
        # Retry fetching another user if attributes do not match
        return fetch_face_with_attributes(gender, nationality)

# Function to enhance a blurred image using Unsharp Mask
def enhance_image(image):
    # Apply Unsharp Mask filter for better image enhancement
    enhanced_img = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    return enhanced_img

# Streamlit app
st.title("DCGAN Human Face Generator and Image Enhancer")

# Sidebar for options
st.sidebar.title("Choose an Option")
option = st.sidebar.radio("Select an Option", ["Generate DCGAN Image", "Enhance Blurred Image"])

if option == "Generate DCGAN Image":
    # Input fields for face attributes
    gender = st.selectbox("Select Gender", ["Male", "Female"])
    nationality = st.selectbox("Select Nationality", ["USA", "UK", "France", "Denmark", "Australia", "Canada"])

    # Button to fetch face
    if st.button("Generate Face"):
        image = fetch_face_with_attributes(gender, nationality)
        if image:
            st.image(image, caption='Generated Human Face', use_column_width=True)

            # Prepare image data for download
            img_bytes_png = BytesIO()
            img_bytes_jpg = BytesIO()
            image.save(img_bytes_png, format='PNG')
            image.save(img_bytes_jpg, format='JPEG')

            # Provide download buttons
            st.write("Download as:")
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(label="Download as PNG", data=img_bytes_png.getvalue(),
                                   file_name="generated_face.png", mime="image/png")
            with col2:
                st.download_button(label="Download as JPG", data=img_bytes_jpg.getvalue(),
                                   file_name="generated_face.jpg", mime="image/jpeg")

elif option == "Enhance Blurred Image":
    # File uploader for blurred image
    uploaded_file = st.file_uploader("Upload a Blurred Image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Load the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Button to enhance the image
        if st.button("Enhance Image"):
            enhanced_image = enhance_image(image)
            st.image(enhanced_image, caption="Enhanced Image", use_column_width=True)

            # Prepare enhanced image data for download
            enhanced_img_bytes_png = BytesIO()
            enhanced_img_bytes_jpg = BytesIO()
            enhanced_image.save(enhanced_img_bytes_png, format='PNG')
            enhanced_image.save(enhanced_img_bytes_jpg, format='JPEG')

            # Provide download buttons for enhanced images
            st.write("Download Enhanced Image as:")
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(label="Download as PNG", data=enhanced_img_bytes_png.getvalue(),
                                   file_name="enhanced_image.png", mime="image/png")
            with col2:
                st.download_button(label="Download as JPG", data=enhanced_img_bytes_jpg.getvalue(),
                                   file_name="enhanced_image.jpg", mime="image/jpeg")
