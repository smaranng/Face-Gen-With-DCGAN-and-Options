# 👤 Face-Gen-With-DCGAN-and-Options

---


This project demonstrates a human face generation application that combines a Deep Convolutional Generative Adversarial Network (DCGAN) with a user-friendly interface built using Streamlit. The application allows users to generate realistic human faces based on specific attributes, such as gender and nationality.

# Table of Contents
Features

Technologies Used

Installation

Usage

Directory Structure

Screenshots

License

# Features
**1. Dynamic Face Generation:**

Users can select attributes (gender and nationality) to generate human faces.

Fetches face data from the Random User API or from a trained DCGAN model.
---


# User-Friendly Interface:

Built with Streamlit for a clean and intuitive web app.

# Download Functionality:

Users can download generated faces as PNG or JPG.

# Extendable Backend:

Can integrate a pre-trained DCGAN model for face generation instead of relying on an external API.

# Technologies Used
Python 3.9

Streamlit

Requests (for API communication)

Pillow (for image processing)

PyTorch (if integrated with DCGAN)

# Installation
Clone this repository:

git clone https://github.com/smaranng/Face-Gen-With-DCGAN-and-Options.git
cd Face-Gen-With-DCGAN-and-Options
Set up a Python virtual environment:

python3 -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`
Install dependencies:

pip install -r requirements.txt
Run the Streamlit app:

streamlit run dcgan_1_streamlit.py or dcgan_2.py
Usage
Open the app in your browser (usually at http://localhost:8501).

Select gender and nationality from the dropdown menus.

Click the "Generate Face" button to see a generated face.

Use the "Download as PNG" or "Download as JPG" buttons to save the face.

Directory Structure


project-folder/


|
├── dcgan_1_streamlit.py, dcgan_2.py                # Main Streamlit application


├── generator.pth         # (Optional) Pre-trained DCGAN generator model


├── home.html     # Home page for the project


├── README.md             # Project documentation


└── dataset/               # Images organized for training the model

Main Page


Generated Face


License
This project is licensed under the MIT License. See the LICENSE file for details.

# Future Enhancements
Integrate a fully trained DCGAN model for offline face generation.

Add options for users to customize other attributes (e.g., age, emotion).

Improve the UI design for better accessibility and aesthetics.

Extend the application to support multilingual interfaces.


