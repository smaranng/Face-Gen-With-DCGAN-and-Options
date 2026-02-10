

# 👤 Face Generation with DCGAN and Attribute Options

## Overview

This project demonstrates a human face generation application that combines a **Deep Convolutional Generative Adversarial Network (DCGAN)** with a user-friendly interface built using **Streamlit**. The application allows users to generate realistic human faces based on specific attributes such as **gender** and **nationality**.

The system can either:

* Fetch face images from the **Random User API**, or
* Generate faces using a **pre-trained DCGAN model** for offline face synthesis.

---

## 📑 Table of Contents

* Features
* Technologies Used
* Installation
* Usage
* Directory Structure
* Screenshots
* Future Enhancements
* License

---

## ✨ Features

### 1. Dynamic Face Generation

* Select attributes such as **gender** and **nationality**
* Generate realistic human faces
* Supports both API-based and DCGAN-based generation

### 2. User-Friendly Interface

* Built with **Streamlit**
* Clean and intuitive web interface
* Easy interaction for non-technical users

### 3. Download Functionality

* Download generated faces in **PNG** or **JPG** format

### 4. Extendable Backend

* Easily integrate a fully trained DCGAN model
* Can switch between API-based and offline generation

---

## 🛠 Technologies Used

* Python 3.9
* Streamlit
* Requests (for API communication)
* Pillow (for image processing)
* PyTorch (for DCGAN integration)

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/smaranng/Face-Gen-With-DCGAN-and-Options.git
cd Face-Gen-With-DCGAN-and-Options
```

### 2. Set Up a Virtual Environment

```bash
python3 -m venv env
```

Activate the environment:

**On Windows:**

```bash
env\Scripts\activate
```

**On macOS/Linux:**

```bash
source env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App

```bash
streamlit run dcgan_1_streamlit.py
```

or

```bash
streamlit run dcgan_2.py
```

---

## ▶️ Usage

1. Open the app in your browser (usually at `http://localhost:8501`)
2. Select **gender** and **nationality** from the dropdown menus
3. Click **Generate Face** to create a new face
4. Use **Download as PNG** or **Download as JPG** to save the image

---

## 📁 Directory Structure

```
project-folder/
│
├── dcgan_1_streamlit.py, dcgan_2.py   # Main Streamlit application
├── generator.pth                     # (Optional) Pre-trained DCGAN generator model
├── home.html                         # Home page for the project
├── README.md                         # Project documentation
└── dataset/                          # Images organized for training the model
```

---

## 🖼 Screenshots

* Main Page
* Generated Face Output

*(Add screenshots here for better GitHub presentation)*

---

## 🚀 Future Enhancements

* Integrate a fully trained DCGAN model for complete offline face generation
* Add more customization options (e.g., age, emotion, facial expressions)
* Improve UI/UX design for better accessibility
* Add multilingual interface support

---

## 📄 License

This project is licensed under the **MIT License**.
See the `LICENSE` file for more details.

---

✅ Make it more academic/research-style
✅ Or simplify it for recruiters/portfolio use
