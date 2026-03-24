# ✨ Crop Shield AI - Crop Disease Detection

[![GitHub Repo](https://img.shields.io/badge/GitHub-Crop--Shield--Ai-green?logo=github)](https://github.com/HarshitShira001/Crop-Shield-Ai)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Backend-black?logo=flask)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-ML-orange?logo=tensorflow)](https://www.tensorflow.org/)

**Crop Shield AI** is an advanced AI-powered crop disease detection system designed to empower farmers and agricultural enthusiasts. By leveraging **Deep Learning (CNN)** and **Computer Vision**, it provides real-time, accurate diagnosis of plant diseases through leaf image analysis, followed by actionable treatment recommendations.

---

## 🚀 Key Features

- ✅ **AI-Powered Diagnosis**: Instant detection of crop diseases using state-of-the-art Deep Learning.
- ✅ **Treatment Recommendations**: Get AI-generated advice on treatment and prevention for detected diseases.
- ✅ **Support for 17 Categories**: Precision detection across major crops including Corn, Potato, Rice, Sugarcane, and Wheat.
- ✅ **Contribution Portal**: A user-friendly interface to contribute new data for continuous model improvement.
- ✅ **Clean UI/UX**: A modern, mobile-responsive web interface built with Flask and Vanilla CSS.

---

## 🛠️ Tech Stack

- **Backend**: Python, [Flask](https://flask.palletsprojects.com/)
- **Machine Learning**: [TensorFlow](https://www.tensorflow.org/), [Keras](https://keras.io/)
- **Data Processing**: NumPy, Pandas, PIL (Pillow)
- **Frontend**: HTML5, CSS3 (Custom Design)
- **Model**: Convolutional Neural Networks (CNN)

---

## 📷 Supported Disease Classes (17)

Crop Shield AI currently identifies the following crop-disease pairs:

| Crop | Disease / State |
| :--- | :--- |
| **Corn** | Common Rust, Gray Leaf Spot, Northern Leaf Blight, Healthy |
| **Potato** | Early Blight, Late Blight, Healthy |
| **Rice** | Brown Spot, Leaf Blast, Neck Blast, Healthy |
| **Sugarcane** | Bacterial Blight, Red Rot, Healthy |
| **Wheat** | Brown Rust, Yellow Rust, Healthy |

---

## 🚀 Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/HarshitShira001/Crop-Shield-Ai.git
cd Crop-Shield-Ai
```

### 2️⃣ Install Dependencies
Ensure you have Python 3.8+ installed.
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application
```bash
python app.py
```
Wait for the terminal to show `✓ Model loaded successfully` and then visit `http://127.0.0.1:5000` in your browser.

---

## 📂 Project Structure
- `app.py`: Core Flask application and ML inference logic.
- `crop_model.keras`: Trained CNN model file.
- `templates/`: HTML templates for UI components.
- `static/`: CSS and uploaded images.
- `crop_detection1.ipynb`: Model training and evaluation notebook.

---

## 🤝 Contributing
We welcome contributions to expand the dataset and improve model accuracy! Use the **Contribute** section in the web app to upload new samples.

---

## 👤 Developer
**Harshit Shira**  
*AI/ML Developer & Agricultural Tech Enthusiast*
