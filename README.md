# 🛡️ AI Content Guardian | Hate Speech Detection

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)
![Flask](https://img.shields.io/badge/Flask-Backend-black)
![License](https://img.shields.io/badge/License-MIT-green)

> **"A Cyber-Secured AI approach to filtering online toxicity."**

## 📌 Overview
**AI Content Guardian** is a deep learning-based web application designed to detect **hate speech** and **offensive language** in real-time. 

Unlike simple keyword filters, this project utilizes a **Long Short-Term Memory (LSTM)** neural network to understand the *context* of sentences. It features a modern, **Glassmorphism-styled UI** that provides instant visual feedback (Green/Red) based on the toxicity level of the input text.

---

## 📸 Screenshots
| **Safe Content (Green)** | **Offensive Content (Red)** |
|:-------------------------:|:---------------------------:|
| *(Upload screenshot here)* | *(Upload screenshot here)* |

---

## ✨ Features
* **🧠 Context-Aware AI:** Uses an LSTM model trained on the Davidson et al. Hate Speech dataset.
* **⚡ Real-Time Analysis:** Instant classification via a Flask REST API.
* **🎨 Cyber-Themed UI:** A responsive, dark-mode interface with glassmorphism effects and animations.
* **📊 Confidence Scoring:** Displays the model's certainty percentage for every prediction.
* **🔍 Smart Preprocessing:** Includes custom text cleaning (stopword handling, regex filtering) for higher accuracy.

---

## 🛠️ Tech Stack
### **Frontend**
* **HTML5 & CSS3:** Custom "Cyber-AI" design with CSS Variables and Animations.
* **JavaScript (ES6):** Async/Await for fetching API results without reloading.

### **Backend**
* **Flask:** Lightweight Python web server to host the model.
* **TensorFlow / Keras:** For building and running the LSTM Deep Learning model.
* **Pickle:** For serializing the tokenizer.

### **Data Processing**
* **Pandas & NumPy:** For dataset manipulation.
* **NLTK:** For natural language preprocessing.

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/astha2211/Hate-Speech-Detection-AI.git](https://github.com/astha2211/Hate-Speech-Detection-AI.git)
cd Hate-Speech-Detection-AI
