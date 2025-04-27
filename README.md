# Wyvernaire

**Wyvernaire** is a web application that classifies monsters from the *Monster Hunter* franchise based on user-uploaded images.  
It leverages a deep learning model trained to recognize monster species and provides detailed information about the predicted monster.

This project combines deep learning, Flask web development, and a clean, responsive UI to deliver an intuitive user experience.

---

## ✨ Features

- 🖼️ Upload an image of a monster  
- 🧠 Predict the monster species using a trained AI model  
- 📜 Display detailed monster information (description, traits)  
- ⚡ Drag-and-drop image upload support  
- 🖥️ Clean and responsive web interface (Bootstrap 5)  
- 🔥 Fast prediction using subprocess-based model inference  
- 🎯 Top-1 monster prediction shown with additional info  

---

## 🛠️ Tech Stack

| Category         | Tools & Libraries               |
|------------------|---------------------------------|
| **Frontend**     | HTML5, Bootstrap 5              |
| **Backend**      | Python 3, Flask                 |
| **Machine Learning** | PyTorch (image classification) |
| **Model Handling**   | Subprocess module (CLI execution) |
| **Data Storage**     | JSON (monster info database)   |

---

## 🗂️ Project Structure

```
Wyvernaire/
├── app.py               # Main Flask application
├── predictor.py         # Deep learning model loading and prediction
├── static/              # Static assets
│   ├── icons/           # UI icons
│   ├── img/             # Monster and UI images
│   └── uploads/         # User-uploaded images
├── templates/           # HTML templates
│   └── index.html
├── models/              # Trained model (.pth checkpoint)
├── monsters_info.json   # Monster information and metadata
├── requirements.txt     # Python package dependencies
├── source/              # Source code files
│   ├── app.py
│   ├── get_data.py
│   ├── img_crawler.py
│   ├── info.py
│   ├── model_v2.py
│   ├── predictor.py
│   ├── test.py
│   └── wiki_crawler.py
└── README.md            # Project documentation
```

---

## 🚀 Setup and Run Locally

### 1. Clone the Repository
```bash
git clone git@github.com:nhatphan244/Wyvernaire.git
cd Wyvernaire
```

### 2. Create and Activate a Virtual Environment (optional)
```bash
python -m venv venv
venv\Scripts\activate    # Windows
# or
source venv/bin/activate # Mac/Linux
```

### 3. Install Dependencies
```bash
pip install --target=packages -r requirements.txt
```

### 4. Start the Flask App
```bash
python app.py
```

By default, the app will run at:
```
http://127.0.0.1:5000/
```

Open the link in your browser to use Wyvernaire!

---

##  How It Works

1. User uploads an image through the web interface.  
2. The image is saved temporarily and passed to `predictor.py`.  
3. `predictor.py` loads the trained PyTorch model and predicts the monster class.  
4. The app matches the prediction with `monsters_info.json` to retrieve monster details.  
5. The predicted monster name and description are displayed to the user.  

---

##  License

This project is intended for educational purposes and personal use only.  
All *Monster Hunter* characters and assets are © Capcom Co., Ltd.

---

##  Acknowledgments

- Monster images and information sourced from *Monster Hunter* games.  
- Thanks to the open-source community for tools and inspiration.
