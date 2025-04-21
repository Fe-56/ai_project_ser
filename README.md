# Speech Emotion Recognition (SER) Application: Audio-based Neural Network for Affective Multiclass Analysis and Labelling using Artificial Intelligence [ANNAMALAI]

This project provides a simple graphical interface for uploading `.mp3` or `.wav` speech audio files and predicting the emotional content of their five-second segments.

## ⬇️ Download the Model
First and foremost, download the model weights. If not, the GUI will not analyze your uploaded speech audio file.

1. Download `final_model.zip` from our [Microsoft Teams](https://sutdapac.sharepoint.com/:u:/s/50.021AIProject/EUcB6OqFi-NPi7iuuMogTkQBjCE2dFcsLpUm-lMFHFr1yg?e=uIOGGe).
2. Move `final_model.zip` to the `gui/` directory, if it is not downloaded and saved there already.
3. Extract the contents of `final_model.zip` into the current `gui/` directory.
4. Once the extraction is complete, there should be `best_meta_ffnn_model.pt`, `checkpoint-22112/` and `checkpoint-55280/`. These files and folders contain the model weights.
5. You may delete `final_model.zip`.

---

## 🔧 Setup Instructions

Follow the steps below to set up and run the project.

### Set up the Python Virtual Environment

> It is recommended to use a virtual environment to avoid dependency conflicts.

1. **Install virtualenv if not already installed:**
```bash
pip install virtualenv
```

2. Create a new virtual envionment named 'venv':
```bash
virtualenv venv
```

3. Activate the virtual environment:
```bash
source venv/bin/activate
```

4. Install project dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚀 Using the GUI

Once the environment is ready, follow these steps to launch the GUI:

1. **Ensure you're inside the virtual environment.**  
   If not, activate it using:

   ```bash
   source venv/bin/activate
   ```

2. From the project root directory, navigate to the GUI folder:

   ```bash
   cd gui
   ```

3. Start the application:

   ```bash
   python app.py
   ```

4. On first run, wait up to **1 minute** for the backend to initialize.

5. **Important:** Clear any saved data and cookies from `localhost` in your browser settings.

6. Open `index.html` in your browser (e.g., drag-and-drop into your browser or use `File > Open`).

7. Upload a `.mp3` or `.wav` speech audio file, and click **Submit** to process and view the predicted emotions for five-second segments.

8. To analyze another file, **refresh the page**, then repeat step 7.

---

## 🔚 Exiting the Virtual Environment

When you are done:

```bash
deactivate
```

---

## 📁 Project Structure

```
ai_project_ser/
│
├── classical_models/   # Our experiments/iterations using classical machine learning models
├── cnn/   # Our experiments/iterations using convolutional neural networks
├── data/   # Datasets required for conducting our experiments
├── ffnn/   # Our experiments/iterations using feed-forward neural networks
├── gui/
│   └── app.py          # Flask backend
│   └── index.html      # Frontend GUI
│   └── ...             # Additional frontend assets
├── README.md           # This file
├── requirements.txt    # Python dependencies
├── transformer_models/   # Our experiments/iterations using transformer models
└── venv/               # Virtual environment (created after setup)
```

---

## 💡 Notes

- For best results, use Google Chrome or Firefox.
- If you're running the app for the first time, initialization may take a little longer due to backend setup.
- Ensure your uploaded audio is clear and within a reasonable duration for better emotion detection accuracy.
