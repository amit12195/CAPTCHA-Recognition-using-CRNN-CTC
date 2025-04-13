# CAPTCHA Recognition using CRNN-CTC

This repository contains a complete pipeline for recognizing numeric CAPTCHA strings using deep learning. The system uses a CRNN-based architecture with preprocessing steps tailored for noisy, distorted CAPTCHA images.

---
###  Step 1: Image Preprocessing
             Pyhton preprocessing/image_preprocessing.py 
            (Make sure to set the input dataset path inside the script before running.)

###  Step 2: Text Recognition Model Training (Optional)
            Visit the crnn_training directory for training the model. This includes:
                  Dataset preparation (after required preprocessing)
                  Model training using either CPU or GPU (depending on availability)

            Detailed step-by-step instructions are available in the user documentation.

### Step 3: Inference
             You can choose from the following options for performing inference:
             **Option 1:** Use the provided Jupyter Notebook for both text detection and recognition.

             **Option 2:** Run the following Python scripts separately:
                  python text_detection.py
                  python text_recognition_v1.py

###  Model Files:
            At the end of the user document, you’ll find a Google Drive link containing the pre-trained text detection and recognition models.
            After downloading, place the models in the ai_models directory.

