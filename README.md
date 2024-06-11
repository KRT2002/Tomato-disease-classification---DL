# Tomato Disease Classification - Deep Learning

## Overview

This project focuses on classifying tomato diseases using Convolutional Neural Networks (CNNs) implemented with TensorFlow. The dataset used is sourced from Kaggle, which includes images of potato, pepper-bell, and tomato diseases categorized in separate directories. Specifically, this project aims to classify tomatoes into ten distinct disease categories based on the dataset directory names.

By leveraging pre-trained models from TensorFlow Hub and applying fine-tuning techniques, the project achieves enhanced model performance, resulting in high accuracy in disease classification.

## Tech Stack

- **Programming Language**: Python
- **Deep Learning Framework**: TensorFlow
- **Additional Libraries**: 
  - TensorFlow Hub
  - Flask (for the optional web interface)
- **Data Science Tools**:
  - Jupyter Notebook (for model development and experimentation)
- **Version Control**:
  - Git (for version control of the project)
- **Virtual Environment**:
  - Python Virtual Environment (for managing project dependencies)

## How to Run the Project

### Step 1: Setup Virtual Environment

1. **Create Virtual Environment (Optional)**:
   - It's recommended to create a virtual environment to manage project dependencies.
     ```bash
     python -m venv env
     ```

2. **Activate Virtual Environment**:
   - On Windows:
     ```bash
     .\env\Scripts\activate
     ```
   - On macOS and Linux:
     ```bash
     source env/bin/activate
     ```

### Step 2: Data Installation

1. **Download the dataset from Kaggle**:
   - Visit the [dataset link](https://www.kaggle.com/datasets/arjuntejaswi/plant-village).
   - Download the dataset and extract only the tomato dataset folder.

2. **Organize the dataset**:
   - Create a directory named `data`.
   - Move the extracted tomato dataset folder into the `data` directory. Ensure the directory structure is as follows:
     ```
     data/
         ├── Tomato___Bacterial_spot/
         ├── Tomato___Early_blight/
         ├── Tomato___Late_blight/
         ├── Tomato___Leaf_Mold/
         ├── Tomato___Septoria_leaf_spot/
         ├── Tomato___Spider_mites Two-spotted_spider_mite/
         ├── Tomato___Target_Spot/
         ├── Tomato___Tomato_Yellow_Leaf_Curl_Virus/
         ├── Tomato___Tomato_mosaic_virus/
         └── Tomato___healthy/
     ```

### Step 3: Install the Requirements

Ensure you have Python installed. Then, install the required libraries using the following command:
```bash
pip install -r requirements.txt
```

### Step 4: Download and Run the Jupyter Notebook

1. **Download the notebook (.ipynb)**:
   - Obtain the Jupyter notebook file associated with this project.

2. **Run the notebook**:
   - Launch Jupyter Notebook:
     ```bash
     jupyter notebook
     ```
   - Open the downloaded notebook file and execute the cells to train and test the model.

### Step 5: Flask Application

A Flask application has been added for serving the model predictions via a web interface. To run the Flask app:

1. Navigate to the directory containing the Flask app files.
2. Run the following command to start the server:

```bash
python app.py
```

3. Open your browser and navigate to http://127.0.0.1:5000/ to interact with the app.

## Flask App Structure
1. app.py: Contains the code for running the Flask application.
2. templates/index.html: The main HTML page for the web interface.
3. static/styles.css: CSS file for styling the web interface.

### Model Training and Fine-Tuning

The model training process involves several key steps:

- **Loading and Preprocessing Data**: Images are loaded and preprocessed to match the input requirements of the pre-trained models.
- **Using Pre-trained Models**: Pre-trained models from TensorFlow Hub are utilized. These models have been trained on large datasets and can be fine-tuned for specific tasks like tomato disease classification.
- **Fine-Tuning**: Fine-tuning involves unfreezing some of the top layers of the pre-trained model and training them on the tomato dataset to improve performance.
- **Evaluation**: The model is evaluated using metrics such as accuracy, precision, recall and f1-score to ensure it performs well on the test data.

#### Tips for Best Results

- **Data Augmentation**: Apply data augmentation techniques such as rotation, flipping, and zooming to increase the diversity of your training data.
- **Learning Rate Tuning**: Experiment with different learning rates for fine-tuning to achieve the best model performance.
- **Model Checkpoints**: Save model checkpoints during training to prevent loss of progress in case of interruptions.
