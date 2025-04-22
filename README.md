
# Concrete Compressive Strength Predictor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview

This project predicts the compressive strength of concrete based on its constituent components and age using Machine Learning. A Linear Regression model is trained on the Concrete Compressive Strength dataset from the UCI Machine Learning Repository. The project includes scripts for training the model and a web application built with Streamlit for interactive predictions.

**🎥 Live Demo:**  
[![Watch the Demo](https://img.youtube.com/vi/5S0UmTECWqE/0.jpg)](https://www.youtube.com/watch?v=5S0UmTECWqE)

**Repository:** [http://github.com/karanji55/Compressive-Strength-Predictor](http://github.com/karanji55/Compressive-Strength-Predictor)

## Dataset

-   **Source:** UCI Machine Learning Repository
    
-   **Name:** Concrete Compressive Strength Dataset
    
-   **Link:** [https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength)
    
-   **Description:** Contains 1030 instances with 8 quantitative input variables and 1 quantitative output variable. The dataset file (`Concrete_Data.xls`) is downloaded automatically by the training script if not present locally.
    

## Features (Input Variables)

1.  Cement (kg in a m³ mixture)
    
2.  Blast Furnace Slag (kg in a m³ mixture)
    
3.  Fly Ash (kg in a m³ mixture)
    
4.  Water (kg in a m³ mixture)
    
5.  Superplasticizer (kg in a m³ mixture)
    
6.  Coarse Aggregate (kg in a m³ mixture)
    
7.  Fine Aggregate (kg in a m³ mixture)
    
8.  Age (day)
    

## Target Variable (Output)

-   **Concrete compressive strength (MPa)**
    

## Model

-   **Type:** Linear Regression
    
-   **Library:** scikit-learn
    
-   **Evaluation on test set:**
    
    -   R-squared (R²): ~0.63
        
    -   RMSE: ~9.80 MPa
        

## Technology Stack

-   Python 3.x
    
-   pandas
    
-   scikit-learn
    
-   joblib
    
-   Streamlit
    
-   requests
    
-   openpyxl
    

## Project Structure

```
.
├── app.py
├── train_model.py
├── concrete_strength_model.joblib
├── requirements.txt
├── Concrete_Data.xls
└── README.md

```

## Setup and Installation

1.  **Clone the Repository:**
    
    ```bash
    git clone https://github.com/karanji55/Compressive-Strength-Predictor.git
    cd Compressive-Strength-Predictor
    
    ```
    
2.  **Set Up Virtual Environment:**
    
    ```bash
    python -m venv venv
    # Activate it
    source venv/bin/activate  # On macOS/Linux
    .\venv\Scripts\activate   # On Windows
    
    ```
    
3.  **Install Dependencies:**
    
    ```bash
    pip install -r requirements.txt
    
    ```
    

## Usage

1.  **Train the Model:**
    
    ```bash
    python train_model.py
    
    ```
    
2.  **Run the App:**
    
    ```bash
    streamlit run app.py
    
    ```
    

## Acknowledgements

-   Dataset from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
    
-   Original data by Prof. I-Cheng Yeh, Department of Civil Engineering, Tamkang University, Taiwan
