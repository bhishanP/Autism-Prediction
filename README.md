# Autism Screening Classification

This repository contains a machine learning project aimed at classifying individuals based on autism screening results. The project utilizes a dataset from Kaggle and employs various data preprocessing, exploratory data analysis (EDA), and machine learning techniques to build a predictive model.

## Table of Contents

- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Modeling](#modeling)
- [Deployment](#deployment)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Dataset

The dataset used in this project is available on Kaggle: [Autism Screening Dataset](https://www.kaggle.com/datasets/faizunnabi/autism-screening). It contains various attributes related to autism screening.

## Project Structure
```
Autism_Screening_Classification
│
├── Autism_Data.csv  # Dataset file
│
├── Data_explore.ipynb         # Jupyter notebook for EDA
├── model.ipynb           # Jupyter notebook for model building
├── app.py            # Flask application for model deployment
│
│
├── sample_input.json      # Sample input data for the Flask app
├── best_model.pkl         # Trained model file
├── README.md              # Project README file
└── LICENSE                # Project license file

```
## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/bhishanP/Autism-Prediction.git
    cd Autism-Prediction
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install the required packages:
    ```bash
    pip install flask
    ```

## Data Preprocessing

The data preprocessing steps include:

1. Loading the dataset from the ARFF/CSV file.
2. Handling missing values in `ethnicity`, `age`, and `relation`.
3. Converting `age` to numeric and replacing outliers with the median.

The preprocessing steps are implemented in the `Data_explore.ipynb` notebook.

## Exploratory Data Analysis (EDA)

EDA includes:

1. Generating a correlation matrix to understand the relationships between features.
2. Creating pair plots to visualize the relationships between numerical features.
3. Grouping and aggregating data to analyze categorical features.

The EDA is documented in the `Visual.ipynb` notebook.

## Modeling

The modeling steps include:

1. Splitting the data into training and testing sets.
2. Training various machine learning models.
3. Evaluating the models and selecting the best one.
4. Saving the best model using `joblib`.

The modeling steps are documented in the `model.ipynb` notebook.

## Deployment

The best model is deployed using a Flask application:

1. The Flask app (`app.py`) provides an endpoint for making predictions.

To run the Flask app locally:
```bash
python app.py
