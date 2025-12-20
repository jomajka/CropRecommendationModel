# Crop Recommendation System (Machine Learning)
An interactive machine learning–based crop recommendation system built in Python.
The application analyzes soil nutrients and environmental conditions to recommend an optimal crop using a Random Forest classifier, with built-in data visualizations and an interactive user interface.
## Overview
Selecting the right crop based on soil and climate conditions can significantly enhance agricultural yields. This project utilizes a supervised machine learning model trained on agricultural data to predict the most suitable crop based on user-defined inputs.
The application includes:
- Exploratory data analysis (EDA)
- Model training and evaluation
- Interactive prediction using sliders and buttons
## Features
- Trains a Random Forest Classifier on labeled crop data 
- Displays exploratory visualizations
- Calculates and prints model accuracy
- Interactive UI for real-time crop prediction using ipywidgets

## Technologies Used
- Python 3
- pandas – data loading and manipulation
- scikit-learn – machine learning model and evaluation
- matplotlib – data visualization
- ipywidgets – interactive user interface
- Jupyter Notebook – recommended runtime environment

## Dataset
The project uses a CSV dataset (Crop_recommendation.csv) containing agricultural records with the following features:
- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- Soil pH
- Rainfall

**Target Label:** Crop type

## Installation & Setup
1.	Clone the repository:\
git clone https://github.com/yourusername/crop-recommendation-system.git \
cd crop-recommendation-system
2.	(Recommended) Create and activate a virtual environment:\
python -m venv venv\
source venv/bin/activate   # Windows: venv\Scripts\activate
3.	Install required libraries:\
pip install pandas scikit-learn matplotlib ipywidgets\
Note: This project is designed to run in a Jupyter Notebook environment for full interactivity.
## How to Run
1.	Open Jupyter Notebook:
jupyter notebook
2.	Open the notebook containing the project code.
3.	Run all cells. The program will:
   - Load the dataset
   - Display exploratory plots
   - Train the machine learning model
   - Print model accuracy
   - Display interactive sliders and a Predict Crop button
________________________________________
## Example Output
Model Accuracy: 0.96\
Recommended Crop: Rice\
Users can adjust input values using sliders and click Predict Crop to generate recommendations.

## Machine Learning Approach
- Problem Type: Multiclass classification
- Algorithm: Random Forest Classifier
- Train/Test Split: 80% training / 20% testing
- Evaluation Metric: Accuracy score

## Author
Jonathon Majka\
B.S. Computer Science – Western Governors University\
GitHub: github.com/jomajka
## License
This project is for educational purposes.
