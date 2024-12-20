# Artificial Neural Network based Stroke Prediction Model with Hyperparameter Optimization using Multi-Objective Genetic Algorithm 

This repository contains a Python implementation of a Multi-Objective Genetic Algorithm (MOGA) to optimize the hyperparameters of an Artificial Neural Network (ANN) built for Stroke Prediction. The MOGA aims to maximize accuracy and minimize loss simultaneously.

## Repository Structure

- **Data/**: Contains the initial data preprocessing code and the original dataset along with the cleaned dataset as well as the final balanced dataset used by the model.
- **Frontend_UI/**: Contains the code for implementing the user interface. Additionally includes pictures of the deployed UI within another folder.
- **Model_Results_images/**: Contains images of the final GA implemented model performance.
- **base_ann_model.ipynb**: This notebook contains the base ANN model without any optimization, serving as a baseline for comparison with the optimized model.
- **moga_optimized_model.ipynb**: This notebook implements the Multi-Objective Genetic Algorithm for optimizing the ANN's hyperparameters.

## File Infromation

- **data_cleaning.py**: Codes used for data preprocessing/data cleaning. Also includes the implementation of SMOTE to address class imbalance.
- **get_hyperparameters_ui.py**: Contains the code for implementing the user interface where optimized hyperparameters are presented.
- **moga_model_for_ui.py**: Code is the same as in "moga_optimized_model, the code has been copied into a .py file for easy deployment on Streamlit.
- **stroke_ui.py**: Contains code of the user interface for predicting stroke risk using an Artificial Neural Network (ANN).
- **base_ann_model.ipynb**: This notebook contains the base ANN model without any optimization, serving as a baseline for comparison with the optimized model.
- **moga_optimized_model.ipynb**: This notebook implements the Multi-Objective Genetic Algorithm for optimizing the ANN's hyperparameters.

