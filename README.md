# Urban Surface Temperature Estimation using Machine Learning

**Bachelor Thesis Project - Alma Mater Studiorum University of Bologna**

## Overview
This repository contains the code and data for my Bachelor's thesis: *"Machine Learning for the estimation of urban surface temperature from satellite indices: application to Bologna"*. 

The project focuses on estimating the normalized Land Surface Temperature (LST z-score) using machine learning techniques applied to satellite-derived indices. The analysis uses data from the TALEA project (Summer 2024, Bologna municipality) to understand the drivers of urban heat. 

The predictive pipeline leverages several explanatory variables, including mean NDVI (vegetation cover), mean Albedo (surface reflectance), and custom composite indices (`heat_veg_index`, `heat_retention_index`).

### Machine Learning Models Compared:
- **Linear Regression**
- **Random Forest Regressor**
- **Gradient Boosting Regressor**

## Repository Contents
- **`data.zip`**: Contains all the raw and processed datasets required to run the experiments.
- **`Jupyter Notebook`**: The main notebook containing the complete pipeline (data loading, preprocessing, model training, evaluation, and visualizations) along with descriptive text blocks explaining the methodology.

## How to Run
The project is designed to be easily executed in **Google Colab**.

1. Upload and open the notebook in [Google Colab](https://colab.research.google.com/).
2. In the Colab file explorer (left panel), create a new folder named `data`.
3. Extract the `data.zip` archive locally and upload all its contents into the newly created `data` folder on Colab.
4. Run the code cells sequentially from top to bottom.
