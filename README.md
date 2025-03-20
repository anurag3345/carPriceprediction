# Car Price Prediction Model

## Overview
This project is a Machine Learning model designed to predict the selling price of used cars based on various attributes such as year of manufacture, present price, kilometers driven, fuel type, seller type, transmission, and ownership history. 

## Dataset
The dataset used for training the model includes the following features:
- **Car_Name**: The name of the car (not used in the model directly).
- **Year**: The manufacturing year of the car.
- **Selling_Price**: The price at which the car is sold (Target Variable).
- **Present_Price**: The current market price of the car.
- **Kms_Driven**: The total distance traveled by the car in kilometers.
- **Fuel_Type**: The type of fuel used (Petrol/Diesel).
- **Seller_Type**: The type of seller (Dealer/Individual).
- **Transmission**: Transmission type (Manual/Automatic).
- **Owner**: The number of previous owners (0 for first owner, 1 for second owner, etc.).

## Model Performance
The model was evaluated using common regression metrics:
- **Mean Absolute Error (MAE)**: 1.258
- **Mean Squared Error (MSE)**: 3.493
- **R-squared Score (R2 Score)**: 0.829

A high R2 score of 0.829 indicates that the model explains approximately 83% of the variance in car prices, demonstrating good predictive accuracy.

## Model File
The trained model has been saved as a `.pkl` file, allowing for easy reuse and deployment. To load and use the model, follow these steps:
```python
import pickle

# Load the model
with open('car_price_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Use the model for predictions
predicted_price = model.predict([[2014, 5.59, 27000, 'Petrol', 'Dealer', 'Manual', 0]])
print(predicted_price)
```

## Dependencies
The model is implemented in Python using the following libraries:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- pickle (for saving/loading the model)

## Usage
1. Load the dataset into a Pandas DataFrame.
2. Preprocess the data (handling missing values, encoding categorical variables, and feature scaling if needed).
3. Train the model using a suitable regression algorithm (e.g., Linear Regression, Random Forest, or XGBoost).
4. Save the trained model as a `.pkl` file.
5. Load the `.pkl` file to make predictions on new data.
6. Evaluate the model using performance metrics.

## Future Improvements
- Improve feature selection and engineering.
- Experiment with advanced ML algorithms such as XGBoost and Neural Networks.
- Fine-tune hyperparameters for better performance.
- Expand the dataset with more features such as car condition, insurance status, and service history.

## Contact
For any queries or contributions, feel free to reach out 

