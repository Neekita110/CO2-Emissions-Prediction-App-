# CO2 Emissions Prediction App | CLL725 Project Work

## Overview
The CO2 Emissions Prediction App is designed to estimate a vehicle's CO2 emissions based on various parameters, offering users insights into the environmental impact of their vehicles. Developed as part of my CLL725 Project Work, this app applies machine learning techniques to provide real-time CO2 emissions predictions, which can inform decisions around vehicle usage, fuel consumption, or future vehicle purchases.

## Key Features
- **Engine Size**: Enter engine displacement in liters.
- **Vehicle Age**: Specify the vehicle's age in years.
- **Fuel Consumption**: Input the amount of fuel consumed per 100 kilometers.
- **Energy Consumption**: Indicate energy usage if applicable.
- **Fuel Type**: Select from various fuel options.
- **Transmission Type**: Choose between automatic or manual transmission.
- **Make**: Select from categories like Luxury, Premium, or Sports.
- **Vehicle Class**: Define the vehicle type (SUV, Sedan, Truck, etc.).
- **Kilometers Driven Per Day**: Input the average daily distance driven.

The app is built using Streamlit, which provides an interactive and user-friendly interface. The underlying model uses Linear Regression, developed with Scikit-learn, to analyze and predict CO2 emissions.

## Technical Stack
- **Languages**: Python
- **Libraries**: Streamlit, Scikit-Learn, Pandas, NumPy
- **Machine Learning Model**: Linear Regression trained on vehicle data



This project demonstrates how predictive models can be used to assess environmental factors and how these models can be deployed through an easy-to-use web application. I hope this app can help users make more environmentally conscious decisions by understanding the CO2 output of their vehicles.
