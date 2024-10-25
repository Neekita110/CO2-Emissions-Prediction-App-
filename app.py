import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Set page title
st.set_page_config(page_title="CLL725 Project Work")

# Load the data
data = pd.read_csv('CO2_file.csv', index_col=0)

# Split the data into features and target
X = data.drop(['CO2_Emissions'], axis=1)
y = data['CO2_Emissions']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Train the model
reg = LinearRegression()
reg.fit(X_train, y_train)

# Save the model
filename = 'model.pkl'
pickle.dump(reg, open(filename, 'wb'))

# Load the model with the correct encoding
model = pickle.load(open('finalized_model.pkl', 'rb'))

# Streamlit app
def predict_co2_emissions(Engine_Size, Vehicle_Age, Fuel_Consumption, Energy_Consumption, Fuel_Consumption_Comb,
                           km_driven_per_day, Fuel_Type, Transmission, Make, Vehicle_Class):
    st.title('CLL725 Project')
    st.write('Please fill in the details of your car to predict CO2 emissions.')

    # Preprocess inputs
    fuel_type_values = {'Type_E': [1, 0, 0], 'Type_X': [0, 1, 0], 'Type_Z': [0, 0, 1]}
    fuel_type_input = fuel_type_values[Fuel_Type]

    transmission_values = {Transmission: 1}
    transmission_input = [transmission_values.get(key, 0) for key in ['A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'AM5', 'AM6', 'AM7', 'AM8', 'AM9', 'AS10', 'AS4', 'AS5', 'AS6', 'AS7', 'AS8', 'AS9', 'AV', 'AV6', 'AV7', 'AV8', 'AV10', 'M5', 'M6', 'M7']]

    make_values = {'Luxury': [1, 0, 0], 'Premium': [0, 1, 0], 'Sports': [0, 0, 1]}
    make_input = make_values[Make]

    vehicle_class_values = {'SUV': [1, 0, 0], 'Sedan': [0, 1, 0], 'Truck': [0, 0, 1]}
    vehicle_class_input = vehicle_class_values[Vehicle_Class]

    # Make prediction
    prediction_input = np.array([Engine_Size, Vehicle_Age, Fuel_Consumption, Energy_Consumption,
                                 Fuel_Consumption_Comb, km_driven_per_day] + fuel_type_input +
                                transmission_input + make_input + vehicle_class_input).reshape(1, -1)
    prediction = model.predict(prediction_input)

    # Display prediction
    st.write(f'Predicted CO2 Emissions by the car: {np.round(prediction[0], 2)}')

if __name__ == "__main__":
    st.sidebar.title('CLL725 Project')
    st.sidebar.write('Please fill in the details of your car to predict CO2 emissions.')
    st.sidebar.write('---')

    # Layout design
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.write('')
        Engine_Size = st.number_input('Engine Size')
        Vehicle_Age = st.number_input('Vehicle Age')
        Fuel_Consumption = st.number_input('Fuel Consumption')
        Energy_Consumption = st.number_input('Energy Consumption')
        Fuel_Consumption_Comb = st.number_input('Fuel Consumption Comb')
        km_driven_per_day = st.number_input('km driven per day')

    with col2:
        st.write('')
        Fuel_Type = st.selectbox('Fuel Type', ['Type_E', 'Type_X', 'Type_Z'])
        Transmission = st.selectbox('Transmission', ['A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'AM5', 'AM6', 'AM7', 'AM8', 'AM9', 'AS10', 'AS4', 'AS5', 'AS6', 'AS7', 'AS8', 'AS9', 'AV', 'AV6', 'AV7', 'AV8', 'AV10', 'M5', 'M6', 'M7'])
        Make = st.selectbox('Make', ['Luxury', 'Premium', 'Sports'])
        Vehicle_Class = st.selectbox('Vehicle Class', ['SUV', 'Sedan', 'Truck'])

    st.sidebar.write('---')
    if st.sidebar.button('Predict'):
        predict_co2_emissions(Engine_Size, Vehicle_Age, Fuel_Consumption, Energy_Consumption,
                               Fuel_Consumption_Comb, km_driven_per_day, Fuel_Type, Transmission, Make, Vehicle_Class)
