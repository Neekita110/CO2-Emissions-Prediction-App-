import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

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
model = pickle.load(open('model.pkl', 'rb'))


# Flask app route for home page
@app.route('/')
def home():
    return render_template('index.html')


# Flask app route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Capture input data from form
    Engine_Size = float(request.form['Engine_Size'])
    Vehicle_Age = float(request.form['Vehicle_Age'])
    Fuel_Consumption = float(request.form['Fuel_Consumption'])
    Energy_Consumption = float(request.form['Energy_Consumption'])
    Fuel_Consumption_Comb = float(request.form['Fuel_Consumption_Comb'])
    km_driven_per_day = float(request.form['km_driven_per_day'])
    Fuel_Type = request.form['Fuel_Type']
    Transmission = request.form['Transmission']
    Make = request.form['Make']
    Vehicle_Class = request.form['Vehicle_Class']

    # Preprocess inputs
    fuel_type_values = {'Type_E': [1, 0, 0], 'Type_X': [0, 1, 0], 'Type_Z': [0, 0, 1]}
    fuel_type_input = fuel_type_values[Fuel_Type]

    transmission_values = {Transmission: 1}
    transmission_input = [transmission_values.get(key, 0) for key in
                          ['A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'AM5', 'AM6', 'AM7', 'AM8', 'AM9', 'AS10', 'AS4', 'AS5',
                           'AS6', 'AS7', 'AS8', 'AS9', 'AV', 'AV6', 'AV7', 'AV8', 'AV10', 'M5', 'M6', 'M7']]

    make_values = {'Luxury': [1, 0, 0], 'Premium': [0, 1, 0], 'Sports': [0, 0, 1]}
    make_input = make_values[Make]

    vehicle_class_values = {'SUV': [1, 0, 0], 'Sedan': [0, 1, 0], 'Truck': [0, 0, 1]}
    vehicle_class_input = vehicle_class_values[Vehicle_Class]

    # Prepare data for prediction
    prediction_input = np.array([Engine_Size, Vehicle_Age, Fuel_Consumption, Energy_Consumption,
                                 Fuel_Consumption_Comb, km_driven_per_day] + fuel_type_input +
                                transmission_input + make_input + vehicle_class_input).reshape(1, -1)

    # Make prediction
    prediction = model.predict(prediction_input)

    # Render the result in the template
    return render_template('index.html',
                           prediction_text=f'Predicted CO2 Emissions by the car: {np.round(prediction[0], 2)}')


if __name__ == "__main__":
    app.run(debug=True)
