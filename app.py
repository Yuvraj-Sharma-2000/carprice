import json
import pickle
from sklearn.preprocessing import StandardScaler

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
## Load the model
regmodel=pickle.load(open('regmodel.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')


def preprocess_dict(data):
    # Drop the 'Car_Name' key from the dictionary
    data.pop('Car_Name', None)

    # Calculate the 'Age' key based on the 'Year' key
    data['Age'] = 2023 - data['Year']

    # Drop the 'Year' key from the dictionary
    data.pop('Year', None)

    # Add Fuel_Type_Diesel, Fuel_Type_Petrol, and Transmission_Manual keys
    fuel_type = data.get('Fuel_Type', None)
    if fuel_type is not None and isinstance(fuel_type, str):
        if fuel_type.lower() == 'diesel':
            data['Fuel_Type_Diesel'] = 1
            data['Fuel_Type_Petrol'] = 0
        elif fuel_type.lower() == 'petrol':
            data['Fuel_Type_Diesel'] = 0
            data['Fuel_Type_Petrol'] = 1
        else:
            print(f"Could not determine Fuel_Type_Diesel and Fuel_Type_Petrol for fuel type '{fuel_type}'")

    transmission = data.get('Transmission', None)
    if transmission is not None and isinstance(transmission, str):
        if transmission.lower() == 'manual':
            data['Transmission_Manual'] = 1
        else:
            data['Transmission_Manual'] = 0
    else:
        print(f"Invalid value '{transmission}' for key 'Transmission'")

    data.pop('Seller_Type', None)
    data.pop('Fuel_Type', None)
    data.pop('Transmission', None)

    return data


@app.route('/predict_api',methods=['POST'])
def predict_api():

    data=request.json['data']

    data = preprocess_dict(data)

    print(np.array(list(data.values())).reshape(1,-1))

    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.predict(new_data)
    print(output[0])

    return jsonify(output[0])


@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=regmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="The House price prediction is {}".format(output))


if __name__=="__main__":
    app.run(debug=True)