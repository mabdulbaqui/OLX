from distutils.log import debug
from flask import Flask, render_template, request
from helpers.dummies import *
import joblib

app = Flask(__name__)
app.debug = True
model = joblib.load('models/model.h5')
scaler = joblib.load('models/scaler.h5')



@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    all_data = request.form
    area = int(all_data['area'])
    meterprice = int(all_data['meterprice'])
    down_payment = int(all_data['down_payment'])
    electricity_meter = int(all_data['electricity_meter'])
    balcony = int(all_data['balcony'])
    water_meter = int(all_data['water_meter'])
    elevator = int(all_data['elevator'])
    security=int(all_data['security'])
    private_garden = int(all_data['private_garden'])
    natural_gas = int(all_data['natural_gas'])
    pool = int(all_data['pool'])
    location = location_dummies[all_data['location']]
    compound = compound_dummies[all_data['compound']]
    bedrooms = bedrooms_dummies[all_data['bedrooms']]
    level = level_dummies[all_data['level']]
    bathrooms = bathrooms_dummies[all_data['bathrooms']]
    payment_option = payment_option_dummies[all_data['payment_option']]
    delivery_term = delivery_term_dummies[all_data['delivery_term']]
    month=month_dummies[all_data['month']]
    
    
    
    
    x = [ area  , down_payment , electricity_meter , balcony , water_meter  , elevator , security,natural_gas , private_garden , pool , meterprice]
    x += location + compound + bedrooms + month + level + bathrooms + payment_option + delivery_term 


    x = scaler.transform([x])
    price = round(model.predict(x)[0])
    
    return render_template('prediction.html', price=price)









if __name__ == "__main__":  
    app.run(debug=True)
