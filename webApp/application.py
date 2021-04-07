from flask import Flask, render_template, flash, request, jsonify, Markup
import logging, io, os, sys
import pandas as pd
import numpy as np
import scipy
import pickle
from modules.custom_transformers import *

application = Flask(__name__)

np.set_printoptions(precision=2)

#Model features
gbm_model0 = None
gbm_model1 = None
gbm_model2 = None
gbm_model3 = None
gbm_model4 = None

features = ['Brand', 'Model', 'Year', 'Kilometers_Driven', 
        'Fuel_Type', 'Transmission', 'Owner_Type', 'Mileage', 'Engine', 
        'Power', 'Seats']


@application.before_first_request
def startup():

    global gbm_model0, gbm_model1, gbm_model2, gbm_model3, gbm_model4, model2brand
    
    # gbm model
    with open('static/GBM_Regressor_0.pkl', 'rb') as f:
        gbm_model0 = pickle.load(f)

    with open('static/GBM_Regressor_1.pkl', 'rb') as f:
        gbm_model1 = pickle.load(f)

    with open('static/GBM_Regressor_2.pkl', 'rb') as f:
        gbm_model2 = pickle.load(f)

    with open('static/GBM_Regressor_3.pkl', 'rb') as f:
        gbm_model3 = pickle.load(f)

    with open('static/GBM_Regressor_4.pkl', 'rb') as f:
        gbm_model4 = pickle.load(f)

        # min, max, default values to categories mapping dictionary
    with open('static/Dictionaries.pkl', 'rb') as f:
        default_dict,min_dict, max_dict, default_dict_mapped = pickle.load(f)

    # Encoded values to categories mapping dictionary
    with open('static/Encoded_dicts.pkl', 'rb') as f:
        brands_Encdict,models_Encdict,locations_Encdict,fuel_types_Encdict,transmissions_Encdict,owner_types_Encdict = pickle.load(f)

    with open('static/model2brand.pkl', 'rb') as f:
        model2brand = pickle.load(f)

@application.errorhandler(500)
def server_error(e):
    logging.exception('some eror')
    return """
    And internal error <pre>{}</pre>
    """.format(e), 500

@application.route("/", methods=['POST', 'GET'])
def index():
     # Encoded values to categories mapping dictionary
      # Encoded values to categories mapping dictionary
    with open('static/Encoded_dicts.pkl', 'rb') as f:
        brands_Encdict,models_Encdict,locations_Encdict,fuel_types_Encdict,transmissions_Encdict,owner_types_Encdict = pickle.load(f)


    return render_template( 'index.html', model2brand = model2brand,models_Encdict = models_Encdict,locations_Encdict = locations_Encdict, fuel_types_Encdict = fuel_types_Encdict, transmissions_Encdict = transmissions_Encdict, owner_types_Encdict = owner_types_Encdict, brands_Encdict = brands_Encdict,price_prediction = 17.09)



# accepts either deafult values or user inputs and outputs prediction 
@application.route('/background_process', methods=['POST', 'GET'])
def background_process():
    Brand = request.args.get('Brand')                                        
    Model = request.args.get('Model')
    Location = request.args.get('Location')  
    types = request.args.get(('Types'))                                
    Year = int(request.args.get('Year'))                                          
    Kilometers_Driven = float(request.args.get('Kilometers_Driven'))                
    Fuel_Type = request.args.get('Fuel_Type')
    Transmission = request.args.get('Transmission')
    Owner_Type = request.args.get('Owner_Type')
    Mileage = float(request.args.get('Mileage'))                                    
    Engine = float(request.args.get('Engine'))                                      
    Power = float(request.args.get('Power'))                                        
    Seats = float(request.args.get('Seats'))

	# values stroed in list later to be passed as df while prediction
    user_vals = [Brand, Model, Location, Year, Kilometers_Driven, 
        Fuel_Type, Transmission, Owner_Type, Mileage, Engine, 
        Power, Seats]


    x_test_tmp = pd.DataFrame([user_vals],columns = features)
    float_formatter = "{:.2f}".format

    if types == "Sedan":
        pred = float_formatter(np.exp(gbm_model0.predict(x_test_tmp[features])[0])*2000)
    elif types == "Suv":
        pred = float_formatter(np.exp(gbm_model1.predict(x_test_tmp[features])[0])*2000)
    elif types == "Trucks":
        pred = float_formatter(np.exp(gbm_model2.predict(x_test_tmp[features])[0])*2000)
    elif types == "Sports":
        pred = float_formatter(np.exp(gbm_model3.predict(x_test_tmp[features])[0])*2000)
    else:
        pred = float_formatter(np.exp(gbm_model4.predict(x_test_tmp[features])[0])*2000)


    return jsonify({'price_prediction':pred})

# when running app locally
if __name__ == '__main__':
    application.run(debug = True) 