from flask import Flask, request
import numpy as np
import pandas as pd
import json
import pickle
import os

app = Flask(__name__)
# Load model and scaler files
model_path = os.path.join(os.path.pardir, 'models')
model_filepath = os.path.join(model_path, 'lr_model.pkl')
scaler_filepath = os.path.join(model_path, 'lr_scaler.pkl')

scaler = pickle.load(open(scaler_filepath))
model = pickle.load(open(model_filepath))

# columns
columns = ['Age', 'Fare', 'Family_size', 'Is_mother', 'Is_male',
       'Deck_A', 'Deck_B', 'Deck_C', 'Deck_D', 'Deck_E', 'Deck_F', 'Deck_G',
       'Deck_Z', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Title_Lady',
       'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Officer',
       'Title_Sir', 'Fare_Bin_Very_Low', 'Fare_Bin_Low', 'Fare_Bin_High',
       'Fare_Bin_Very_High', 'Embarked_C', 'Embarked_Q', 'Embarked_S',
       'Is_adult']

@pp.route('/api', methods=['POST'])
def make_predictions():
    # read json object and convert to json string
    data = json.dumps(requests.get_json(force=True))
    # create pandas dataframe using json string
    df = pd.read_json(data)
    # extract passengerids
    passenger_ids = df.PassengerId.ravel()
    # actual survivedd values
    actuals = df.Survived.ravel()
    # exctract required columns and convert to matrix
    X = df[columns].to_numpy().astype('float')
    # transform the input
    X_scaled = scaler.transform(X)
    # make predictions
    predictions = model.predict(X_scaled)
    # create response dataframe
    df_response = pd.DataFrame({'PassengerId': passenger_ids, 'Predicted': predictions, 'Actuals': actuals})
    # return json
    return df_response.to_json()
if __name__ == "__main__":
    # host flask app at port 10001
    app.run(port=10001, debug=True)
    
