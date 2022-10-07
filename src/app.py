from flask import Flask, jsonify, request
from flask_restful import Resource, Api, reqparse
from itsdangerous import json
import pandas as pd
import numpy
import pickle

loan_model = pickle.load(open('loan_model.sav', 'rb'))

app = Flask(__name__)
api = Api(app)

# create custom class
class RawFeats:
    def __init__(self, feats):
        self.feats = feats

    def fit(self, X, y = None):
        pass


    def transform(self, X, y = None):
        return X[self.feats]

    def fit_transform(self, X, y = None):
        self.fit(X)
        return self.transform(X)

# create endpoint for communicating with ML model
class Scoring(Resource):
    def post(self):
        json_data = request.get_json()
        df = pd.DataFrame(json_data.values(), index = json_data.keys()).transpose()
        # getting predictions from our model.
        # it is much simpler because we used pipelines during development
        res = loan_model.predict(df)
        # we cannot send numpy array as a result
        return res.tolist() 

# assign endpoint
api.add_resource(Scoring, '/scoring')

# create application run when api.py file is run directly and not imported as a module from another script
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)