import os
import shutil
import time
import traceback
import json

from flask import Flask, request, jsonify

import pandas as pd
import sqlalchemy
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import joblib

engine=sqlalchemy.create_engine('postgresql://postgres:123@localhost/honeypot')
con=engine.connect()


model_directory = 'model'
location = '\\Users\\aycakaya\\PycharmProjects\\MLmodel\\model'
model_file_name = os.path.join(location, 'model.pkl')
model_columns_file_name = os.path.join(location, 'model_columns.pkl')

# These will be populated at training time
model_columns = None
clf = None

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if clf:
        try:
            #json_ = request.json
            json_="\\Users\\aycakaya\\PycharmProjects\\MLmodel\\predict.json"
            df = pd.read_json(json_)
            query = pd.get_dummies(df)

            # https://github.com/amirziai/sklearnflask/issues/3
            # Thanks to @lorenzori
            query = query.reindex(columns=model_columns, fill_value=0)

            prediction = list(clf.predict(query))

            # Converting to int from int64
            return jsonify({"prediction": list(map(int, prediction))})

        except Exception as e:

            return jsonify({'error': str(e), 'trace': traceback.format_exc()})
    else:
        print('train first')
        return 'no model here'




if __name__ == '__main__':
    try:
        clf = joblib.load(model_file_name)
        print('model loaded')
        model_columns = joblib.load(model_columns_file_name)
        print('model columns loaded')

    except Exception as e:
        print('No model here')
        print('Train first')
        print(str(e))
        clf = None
    app.run(debug=True)
