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

@app.route('/predict', methods=['GET','POST'])
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

@app.route('/train', methods=['GET'])
def train():
    #reading data from database
    query = 'select * from "Activity"'
    selected = ['SERVICE', 'SOURCE_BYTES', 'DESTINATION_BYTES', 'ATTACK',
                'SOURCE_PORT_NUMBER', 'DESTINATION_PORT_NUMBER', 'PROTOCOL']
    frame = pd.read_sql(query, con=engine, columns=selected)
    frame=frame.drop(['SOURCE_IP_ADDRESS','DESTINATION_IP_ADDRESS','TIMESTAMP',
                      'LOCATION_ABB','LOCATION'], axis=1)
    print(frame.columns)

    # Get categorical features for frame
    categorical_features = list(frame.columns[frame.dtypes == object].values)
    print('categorical features:')
    print(categorical_features)
    # Get numerical features for frame
    numerical_features = list(frame.columns[frame.dtypes != object].values[:-1])
    print('numerical_features')
    print(numerical_features)

    # dropping high cardinality columns for frame
    cardinality = (frame[categorical_features].nunique() / frame[categorical_features].count())
    to_drop = cardinality.index[cardinality > 0.5].values
    for feat in to_drop:
        frame = frame.drop(feat, axis=1)
        categorical_features.remove(feat)
    print('dropped high cardinality values!')

    # train-test split for frame
    X = frame.drop('ATTACK', axis=1)
    y = frame['ATTACK']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print('Splitted the dataset into test and train')

    # replacing inf and -inf values with Nan for frame
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    print('Transformed inf values into Nan in X_train')

    # Perform Weight of Evidence transformation on categorical features
    unseen_val = 0
    alpha = 1
    transform_table = {}
    for feat in categorical_features:
        vals = y_train.groupby(X_train[feat]).agg(['sum', 'count'])
        numerator = (vals['sum'] + alpha) / (y_train.sum() + 2 * alpha)
        denominator = ((vals['count'] - vals['sum']) + alpha) / (y_train.count() - y_train.sum() + 2 * alpha)
        transform_table[feat] = np.log(numerator / denominator)
    # encode datasets
    X_train_c = X_train.copy()
    X_test_c = X_test.copy()
    for feat in categorical_features:
        X_train_c.loc[:, feat] = X_train_c[feat].map(transform_table[feat]).fillna(unseen_val)
        X_test_c.loc[:, feat] = X_test_c[feat].map(transform_table[feat]).fillna(unseen_val)

    print('Performed WOE transformation on categorical features')

    # replacing inf and -inf values with Nan
    X_train_c = X_train_c.replace([np.inf, -np.inf], np.nan)
    print('Transformed inf values into Nan in X_train')
    X_test_c = X_test_c.replace([np.inf, -np.inf], np.nan)
    print('Transformed inf values into Nan in X_test')


    # Filling missing data
    # use median to fill missing values
    imputer = SimpleImputer(strategy='median')
    print('Imputer function is done!')
    # fit on training dataset
    imputer.fit(X_train_c)
    print('X_train is fitted to the imputer')
    # fill missing values
    X_train_c = pd.DataFrame(imputer.transform(X_train_c), columns=X.columns)
    X_test_c = pd.DataFrame(imputer.transform(X_test_c), columns=X.columns)
    print('Missing values are filled')

    # Remove highly correlated features
    corr_table = X_train_c.corr()
    triu = corr_table.where(np.triu(np.ones(corr_table.shape), k=1).astype(bool))
    to_drop = [feat for feat in triu.columns if any(triu[feat] > 0.95)]

    X_train_c = X_train_c.drop(to_drop, axis=1)
    X_test_c = X_test_c.drop(to_drop, axis=1)

    for feat in to_drop:
        if feat in categorical_features:
            categorical_features.remove(feat)
        else:
            numerical_features.remove(feat)
    print('Highly correlated values are dropped')

    # capture a list of columns that will be used for prediction
    global model_columns
    model_columns = list(X.columns)
    joblib.dump(model_columns, model_columns_file_name)
    print('Columns are captured for prediction')

    # Making predictions
    global clf
    clf = RandomForestClassifier(random_state=42, n_jobs=-1)
    print('CLF is made!')
    start = time.time()
    clf.fit(X_train_c, y_train)
    print('Model is fitted into CLF')
    joblib.dump(clf, model_file_name)

    message1 = 'Trained in %.5f seconds' % (time.time() - start)
    message2 = 'Model training score: %s' % clf.score(X_train_c, y_train)
    return_message = 'Success. \n{0}. \n{1}.'.format(message1, message2)
    return return_message

@app.route('/wipe', methods=['GET'])
def wipe():
    try:
        shutil.rmtree('model')
        os.makedirs(model_directory)
        return 'Model wiped'

    except Exception as e:
        print(str(e))
        return 'Could not remove and recreate the model directory'

@app.route('/')
def hello_world():
    return 'Welcome to our ML model!'


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
