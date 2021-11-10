import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

app = Flask(__name__)
model = load_model("BTC-predict.h5")

import yfinance as yf

stock = yf.Ticker("BTC-USD")
hist = stock.history(period="5y")
df=hist
df['Date']=df.index
df=df.reset_index(drop=True)


d=30
ahead=10
n=int(hist.shape[0]*0.8)
training_set = df.iloc[:n, 1:2].values
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

@app.route('/')
def home():
    return render_template('index.html')

app.listen(process.env.PORT || 3000, function(){
  console.log("Express server listening on port %d in %s mode", this.address().port, app.settings.env);
});

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
#    int_features = [int(x) for x in request.form.values()]
#    final_features = [np.array(int_features)]

    df.loc[len(df)] = df.loc[len(df) - 1]
    dataset_train = df.iloc[:n, 1:2]
    dataset_test = df.iloc[n:, 1:2]
    dataset_total = pd.concat((dataset_train, dataset_test), axis=0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - d:].values
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)
    X_test = []
    for i in range(d, inputs.shape[0]):
        X_test.append(inputs[i - d:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    print(X_test.shape)
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    print("Tomorrow's predicted price = $", float(predicted_stock_price[-1]))
    #return float(predicted_stock_price[-1])
    return render_template('index.html', prediction_text='tomorrows prediction $ {}'.format(float(predicted_stock_price[-1])))
    #prediction = model.predict()

    #output = round(prediction[0], 2)

    #return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
