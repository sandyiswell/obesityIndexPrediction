from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open('cdc_model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def prediction():
    data1 = request.form['lowConfidenceLimit']
    data2 = request.form['sampleSize']
    data3 = request.form['class']
    data4 = request.form['topic']
    df = np.array([[data1, data2, data3, data4]])
    # The model ingests dataframe object for prediction with column names.
    pred = round(model.predict(pd.DataFrame(df, columns=['Low_Confidence_Limit', 'Sample_Size', 'Class', 'Topic']))[0], 2)
    # return render_template('result.html', data=pred)
    return render_template('result.html', data='Predicted Body Mass Index is: {}'.format(pred))


if __name__ == "__main__":
    app.run(debug=True)



