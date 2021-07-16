
from flask import Flask ,request,jsonify,render_template

import pickle
import numpy as np



app=Flask(__name__)

model=pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index1.html')



@app.route('/predict',methods=['POST'])
def predict():
    int_features=[int(x) for x in request.form.values()] #3no feature experience,score ko i/p lega
    final_features=[np.array(int_features)]  #3no ko 2d array k rup me store krega final _feature me
    prediction=model.predict(final_features)  #then predict krega 


    


    output=round(prediction[0],2)
    return render_template('index1.html',prediction_text='employ salary is {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.get_json(force=True)
    prediction=model.predict([np.array(list(data.values()))])


    output=prediction[0]
    return jsonify(output)

    if(__name__)=='__main__':
        app.run(port=5000)