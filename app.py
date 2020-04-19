import sklearn
import numpy as np
import pickle
from flask import Flask,request,render_template,url_for,jsonify
from sklearn.externals import joblib

churn_pred = joblib.load("./churn_pred.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/result',methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        #return render_template('home.html', Pred_Value="Hai")
        int_features = [int(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
      
        pred_Value = churn_pred.predict(final_features)
        output = round(pred_Value[0], 2)
  
        if int(output)== 1:
            pred_val ='Yes'
        else: 
            pred_val ='No' 
    
        return render_template('result.html', Pred_Value=pred_val)
        

if __name__ == '__main__':
    app.run(host='0.0.0.0',port="8400",debug=True)
