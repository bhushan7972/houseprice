from flask import Flask, render_template, request
from joblib import dump,load
import numpy as np

app = Flask(__name__)

@app.route('/')
def student():
   return render_template('student.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      result = request.form

      model1 = load('DecisionTree.joblib')
      model2 = load('RandomForest.joblib')
      #model3=load('linerm2.joblib')
      capita_crime = request.form['capita_crime']
      residential_land_zoned = request.form['residential_land_zoned']
      non_retail_business =request.form['non_retail_business']
      River = request.form['River']
      nitric_oxides_concentration = request.form['nitric_oxides_concentration']
      number_of_rooms = request.form['number_of_rooms']
      AGE =request.form['AGE']
      distances_to_five_Boston_employment_centres =request.form['distances_to_five_Boston_employment_centres']
      accessibility_to_radial_highways =request.form['accessibility_to_radial_highways']
      TAX = request.form['TAX']
      PTRATIO = request.form['PTRATIO']
      Bk = request.form['Bk']
      LOWERSTAT =request.form['LOWERSTAT']

      feauterdata = np.array([[capita_crime, residential_land_zoned, non_retail_business, River,
                               nitric_oxides_concentration, number_of_rooms, AGE,
                               distances_to_five_Boston_employment_centres, accessibility_to_radial_highways, TAX,
                               PTRATIO, Bk, LOWERSTAT]])

      #l1= model3.predict(feauterdata)
      decisiiontreepre=model1.predict(feauterdata)
      RandomForest = model2.predict(feauterdata)



      return render_template("result.html", result1 =RandomForest,result0 = decisiiontreepre,result=result)

if __name__ == '__main__':
   app.run(debug = True)