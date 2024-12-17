    
import numpy as np
from flask import Flask, request, render_template
import pickle


app = Flask(__name__)
model = pickle.load(open('Logistic_Regression.pkl', 'rb'))
print(type(model))  # برای بررسی نوع مدل

def get_dummies(value, possible_values):
    return [1 if val == value else 0 for val in possible_values]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    GridFins = float(request.form.get('GridFins'))
    Reused = float(request.form.get('Reused'))
    Legs = float(request.form.get('Legs'))
    Block = float(request.form.get('Block'))
    ReusedCount = float(request.form.get('ReusedCount'))
    PayloadMass = float(request.form.get('PayloadMass'))
    Orbit = request.form.get('Orbit')
    LaunchSite = request.form.get('LaunchSite')
    Outcome = request.form.get('Outcome')
    LandingPad = request.form.get('LandingPad')
    #Serial = request.form.get('Serial')
    print(Orbit)

    # Get Dummies for the two categorical features
    Orbit_encoded = get_dummies(Orbit, 
                                ['Orbit_ES-L1', 'Orbit_GEO', 'Orbit_GTO',
              'Orbit_HEO', 'Orbit_ISS', 'Orbit_LEO', 
              'Orbit_MEO', 'Orbit_PO', 'Orbit_SO', 
              'Orbit_SSO', 'Orbit_VLEO',])
    LaunchSite_encoded = get_dummies(LaunchSite, 
                                     ['LaunchSite_CCAFS SLC 40',
                                      'LaunchSite_KSC LC 39A',
                                      'LaunchSite_VAFB SLC 4E'])
    
    Outcome_encoded = get_dummies(Outcome, 
                                     ['Outcome_False ASDS', 'Outcome_False Ocean',
               'Outcome_False RTLS', 'Outcome_None ASDS', 'Outcome_None None', 
               'Outcome_True ASDS', 'Outcome_True Ocean', 'Outcome_True RTLS'])

    LandingPad_encoded = get_dummies(LandingPad , ['LandingPad_5e9e3032383ecb267a34e7c7',
       'LandingPad_5e9e3032383ecb554034e7c9',
       'LandingPad_5e9e3032383ecb6bb234e7ca',
       'LandingPad_5e9e3032383ecb761634e7cb',])
    

    
    print(Orbit_encoded)
    # Combining all features
    final_features = [0] +  [GridFins] + [Reused] + [Legs] +[PayloadMass] + [Block]  + [ReusedCount] + Orbit_encoded + LaunchSite_encoded + Outcome_encoded + LandingPad_encoded + [0]
    print(f'---->  {final_features}')
    final_features = [np.array(final_features)]
    print(f'---->  {final_features}')
    prediction = model.predict(final_features)
    print(prediction)

    output = round(prediction[0], 2)
    print(output)

    return render_template('index.html', prediction_text='فرود  {} '.format(prediction))

if __name__ == '__main__':
    app.run(debug=True, port=5012)
