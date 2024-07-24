from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the trained model
model = joblib.load('best_model.pkl')

# Initialize LabelEncoders
label_encoders = {}

# Function to apply label encoding to categorical columns
def apply_label_encoding(df):
    for col in ['Gender', 'Ethnicity', 'Jaundice', 'Autism', 'Country', 'Used_app_before', 'Age_desc', 'Relation']:
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col])
    return df

# Define the route for the home page
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html',input_data={})
# Define the route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the request
    data = request.form
    
    # Prepare input data
    input_data = {
        "A1_Score": data['A1_Score'],
        "A2_Score": data['A2_Score'],
        "A3_Score": data['A3_Score'],
        "A4_Score": data['A4_Score'],
        "A5_Score": data['A5_Score'],
        "A6_Score": data['A6_Score'],
        "A7_Score": data['A7_Score'],
        "A8_Score": data['A8_Score'],
        "A9_Score": data['A9_Score'],
        "A10_Score": data['A10_Score'],
        "Age": data['age'],
        "Gender": data['gender'],
        "Ethnicity": data['ethnicity'],
        "Jaundice": data['jaundice'],
        "Autism": data['autism'],
        "Country": data['country'],
        "Used_app_before": data['used_app_before'],
        "Result": data['result'],
        "Age_desc": data['age_desc'],
        "Relation": data['relation']
    }
    
    # Convert input_data to a DataFrame
    input_data_df = pd.DataFrame([input_data])
    
    # Apply label encoding
    input_data_df = apply_label_encoding(input_data_df)
    
    # Make predictions
    predictions = model.predict(input_data_df)

    if predictions == 1:
        predictions = "Yes"
    else:
        predictions = "No"
    
    # Convert predictions to a list and return as JSON
    # return jsonify(predictions.tolist())
    return render_template('index.html', prediction_text='The prediction is {}'.format(predictions),input_data=input_data)

if __name__ == '__main__':
    app.run(debug=True)
