from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
loaded_model = joblib.load('cancer_prediction_model.pkl')

# Define feature names corresponding to your symptoms
# Define feature names corresponding to your symptoms
feature_names = ['Lump','Cough','Chest Pain','Shortness of Breath','Weight Loss','Fatigue','Changes in Bowel Habits','Changes in Urination','Skin Changes','Jaundice','Skin Lesions','Infections','Bruising/Bleeding','Swollen Lymph Nodes','Blood in Urine','Pain','Fever','Headache','Nausea','Vomiting']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get user input from the form and convert to integer
            user_input = [int(request.form.get(feature, 0)) for feature in feature_names]
            
            print("Received data:", user_input) 

            # Create a DataFrame from user input
            user_data = pd.DataFrame([user_input], columns=feature_names, dtype=int)
            
            # Use the loaded model for predictions
            predicted_cancer_type = loaded_model.predict(user_data)
            print(predicted_cancer_type)
            return str(predicted_cancer_type[0])
        except ValueError:
            print("ValueError:", e)
            return "Invalid input. Please enter integers for all fields."
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
