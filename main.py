import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings

# Suppress the warning
warnings.filterwarnings("ignore")

data = pd.read_csv('cancer.txt', delimiter=',', encoding='latin1')
data.columns = ['Cancer_Type','Lump','Cough','Chest Pain','Shortness of Breath','Weight Loss','Fatigue','Changes in Bowel Habits','Changes in Urination','Skin Changes','Jaundice','Skin Lesions','Infections','Bruising/Bleeding','Swollen Lymph Nodes','Blood in Urine','Pain','Fever','Headache','Nausea','Vomiting']
X = data.drop(columns=['Cancer_Type'])
y = data['Cancer_Type']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the model: {accuracy * 100:.2f}%")


# Save the model to a file
joblib.dump(model, 'cancer_prediction_model.pkl')

# Now, you can use the trained model to make predictions for new data
# Replace the values in the 'new_data' variable with the symptoms (0 or 1)
new_data = np.array([[0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 ,0, 0, 0]])  # Example symptoms

predicted_cancer_type = model.predict(new_data)
print(f"Predicted Cancer Type: {predicted_cancer_type[0]}")


import json

model_json = model.to_json()  # Assuming your model supports conversion to JSON
with open('cancer_prediction_model.json', 'w') as json_file:
    json.dump(model_json, json_file)

# Optionally, save the feature names as well
with open('feature_names.json', 'w') as json_file:
    json.dump(X.columns.tolist(), json_file)