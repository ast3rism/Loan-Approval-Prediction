import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

# Load and preprocess data
def load_data():
    data = pd.read_csv('LoanApprovalPrediction.csv')
    data.drop(['Loan ID'], axis=1, inplace=True)
    return data

def preprocess_data(data):
    label_encoder = LabelEncoder()
    obj = (data.dtypes == 'object')
    for col in list(obj[obj].index):
        data[col] = label_encoder.fit_transform(data[col])
    
    for col in data.columns:
        data[col] = data[col].fillna(data[col].mean())
    
    X = data.drop(['Loan Status'], axis=1)
    Y = data['Loan Status']
    return X, Y

# Load data and preprocess it
data = load_data()
X, Y = preprocess_data(data)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# Train models
models = {
    # 'K-Neighbors Classifier': KNeighborsClassifier(n_neighbors=3),
    'Random Forest Classifier': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(solver='liblinear')
}

trained_models = {}
for model_name, model in models.items():
    model.fit(X_train, Y_train)
    trained_models[model_name] = model

# Streamlit app
st.title('Loan Approval Prediction App')

# User inputs
st.header('Applicant Information')
applicant_info = {}
for feature in X.columns:
    if feature == 'Gender':
        applicant_info[feature] = st.selectbox(feature, ['Male', 'Female'])
    elif feature == 'Married':
        applicant_info[feature] = st.selectbox(feature, ['Yes', 'No'])
    elif feature == 'Education':
        applicant_info[feature] = st.selectbox(feature, ['Graduate', 'Not Graduate'])
    elif feature == 'Property Area':
        applicant_info[feature] = st.selectbox(feature, ['Urban', 'Semiurban', 'Rural'])
    elif feature in data.select_dtypes(include=['object']).columns:
        choices = sorted(data[feature].unique())  # Use sorted unique values for other categorical features
        applicant_info[feature] = st.selectbox(feature, choices)
    else:
        applicant_info[feature] = st.number_input(feature, value=float(data[feature].mean()))

# Convert user input to DataFrame
user_data = pd.DataFrame([applicant_info])
user_data = user_data[X.columns]  # Ensure column order matches

# Encode categorical inputs
label_encoder = LabelEncoder()
for feature in user_data.select_dtypes(include=['object']).columns:
    user_data[feature] = label_encoder.fit_transform(user_data[feature])

# Scale user input
user_data_scaled = scaler.transform(user_data)

# Display model accuracies
st.header('Model Accuracies')
for model_name, model in trained_models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, y_pred)
    st.write(f'{model_name}: {accuracy:.2f}')

# Make predictions
if st.button('Predict Loan Approval'):
    st.header('Loan Approval Predictions')
    for model_name, model in trained_models.items():
        prediction = model.predict(user_data_scaled)[0]
        result = 'Approved' if prediction == 1 else 'Rejected'
        st.write(f'{model_name}: {result}')