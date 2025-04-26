from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Dropout

app = Flask(__name__)

# Load the dataset for model training
def load_data():
    df = pd.read_csv('Loan_default.csv', header=None)
    column_names = ['ID', 'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 
                    'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTI', 
                    'Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 
                    'HasDependents', 'LoanPurpose', 'HasCoSigner', 'Default']
    df.columns = column_names
    return df

# Data preprocessing
def preprocess_data(df):
    # Drop ID column
    df = df.drop('ID', axis=1)
    
    # Convert categorical variables
    categorical_cols = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 
                        'HasDependents', 'LoanPurpose', 'HasCoSigner']
    for col in categorical_cols:
        df[col] = df[col].astype('category')
    
    # Convert binary columns to numeric
    binary_cols = ['HasMortgage', 'HasDependents', 'HasCoSigner', 'Default']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0} if col != 'Default' else {1: 1, 0: 0})
    
    # Feature engineering
    df['DebtToIncome'] = df['LoanAmount'] / df['Income']
    df['LoanToIncome'] = df['LoanAmount'] / df['Income']
    df['MonthlyPayment'] = (df['LoanAmount'] * (df['InterestRate']/100/12) * 
                           (1 + df['InterestRate']/100/12)**(df['LoanTerm'])) / \
                          ((1 + df['InterestRate']/100/12)**(df['LoanTerm']) - 1)
    df['PaymentToIncome'] = df['MonthlyPayment'] / (df['Income'] / 12)
    
    # Handle infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(), inplace=True)
    
    return df

# Create and train models
def train_models(df):
    # Split data
    X = df.drop('Default', axis=1)
    y = df['Default']
    
    # Identify numerical and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['category']).columns.tolist()
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Fit the preprocessor on data
    X_processed = preprocessor.fit_transform(X)
    
    # Create and train ANN model
    def create_ann_model(input_dim):
        model = Sequential([
            Dense(64, activation='relu', input_dim=input_dim),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        return model
    
    input_dim = X_processed.shape[1]
    ann_model = create_ann_model(input_dim)
    ann_model.fit(X_processed, y, epochs=10, batch_size=32, verbose=0)
    
    # Create and train MLP model
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=32,
        learning_rate='adaptive',
        max_iter=10,
        random_state=42
    )
    mlp_model.fit(X_processed, y)
    
    # Save models and preprocessor
    ann_model.save('ann_model.h5')
    joblib.dump(mlp_model, 'mlp_model.pkl')
    joblib.dump(preprocessor, 'preprocessor.pkl')
    
    return preprocessor, ann_model, mlp_model

# Load or train models
try:
    preprocessor = joblib.load('preprocessor.pkl')
    ann_model = load_model('ann_model.h5')
    mlp_model = joblib.load('mlp_model.pkl')
    print("Models loaded successfully")
except:
    print("Training models...")
    df = load_data()
    df = preprocess_data(df)
    preprocessor, ann_model, mlp_model = train_models(df)
    print("Models trained and saved successfully")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        data = request.form.to_dict()
        
        # Convert form data to appropriate types
        input_data = {
            'Age': int(data['age']),
            'Income': float(data['income']),
            'LoanAmount': float(data['loanAmount']),
            'CreditScore': int(data['creditScore']),
            'MonthsEmployed': int(data['monthsEmployed']),
            'NumCreditLines': int(data['numCreditLines']),
            'InterestRate': float(data['interestRate']),
            'LoanTerm': int(data['loanTerm']),
            'DTI': float(data['dti']),
            'Education': data['education'],
            'EmploymentType': data['employmentType'],
            'MaritalStatus': data['maritalStatus'],
            'HasMortgage': data['hasMortgage'],
            'HasDependents': data['hasDependents'],
            'LoanPurpose': data['loanPurpose'],
            'HasCoSigner': data['hasCoSigner']
        }
        
        # Create DataFrame from input data
        input_df = pd.DataFrame([input_data])
        
        # Feature engineering
        input_df['DebtToIncome'] = input_df['LoanAmount'] / input_df['Income']
        input_df['LoanToIncome'] = input_df['LoanAmount'] / input_df['Income']
        input_df['MonthlyPayment'] = (input_df['LoanAmount'] * (input_df['InterestRate']/100/12) * 
                                     (1 + input_df['InterestRate']/100/12)**(input_df['LoanTerm'])) / \
                                    ((1 + input_df['InterestRate']/100/12)**(input_df['LoanTerm']) - 1)
        input_df['PaymentToIncome'] = input_df['MonthlyPayment'] / (input_df['Income'] / 12)
        
        # Handle infinite values
        input_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        input_df.fillna(0, inplace=True)
        
        # Convert categorical columns
        categorical_cols = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 
                            'HasDependents', 'LoanPurpose', 'HasCoSigner']
        for col in categorical_cols:
            input_df[col] = input_df[col].astype('category')
        
        # Preprocess input data
        input_processed = preprocessor.transform(input_df)
        
        # Make predictions
        ann_pred = float(ann_model.predict(input_processed)[0][0])
        mlp_pred = float(mlp_model.predict_proba(input_processed)[0][1])
        
        # Average prediction
        avg_pred = (ann_pred + mlp_pred) / 2
        
        # Determine risk level
        if avg_pred < 0.2:
            risk_level = "Low Risk"
        elif avg_pred < 0.5:
            risk_level = "Moderate Risk"
        else:
            risk_level = "High Risk"
        
        # Return predictions
        result = {
            'ann_prediction': round(ann_pred * 100, 2),
            'mlp_prediction': round(mlp_pred * 100, 2),
            'average_prediction': round(avg_pred * 100, 2),
            'risk_level': risk_level,
            'derived_features': {
                'debt_to_income': round(float(input_df['DebtToIncome'][0]), 4),
                'loan_to_income': round(float(input_df['LoanToIncome'][0]), 4),
                'monthly_payment': round(float(input_df['MonthlyPayment'][0]), 2),
                'payment_to_income': round(float(input_df['PaymentToIncome'][0]), 4)
            }
        }
        
        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
