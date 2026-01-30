import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

# Load dataset
data = pd.read_csv("heart.csv")

# Drop 'id' column as it's not a feature
data = data.drop("id", axis=1)

# Convert '?' to NaN for proper handling of missing values
data = data.replace('?', pd.NA)

# Identify categorical columns (excluding the target 'num')
categorical_cols = ['sex', 'dataset', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

# Convert potential numerical-like categorical columns to object/category before one-hot encoding
# This ensures columns like 'ca' and 'thal' are treated as categories even if they contain numbers
for col in categorical_cols:
    if col in data.columns:
        data[col] = data[col].astype(str).astype('category')

# Apply one-hot encoding to categorical features
X = pd.get_dummies(data.drop("num", axis=1), columns=categorical_cols, drop_first=True, dtype=int)
y = data["num"]

# Impute missing numerical values (e.g., from 'oldpeak' or other columns that might have NaNs)
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) # Scale X_test using the same scaler fitted on X_train

# Train model
model = LogisticRegression(max_iter=1000) # Increased max_iter for convergence
model.fit(X_train, y_train)

# Save model and scaler
with open("heart_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model and scaler saved successfully")