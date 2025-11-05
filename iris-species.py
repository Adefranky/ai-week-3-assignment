# ===========================================
# Iris Species Classification using Decision Tree (Scikit-learn)
# Dataset: iris.csv
# ===========================================

# Step 1: Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# -------------------------------------------
# Step 2: Load the dataset
# -------------------------------------------
data = pd.read_csv('iris.csv')

print("First 5 rows of the dataset:")
print(data.head())

# -------------------------------------------
# Step 3: Drop the 'Id' column (not useful for prediction)
# -------------------------------------------
if 'Id' in data.columns:
    data = data.drop(columns=['Id'])

# -------------------------------------------
# Step 4: Check for missing values
# -------------------------------------------
print("\nMissing values in each column:")
print(data.isnull().sum())

# If missing values exist, we will impute (replace with mean)
imputer = SimpleImputer(strategy='mean')
data.iloc[:, :-1] = imputer.fit_transform(data.iloc[:, :-1])

# -------------------------------------------
# Step 5: Encode the target labels (Species)
# -------------------------------------------
label_encoder = LabelEncoder()
data['Species'] = label_encoder.fit_transform(data['Species'])

# -------------------------------------------
# Step 6: Separate features (X) and target (y)
# -------------------------------------------
X = data.iloc[:, :-1]  # Features (all except last column)
y = data['Species']    # Target column

# -------------------------------------------
# Step 7: Split data into training and testing sets
# -------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------------
# Step 8: Train the Decision Tree Classifier
# -------------------------------------------
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# -------------------------------------------
# Step 9: Make predictions
# -------------------------------------------
y_pred = model.predict(X_test)

# -------------------------------------------
# Step 10: Evaluate the model
# -------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')

print("\nModel Evaluation Metrics:")
print(f" Accuracy: {accuracy:.2f}")
print(f" Precision: {precision:.2f}")
print(f" Recall: {recall:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# -------------------------------------------
# Step 11: Visualize the Decision Tree
# -------------------------------------------
plt.figure(figsize=(12,8))
plot_tree(model, feature_names=X.columns, class_names=label_encoder.classes_, filled=True)
plt.title("Decision Tree for Iris Classification")
plt.savefig("decision_tree.png")
print("Decision Tree saved as 'decision_tree.png' in your folder.")


# -------------------------------------------
# Step 12: Example Prediction
# -------------------------------------------
print("\nExample prediction for the first test sample:")
sample = X_test.iloc[0:1]  # keeps dataframe structure with column names
predicted_class = label_encoder.inverse_transform(model.predict(sample))
print(f"Predicted species: {predicted_class[0]}")
