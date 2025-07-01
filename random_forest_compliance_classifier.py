import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
df = pd.read_csv("compliance_dataset.csv")  # Use your dataset path

# Step 2: TF-IDF Vectorization of ingredient list
vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(", "))
X = vectorizer.fit_transform(df["ingredient_list"])
y = df["compliance_status"]

# Step 3: Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Train Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Step 5: Make Predictions
y_pred = rf_model.predict(X_test)

# Step 6: Evaluate Model
print("\nClassification Report (Random Forest):\n")
print(classification_report(y_test, y_pred))

# Step 7: Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Greens",
            xticklabels=["Non-compliant", "Compliant"],
            yticklabels=["Non-compliant", "Compliant"])
plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
