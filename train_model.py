import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

csv_path = os.path.join('dataset', 'dna.csv')

if not os.path.exists(csv_path):
    print(f"❌ Error: Could not find '{csv_path}'.")
    exit()

print("📦 Loading dataset...")
df = pd.read_csv(csv_path)

print(f"📊 Dataset loaded successfully with {len(df)} rows and {len(df.columns)} columns.")

# Last column = label, rest = features
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("🤖 Training the Multinomial Naive Bayes model...")
classifier = MultinomialNB(alpha=0.1)
classifier.fit(X_train, y_train)

print("🎯 Evaluating model performance...")
y_pred = classifier.predict(X_test)

print("\n=== Model Results ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("✅ Training complete!")

# Save the trained model
joblib.dump(classifier, "dna_model.pkl")
print("✅ Model saved as dna_model.pkl")