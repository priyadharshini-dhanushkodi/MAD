# ================================
# 1. IMPORT LIBRARIES
# ================================
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


# ================================
# 2. CONVERT TSV TO CSV
# ================================
input_file = "SMSSpamCollection"     # <-- your dataset
output_file = "sms_data.csv"

# Read TSV (tab-separated)
df = pd.read_csv(input_file, sep="\t", header=None)

# Assign column names
df.columns = ["label", "message"]

# Save as CSV
df.to_csv(output_file, index=False)

print("TSV converted to CSV successfully.\n")


# ================================
# 3. LOAD CSV
# ================================
df = pd.read_csv(output_file)

print("First 5 rows:\n", df.head(), "\n")


# ================================
# 4. PREPROCESSING
# ================================
# Convert labels to numbers
df["label"] = df["label"].map({"ham": 0, "spam": 1})

X = df["message"]
y = df["label"]


# ================================
# 5. TRAIN-TEST SPLIT
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ================================
# 6. TEXT VECTORIZATION
# ================================
vectorizer = TfidfVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# ================================
# 7. MODEL TRAINING
# ================================
model = MultinomialNB()
model.fit(X_train_vec, y_train)


# ================================
# 8. PREDICTION & ACCURACY
# ================================
y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)


# ================================
# 9. TEST WITH NEW MESSAGE
# ================================
sample_msg = ["Congratulations! You won a free prize. Call now!"]

sample_vec = vectorizer.transform(sample_msg)
prediction = model.predict(sample_vec)

if prediction[0] == 1:
    print("\nMessage:", sample_msg[0])
    print("Prediction: SPAM")
else:
    print("\nMessage:", sample_msg[0])
    print("Prediction: HAM")