import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# STEP 1: LOAD YOUR DATASET
print("="*60)
print("LOADING DATASET")
print("="*60)

df = pd.read_csv(r'C:\Users\PRAKRATHI\OneDrive\Desktop\OASIS INFOBYTE Internship\DataScience\4. spam.csv', encoding='latin1')
print(df.head())
# Check the actual columns in your dataset
print(f"\nOriginal columns: {df.columns.tolist()}")
print(f"Dataset shape: {df.shape}")

# Most spam datasets have extra unnamed columns - remove them
df = df.iloc[:, :2]  # Keep only first 2 columns

print(f"\nAfter selecting first 2 columns:")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst few rows:\n{df.head()}")

# Check unique labels
print(f"\nUnique labels in dataset: {df['label'].unique()}")
print(f"Label distribution:\n{df['label'].value_counts()}")

# STEP 2: DATA PREPROCESSING
print(f"\n{'='*60}")
print("DATA PREPROCESSING")
print("="*60)

# Convert labels to binary (0 = ham/not spam, 1 = spam)
if 'ham' in df['label'].values or 'spam' in df['label'].values:
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
elif 'Ham' in df['label'].values or 'Spam' in df['label'].values:
    df['label'] = df['label'].map({'Ham': 0, 'Spam': 1})
else:
    print(f"WARNING: Unknown label format. Please check your labels!")

# Remove any missing values
print(f"\nMissing values before cleaning:\n{df.isnull().sum()}")
df = df.dropna()

# Remove duplicates
duplicates_before = df.duplicated().sum()
df = df.drop_duplicates()
print(f"\nDuplicates removed: {duplicates_before}")

print(f"\nAfter cleaning:")
print(f"Dataset shape: {df.shape}")
print(f"Label distribution:\n{df['label'].value_counts()}")
print(f"Label distribution (%):\n{df['label'].value_counts(normalize=True) * 100}")

# Check if we have both classes
unique_labels = df['label'].nunique()
print(f"\nNumber of unique classes: {unique_labels}")
if unique_labels < 2:
    print("⚠️ WARNING: Dataset contains only ONE class! Cannot train a proper classifier.")
    print("This is why you might get 100% accuracy - the model always predicts the same class.")

# STEP 3: SPLIT DATA
print(f"\n{'='*60}")
print("SPLITTING DATA")
print("="*60)

X = df['text']
y = df['label']

# Check if we have both classes before splitting
if unique_labels >= 2:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
else:
    # If only one class, split without stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

print(f"\nTraining set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")
print(f"\nTraining set label distribution:\n{y_train.value_counts()}")
print(f"Testing set label distribution:\n{y_test.value_counts()}")

# STEP 4: FEATURE EXTRACTION 
print(f"\n{'='*60}")
print("FEATURE EXTRACTION")
print("="*60)

vectorizer = TfidfVectorizer(max_features=3000, stop_words='english', min_df=2)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"\nFeature matrix shape: {X_train_tfidf.shape}")
print(f"Number of features: {len(vectorizer.get_feature_names_out())}")

# STEP 5: TRAIN THE MODEL
print(f"\n{'='*60}")
print("TRAINING MODEL")
print("="*60)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✓ Model trained successfully!")
print(f"\nModel Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Detailed analysis
print(f"\nDetailed Analysis:")
print(f"Total test samples: {len(y_test)}")
print(f"Correct predictions: {(y_test == y_pred).sum()}")
print(f"Incorrect predictions: {(y_test != y_pred).sum()}")

# Check if accuracy is suspiciously high
if accuracy == 1.0:
    print("\n⚠️ WARNING: 100% accuracy detected!")
    print("Possible reasons:")
    print("1. Dataset is too small or too simple")
    print("2. Data leakage or duplicate samples")
    print("3. Only one class in test set")
    print("4. Model is overfitting")

print(f"\nClassification Report:")
try:
    report = classification_report(y_test, y_pred, target_names=['Ham (0)', 'Spam (1)'], zero_division=0)
    print(report)
except:
    report = classification_report(y_test, y_pred, zero_division=0)
    print(report)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Ham', 'Spam'], 
            yticklabels=['Ham', 'Spam'])
plt.title(f'Confusion Matrix - Email Spam Detection\nAccuracy: {accuracy*100:.2f}%')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()

# STEP 6: TEST WITH CUSTOM EMAIL
print(f"\n{'='*60}")
print("TESTING WITH SAMPLE EMAILS")
print("="*60)

def predict_spam(email_text):
    """
    Predict if an email is spam or not
    """
    email_tfidf = vectorizer.transform([email_text])
    prediction = model.predict(email_tfidf)
    probability = model.predict_proba(email_tfidf)
    
    result = "SPAM" if prediction[0] == 1 else "HAM (NOT SPAM)"
    confidence = probability[0][prediction[0]] * 100
    
    print(f"Prediction: {result} | Confidence: {confidence:.2f}%")
    return result

sample_emails = [
    "Congratulations! You've won a $1000 gift card. Click here to claim now!",
    "Hi, can we schedule a meeting tomorrow at 3 PM to discuss the project?",
    "URGENT! Your account will be closed. Verify your identity immediately!",
    "Thanks for your email. I'll review the document and get back to you.",
    "FREE MONEY! Act now! Limited time offer! Click here!!!",
    "Let me know if you need any help with the assignment."
]

for i, email in enumerate(sample_emails, 1):
    print(f"\n[Email {i}]: {email[:65]}...")
    predict_spam(email)

print(f"\n{'='*60}")
print("ANALYSIS COMPLETE!")
print("="*60)
print(f"\nDataset: {df.shape[0]} emails")
print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Model: Naive Bayes")
print("="*60)