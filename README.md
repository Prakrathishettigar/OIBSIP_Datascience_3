# 📧 Spam Detection Using Naive Bayes

A machine learning project that classifies **emails as spam or ham (not spam)** using natural language processing and the **Multinomial Naive Bayes** algorithm. This project walks through the complete ML pipeline — from **data cleaning and vectorization** to **model training**, **evaluation**, and **custom email testing**.

---

## 🚀 Project Overview

The **Spam Detection** project leverages **TF-IDF vectorization** and **Naive Bayes classification** to identify spam messages. It includes **data preprocessing**, **feature extraction**, **model evaluation**, and a **custom prediction interface** for testing new emails.

---

## 🧠 Key Features

* 🧹 **Data Preprocessing**
  * Loads and cleans the SMS Spam Collection dataset
  * Handles missing values and removes duplicates
  * Converts labels (`ham`, `spam`) into binary format

* 🧾 **Feature Extraction**
  * Uses **TF-IDF Vectorizer** to convert text into numerical features
  * Removes stop words and limits features to top 3000 terms

* 🤖 **Model Building**
  * Trains a **Multinomial Naive Bayes** classifier
  * Evaluates using **accuracy**, **confusion matrix**, and **classification report**

* 💬 **Custom Email Testing**
  * Accepts user-defined email text
  * Predicts whether it's spam or not
  * Displays prediction with confidence score

---

## 🧰 Tech Stack

| Component    | Technology                                       |
| ------------ | ------------------------------------------------ |
| Language     | Python 3.x                                       |
| Libraries    | Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn |
| Model        | Multinomial Naive Bayes                          |
| Dataset      | SMS Spam Collection (CSV)                        |
| Vectorizer   | TF-IDF (Term Frequency–Inverse Document Frequency) |

---

## 📂 Project Structure

```
spam_detection.py         # Main Python script
spam.csv                  # Dataset file
```

---

## ⚙️ How to Run

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/Spam-Detection-NaiveBayes.git
cd Spam-Detection-NaiveBayes
```

### 2️⃣ Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 3️⃣ Run the Script

```bash
python spam_detection.py
```

---

## 📈 Model Performance

* **Algorithm:** Multinomial Naive Bayes
* **Evaluation Metrics:**
  * Accuracy: ~97–99%
  * Precision, Recall, F1-score for both classes
  * Confusion Matrix for visual insight

---

## 📊 Visualizations

* Confusion Matrix Heatmap
* Label distribution before and after cleaning
* Sample predictions with confidence scores

---

## 🧪 Sample Predictions

```
[Email 1]: Congratulations! You've won a $1000 gift card...
Prediction: SPAM | Confidence: 98.45%

[Email 2]: Hi, can we schedule a meeting tomorrow...
Prediction: HAM (NOT SPAM) | Confidence: 96.12%
```

---

## 💡 Future Enhancements

* Add **web interface** using Streamlit or Flask
* Integrate **word cloud** for spam vs ham visualization
* Experiment with **other models** (e.g., SVM, Logistic Regression)
* Save and load model using `joblib` for deployment

---

## 📜 License

This project is open-source and available under the **MIT License**.
