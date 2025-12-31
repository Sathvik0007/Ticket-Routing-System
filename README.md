# Customer Support Ticket Routing System

An end-to-end machine learning project that automatically routes customer support tickets to the appropriate queue based on ticket text. The project focuses not only on prediction accuracy, but also on understanding label ambiguity and designing a system that behaves safely when predictions are uncertain.

---

## 📋 Problem Overview

Customer support teams receive large volumes of unstructured tickets covering billing, technical issues, product questions, outages, and general inquiries. Manual triage is slow and inconsistent.  
This project explores whether ticket text alone can be used for reliable routing and how to design a system that handles real-world ambiguity.

---

## 📊 Dataset

- **Features:** Support tickets containing `subject`, `body`, and metadata.
- **Targets:** Multiple support queues (Billing, Technical Support, Product Support, etc.).
- **Languages:** Dataset includes both English and German tickets (English used for the baseline model).
- **Preprocessing:**
  - Combined `subject` and `body` into a single text field.
  - Removed data leakage (`answer` column).
  - Dropped sparse `tag_*` columns.

---

## ⚙️ Approach

### Text Representation
- Used **TF-IDF (unigrams + bigrams)** to convert ticket text into numerical features.
- High-dimensional, sparse feature space suitable for linear classifiers.

### Supervised Learning
- **Final Model:** Linear SVM with class weighting.
- Evaluation metrics: precision, recall, and F1-score.
- Strong performance on clearly defined queues; confusion observed in semantically overlapping categories.

### Unsupervised Analysis
- Applied **KMeans clustering** on TF-IDF vectors.
- Compared clusters with true labels to study label noise and overlap.
- Used strictly for analysis and interpretability, not prediction.

### Hybrid Decision Logic (Uncertainty Handling)
- Confidence-aware routing strategy:
  - High-confidence predictions → auto-routing.
  - Low-confidence predictions → manual triage.
- Reflects real-world production constraints where incorrect routing is costly.

---

## 🛠 Tech Stack

- Python
- scikit-learn (TF-IDF, Linear SVM, KMeans)
- Streamlit
- Joblib

---


# 🚀 Running the Project

### 1. Install Dependencies
Make sure Python is installed and you are in the project root directory, then install all required packages:

```bash
pip install -r requirements.txt

```

### 2. Train and Save the Model

Run the training script or notebook to train the model. After training, save the trained artifacts using Joblib:

```python
joblib.dump(svm_model, "svm_model.pkl")
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")

```

This step generates the serialized model and vectorizer files required by the application.

### 3. Launch the Application

Start the Streamlit application from the project root directory:

```bash
streamlit run app.py

```

Once launched, the app will open in your browser and allow you to input customer support tickets and observe how they are routed based on model confidence.


## 💡 Key Insights
 - Linear models perform extremely well on sparse TF-IDF text features.
 - Label ambiguity places a natural ceiling on achievable accuracy.
 - System-level design choices matter more than marginal metric gains.

## 🏁 Conclusion
This project goes beyond basic classification by analyzing label ambiguity, validating model limitations, and designing a realistic hybrid routing system. The emphasis is on correctness, interpretability, and safe behavior under uncertainty.

Link: [https://ticket-routing-system-k8fvdautsawxdek7uh5xlu.streamlit.app/]
