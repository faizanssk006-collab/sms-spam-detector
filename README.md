# 📬 SMS Spam Classification - 2nd Year Project

A clean, humanistic, and lightweight Machine Learning project to classify SMS messages as either **Legitimate (Ham)** or **Spam** using Natural Language Processing (NLP).

---

## 📁 Project Structure

```text
├── sms_spam_collection (1).csv   # Original SMS dataset (5,572 messages)
├── sms_spam_classification.ipynb # Clean, step-by-step Jupyter Notebook
├── index.html                    # Modern web interface for GitHub Pages
├── model_data.js                 # Exported model weights for client-side inference
├── app.py                       # Modern Streamlit Web Application
├── train.py                     # Standalone model training & evaluation script
├── sms_spam_model.joblib        # Trained TF-IDF + Naive Bayes pipeline artifact
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation & instructions
```

---

## 🎯 Key Highlights & Results

- **No Overcomplication:** Direct, explainable pipeline without robotic boilerplate or heavy offline dependencies.
- **Preprocessing:** Casing normalization, URL removal, punctuation removal, and whitespace stripping.
- **Feature Extraction:** `TfidfVectorizer` (sub-linear TF scaling, English stopwords, 3,000 max features).
- **Primary Model:** **Multinomial Naive Bayes**
  - **Accuracy:** `97.39%`
  - **Precision:** `100.0%` *(Crucial for spam filtering — 0 legitimate messages misclassified as spam)*
  - **Recall:** `79.39%`
  - **F1-Score:** `88.51%`
- Compared against **Logistic Regression** (`95.94%` accuracy, `97.8%` precision).

---

## 🌐 Deploy to GitHub Pages (github.io)

This repository includes a standalone, zero-server web version (`index.html` + `model_data.js`) designed to run directly on **GitHub Pages** (`https://<your-username>.github.io/<repo-name>`):

1. Push this project to your GitHub repository.
2. In your GitHub repository, go to **Settings** > **Pages** (under "Code and automation").
3. Under **Build and deployment** > **Source**, choose **Deploy from a branch**.
4. Set the branch to `main` (or `master`) and folder to `/ (root)`.
5. Click **Save**. Within 1–2 minutes, your project will be live at `https://<your-username>.github.io/<repo-name>/`!

---

## 🚀 How to Run Locally

### 1. Running the GitHub Pages Web App Locally
Simply double-click `index.html` in your file explorer, or open it in any web browser. No server required!

### 2. Running the Streamlit Web UI
To start the interactive Streamlit Python app:
```bash
pip install -r requirements.txt
python -m streamlit run app.py
```
Then open your browser at:
👉 **`http://localhost:8501`**

### 3. Running the Jupyter Notebook
Open the notebook in VS Code, JupyterLab, or Google Colab:
- File: [`sms_spam_classification.ipynb`](sms_spam_classification.ipynb)
- Run the cells sequentially to see the data exploration, distribution charts, model comparisons, and predictions.

### 4. Retraining via Command Line (Optional)
If you ever want to retrain the model and save the artifact:
```bash
python train.py
```

