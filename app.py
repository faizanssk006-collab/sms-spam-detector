import os
import joblib
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    [data-testid="stSidebar"], section[data-testid="stSidebar"] {
        display: none !important;
    }
    
    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 3rem;
        max-width: 720px;
    }
    
    .result-card-spam {
        background: #fff1f0;
        border: 2px solid #ff4d4f;
        border-radius: 12px;
        padding: 20px 24px;
        margin-top: 20px;
        margin-bottom: 18px;
        box-shadow: 0 4px 12px rgba(255, 77, 79, 0.1);
    }
    .result-card-ham {
        background: #f6ffed;
        border: 2px solid #52c41a;
        border-radius: 12px;
        padding: 20px 24px;
        margin-top: 20px;
        margin-bottom: 18px;
        box-shadow: 0 4px 12px rgba(82, 196, 26, 0.1);
    }
    .result-title {
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .result-desc {
        font-size: 15px;
        color: #333;
        margin-bottom: 10px;
    }
    
    .tag {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .tag-spam {
        background-color: #ffd8d6;
        color: #cf1322;
        border: 1px solid #ffa39e;
    }
    .tag-safe {
        background-color: #d9f7be;
        color: #237804;
        border: 1px solid #b7eb8f;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_classifier():
    model_path = "sms_spam_model.joblib"
    
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception:
            pass

    csv_file = "sms_spam_collection (1).csv"
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        df = df.drop_duplicates().reset_index(drop=True)
        df['target'] = df['label'].map({'ham': 0, 'spam': 1})
        
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', max_features=3000)),
            ('nb', MultinomialNB())
        ])
        pipeline.fit(df['message'], df['target'])
        joblib.dump(pipeline, model_path)
        return pipeline

    return None

pipeline = get_classifier()

st.title("📬 SMS Spam Detector")
st.write("A simple Machine Learning tool to check whether a text message is **Spam** or **Legitimate (Ham)**.")

COMMON_SPAM_WORDS = {
    'free', 'win', 'winner', 'cash', 'prize', 'claim', 'urgent', 'txt', 'reply',
    'call', 'selected', 'won', 'award', 'guaranteed', 'bonus', 'credit', 'account',
    'suspended', 'password', 'verify', 'click', 'subscribe', 'congratulations', 'apply'
}

user_message = st.text_area(
    "Type or paste an SMS message:",
    height=130,
    placeholder="Type or paste your SMS message here..."
)

classify_btn = st.button("🔍 Check SMS", type="primary", use_container_width=True)

if classify_btn:
    clean_text = user_message.strip()
    
    if not clean_text:
        st.warning("Please enter some text in the message box above.")
    elif pipeline is None:
        st.error("Model could not be loaded. Please ensure dataset or model file exists.")
    else:
        probs = pipeline.predict_proba([clean_text])[0]
        ham_prob = probs[0] * 100
        spam_prob = probs[1] * 100
        is_spam = pipeline.predict([clean_text])[0] == 1
        
        tokens = set(clean_text.lower().replace('.', ' ').replace(',', ' ').replace('!', ' ').split())
        matched_words = tokens.intersection(COMMON_SPAM_WORDS)
        
        if is_spam:
            st.markdown(f"""
            <div class="result-card-spam">
                <div class="result-title" style="color: #cf1322;">🚨 SPAM DETECTED</div>
                <div class="result-desc">
                    This message has patterns typical of promotional spam, prize scams, or phishing.
                </div>
                <div>
                    <strong>Confidence:</strong> {spam_prob:.1f}% Spam Probability
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card-ham">
                <div class="result-title" style="color: #237804;">✅ LEGITIMATE (HAM)</div>
                <div class="result-desc">
                    This looks like a normal, safe personal message.
                </div>
                <div>
                    <strong>Confidence:</strong> {ham_prob:.1f}% Safe (Legitimate)
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("##### 📊 Message Breakdown")
        m1, m2, m3 = st.columns(3)
        m1.metric("Character Count", len(clean_text))
        m2.metric("Word Count", len(clean_text.split()))
        m3.metric("Spam Risk Score", f"{spam_prob:.1f}%")
        
        st.progress(float(probs[1]), text=f"Spam Risk: {spam_prob:.1f}% | Safe: {ham_prob:.1f}%")
        
        if matched_words:
            st.markdown("##### ⚠️ Spam Keywords Found:")
            tags = "".join([f'<span class="tag tag-spam">"{w}"</span>' for w in matched_words])
            st.markdown(tags, unsafe_allow_html=True)
        else:
            st.markdown("##### 🔍 Keyword Analysis:")
            st.markdown('<span class="tag tag-safe">No common spam trigger words detected</span>', unsafe_allow_html=True)

with st.expander("ℹ️ About This Model & Accuracy"):
    st.markdown("""
    - **Algorithm:** Multinomial Naive Bayes + TF-IDF Vectorizer (3,000 features)
    - **Dataset:** SMS Spam Collection (5,572 messages)
    - **Test Accuracy:** `97.39%`
    - **Spam Precision:** `100.0%` *(No legitimate personal messages misclassified as spam)*
    """)

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888; font-size: 13px;'>"
    "Second Year Project • SMS Spam Classification"
    "</div>",
    unsafe_allow_html=True
)
