# app.py - Streamlit application for Technology Term Extraction
import streamlit as st
import joblib
import pickle
import json
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from collections import Counter
import pandas as pd
import html
import os
import math

# ==============================================================================
# 1. CORE CONFIGURATION & CONSTANTS
# ==============================================================================

# Directory containing model artifacts (ensure this folder exists)
RES_DIR = "deployment_resources"

# Define tech keywords for heuristic filtering (hybrid approach)
TECH_KEYWORDS = {
    "ai", "chatgpt", "llm", "model", "mô hình", "dữ liệu", "data",
    "cloud", "server", "api", "hệ thống", "thuật toán",
    "machine", "learning", "deep", "neural", "mạng"
}

# ==============================================================================
# 2. HELPER CLASSES (MONKEY PATCHING)
# ==============================================================================

# Custom InputLayer to handle 'batch_shape' incompatibility between Keras versions.
# This ensures models trained on different envs can load without error.
class FixedInputLayer(tf.keras.layers.InputLayer):
    def __init__(self, batch_shape=None, **kwargs):
        if batch_shape is not None:
            kwargs['batch_input_shape'] = batch_shape
        super(FixedInputLayer, self).__init__(**kwargs)

# Custom DTypePolicy to handle serialization issues in newer Keras/TensorFlow.
class DTypePolicy:
    def __init__(self, name="float32", **kwargs):
        self._name = name
        self._compute_dtype = name
        self._variable_dtype = name

    @property
    def name(self):
        return self._name

    @property
    def compute_dtype(self):
        return self._compute_dtype
    
    @property
    def variable_dtype(self):
        return self._variable_dtype

    def get_config(self):
        return {"name": self._name}

# ==============================================================================
# 3. TEXT PROCESSING FUNCTIONS
# ==============================================================================

def gold_tokenize(sentence, phrase_vocab):
    """
    Tokenizes text based on a learned phrase vocabulary (Longest Matching).
    Args:
        sentence (str): Input raw text.
        phrase_vocab (set): Set of multi-word phrases (e.g., 'machine_learning').
    Returns:
        list: List of tokens (words/phrases).
    """
    words = re.findall(r"\w+|[^\w\s]", sentence, re.UNICODE)
    words = [w for w in words if w.strip()]
    n = len(words)
    i = 0
    tokens = []

    while i < n:
        matched = False
        # Try to match phrases up to 5 words long
        for L in range(5, 0, -1):
            if i + L <= n:
                phrase = "_".join(words[i:i+L]).lower()
                if phrase in phrase_vocab:
                    tokens.append("_".join(words[i:i+L]))
                    i += L
                    matched = True
                    break
        if not matched:
            tokens.append(words[i])
            i += 1
    return tokens

def extract_features(sent, i):
    """
    Extracts linguistic features for a token at index i.
    Used for Feature-based ML models (SVM, LogReg, RF, NB).
    """
    w = sent[i]
    return {
        "word": w,
        "is_first": i == 0,
        "is_last": i == len(sent) - 1,
        "is_cap": w[0].isupper() if len(w) > 0 else False,
        "is_upper": w.isupper(),
        "is_lower": w.islower(),
        "has_underscore": "_" in w,
        "prefix1": w[:1],
        "prefix2": w[:2],
        "suffix1": w[-1:],
        "suffix2": w[-2:],
        "prev_word": sent[i-1] if i > 0 else "",
        "next_word": sent[i+1] if i < len(sent)-1 else ""
    }

def extract_tech_terms_improved(tokens, pos_tags):
    """
    Extracts terms based on POS tags and heuristics.
    Tags: N (Noun), Nb, Np, Ny (Special Nouns).
    """
    terms = []
    current = []
    has_strong_tag = False  # e.g., Nb (Borrowed noun) or Ny (Abbreviation)

    for w, t in zip(tokens, pos_tags):
        w_clean = w.replace("_", " ")
        
        # Check if tag corresponds to a noun type
        if t in {"N", "Nb", "Np", "Ny"}:
            current.append(w_clean)
            if t in {"Nb", "Ny"}:
                has_strong_tag = True
        else:
            if current:
                term = " ".join(current)
                # Filter: Keep if it has strong tag, multiple words, or is a known keyword
                if (
                    has_strong_tag or
                    len(current) >= 2 or
                    any(k in term.lower() for k in TECH_KEYWORDS)
                ):
                    terms.append(term)
                current = []
                has_strong_tag = False

    # Handle the last phrase if sentence ends with a noun
    if current:
        term = " ".join(current)
        if (
            has_strong_tag or
            len(current) >= 2 or
            any(k in term.lower() for k in TECH_KEYWORDS)
        ):
            terms.append(term)

    return terms

# ==============================================================================
# 4. RESOURCE MANAGEMENT (CACHED LOADING)
# ==============================================================================

@st.cache_resource
def load_resources(res_dir=RES_DIR):
    """
    Loads all models and preprocessing artifacts from disk.
    Uses caching to prevent reloading on every user interaction.
    """
    resources = {}
    
    # --- Paths definition ---
    # ML Models
    svm_path = os.path.join(res_dir, "svm_final.joblib")
    logreg_path = os.path.join(res_dir, "logreg_final.joblib")
    rf_path = os.path.join(res_dir, "rf_final.joblib")
    nb_path = os.path.join(res_dir, "nb_final.joblib")
    
    # ML Preprocessing
    vec_path = os.path.join(res_dir, "vec_full.joblib")
    le_path = os.path.join(res_dir, "label_encoder.joblib")
    phrase_path = os.path.join(res_dir, "phrase_vocab.pkl")

    # DL Models & Configs
    word2idx_path = os.path.join(res_dir, "word2idx.json")
    tag2idx_path = os.path.join(res_dir, "tag2idx.json")
    maxlen_path = os.path.join(res_dir, "max_len.json")
    bilstm_path = os.path.join(res_dir, "bilstm.h5")

    # --- Loading ML Artifacts ---
    # Models
    resources["svm"] = joblib.load(svm_path) if os.path.exists(svm_path) else None
    resources["logreg"] = joblib.load(logreg_path) if os.path.exists(logreg_path) else None
    resources["rf"] = joblib.load(rf_path) if os.path.exists(rf_path) else None
    resources["nb"] = joblib.load(nb_path) if os.path.exists(nb_path) else None
    
    # Vectorizer & LabelEncoder (Shared across ML models for fair comparison)
    resources["vec"] = joblib.load(vec_path) if os.path.exists(vec_path) else None
    resources["le"] = joblib.load(le_path) if os.path.exists(le_path) else None

    # Phrase Vocabulary
    if os.path.exists(phrase_path):
        with open(phrase_path, "rb") as f:
            resources["phrase_vocab"] = pickle.load(f)
    else:
        resources["phrase_vocab"] = set()

    # --- Loading DL Artifacts ---
    # Dictionaries
    if os.path.exists(word2idx_path):
        with open(word2idx_path, "r", encoding="utf-8") as f:
            resources["word2idx"] = json.load(f)
    else:
        resources["word2idx"] = {}

    if os.path.exists(tag2idx_path):
        with open(tag2idx_path, "r", encoding="utf-8") as f:
            resources["tag2idx"] = json.load(f)
    else:
        resources["tag2idx"] = {}

    # Inverse mapping for tags
    resources["idx2tag"] = {int(v): k for k, v in resources["tag2idx"].items()} if resources["tag2idx"] else {}

    # Max sequence length
    if os.path.exists(maxlen_path):
        with open(maxlen_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            resources["MAX_LEN"] = int(data.get("MAX_LEN", 0))
    else:
        resources["MAX_LEN"] = None

    # Load Keras Model (Bi-LSTM)
    # Using custom objects to patch version incompatibilities
    if os.path.exists(bilstm_path):
        try:
            custom_objects = {
                'InputLayer': FixedInputLayer, 
                'DTypePolicy': DTypePolicy
            }
            resources["bilstm"] = tf.keras.models.load_model(
                bilstm_path, 
                compile=False, 
                custom_objects=custom_objects
            )
        except Exception as e:
            st.warning(f"Warning: Could not load Bi-LSTM model. Error: {e}")
            resources["bilstm"] = None
    else:
        resources["bilstm"] = None

    return resources

# ==============================================================================
# 5. UI UTILITIES
# ==============================================================================

def highlight_terms_in_text(text, terms, highlight_style="background-color:#fff176;padding:2px;border-radius:3px"):
    """
    Generates HTML to highlight extracted terms within the original text.
    Handles case-insensitivity and sorts by length to avoid partial replacements.
    """
    if not terms:
        return html.escape(text)

    # Sort terms by length (descending) to match longest phrases first
    terms_sorted = sorted(set(terms), key=lambda s: len(s), reverse=True)
    escaped = html.escape(text)

    for term in terms_sorted:
        if not term.strip():
            continue
        term_escaped = html.escape(term)
        pattern = re.compile(re.escape(term_escaped), flags=re.IGNORECASE)
        replacement = f"<span style=\"{highlight_style}\">{term_escaped}</span>"
        escaped = pattern.sub(replacement, escaped)
    
    # Convert newlines to HTML breaks for display
    return escaped.replace("\n", "<br>")

# ==============================================================================
# 6. MAIN APP LOGIC
# ==============================================================================

def main():
    st.set_page_config(page_title="Tech Term Extractor", layout="wide", page_icon="🔍")
    
    st.title("🔎 Trích xuất thuật ngữ công nghệ")
    st.markdown("""
    Ứng dụng Demo cho môn Xử lý ngôn ngữ tự nhiên.
    Hệ thống sử dụng **4 mô hình Machine Learning** và **1 mô hình Deep Learning** để so sánh hiệu quả.
    """)

    # Load resources
    resources = load_resources()

    # --- Sidebar / Control Panel ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📝 Nhập văn bản")
        raw_text = st.text_area(
            "Nội dung văn bản (Tiếng Việt):", 
            height=250, 
            placeholder="Ví dụ: Mô hình ngôn ngữ lớn như ChatGPT đang thay đổi cách chúng ta xử lý dữ liệu..."
        )
        
        # Define available models based on loaded resources
        # The key is the display name, value is the internal resource key
        model_map = {}
        if resources.get("svm"): model_map["SVM (Feature-based)"] = "svm"
        if resources.get("logreg"): model_map["Logistic Regression (Feature-based)"] = "logreg"
        if resources.get("rf"): model_map["Random Forest (Feature-based)"] = "rf"
        if resources.get("nb"): model_map["Naive Bayes (Feature-based)"] = "nb"
        if resources.get("bilstm"): model_map["Bi-LSTM (Deep Learning)"] = "bilstm"

        if not model_map:
            st.error("🚨 Không tìm thấy mô hình nào! Vui lòng kiểm tra thư mục 'deployment_resources'.")
            st.stop()

        model_choice_name = st.radio(
            "🤖 Chọn mô hình dự đoán:", 
            list(model_map.keys()),
            help="Chọn thuật toán để trích xuất thực thể."
        )
        
        # Get internal key (e.g., 'svm', 'bilstm')
        selected_model_key = model_map[model_choice_name]
        
        run_btn = st.button("🚀 Phân tích ngay", type="primary")

    with col2:
        st.subheader("⚙️ Trạng thái hệ thống")
        # Display status of all components
        status_data = [
            {"Component": "Phrase Vocab", "Status": f"✅ {len(resources['phrase_vocab'])} phrases"},
            {"Component": "Vectorizer", "Status": "✅ Ready" if resources.get('vec') else "❌ Missing"},
            {"Component": "LabelEncoder", "Status": "✅ Ready" if resources.get('le') else "❌ Missing"},
            {"Component": "SVM Model", "Status": "✅ Ready" if resources.get('svm') else "❌ Missing"},
            {"Component": "LogReg Model", "Status": "✅ Ready" if resources.get('logreg') else "❌ Missing"},
            {"Component": "Random Forest", "Status": "✅ Ready" if resources.get('rf') else "❌ Missing"},
            {"Component": "Naive Bayes", "Status": "✅ Ready" if resources.get('nb') else "❌ Missing"},
            {"Component": "Bi-LSTM Model", "Status": "✅ Ready" if resources.get('bilstm') else "❌ Missing"},
        ]
        st.table(pd.DataFrame(status_data))

    # --- Processing Logic ---
    if run_btn and raw_text.strip():
        with st.spinner(f"Đang phân tích bằng mô hình {model_choice_name}..."):
            
            # Step 1: Tokenization (Common for all models)
            phrase_vocab = resources.get("phrase_vocab", set())
            tokens = gold_tokenize(raw_text, phrase_vocab)
            
            preds = [] # List to store predicted tags for each token

            # Step 2: Prediction Dispatch
            
            # --- CASE A: Machine Learning Models (Feature-based) ---
            if selected_model_key in ["svm", "logreg", "rf", "nb"]:
                if not resources.get("vec") or not resources.get("le"):
                    st.error("Thiếu Vectorizer hoặc LabelEncoder cho mô hình ML.")
                    st.stop()
                
                # Feature Extraction
                feats = [extract_features(tokens, i) for i in range(len(tokens))]
                # Vectorization (Shared across ML models -> FAIR COMPARISON)
                Xv = resources["vec"].transform(feats)
                
                # Prediction
                model = resources[selected_model_key]
                preds_idx = model.predict(Xv)
                
                # Decoding (Index -> Label)
                preds = resources["le"].inverse_transform(preds_idx)

            # --- CASE B: Deep Learning Model (Sequence-based) ---
            elif selected_model_key == "bilstm":
                w2i = resources["word2idx"]
                # Handle OOV (Out of Vocabulary) - use UNK token
                unk_idx = w2i.get("UNK", w2i.get("unk", 1))
                # Convert tokens to IDs (Lowercasing to match training data)
                ids = [w2i.get(w.lower(), unk_idx) for w in tokens]
                
                MAX_LEN = resources["MAX_LEN"]
                idx2tag = resources["idx2tag"]
                
                # Chunking Strategy: Handle texts longer than MAX_LEN
                # We split text into chunks of MAX_LEN, predict, then merge results
                final_preds_idx = []
                num_chunks = math.ceil(len(ids) / MAX_LEN) if len(ids) > 0 else 0
                
                if num_chunks == 0:
                    final_preds_idx = []
                else:
                    for i in range(num_chunks):
                        start = i * MAX_LEN
                        end = start + MAX_LEN
                        chunk_ids = ids[start:end]
                        actual_len = len(chunk_ids)
                        
                        # Pad chunk to fit model input shape
                        X_chunk = pad_sequences([chunk_ids], maxlen=MAX_LEN, padding="post", dtype="int32")
                        
                        # Predict
                        probs = resources["bilstm"].predict(X_chunk, verbose=0)
                        
                        # Get best tag index (argmax), truncate padding
                        chunk_pred_idx = np.argmax(probs[0], axis=1)[:actual_len]
                        final_preds_idx.extend(chunk_pred_idx)
                
                # Decode IDs to Tags
                preds = [idx2tag.get(int(i), "O") for i in final_preds_idx]

            # Step 3: Extract Terms from Tags
            terms = extract_tech_terms_improved(tokens, preds)
            term_counts = Counter(terms)

            # --- Visualization ---
            st.divider()
            
            # 1. Highlighted Text
            st.subheader("📄 Kết quả Highlight")
            highlighted_html = highlight_terms_in_text(raw_text, terms)
            st.markdown(
                f'<div style="background-color:#f9f9f9; padding:15px; border-radius:5px; line-height:1.6; border:1px solid #ddd;">{highlighted_html}</div>', 
                unsafe_allow_html=True
            )

            # 2. Detailed Token-Tag Table (Expandable)
            with st.expander("xem chi tiết gán nhãn (Token & POS Tags)"):
                df_tags = pd.DataFrame({
                    "Token": [t.replace("_", " ") for t in tokens],
                    "Predicted Tag": preds
                })
                st.dataframe(df_tags.T, use_container_width=True)

            # 3. WordCloud
            st.subheader("☁️ WordCloud Thuật ngữ")
            col_wc1, col_wc2 = st.columns([3, 1])
            
            with col_wc1:
                try:
                    from wordcloud import WordCloud
                    if term_counts:
                        # Attempt to load font to support Vietnamese
                        font_path = os.path.join(RES_DIR, "Roboto.ttf")
                        if not os.path.exists(font_path):
                            font_path = None # Fallback (might show squares for VN characters)
                        
                        wc = WordCloud(
                            width=800, height=400, 
                            background_color="white", 
                            font_path=font_path,
                            collocations=False
                        )
                        wc.generate_from_frequencies(term_counts)
                        
                        # Use to_image() to convert to PIL Image directly
                        # This avoids Matplotlib/Numpy version conflicts
                        st.image(wc.to_image(), use_container_width=True)
                    else:
                        st.info("Không tìm thấy thuật ngữ nào để tạo WordCloud.")
                except Exception as e:
                    st.warning("Không thể tạo WordCloud. Vui lòng kiểm tra thư viện hoặc font chữ.")
                    st.caption(f"Error details: {e}")

            with col_wc2:
                # Top Terms Table
                if term_counts:
                    st.write("**Top thuật ngữ:**")
                    df_top = pd.DataFrame(term_counts.most_common(10), columns=["Thuật ngữ", "Tần suất"])
                    st.dataframe(df_top, hide_index=True)
                    
                    # Download CSV Button
                    csv = pd.DataFrame(term_counts.most_common(), columns=["Term", "Frequency"]).to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        "📥 Tải CSV",
                        csv,
                        "extracted_terms.csv",
                        "text/csv",
                        key='download-csv'
                    )

# Entry point
if __name__ == "__main__":
    main()