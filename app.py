# app.py - Streamlit app for Technology Term Extraction
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

# ===========================
# --- COPY-PASTE FUNCTIONS (exact) ---
# These functions are copied from your training code (must remain identical)
# 1) gold_tokenize
def gold_tokenize(sentence, phrase_vocab):
    words = re.findall(r"\w+|[^\w\s]", sentence, re.UNICODE)
    words = [w for w in words if w.strip()]
    n = len(words)
    i = 0
    tokens = []

    while i < n:
        matched = False
        for L in range(5, 0, -1):  # up to 5-word phrases
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

# 2) extract_features
def extract_features(sent, i):
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

# 3) extract_tech_terms_improved + TECH_KEYWORDS
TECH_KEYWORDS = {
    "ai", "chatgpt", "llm", "model", "mô hình", "dữ liệu", "data",
    "cloud", "server", "api", "hệ thống", "thuật toán",
    "machine", "learning", "deep", "neural", "mạng"
}

def extract_tech_terms_improved(tokens, pos_tags):
    terms = []
    current = []
    has_strong_tag = False  # Nb hoặc Ny

    for w, t in zip(tokens, pos_tags):
        w_clean = w.replace("_", " ")
        w_lower = w_clean.lower()

        if t in {"N", "Nb", "Np", "Ny"}:
            current.append(w_clean)
            if t in {"Nb", "Ny"}:
                has_strong_tag = True
        else:
            if current:
                term = " ".join(current)
                if (
                    has_strong_tag or
                    len(current) >= 2 or
                    any(k in term.lower() for k in TECH_KEYWORDS)
                ):
                    terms.append(term)
                current = []
                has_strong_tag = False

    if current:
        term = " ".join(current)
        if (
            has_strong_tag or
            len(current) >= 2 or
            any(k in term.lower() for k in TECH_KEYWORDS)
        ):
            terms.append(term)

    return terms

# ===========================
# --- Resource loader (cached) ---
RES_DIR = "deployment_resources"

# [FIX 1] Lớp InputLayer tùy chỉnh (như lần trước)
class FixedInputLayer(tf.keras.layers.InputLayer):
    def __init__(self, batch_shape=None, **kwargs):
        if batch_shape is not None:
            kwargs['batch_input_shape'] = batch_shape
        super(FixedInputLayer, self).__init__(**kwargs)

# [FIX 2] Lớp DTypePolicy giả lập (Mới bổ sung)
# Giúp Keras cũ đọc được config 'dtype' của Keras 3
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

@st.cache_resource
def load_resources(res_dir=RES_DIR):
    resources = {}
    # SVM artifacts
    svm_path = os.path.join(res_dir, "svm_final.joblib")
    vec_path = os.path.join(res_dir, "vec_full.joblib")
    le_path = os.path.join(res_dir, "label_encoder.joblib")

    # Pickle/json artifacts
    phrase_path = os.path.join(res_dir, "phrase_vocab.pkl")
    word2idx_path = os.path.join(res_dir, "word2idx.json")
    tag2idx_path = os.path.join(res_dir, "tag2idx.json")
    maxlen_path = os.path.join(res_dir, "max_len.json")

    # Keras model
    bilstm_path = os.path.join(res_dir, "bilstm.h5")

    # Load SVM
    if os.path.exists(svm_path):
        resources["svm"] = joblib.load(svm_path)
    else:
        resources["svm"] = None

    # Load vec and label encoder
    if os.path.exists(vec_path):
        resources["vec"] = joblib.load(vec_path)
    else:
        resources["vec"] = None

    if os.path.exists(le_path):
        resources["le"] = joblib.load(le_path)
    else:
        resources["le"] = None

    # Load phrase vocab
    if os.path.exists(phrase_path):
        with open(phrase_path, "rb") as f:
            resources["phrase_vocab"] = pickle.load(f)
    else:
        resources["phrase_vocab"] = set()

    # Load word2idx and tag2idx
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

    # Build idx2tag
    resources["idx2tag"] = {int(v): k for k, v in resources["tag2idx"].items()} if resources["tag2idx"] else {}

    # Load MAX_LEN
    if os.path.exists(maxlen_path):
        with open(maxlen_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            resources["MAX_LEN"] = int(data.get("MAX_LEN", 0))
    else:
        resources["MAX_LEN"] = None

    # Load BiLSTM Keras model (FIXED với cả InputLayer và DTypePolicy)
    if os.path.exists(bilstm_path):
        try:
            # Tạo từ điển chứa các lớp tùy chỉnh/giả lập
            custom_objects = {
                'InputLayer': FixedInputLayer,
                'DTypePolicy': DTypePolicy
            }
            
            resources["bilstm"] = tf.keras.models.load_model(
                bilstm_path, 
                compile=False,
                custom_objects=custom_objects
            )
            # Log thành công (tùy chọn)
            # print("Bi-LSTM loaded successfully with patches.")
        except Exception as e:
            st.warning(f"Không thể load Bi-LSTM model: {e}")
            resources["bilstm"] = None
    else:
        resources["bilstm"] = None

    return resources

# ===========================
# --- Utilities for UI ---
def highlight_terms_in_text(text, terms, highlight_style="background-color:#fff176;padding:2px;border-radius:3px"):
    """
    Highlight all occurrences of terms in text (case-insensitive). Returns HTML string.
    We escape input before replacements to avoid injection.
    """
    if not terms:
        return html.escape(text)

    # sort terms by length desc to replace longest first
    terms_sorted = sorted(set(terms), key=lambda s: len(s), reverse=True)
    escaped = html.escape(text)

    # For each term, replace all occurrences (case-insensitive)
    for term in terms_sorted:
        if not term.strip():
            continue
        term_escaped = html.escape(term)
        pattern = re.compile(re.escape(term_escaped), flags=re.IGNORECASE)
        replacement = f"<span style=\"{highlight_style}\">{term_escaped}</span>"
        escaped = pattern.sub(replacement, escaped)
    return escaped

# ===========================
# --- Streamlit UI ---
st.set_page_config(page_title="Tech Term Extractor", layout="wide")
st.title("🔎 Trích xuất thuật ngữ công nghệ (POS → Terms)")

resources = load_resources()

col1, col2 = st.columns([2,1])

with col1:
    st.subheader("Nhập văn bản")
    raw_text = st.text_area("Văn bản (tiếng Việt):", height=220, placeholder="Nhập (hoặc paste) văn bản cần phát hiện thuật ngữ...")
    model_choice = st.radio("Chọn mô hình:", ("SVM (feature-based)", "Bi-LSTM"))
    run_btn = st.button("Phân tích")

with col2:
    st.subheader("Tài nguyên đã load")
    st.write({
        "SVM loaded": bool(resources.get("svm")),
        "Vec loaded": bool(resources.get("vec")),
        "LabelEncoder": bool(resources.get("le")),
        "Bi-LSTM loaded": bool(resources.get("bilstm")),
        "Phrase vocab size": len(resources.get("phrase_vocab", [])),
        "MAX_LEN": resources.get("MAX_LEN")
    })

if run_btn and raw_text.strip():
    # Tokenize with gold_tokenize using loaded phrase_vocab
    phrase_vocab = resources.get("phrase_vocab", set())
    tokens = gold_tokenize(raw_text, phrase_vocab)

    # Choose pipeline
    if model_choice.startswith("SVM"):
        if resources.get("svm") is None or resources.get("vec") is None or resources.get("le") is None:
            st.error("Thiếu artifacts SVM (svm_final / vec_full / label_encoder). Vui lòng kiểm tra thư mục deployment_resources.")
        else:
            feats = [extract_features(tokens, i) for i in range(len(tokens))]
            Xv = resources["vec"].transform(feats)
            preds_idx_or_label = resources["svm"].predict(Xv)
            # svm uses LabelEncoder previously -> inverse_transform to labels
            preds = resources["le"].inverse_transform(preds_idx_or_label)
    else:  # Bi-LSTM
        if resources.get("bilstm") is None or not resources.get("word2idx") or not resources.get("idx2tag"):
            st.error("Thiếu artifacts Bi-LSTM. Vui lòng kiểm tra resources.")
            preds = ["O"] * len(tokens)
        else:
            w2i = resources["word2idx"]
            # Lấy index của UNK (ưu tiên lowercase 'unk' nếu có)
            unk_idx = w2i.get("UNK", w2i.get("unk", 1))
            
            # [FIX 1] Chuyển về chữ thường (lower) khi tra từ điển để tránh bị UNK oan
            ids = [w2i.get(w.lower(), unk_idx) for w in tokens]
            
            MAX_LEN = resources["MAX_LEN"]
            idx2tag = resources["idx2tag"]

            # [FIX 2] Cắt token thành từng đoạn nhỏ (chunks) nếu dài hơn MAX_LEN
            # Thay vì cắt cụt, ta trượt cửa sổ hoặc chia nhỏ để model xử lý hết
            final_preds_idx = []
            
            # Chia danh sách token thành các đoạn có độ dài MAX_LEN
            # Lưu ý: Cách tốt nhất là tách câu, nhưng ở đây ta cắt theo độ dài để đơn giản hóa
            import math
            num_chunks = math.ceil(len(ids) / MAX_LEN)
            
            if num_chunks == 0:
                final_preds_idx = []
            else:
                for i in range(num_chunks):
                    # Lấy đoạn con
                    start = i * MAX_LEN
                    end = start + MAX_LEN
                    chunk_ids = ids[start:end]
                    actual_chunk_len = len(chunk_ids)
                    
                    # Pad đoạn con này cho đủ MAX_LEN để đưa vào model
                    X_chunk = pad_sequences([chunk_ids], maxlen=MAX_LEN, padding="post", dtype="int32")
                    
                    # Dự đoán
                    probs = resources["bilstm"].predict(X_chunk, verbose=0)
                    
                    # Chỉ lấy kết quả của những từ thực tế (bỏ phần padding)
                    chunk_pred_idx = np.argmax(probs[0], axis=1)[:actual_chunk_len]
                    final_preds_idx.extend(chunk_pred_idx)

            # Map ngược từ ID sang Tag
            preds = [idx2tag.get(int(i), "O") for i in final_preds_idx]

    # Extract terms
    terms = extract_tech_terms_improved(tokens, preds)
    term_counts = Counter(terms)

    # Visualization: highlighted text
    highlighted_html = highlight_terms_in_text(raw_text, terms)
    st.subheader("📝 Kết quả (highlight các thuật ngữ được phát hiện)")
    st.markdown(highlighted_html, unsafe_allow_html=True)

    # Show per-token tagging table
    st.subheader("🔖 Gán nhãn token")
    token_display = []
    for t, p in zip(tokens, preds):
        token_display.append({"token": t.replace("_", " "), "tag": p})
    st.dataframe(pd.DataFrame(token_display), use_container_width=True)

    # WordCloud (Final Fix: Dùng PIL Image trực tiếp)
    st.subheader("☁️ WordCloud (thuật ngữ)")
    try:
        from wordcloud import WordCloud
        if term_counts:
            # Tạo đối tượng WordCloud
            wc = WordCloud(width=800, height=400, background_color="white", collocations=False)
            wc.generate_from_frequencies(term_counts)
            
            # [QUAN TRỌNG] Lấy đối tượng ảnh gốc (PIL Image)
            # Thay vì dùng .to_array() (gây lỗi với Numpy cũ), ta dùng .to_image()
            pil_image = wc.to_image()
            
            # Streamlit hỗ trợ hiển thị trực tiếp PIL Image
            st.image(pil_image, caption="Biểu đồ tần suất thuật ngữ", use_container_width=True)
            
        else:
            st.info("Không tìm thấy thuật ngữ để tạo WordCloud.")
    except Exception as e:
        st.info("Có lỗi khi tạo WordCloud.")
        st.write("Chi tiết lỗi:", str(e))

    # Top terms table
    st.subheader("📊 Top thuật ngữ trong văn bản")
    if term_counts:
        df_top = pd.DataFrame(term_counts.most_common(), columns=["term", "frequency"])
        st.table(df_top.head(20))
        # also provide CSV download
        csv = df_top.to_csv(index=False, encoding="utf-8-sig")
        st.download_button("Tải CSV thuật ngữ", csv, file_name="extracted_terms.csv", mime="text/csv")
    else:
        st.write("Không phát hiện thuật ngữ nào theo quy tắc hiện tại.")
