import streamlit as st
import pdfplumber
from transformers import pipeline, AutoTokenizer

st.set_page_config(
    page_title="LawLens - Legal Document Analyzer",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ LawLens - Legal Document Analyzer")
st.write(
    "Upload a **legal PDF document** and let Us help you by:\n"
    "- 📑 Extracting the text\n"
    "- 📝 Summarizing the content\n"
    "- ⚠️ Highlighting potentially risky clauses"
)

@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
    return summarizer, classifier, tokenizer

summarizer, classifier, tokenizer = load_models()

# ----------------- TEXT CHUNKING -----------------
def chunk_text(text, max_tokens=900):
    """Split text into smaller chunks for summarization."""
    tokens = tokenizer.encode(text, truncation=False)
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        yield tokenizer.decode(chunk_tokens, skip_special_tokens=True)

# ----------------- FILE UPLOAD -----------------
uploaded_file = st.file_uploader("📂 Upload a legal PDF", type=["pdf"])

if uploaded_file:
    st.success(f"✅ Uploaded: {uploaded_file.name}")

    # Extract text
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    if text.strip() == "":
        st.error("⚠️ Could not extract text from this PDF. Try another file.")
    else:
        # Tabs for workflow
        tab1, tab2, tab3 = st.tabs(
            ["📑 Extracted Text", "📝 AI Summary", "⚠️ Risky Clauses"]
        )

        # -------- Extracted Text --------
        with tab1:
            st.subheader("📑 Extracted Text")
            st.write(text[:5000] + ("..." if len(text) > 5000 else ""))

        # -------- AI Summary --------
        with tab2:
            st.subheader("📝 AI Summary")
            if st.button("🔍 Generate Summary"):
                with st.spinner("⏳ Summarizing document..."):
                    chunks = list(chunk_text(text))
                    summaries = [
                        summarizer(chunk, max_length=200, min_length=50, do_sample=False)[0]["summary_text"]
                        for chunk in chunks
                    ]
                    final_summary = " ".join(summaries)

                st.success("✅ Summary generated!")
                st.write(final_summary)

        # -------- Risky Clauses --------
        with tab3:
            st.subheader("⚠️ Risky Clauses")
            with st.spinner("🔍 Scanning document for risky clauses..."):
                candidate_labels = [
                    "liability",
                    "termination",
                    "payment",
                    "confidentiality",
                    "arbitration",
                    "indemnity",
                ]
                sentences = text.split(". ")
                risky = []
                for sent in sentences:
                    if len(sent.strip()) > 20:
                        result = classifier(sent, candidate_labels)
                        if max(result["scores"]) > 0.8:  # confidence threshold
                            risky.append(
                                (sent.strip(), result["labels"][result["scores"].index(max(result["scores"]))])
                            )

            if risky:
                for clause, label in risky:
                    st.warning(f"**{label.upper()} Clause:** {clause}")
            else:
                st.success("✅ No risky clauses detected.")



