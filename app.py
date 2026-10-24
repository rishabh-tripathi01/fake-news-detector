import html, json, pickle, re
import numpy as np
import pandas as pd
import streamlit as st
from utils import clean_text

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

@st.cache_resource
def load_model():
    with open("fake_news_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_metrics():
    try:
        with open("metrics.json") as f:
            return json.load(f)
    except Exception:
        return None

model = load_model()
tfidf, clf = model.named_steps["tfidf"], model.named_steps["clf"]
metrics = load_metrics()

def explain(text):
    """Har word ka contribution: + = Real ki taraf, - = Fake ki taraf."""
    vec = tfidf.transform([clean_text(text)])
    contrib = vec.toarray()[0] * clf.coef_[0]
    vocab = tfidf.get_feature_names_out()
    idx = np.nonzero(contrib)[0]
    return {vocab[i]: float(contrib[i]) for i in idx}

def highlight(text, contrib):
    out = []
    for tok in re.findall(r"\w+|\W+", text):
        c = contrib.get(tok.lower())
        safe = html.escape(tok)
        if c is None or abs(c) < 0.02 or not tok.strip():
            out.append(safe)
        else:
            strength = min(abs(c) * 4, 0.75) + 0.15
            color = f"rgba(46,160,67,{strength})" if c > 0 else f"rgba(248,81,73,{strength})"
            out.append(f"<span style='background:{color};padding:1px 3px;border-radius:4px'>{safe}</span>")
    return "".join(out)

st.title("📰 Fake News Detector")
st.caption("TF-IDF + Logistic Regression  •  Explainable: dekho model ne kaun se words pe bharosa kiya")

tab1, tab2, tab3 = st.tabs(["🔍 Check News", "📊 Model Report", "ℹ️ About"])

with tab1:
    EXAMPLES = {
        "Fake example": "Shocking! Secret miracle drink makes you lose 20 kilos in 3 days, doctors hate it and the government is hiding the truth from you",
        "Real example": "The central bank kept interest rates unchanged on Wednesday, citing steady inflation and a stable job market, officials said at a press conference",
    }
    choice = st.radio("Example try karo (optional):", ["Apna text likhunga"] + list(EXAMPLES), horizontal=True)
    text = st.text_area("News text yahan paste karo:", value=EXAMPLES.get(choice, ""), height=180)

    if st.button("Check karo 🔍", type="primary"):
        words = len(text.split())
        if words < 5:
            st.warning("Thoda lamba text daalo (kam se kam 5 words).")
        else:
            proba = model.predict_proba([clean_text(text)])[0]
            fake_p, real_p = proba[0] * 100, proba[1] * 100
            if real_p >= fake_p:
                st.success(f"✅ Ye news REAL jaisi lag rahi hai ({real_p:.1f}% confidence)")
            else:
                st.error(f"🚨 Ye news FAKE jaisi lag rahi hai ({fake_p:.1f}% confidence)")
            st.progress(int(real_p), text=f"Real: {real_p:.1f}%   |   Fake: {fake_p:.1f}%")

            if words < 30:
                st.info("⚠️ Text chhota hai, model ko lambe articles pe train kiya gaya tha, isliye result kam bharosemand ho sakta hai.")

            contrib = explain(text)
            st.subheader("🧠 Model ne kya dekha?")
            st.markdown("🟢 **Green** = Real ki taraf  |  🔴 **Red** = Fake ki taraf")
            st.markdown(highlight(text, contrib), unsafe_allow_html=True)

            if contrib:
                s = pd.Series(contrib).sort_values()
                top = pd.concat([s.head(5), s.tail(5)])
                st.caption("Sabse zyada asar karne wale words (negative = Fake, positive = Real)")
                st.bar_chart(top)
            st.caption("Dhyan do: ye model facts verify nahi karta, sirf likhne ke patterns dekhta hai.")

with tab2:
    if metrics:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{metrics['accuracy']}%")
        c2.metric("Precision", f"{metrics['precision']}%")
        c3.metric("Recall", f"{metrics['recall']}%")
        c4.metric("F1 Score", f"{metrics['f1']}%")
        st.write(f"Dataset: **{metrics['n_articles']:,}** articles  |  Test set: **{metrics['n_test']:,}**")
        st.subheader("Confusion Matrix")
        cm = pd.DataFrame(metrics["confusion_matrix"],
                          index=["Asli: Fake", "Asli: Real"],
                          columns=["Model bola: Fake", "Model bola: Real"])
        st.table(cm)
        st.caption("Diagonal wale numbers sahi predictions hain, baaki galtiyan.")
    else:
        st.info("metrics.json nahi mili. train_model.py dobara chalao.")

with tab3:
    st.markdown("""
**Ye kaise kaam karta hai?**
1. News text ko **TF-IDF** se numbers mein badla jaata hai (important words ko zyada weight).
2. **Logistic Regression** un numbers se decide karta hai: Real ya Fake.
3. Har word ka asar nikaal ke upar highlight kiya jaata hai (explainability).

**Honest limitations**
- Dataset mein se **Reuters tags aur links hata diye** gaye, warna model shortcut seekh leta hai.
- Model **facts check nahi karta**, sirf language ke patterns dekhta hai.
- Chhote text aur naye topics pe galat ho sakta hai.
- Dataset sirf **English** news ka hai.
""")
