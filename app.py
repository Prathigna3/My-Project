import streamlit as st
from model import predict

st.title("📰 Fake News Detection")

text_input = st.text_area("Enter News Content:", height=200)

if st.button("Predict"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        prediction, probs = predict(text_input)
        confidence = probs[0][prediction].item()
        if prediction == 0:
            st.error(f"⚠️ This news is likely **FAKE** (Confidence: {confidence:.2%})")
        else:
            st.success(f"✅ This news is likely **REAL** (Confidence: {confidence:.2%})")
