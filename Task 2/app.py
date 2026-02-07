import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import re
import pickle

# ---------- Load Model ----------
@st.cache_resource
def load():
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=8
    )

    model.load_state_dict(torch.load("model.pt", map_location='cpu'))
    model.eval()

    return tokenizer, model

tokenizer, model = load()

# ---------- Priority ----------
def infer_priority(text):

    text = text.lower()

    high = ['down','failed','urgent','asap','crash','error','not working']
    medium = ['issue','problem','reset','access','slow']

    if any(w in text for w in high):
        return "High"
    elif any(w in text for w in medium):
        return "Medium"
    else:
        return "Low"

# ---------- Prediction ----------
def predict(text):

    enc = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        out = model(**enc)

    cat_id = torch.argmax(out.logits).item()

    classes = ['Access','Admin rights','HR Support',
               'Hardware','Internal Project',
               'Misc','Purchase','Storage']

    return classes[cat_id], infer_priority(text)

# ---------- UI ----------
st.title("🎯 Support Ticket Classifier")

text = st.text_area("Enter Ticket Description")

if st.button("Analyze Ticket"):

    if text.strip() == "":
        st.warning("Please enter ticket text")

    else:
        cat, pri = predict(text)

        st.success(f"Category: {cat}")
        st.info(f"Priority: {pri}")
