import streamlit as st
from transformers import pipeline

st.title("CollabMatch Event Tagger")
st.write("Describe your event and we'll tag it so we can match you with the right businesses.")

CATEGORIES = ["food and drink", "music", "cultural", "fitness", "tech", "fashion", "charity"]

@st.cache_resource
def load_model():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

classifier = load_model()

text = st.text_input("Event description")

if st.button("Tag my event"):
    if not text.strip():
        st.warning("Add a description first.")
    else:
        try:
            with st.spinner("Tagging..."):
                result = classifier(text, CATEGORIES, multi_label=True)
            st.subheader("Best fit categories")
            for label, score in zip(result["labels"][:3], result["scores"][:3]):
                st.write(f"**{label.capitalize()}**: {score:.0%}")
                st.progress(score)
        except Exception as e:
            st.error(f"Something went wrong: {e}")