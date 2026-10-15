import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


st.set_page_config(
    page_icon="",
    page_title="Movie Review Sentiment Analyzer",
    layout="centered"
)


# MODEL-PATH
MODEL_PATH = "RashidatAbdullahi/sentiment_model"

# MODEL LOADING

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer , model


# APP HEADING

st.title("Movie Review Sentiment Analyzer")
st.write("""
    Enter a movie review and let the AI predict whether
      it expresses a positive or negative sentiment
"""
)

try:
    tokenizer , model = load_model()
except Exception as error:
    st.error(
        """Could not load the model. 
        Check that the sentiment_model contains a saved model and tokenizer"""
            )
        
    st.exception(error)
    st.stop()

review = st.text_area(
    "Enter a movie review:" ,
    height = 150
)

if st.button("Analyze" , type= "primary"):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        inputs = tokenizer(
            review,
            return_tensors = "pt",
            truncation = True,
            padding = True,
            max_length = 128
        )


    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=1,
    )[0]

    # get highest probability
    predicted_class = int(torch.argmax(probabilities).item())
    confidence = float(probabilities[predicted_class].item())


    if predicted_class ==1:
        sentiment = "Positive"
        st.success(f"Sentiment: {sentiment}")
    else:
        sentiment = "Negative"
        st.error(f"Sentiment: {sentiment}")


    st.metric(
        label = "Model confidence",
        value = f"{confidence * 100:.2f}%"
    )

    st.subheader("Prediction Probabilities")
    st.write(f"Negative: {probabilities[0].item() * 100:.2f}%")
    st.progress(float(probabilities[0].item()))

    st.write(f"Positive: {probabilities[0].item() * 100:.2f}%")
    st.progress(float(probabilities[0].item()))


    st.caption("""
            Confidence is just the model probility,
            not a guarantee that the prediction is correct

""")





