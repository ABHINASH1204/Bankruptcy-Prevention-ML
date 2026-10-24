
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Bankruptcy Risk Predictor",
    page_icon="🏢",
    layout="centered"
)

st.title("🏢 Bankruptcy Risk Predictor")
st.write(
    "Enter the six company risk factors to predict "
    "the company's bankruptcy classification."
)

MODEL_PATH = Path(__file__).parent / "deployment_artifact.pkl"

@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)

try:
    artifact = load_artifact()
    model = artifact["model"]
    features = artifact["features"]
    model_name = artifact.get("model_name", "Trained Model")
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

st.caption(f"Model: {model_name}")
st.subheader("Company Information")

risk_features = {
    "industrial_risk",
    "management_risk",
    "operating_risk",
}

with st.form("prediction_form"):
    values = {}

    for feature in features:
        label = feature.replace("_", " ").title()

        if feature in risk_features:
            help_text = "0 = low risk, 0.5 = medium risk, 1 = high risk"
        else:
            help_text = "0 = low level, 0.5 = medium level, 1 = high level"

        values[feature] = st.selectbox(
            label,
            options=[0.0, 0.5, 1.0],
            index=1,
            help=help_text,
        )

    submitted = st.form_submit_button(
        "Predict Bankruptcy Risk",
        type="primary",
        use_container_width=True,
    )

if submitted:
    input_df = pd.DataFrame([values], columns=features)

    try:
        prediction = model.predict(input_df)[0]
        st.subheader("Prediction")

        if str(prediction) == "1" or str(prediction).lower() == "bankruptcy":
            st.error("Predicted class: Bankruptcy")
        else:
            st.success("Predicted class: Non-bankruptcy")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            classes = list(model.classes_)

            bankruptcy_class = next(
                (c for c in classes
                 if str(c).lower() == "bankruptcy"
                 or str(c) == "1"),
                None
            )

            if bankruptcy_class is not None:
                index = classes.index(bankruptcy_class)
                st.metric(
                    "Estimated bankruptcy probability",
                    f"{probabilities[index]:.1%}"
                )

        with st.expander("View submitted feature values"):
            st.dataframe(input_df, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.divider()
st.caption(
    "Educational project only. This prediction is not financial advice."
)
