import pandas as pd
import streamlit as st

from api_client import predict_churn


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Customer Churn Intelligence",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
# Page title
# --------------------------------------------------

st.title(
    "AI Customer Churn Intelligence Platform"
)

st.write(
    "Predict customer churn, understand the main "
    "risk factors, and generate AI-powered "
    "retention recommendations."
)

st.divider()


# --------------------------------------------------
# Customer input form
# --------------------------------------------------

st.subheader(
    "Customer Information"
)


with st.form(
    "customer_form"
):

    col1, col2, col3 = st.columns(3)


    # --------------------------------------------------
    # Column 1
    # --------------------------------------------------

    with col1:

        gender = st.selectbox(
            "Gender",
            [
                "Female",
                "Male"
            ]
        )

        seniorcitizen = st.selectbox(
            "Senior Citizen",
            [
                0,
                1
            ],
            format_func=lambda x:
            "Yes" if x == 1 else "No"
        )

        partner = st.selectbox(
            "Partner",
            [
                "Yes",
                "No"
            ]
        )

        dependents = st.selectbox(
            "Dependents",
            [
                "Yes",
                "No"
            ]
        )

        tenure = st.number_input(
            "Tenure (Months)",
            min_value=0,
            max_value=72,
            value=12
        )

        phoneservice = st.selectbox(
            "Phone Service",
            [
                "Yes",
                "No"
            ]
        )


    # --------------------------------------------------
    # Column 2
    # --------------------------------------------------

    with col2:

        multiplelines = st.selectbox(
            "Multiple Lines",
            [
                "No",
                "Yes",
                "No phone service"
            ]
        )

        internetservice = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )

        onlinesecurity = st.selectbox(
            "Online Security",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        onlinebackup = st.selectbox(
            "Online Backup",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        deviceprotection = st.selectbox(
            "Device Protection",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        techsupport = st.selectbox(
            "Technical Support",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )


    # --------------------------------------------------
    # Column 3
    # --------------------------------------------------

    with col3:

        streamingtv = st.selectbox(
            "Streaming TV",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        streamingmovies = st.selectbox(
            "Streaming Movies",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        paperlessbilling = st.selectbox(
            "Paperless Billing",
            [
                "Yes",
                "No"
            ]
        )

        paymentmethod = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthlycharges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0,
            step=1.0
        )

        totalcharges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=840.0,
            step=10.0
        )


    # --------------------------------------------------
    # Submit button
    # --------------------------------------------------

    submitted = st.form_submit_button(
        "Predict Churn",
        use_container_width=True
    )


# --------------------------------------------------
# Prediction section
# --------------------------------------------------

if submitted:

    # --------------------------------------------------
    # Prepare customer JSON
    # --------------------------------------------------

    customer_data = {

        "gender": gender,

        "seniorcitizen": seniorcitizen,

        "partner": partner,

        "dependents": dependents,

        "tenure": tenure,

        "phoneservice": phoneservice,

        "multiplelines": multiplelines,

        "internetservice": internetservice,

        "onlinesecurity": onlinesecurity,

        "onlinebackup": onlinebackup,

        "deviceprotection": deviceprotection,

        "techsupport": techsupport,

        "streamingtv": streamingtv,

        "streamingmovies": streamingmovies,

        "contract": contract,

        "paperlessbilling": paperlessbilling,

        "paymentmethod": paymentmethod,

        "monthlycharges": monthlycharges,

        "totalcharges": totalcharges
    }


    # --------------------------------------------------
    # Call FastAPI
    # --------------------------------------------------

    with st.spinner(
        "Analyzing customer churn risk..."
    ):

        result = predict_churn(
            customer_data
        )


    # --------------------------------------------------
    # Handle API error
    # --------------------------------------------------

    if not result["success"]:

        st.error(
            "Could not connect to the prediction API."
        )

        st.code(
            result["error"]
        )

        st.stop()


    # --------------------------------------------------
    # API response
    # --------------------------------------------------

    data = result["data"]


    # --------------------------------------------------
    # Prediction result
    # --------------------------------------------------

    st.divider()

    st.subheader(
        "Prediction Result"
    )


    probability = data[
        "churn_probability_percent"
    ]

    prediction = data[
        "prediction"
    ]

    risk_category = data[
        "risk_category"
    ]


    metric1, metric2, metric3 = (
        st.columns(3)
    )


    metric1.metric(
        "Churn Probability",
        f"{probability:.2f}%"
    )

    metric2.metric(
        "Prediction",
        prediction
    )

    metric3.metric(
        "Risk Category",
        risk_category
    )


    # --------------------------------------------------
    # Churn risk bar
    # --------------------------------------------------

    st.write(
        "### Churn Risk"
    )

    st.progress(
        min(
            max(
                probability / 100,
                0.0
            ),
            1.0
        )
    )


    # --------------------------------------------------
    # SHAP explanation
    # --------------------------------------------------

    st.divider()

    st.subheader(
        "Why did the model make this prediction?"
    )


    # --------------------------------------------------
    # Risk increasing factors
    # --------------------------------------------------

    risk_factors = data.get(
        "top_risk_factors",
        []
    )


    if risk_factors:

        risk_df = pd.DataFrame(
            risk_factors
        )

        st.write(
            "#### Factors Increasing Churn Risk"
        )

        st.dataframe(
            risk_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No major risk-increasing factors found."
        )


    # --------------------------------------------------
    # Protective factors
    # --------------------------------------------------

    protective_factors = data.get(
        "top_protective_factors",
        []
    )


    if protective_factors:

        protective_df = pd.DataFrame(
            protective_factors
        )

        st.write(
            "#### Factors Reducing Churn Risk"
        )

        st.dataframe(
            protective_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No major protective factors found."
        )


    # --------------------------------------------------
    # AI retention recommendation
    # --------------------------------------------------

    st.divider()

    st.subheader(
        "AI Retention Recommendation"
    )


    ai_status = data.get(
        "ai_status",
        "unavailable"
    )

    ai_recommendation = data.get(
        "ai_recommendation",
        "No AI recommendation available."
    )


    if ai_status == "available":

        st.success(
            "AI recommendation generated successfully."
        )

    else:

        st.warning(
            "AI recommendation service is currently unavailable."
        )


    st.markdown(
        ai_recommendation
    )