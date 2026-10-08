import pandas as pd
import plotly.express as px
import streamlit as st

from api_client import (
    predict_churn,
    get_prediction_history,
    get_high_risk_history
)


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


    metric1, metric2, metric3 = st.columns(
        3
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
    # Churn risk progress
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

    st.caption(
        "Generated using the ML prediction, SHAP explanation, "
        "and retrieved company retention policies."
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


    # --------------------------------------------------
    # Retrieved retention policies
    # --------------------------------------------------

    retrieved_policies = data.get(
        "retrieved_policies",
        []
    )

    if retrieved_policies:

        st.divider()

        st.subheader(
            "Retention Policies Used"
        )

        st.write(
            "These company policies were retrieved by "
            "the RAG system and used to generate the "
            "AI recommendation."
        )

        with st.expander(
            "View Retention Policies Used"
        ):

            for i, policy in enumerate(
                retrieved_policies,
                start=1
            ):

                st.markdown(
                    f"### Policy {i}"
                )

                st.text(
                    policy
                )

                if i < len(
                    retrieved_policies
                ):
                    st.divider()


# --------------------------------------------------
# Prediction History & Analytics
# --------------------------------------------------

st.divider()

st.header(
    "Prediction History & Analytics"
)


history_tab, high_risk_tab, analytics_tab = st.tabs(
    [
        "Recent Predictions",
        "High-Risk Customers",
        "Analytics"
    ]
)


# --------------------------------------------------
# Recent predictions
# --------------------------------------------------

with history_tab:

    st.subheader(
        "Recent Predictions"
    )

    history_result = get_prediction_history(
        limit=20
    )

    if not history_result["success"]:

        st.error(
            "Could not load prediction history."
        )

        st.code(
            history_result["error"]
        )

    else:

        history_data = history_result[
            "data"
        ]

        if not history_data:

            st.info(
                "No prediction history found."
            )

        else:

            history_df = pd.DataFrame(
                history_data
            )

            st.dataframe(
                history_df,
                use_container_width=True,
                hide_index=True
            )


# --------------------------------------------------
# High-risk customers
# --------------------------------------------------

with high_risk_tab:

    st.subheader(
        "High-Risk Customers"
    )

    high_risk_result = get_high_risk_history(
        limit=20
    )

    if not high_risk_result["success"]:

        st.error(
            "Could not load high-risk history."
        )

        st.code(
            high_risk_result["error"]
        )

    else:

        high_risk_data = high_risk_result[
            "data"
        ]

        if not high_risk_data:

            st.info(
                "No High or Critical risk "
                "predictions found yet."
            )

        else:

            high_risk_df = pd.DataFrame(
                high_risk_data
            )

            st.dataframe(
                high_risk_df,
                use_container_width=True,
                hide_index=True
            )


# --------------------------------------------------
# Analytics
# --------------------------------------------------

with analytics_tab:

    st.subheader(
        "Prediction Analytics"
    )

    analytics_result = get_prediction_history(
        limit=100
    )

    if not analytics_result["success"]:

        st.error(
            "Could not load analytics data."
        )

    else:

        analytics_data = analytics_result[
            "data"
        ]

        if not analytics_data:

            st.info(
                "Not enough prediction data yet."
            )

        else:

            analytics_df = pd.DataFrame(
                analytics_data
            )


            # --------------------------------------------------
            # Metrics
            # --------------------------------------------------

            total_predictions = len(
                analytics_df
            )

            churn_predictions = (
                analytics_df[
                    "prediction"
                ] == "Churn"
            ).sum()

            average_risk = (
                analytics_df[
                    "churn_probability"
                ]
                .mean()
                * 100
            )

            high_risk_count = (
                analytics_df[
                    "risk_category"
                ]
                .isin(
                    [
                        "High",
                        "Critical"
                    ]
                )
                .sum()
            )


            m1, m2, m3, m4 = st.columns(
                4
            )

            m1.metric(
                "Total Predictions",
                total_predictions
            )

            m2.metric(
                "Predicted Churn",
                int(
                    churn_predictions
                )
            )

            m3.metric(
                "Average Churn Risk",
                f"{average_risk:.2f}%"
            )

            m4.metric(
                "High-Risk Customers",
                int(
                    high_risk_count
                )
            )


            # --------------------------------------------------
            # Risk Distribution
            # --------------------------------------------------

            risk_counts = (
                analytics_df[
                    "risk_category"
                ]
                .value_counts()
            )

            risk_chart_df = (
                risk_counts
                .reset_index()
            )

            risk_chart_df.columns = [
                "Risk Category",
                "Count"
            ]

            fig_risk = px.bar(
                risk_chart_df,
                x="Risk Category",
                y="Count",
                text="Count",
                title="Risk Distribution"
            )

            fig_risk.update_traces(
                textposition="outside"
            )

            fig_risk.update_layout(
                height=400,
                xaxis_title="Risk Category",
                yaxis_title="Number of Predictions"
            )

            st.plotly_chart(
                fig_risk,
                use_container_width=True
            )


            # --------------------------------------------------
            # Prediction Distribution
            # --------------------------------------------------

            prediction_counts = (
                analytics_df[
                    "prediction"
                ]
                .value_counts()
            )

            prediction_chart_df = (
                prediction_counts
                .reset_index()
            )

            prediction_chart_df.columns = [
                "Prediction",
                "Count"
            ]

            fig_prediction = px.bar(
                prediction_chart_df,
                x="Prediction",
                y="Count",
                text="Count",
                title="Prediction Distribution"
            )

            fig_prediction.update_traces(
                textposition="outside"
            )

            fig_prediction.update_layout(
                height=400,
                xaxis_title="Prediction",
                yaxis_title="Number of Predictions"
            )

            st.plotly_chart(
                fig_prediction,
                use_container_width=True
            )


            # --------------------------------------------------
            # Average churn risk by contract
            # --------------------------------------------------

            contract_risk = (
                analytics_df
                .groupby(
                    "contract"
                )[
                    "churn_probability"
                ]
                .mean()
                .mul(100)
                .reset_index()
            )

            contract_risk.columns = [
                "Contract",
                "Average Churn Risk"
            ]

            fig_contract = px.bar(
                contract_risk,
                x="Contract",
                y="Average Churn Risk",
                text="Average Churn Risk",
                title="Average Churn Risk by Contract"
            )

            fig_contract.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside"
            )

            fig_contract.update_layout(
                height=400,
                xaxis_title="Contract Type",
                yaxis_title="Average Churn Risk (%)"
            )

            st.plotly_chart(
                fig_contract,
                use_container_width=True
            )