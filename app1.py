import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go
import pandas as pd


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Package Prediction AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("model.joblib")


try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    error_message = str(e)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🎯 Package AI")

    st.divider()

    st.subheader("📌 About")

    st.write(
        """
        This Machine Learning application predicts
        a student's expected salary package based
        on their CGPA.
        """
    )

    st.divider()

    st.subheader("🤖 Model Status")

    if model_loaded:
        st.success("Model Loaded Successfully")
    else:
        st.error("Model Loading Failed")

    st.divider()

    st.subheader("📊 Input Feature")

    st.info("CGPA")

    st.write("Minimum: **0.0**")
    st.write("Maximum: **10.0**")

    st.divider()

    st.caption("Python • Machine Learning • Streamlit")
    st.caption("Package Prediction System")


# =========================================================
# MAIN HEADER
# =========================================================

st.title("🎯 Package Prediction AI")

st.subheader(
    "Predict an expected salary package using Machine Learning"
)

st.write(
    "Enter the student's CGPA below and generate an instant prediction."
)

st.divider()


# =========================================================
# MODEL CHECK
# =========================================================

if not model_loaded:

    st.error(f"Unable to load model: {error_message}")

    st.stop()


# =========================================================
# TOP KPI CARDS
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="🤖 Model",
        value="Active",
        delta="Ready"
    )

with col2:
    st.metric(
        label="📊 Feature",
        value="CGPA"
    )

with col3:
    st.metric(
        label="📈 CGPA Range",
        value="0 - 10"
    )

with col4:
    st.metric(
        label="⚡ Prediction",
        value="Instant"
    )


st.write("")


# =========================================================
# INPUT SECTION
# =========================================================

input_col, performance_col = st.columns([1, 1])


with input_col:

    st.subheader("🎓 Student Information")

    cgpa = st.number_input(
        "Enter CGPA",
        min_value=0.0,
        max_value=10.0,
        value=7.5,
        step=0.1
    )

    st.write("Academic Performance")

    st.progress(cgpa / 10)

    if cgpa >= 9:
        performance = "Excellent 🌟"
    elif cgpa >= 8:
        performance = "Very Good ⭐"
    elif cgpa >= 7:
        performance = "Good 👍"
    elif cgpa >= 6:
        performance = "Average 🙂"
    else:
        performance = "Needs Improvement 📚"

    st.info(f"Current Performance: **{performance}**")


with performance_col:

    st.subheader("📊 CGPA Performance Gauge")

    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=cgpa,
            title={
                "text": "CGPA"
            },
            gauge={
                "axis": {
                    "range": [0, 10]
                },
                "bar": {
                    "thickness": 0.7
                },
                "steps": [
                    {
                        "range": [0, 5],
                    },
                    {
                        "range": [5, 7],
                    },
                    {
                        "range": [7, 8.5],
                    },
                    {
                        "range": [8.5, 10],
                    }
                ]
            }
        )
    )

    fig_gauge.update_layout(
        height=280,
        margin=dict(
            l=20,
            r=20,
            t=50,
            b=20
        )
    )

    st.plotly_chart(
        fig_gauge,
        use_container_width=True
    )


st.divider()


# =========================================================
# PREDICTION BUTTON
# =========================================================

button_col1, button_col2, button_col3 = st.columns(
    [1, 2, 1]
)

with button_col2:

    predict = st.button(
        "🚀 PREDICT PACKAGE",
        use_container_width=True,
        type="primary"
    )


# =========================================================
# PREDICTION
# =========================================================

if predict:

    try:

        with st.spinner(
            "Analyzing CGPA and generating prediction..."
        ):

            input_data = np.array([[cgpa]])

            prediction = model.predict(input_data)

            predicted_package = prediction[0]


        # -------------------------------------------------
        # Convert prediction to number if possible
        # -------------------------------------------------

        try:
            package_value = float(predicted_package)

            package_display = f"₹ {package_value:.2f} LPA"

        except:

            package_value = None

            package_display = str(predicted_package)


        st.success("Prediction generated successfully! 🎉")

        st.write("")


        # =================================================
        # PREDICTED PACKAGE CARD
        # =================================================

        st.subheader("🎯 Predicted Package")

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:

            st.metric(
                label="💰 Expected Package",
                value=package_display
            )

        with result_col2:

            st.metric(
                label="🎓 Student CGPA",
                value=f"{cgpa:.1f}"
            )

        with result_col3:

            st.metric(
                label="🤖 Prediction Status",
                value="Generated",
                delta="Success"
            )


        st.write("")

        # Large result message
        st.info(
            f"Based on a CGPA of **{cgpa:.1f}**, "
            f"the predicted package is **{package_display}**."
        )


        st.divider()


        # =================================================
        # PACKAGE VS CGPA CHART
        # =================================================

        if package_value is not None:

            st.subheader(
                "📈 CGPA vs Predicted Package"
            )

            st.write(
                "This chart shows how the model predicts "
                "the package across different CGPA values."
            )


            # Generate CGPA values
            cgpa_values = np.arange(
                0,
                10.1,
                0.5
            )


            # Generate predictions
            package_predictions = model.predict(
                cgpa_values.reshape(-1, 1)
            )


            chart_data = pd.DataFrame(
                {
                    "CGPA": cgpa_values,
                    "Predicted Package": package_predictions
                }
            )


            # Plotly chart
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=chart_data["CGPA"],
                    y=chart_data["Predicted Package"],
                    mode="lines+markers",
                    name="Predicted Package",
                    line=dict(
                        width=4
                    ),
                    marker=dict(
                        size=7
                    )
                )
            )


            # Highlight current prediction
            fig.add_trace(
                go.Scatter(
                    x=[cgpa],
                    y=[package_value],
                    mode="markers",
                    name="Your Prediction",
                    marker=dict(
                        size=16,
                        symbol="star"
                    )
                )
            )


            fig.update_layout(
                title="Predicted Package Trend",
                xaxis_title="CGPA",
                yaxis_title="Package (LPA)",
                hovermode="x unified",
                height=450
            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )


        # =================================================
        # PREDICTION SUMMARY
        # =================================================

        st.divider()

        st.subheader("📋 Prediction Summary")

        summary_col1, summary_col2 = st.columns(2)

        with summary_col1:

            st.write("### Student Details")

            st.write(
                f"**CGPA:** {cgpa:.1f}"
            )

            st.write(
                f"**Performance:** {performance}"
            )

            st.write(
                "**Feature Used:** CGPA"
            )


        with summary_col2:

            st.write("### Prediction Details")

            st.write(
                f"**Predicted Package:** {package_display}"
            )

            st.write(
                "**Model Status:** Active"
            )

            st.write(
                "**Prediction:** Successful"
            )


        # =================================================
        # PACKAGE CATEGORY
        # =================================================

        if package_value is not None:

            st.divider()

            st.subheader("💼 Package Category")

            if package_value >= 10:

                st.success(
                    "🔥 High Package Range"
                )

            elif package_value >= 6:

                st.info(
                    "🚀 Good Package Range"
                )

            elif package_value >= 3:

                st.warning(
                    "📊 Moderate Package Range"
                )

            else:

                st.error(
                    "📚 Entry-Level Package Range"
                )


    except Exception as e:

        st.error(
            f"Prediction Error: {e}"
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🎯 Package Prediction AI | "
    "Built with Python, Machine Learning, Plotly & Streamlit"
)