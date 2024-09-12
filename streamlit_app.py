import streamlit as st
import pandas as pd
from io import BytesIO
from prediction_code import load_and_process_data

# Set up the Streamlit app title and description
st.set_page_config(page_title="Momentous Demand Forecast Tool", layout="wide")

# Function to load predictions
@st.cache_data
def load_predictions():
    import time
    time.sleep(3)  # Simulating a delay for testing
    final_preds_df_com_agg, final_preds_df_ama_agg = load_and_process_data()
    return final_preds_df_com_agg, final_preds_df_ama_agg

# Function to convert DataFrame to CSV
def convert_df_to_csv(df):
    output = BytesIO()
    df.to_csv(output, index=False)
    return output.getvalue().decode('utf-8')

# Initialize session state to manage UI elements and results
if "predictions_available" not in st.session_state:
    st.session_state["predictions_available"] = False

# Function to reset session state for rerun
def reset_state():
    st.session_state["predictions_available"] = False
    st.session_state.pop("final_preds_df_com_agg", None)
    st.session_state.pop("final_preds_df_ama_agg", None)
    st.query_params()  # Updated with the new method

# Homepage with logos
st.image('momentous-logo.png', width=100)  # Placeholder for client logo

# Add the Powered by Saras Analytics to the top-right
st.markdown(
    "<div style='text-align: right; color: blue;'>Powered by Saras Analytics</div>",
    unsafe_allow_html=True
)

st.title('Momentous Demand Forecast Tool')
st.markdown("""
**This app allows you to view and download SKU-wise forecasts for units sold on both website (.com) and Amazon sales channels.**
""")

# Button to initiate predictions
if not st.session_state["predictions_available"]:
    if st.button('Initiate Model Results'):
        with st.spinner('Running predictions... This may take some time.'):
            final_preds_df_com_agg, final_preds_df_ama_agg = load_predictions()
            st.session_state["predictions_available"] = True
            st.session_state["final_preds_df_com_agg"] = final_preds_df_com_agg.round(0)
            st.session_state["final_preds_df_ama_agg"] = final_preds_df_ama_agg.round(0)
        st.success('Predictions are now available!')

# Show the predictions once they are available
if st.session_state["predictions_available"]:
    st.markdown("#### Forecast results are available below:")

    tab1, tab2 = st.tabs([".com Predictions", "Amazon Predictions"])

    # .com Predictions Tab
    with tab1:
        st.header('.com Prediction Results')
        st.write("Here are the aggregated results for the website sales channel (.com).")
        
        # Display the .com DataFrame
        st.dataframe(st.session_state["final_preds_df_com_agg"])

        # Provide an option to download .com CSV
        com_csv = convert_df_to_csv(st.session_state["final_preds_df_com_agg"])
        st.download_button(
            label="Download .com Predictions as CSV",
            data=com_csv,
            file_name='com_predictions.csv',
            mime='text/csv'
        )

    # Amazon Predictions Tab
    with tab2:
        st.header('Amazon Prediction Results')
        st.write("Here are the aggregated results for the Amazon sales channel.")
        
        # Display the Amazon DataFrame
        st.dataframe(st.session_state["final_preds_df_ama_agg"])

        # Provide an option to download Amazon CSV
        amazon_csv = convert_df_to_csv(st.session_state["final_preds_df_ama_agg"])
        st.download_button(
            label="Download Amazon Predictions as CSV",
            data=amazon_csv,
            file_name='amazon_predictions.csv',
            mime='text/csv'
        )

    # Button to rerun the predictions and reset the app
    if st.button('Rerun Predictions'):
        reset_state()

# Footer
st.markdown("""
---
**Note**: This app enables you to view and download prediction results. Make sure to upload the latest dataset and model files to ensure up-to-date predictions.
""")
