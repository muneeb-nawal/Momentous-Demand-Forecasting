# import streamlit as st
# import pandas as pd
# from io import BytesIO
# from prediction_code import load_and_process_data

# # Set up the Streamlit app title and description
# st.title('Prediction Results Viewer')
# st.markdown("""
# This app allows you to view and download prediction results for both website (.com) and Amazon sales channels.
# """)

# # Function to load predictions
# @st.cache_data
# def load_predictions():
#     # Simulate long running process for testing spinner (Optional: Remove in production)
#     import time
#     time.sleep(3)  # Simulating a delay
#     final_preds_df_com_agg, final_preds_df_ama_agg = load_and_process_data()
#     return final_preds_df_com_agg, final_preds_df_ama_agg

# # Function to convert DataFrame to CSV
# def convert_df_to_csv(df):
#     output = BytesIO()
#     df.to_csv(output, index=False)
#     return output.getvalue().decode('utf-8')

# # Initialize session state to manage UI elements and results
# if "predictions_available" not in st.session_state:
#     st.session_state["predictions_available"] = False

# # Function to reset session state for rerun
# def reset_state():
#     st.session_state["predictions_available"] = False
#     st.session_state.pop("final_preds_df_com_agg", None)
#     st.session_state.pop("final_preds_df_ama_agg", None)
#     st.rerun()  # Ensure the app resets fully

# # Button to initiate predictions
# if not st.session_state["predictions_available"]:
#     # Add an explanation message before initiating
#     st.markdown("### Click the button to run predictions:")
    
#     # Add a button to initiate the process
#     if st.button('Initiate Run'):
#         with st.spinner('Running predictions... Please wait.'):
#             # Load the predictions (this might take time)
#             final_preds_df_com_agg, final_preds_df_ama_agg = load_predictions()
            
#             # Store results in session state and round numbers
#             st.session_state["predictions_available"] = True
#             st.session_state["final_preds_df_com_agg"] = final_preds_df_com_agg.round(0)
#             st.session_state["final_preds_df_ama_agg"] = final_preds_df_ama_agg.round(0)

#         # Once available, immediately show the predictions
#         st.success('Predictions are now available!')

#         # Display results immediately
#         st.experimental_rerun()  # Force rerun to show the tabs and results

# # Display the predictions once they are available
# if st.session_state["predictions_available"]:
#     # Display the prediction tabs when predictions are available
#     st.markdown("### Predictions loaded successfully.")
    
#     tab1, tab2 = st.tabs([".com Predictions", "Amazon Predictions"])

#     # .com Predictions Tab
#     with tab1:
#         st.header('.com Prediction Results')
#         st.write("Here are the aggregated results for the website sales channel (.com).")

#         # Display the .com DataFrame
#         st.dataframe(st.session_state["final_preds_df_com_agg"])

#         # Provide an option to download .com CSV
#         com_csv = convert_df_to_csv(st.session_state["final_preds_df_com_agg"])
#         st.download_button(
#             label="Download .com Predictions as CSV",
#             data=com_csv,
#             file_name='com_predictions.csv',
#             mime='text/csv'
#         )

#     # Amazon Predictions Tab
#     with tab2:
#         st.header('Amazon Prediction Results')
#         st.write("Here are the aggregated results for the Amazon sales channel.")

#         # Display the Amazon DataFrame
#         st.dataframe(st.session_state["final_preds_df_ama_agg"])

#         # Provide an option to download Amazon CSV
#         amazon_csv = convert_df_to_csv(st.session_state["final_preds_df_ama_agg"])
#         st.download_button(
#             label="Download Amazon Predictions as CSV",
#             data=amazon_csv,
#             file_name='amazon_predictions.csv',
#             mime='text/csv'
#         )

#     # Button to rerun the predictions and reset the app
#     if st.button('Rerun Predictions'):
#         reset_state()

# # Footer
# st.markdown("""
# ---
# **Note**: This app enables you to view and download prediction results. Make sure to upload the latest dataset and model files to ensure up-to-date predictions.
# """)


import streamlit as st
import pandas as pd
from io import BytesIO
from prediction_code import load_and_process_data

# Set up the Streamlit app title and description
st.title('Prediction Results Viewer')
st.markdown("""
This app allows you to view and download prediction results for both website (.com) and Amazon sales channels.
""")

# Function to load predictions
@st.cache_data
def load_predictions():
    # Simulate long running process for testing spinner (Optional: Remove in production)
    import time
    time.sleep(3)  # Simulating a delay
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
    st.experimental_set_query_params()  # Ensure the app resets fully

# Button to initiate predictions
if not st.session_state["predictions_available"]:
    # Add an explanation message before initiating
    st.markdown("### Click the button to run predictions:")
    
    # Add a button to initiate the process
    if st.button('Initiate Run'):
        with st.spinner('Running predictions... Please wait.'):
            # Load the predictions (this might take time)
            final_preds_df_com_agg, final_preds_df_ama_agg = load_predictions()
            
            # Store results in session state and round numbers
            st.session_state["predictions_available"] = True
            st.session_state["final_preds_df_com_agg"] = final_preds_df_com_agg.round(0)
            st.session_state["final_preds_df_ama_agg"] = final_preds_df_ama_agg.round(0)

        # Once available, immediately show the predictions
        st.success('Predictions are now available!')

# Display the predictions once they are available
if st.session_state["predictions_available"]:
    # Display the prediction tabs when predictions are available
    st.markdown("### Predictions loaded successfully.")
    
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
