# import streamlit as st
# import pandas as pd
# from io import BytesIO
# from prediction_code import load_and_process_data

# # Set up the Streamlit app title and description
# st.set_page_config(page_title="Momentous Demand Forecast Tool", layout="wide")

# # Function to load predictions
# @st.cache_data
# def load_predictions():
#     import time
#     time.sleep(3)  # Simulating a delay for testing
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
#     st.query_params()  # Updated with the new method

# # Homepage with logos
# st.image('momentous-logo.png', width=100)  # Placeholder for client logo

# # Add the Powered by Saras Analytics to the top-right
# st.markdown(
#     "<div style='text-align: right; color: blue;'>Powered by Saras Analytics</div>",
#     unsafe_allow_html=True
# )

# st.title('Momentous Demand Forecast Tool')
# st.markdown("""
# **This app allows you to view and download SKU-wise forecasts for units sold on both website (.com) and Amazon sales channels.**
# """)

# # Button to initiate predictions
# if not st.session_state["predictions_available"]:
#     if st.button('Initiate Model Results'):
#         with st.spinner('Running predictions... This may take some time.'):
#             final_preds_df_com_agg, final_preds_df_ama_agg = load_predictions()
#             st.session_state["predictions_available"] = True
#             st.session_state["final_preds_df_com_agg"] = final_preds_df_com_agg.round(0)
#             st.session_state["final_preds_df_ama_agg"] = final_preds_df_ama_agg.round(0)
#         st.success('Predictions are now available!')

# # Show the predictions once they are available
# if st.session_state["predictions_available"]:
#     st.markdown("#### Forecast results are available below:")

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

######################################################################################################
######################################################################################################
######################################################################################################
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from prediction_code import load_and_process_data
import base64

# Set page config
st.set_page_config(page_title="Momentous Demand Forecast Tool", layout="wide")

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }
    
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 20px;
        background-color: #000000;
    }
    .logo-container {
        display: flex;
        align-items: center;
    }
    .logo-container img {
        height: 30px;
        margin-right: 10px;
    }
    .header-text {
        color: white;
        font-size: 24px;
        font-weight: bold;
    }
    .right-container {
        display: flex;z
        align-items: center;
    }
    .logout-button {
        background-color: transparent;
        color: white;
        border: none;
        cursor: pointer;
        margin-right: 20px;
    }
    .powered-by {
        color: white;
        margin-right: 10px;
    }
    
    .stButton>button {
        background-color: #1E90FF;
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        transition: background-color 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #1C86EE;
    }
    
    .styled-table {
        border-collapse: collapse;
        margin: 25px 0;
        font-size: 0.9em;
        font-family: sans-serif;
        min-width: 400px;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
    }
    
    .styled-table thead tr {
        background-color: #1E90FF;
        color: #ffffff;
        text-align: left;
    }
    
    .styled-table th,
    .styled-table td {
        padding: 12px 15px;
    }
    
    .styled-table tbody tr {
        border-bottom: 1px solid #dddddd;
    }

    .styled-table tbody tr:nth-of-type(even) {
        background-color: #f3f3f3;
    }

    .styled-table tbody tr:last-of-type {
        border-bottom: 2px solid #1E90FF;
    }
</style>
""", unsafe_allow_html=True)

# Custom header
def custom_header():
    header_html = f"""
    <div class="header-container">
        <div class="logo-container">
            <img src="data:image/png;base64,{base64.b64encode(open('momentous-logo.png', 'rb').read()).decode()}" alt="Momentous Logo">
            <span class="header-text">Momentous</span>
        </div>
        <div class="right-container">
            <button class="logout-button" onclick="handleLogout()">Logout</button>
            <span class="powered-by">Powered by</span>
            <img src="data:image/png;base64,{base64.b64encode(open('saras-logo.png', 'rb').read()).decode()}" alt="Saras Logo" height="20">
        </div>
    </div>
    <script>
    function handleLogout() {{
        window.parent.postMessage({{type: 'streamlit:setComponentValue', value: 'logout'}}, '*');
    }}
    </script>
    """
    st.markdown(header_html, unsafe_allow_html=True)

    # Handle logout action
    if st.session_state.get('widget_value') == 'logout':
        st.session_state.logged_in = False
        st.experimental_rerun()

# Login page
def login_page():
    st.markdown("<h1 style='text-align: center;'>Login</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image("momentous-logo.png", width=200)
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if email == "saras@livemomentous.com" and password == "df-saras123":
                st.session_state.logged_in = True
                st.experimental_rerun()
            else:
                st.error("Incorrect email or password. Please try again.")

# Main app
def main_app():
    custom_header()

    st.title("Momentous Demand Forecast Tool")
    st.markdown("This app allows you to view and download SKU-wise forecasts for units sold on both website (.com) and Amazon sales channels.")

    # Sales Channels
    st.subheader("Sales Channels")
    # col1, col2 = st.columns(2)
    # with col1:
    #     website_channel = st.checkbox("Website (.com)", value=True)
    # with col2:
    #     amazon_channel = st.checkbox("Amazon", value=True)

    # Marketing Channels
    st.subheader("Marketing Channels")
    spend_option = st.radio("Choose spend option:", ("Default Spends", "Custom Spends"))
    
    channels = ["Spends_Meta", "Spends_Google", "Spends_Amazon", "Spends_Audio_SponCon", "Spends_Partnerships", "Spends_Others"]
    months = ["current month", "month 1", "month 2", "month 3", "month 4"]
    
    if 'marketing_data' not in st.session_state or spend_option == "Default Spends":
        st.session_state.marketing_data = pd.DataFrame({
            'Channel': channels,
            'current month': [76904.46, 169527.61, 214805.7733, 355941.7067, 449181.4467, 7000.0],
            'month 1': [110000.0, 175000.0, 212215.0, 219228.0, 306728.0, 15000.0],
            'month 2': [120000.0, 225000.0, 233437.0, 254928.0, 364928.0, 20000.0],
            'month 3': [85000.0, 175000.0, 256781.0, 224228.0, 331728.0, 65000.0],
            'month 4': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        })

    if spend_option == "Custom Spends":
        edited_df = st.data_editor(
            st.session_state.marketing_data, 
            num_rows="fixed",
            hide_index=True,
            column_config={
                "Channel": st.column_config.TextColumn(
                    "Marketing Channels",
                    width="medium",
                    required=True,
                ),
                **{month: st.column_config.NumberColumn(
                    month,
                    min_value=0.0,
                    max_value=1000000.0,
                    step=0.01,
                    format="%.2f"
                ) for month in months}
            }
        )
        st.session_state.marketing_data = edited_df

    # # Product Selection
    # st.subheader("Product Selection")
    # product_selection = st.radio("Select products", ("Select all products", "Custom product selection"))

    # if product_selection == "Custom product selection":
    #     # You would need to populate this list with actual product names
    #     products = ["Product 1", "Product 2", "Product 3", "Product 4"]
    #     selected_products = st.multiselect("Select products", products)

    # Generate Results button
    if st.button('Generate Results'):
        with st.spinner('Running predictions... This may take some time.'):
            # Convert DataFrame to numeric, excluding the 'Channel' column
            numeric_df = st.session_state.marketing_data.copy()
            numeric_df[months] = numeric_df[months].apply(pd.to_numeric, errors='coerce')
            
            final_preds_df_com_agg, final_preds_df_ama_agg = load_and_process_data(numeric_df)
            st.session_state.predictions_available = True
            st.session_state.final_preds_df_com_agg = final_preds_df_com_agg
            st.session_state.final_preds_df_ama_agg = final_preds_df_ama_agg
        st.success('Predictions are now available!')
        st.experimental_rerun()

    # Display Results
    if st.session_state.get('predictions_available', False):
        st.markdown("#### Forecast Results")
        
        # Filter options
        # col1, col2, col3, col4 = st.columns(4)
        # with col1:
        #     sales_channel_filter = st.selectbox("Sales channel", ["All", "Website", "Amazon"])
        # with col2:
        #     marketing_channel_filter = st.selectbox("Marketing channel", ["All"] + channels)
        # with col3:
        #     product_category_filter = st.selectbox("Product category", ["All", "Category 1", "Category 2"])
        # with col4:
        #     product_filter = st.selectbox("Product selection", ["All"] + (selected_products if product_selection == "Custom product selection" else ["All products"]))

        tab1, tab2 = st.tabs([".com Predictions", "Amazon Predictions"])

        with tab1:
            st.dataframe(st.session_state.final_preds_df_com_agg)
            com_csv = convert_df_to_csv(st.session_state.final_preds_df_com_agg)
            st.download_button(label="Download .com Predictions as CSV",
                               data=com_csv,
                               file_name='com_predictions.csv',
                               mime='text/csv')

        with tab2:
            st.dataframe(st.session_state.final_preds_df_ama_agg)
            ama_csv = convert_df_to_csv(st.session_state.final_preds_df_ama_agg)
            st.download_button(label="Download Amazon Predictions as CSV",
                               data=ama_csv,
                               file_name='amazon_predictions.csv',
                               mime='text/csv')

    # Footer
    st.markdown("---")
    st.markdown("**Note**: This app enables you to view and download prediction results. Make sure to upload the latest dataset and model files to ensure up-to-date predictions.")

# Helper function to convert DataFrame to CSV
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# Main execution
if __name__ == "__main__":
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        main_app()
    else:
        login_page()