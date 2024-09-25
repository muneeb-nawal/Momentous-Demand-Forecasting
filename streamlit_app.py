import streamlit as st
import pandas as pd
import base64
from io import BytesIO
import time
from prediction_code import load_and_process_data
import numpy as np


# Set page config
st.set_page_config(page_title="Momentous Demand Forecast Tool", layout="wide")

# Convert the provided logo to base64
def img_to_base64(img_path):
    with open(img_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

momentous_logo_base64 = img_to_base64("momentous_logo_BG.png")

# Custom CSS
st.markdown("""
<style>
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 5px 20px;
        background-color: #000000;
        height: 40px;
    }
    .logo-container img {
        height: 30px;
        width: auto;
    }
    .right-container {
        display: flex;
        align-items: center;
    }
    .logout-button {
        background-color: transparent !important;
        color: white !important;
        border: 1px solid white !important;
        padding: 2px 8px !important;
        border-radius: 4px;
        font-size: 11px !important;
        transition: background-color 0.3s, color 0.3s;
    }
    .logout-button:hover {
        background-color: white !important;
        color: black !important;
    }
    .separator {
        border-top: 1px solid #e0e0e0;
        margin: 0;
        padding: 0;
    }
    .powered-by {
        color: #666;
        font-size: 14px;
        text-align: right;
        padding-right: 20px;
        margin-top: 1px;
    }
    .powered-by img {
        height: 18px;
        vertical-align: middle;
        margin-left: 2px;
    }
    .main-header {
        font-size: 22px;
        font-weight: bold;
        margin-top: 10px;
        margin-bottom: 5px;
    }
    .stButton > button {
        height: auto;
    }
</style>
""", unsafe_allow_html=True)

def custom_header():
    col1, col2, col3 = st.columns([1, 5, 1])
    
    with col1:
        st.image("momentous_logo_BG.png", width=80)
    
    with col3:
        if st.button("Logout", key="logout_button"):
            handle_logout()

    st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 5, 1])
    with col3:
        st.markdown(
            '<div class="powered-by">Powered by <img src="data:image/png;base64,{}" alt="Saras Logo"></div>'.format(
                base64.b64encode(open('saras-logo.png', 'rb').read()).decode()
            ),
            unsafe_allow_html=True
        )

def handle_logout():
    st.session_state.clear()
    st.session_state.logged_in = False
    st.session_state.logout_message = "You have been successfully logged out."
    time.sleep(0.1)  # Small delay to ensure the message is displayed
    st.rerun()

        
def login_page():
    if 'logout_message' in st.session_state:
        st.success(st.session_state.logout_message)
        del st.session_state.logout_message

    col1, col2, col3 = st.columns([1,2,1])
    
    with col2:
        st.markdown(
            f"""
            <div style="text-align: center; padding: 20px;">
                <img src="data:image/png;base64,{base64.b64encode(open('momentous-logo.png', 'rb').read()).decode()}" alt="Momentous Logo" width="200">
            </div>
            """,
            unsafe_allow_html=True
        )
        
        with st.form("login_form"):
            email = st.text_input("Email", key="email")
            password = st.text_input("Password", type="password", key="password")
            submit_button = st.form_submit_button("Login")

        if submit_button:
            if email == "saras@livemomentous.com" and password == "df-saras123":
                st.session_state.logged_in = True
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Incorrect email or password. Please try again.")
def format_currency(value):
    return f"${value:,.0f}"

def format_channel_name(name):
    return name.replace('_', ' ')

def main_app():
    custom_header()

    st.title("Momentous Demand Forecast Tool")
    st.markdown("This app allows you to view SKU-wise forecasts for units sold on both website (.com) and Amazon sales channels.")

    # Initialize run counter if not exists
    if 'run_counter' not in st.session_state:
        st.session_state.run_counter = 0
    if 'view_option' not in st.session_state:
        st.session_state.view_option = "Split by Sales Channel"

    # Determine if the expander should be expanded
    expand_spends = st.session_state.run_counter == 0

    with st.expander("Update Spends for Marketing Channels", expanded=expand_spends):
        st.markdown("""
        <div style='background-color: #f0f2f6; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
        <h4 style='margin-top: 0;'>Note:</h4>
        <ul>
            <li>The table below shows the default spend values.</li>
            <li>You can edit these values directly in the table if you wish to customize them.</li>
            <li>All changes are automatically saved.</li>
            <li>If a value doesn't update immediately, try editing it again.</li>
            <li>Click 'Generate Results' when you're ready to proceed with the current spend values.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        channels = ["Spends_Meta", "Spends_Google", "Spends_Amazon", "Spends_Audio_SponCon", "Spends_Partnerships", "Spends_Others"]
        months = ["current month", "month 1", "month 2", "month 3", "month 4"]
        
        if 'marketing_data' not in st.session_state:
            st.session_state.marketing_data = pd.DataFrame({
                'Channel': channels,
                'current month': [76904.46, 169527.61, 214805.7733, 355941.7067, 449181.4467, 7000.0],
                'month 1': [110000.0, 175000.0, 212215.0, 219228.0, 306728.0, 15000.0],
                'month 2': [120000.0, 225000.0, 233437.0, 254928.0, 364928.0, 20000.0],
                'month 3': [85000.0, 175000.0, 256781.0, 224228.0, 331728.0, 65000.0],
                'month 4': [85000.0, 175000.0, 256781.0, 224228.0, 331728.0, 65000.0]
            })

        display_df = st.session_state.marketing_data.copy()
        display_df['Channel'] = display_df['Channel'].apply(lambda x: x.replace('_', ' '))
        
        edited_df = st.data_editor(
            display_df,
            num_rows="fixed",
            hide_index=True,
            key="custom_spends_editor",
            column_config={
                "Channel": st.column_config.TextColumn(
                    "Marketing Channels",
                    width="medium",
                    required=True,
                ),
                **{col: st.column_config.NumberColumn(
                    col,
                    min_value=0.0,
                    max_value=1000000.0,
                    step=0.01,
                    format="$%.0f"
                ) for col in months}
            },
            disabled=["Channel"]
        )
        
        st.session_state.marketing_data[months] = edited_df[months]

    # Generate Results button
    if st.session_state.run_counter > 0:
        st.warning("⚠️ Warning: Proceeding with generate results operation will overwrite the current results. Please ensure that you have saved the results before continuing.")

    if st.button('Generate Results'):
        with st.spinner('Running predictions... This may take some time.'):
            numeric_df = st.session_state.marketing_data.copy()
            numeric_df[months] = numeric_df[months].apply(pd.to_numeric, errors='coerce')
            
            final_preds_df_com_agg, final_preds_df_ama_agg, \
            final_preds_df_dotcom_new, final_preds_df_dotcom_ret, final_preds_df_dotcomsubs_subsplit, \
            final_preds_df_ama_new, final_preds_df_ama_ret, final_preds_df_amasubs_subsplit = load_and_process_data(numeric_df)

            st.session_state.predictions_available = True
            st.session_state.final_preds_df_com_agg = final_preds_df_com_agg
            st.session_state.final_preds_df_ama_agg = final_preds_df_ama_agg
            st.session_state.final_preds_df_dotcom_new = final_preds_df_dotcom_new
            st.session_state.final_preds_df_dotcom_ret = final_preds_df_dotcom_ret
            st.session_state.final_preds_df_dotcomsubs_subsplit = final_preds_df_dotcomsubs_subsplit
            st.session_state.final_preds_df_ama_new = final_preds_df_ama_new
            st.session_state.final_preds_df_ama_ret = final_preds_df_ama_ret
            st.session_state.final_preds_df_amasubs_subsplit = final_preds_df_amasubs_subsplit
            st.session_state.run_counter += 1
        st.success('Predictions are now available!')
        st.rerun()  # Use experimental_rerun() instead of rerun()

    st.markdown("---")

    # Display Results
    if st.session_state.get('predictions_available', False):
        st.markdown("#### Forecast Results")
        
        st.markdown("**🔔 Important: To save your results, please use the download functionality in each table. Results are not automatically saved.**")
        
        # Move the radio button outside of the if statement
        st.session_state.view_option = st.radio(
            "Choose view option:", 
            ("Split by Sales Channel", "Split by Order Type"), 
            key="view_option_radio",
            index=0 if st.session_state.view_option == "Split by Sales Channel" else 1
        )

        tab1, tab2 = st.tabs([".com Predictions", "Amazon Predictions"])

        def display_dataframe_with_totals(df):
            df_display = df.copy()
            df_display['Total'] = df_display.select_dtypes(include=[np.number]).sum(axis=1)
            df_display = df_display.sort_values('Total', ascending=False).reset_index(drop=True)
            numeric_cols = df_display.select_dtypes(include=[np.number]).columns
            df_display[numeric_cols] = df_display[numeric_cols].applymap(lambda x: f"{x:,.0f}")
            if 'Channel' in df_display.columns:
                df_display['Channel'] = df_display['Channel'].apply(lambda x: x.replace('_', ' '))
            st.dataframe(df_display)

        with tab1:
            if st.session_state.view_option == "Split by Sales Channel":
                display_dataframe_with_totals(st.session_state.final_preds_df_com_agg)
            else:
                st.subheader("New Customer Predictions")
                display_dataframe_with_totals(st.session_state.final_preds_df_dotcom_new)
                st.subheader("Returning Customer Predictions")
                display_dataframe_with_totals(st.session_state.final_preds_df_dotcom_ret)
                st.subheader("Subscription Customer Predictions")
                display_dataframe_with_totals(st.session_state.final_preds_df_dotcomsubs_subsplit)

        with tab2:
            if st.session_state.view_option == "Split by Sales Channel":
                display_dataframe_with_totals(st.session_state.final_preds_df_ama_agg)
            else:
                st.subheader("New Customer Predictions")
                display_dataframe_with_totals(st.session_state.final_preds_df_ama_new)
                st.subheader("Returning Customer Predictions")
                display_dataframe_with_totals(st.session_state.final_preds_df_ama_ret)
                st.subheader("Subscription Customer Predictions")
                display_dataframe_with_totals(st.session_state.final_preds_df_amasubs_subsplit)


    st.markdown("**Note**: This app enables you to view prediction results. Make sure to upload the latest dataset and model files to ensure up-to-date predictions.")

# Main execution
if __name__ == "__main__":
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'widget_value' not in st.session_state:
        st.session_state.widget_value = None

    if st.session_state.logged_in:
        main_app()
    else:
        login_page()