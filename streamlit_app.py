import streamlit as st
import pandas as pd
import base64
from io import BytesIO
import time
from prediction_code import load_and_process_data
import numpy as np
from datetime import datetime, timedelta


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
        
        # Define the base table
        base_table = pd.DataFrame({
            'order month': pd.date_range(start='9/1/2024', periods=16, freq='MS'),
            'Spends_Amazon': [224372.24] + [212215] + [233437] + [256781]*13,
            'Spends_Audio_SponCon': [352848.56] + [219228] + [254928] + [224228]*13,
            'Spends_Google': [167742.84] + [175000] + [225000] + [175000]*13,
            'Spends_Meta': [129333.12] + [110000] + [120000] + [85000]*13,
            'Spends_Others': [3000] + [15000] + [20000] + [65000]*13,
            'Spends_Partnerships': [449602.39] + [306728] + [364928] + [331728]*13
        })

        # Function to get the dynamic month label
        def get_dynamic_month(date):
            today = datetime.now().replace(day=1)
            months_diff = (date.year - today.year) * 12 + date.month - today.month
            if months_diff == 0:
                return "current month"
            elif 1 <= months_diff <= 4:
                return f"month {months_diff}"
            else:
                return None

        # Apply the dynamic month labeling
        base_table['dynamic_month'] = base_table['order month'].apply(get_dynamic_month)

        # Filter to only include rows with a dynamic month label
        display_df = base_table[base_table['dynamic_month'].notna()].copy()

        # Prepare the data for UI display
        ui_df = display_df.melt(id_vars=['order month', 'dynamic_month'], 
                                var_name='Marketing Channels', 
                                value_name='Spend')

        # Pivot the melted dataframe to get dates as columns
        ui_df = ui_df.pivot(index='Marketing Channels', columns='order month', values='Spend')
        ui_df = ui_df.reset_index()

        # Rename columns to date strings
        ui_df.columns.name = None
        ui_df.columns = ['Marketing Channels'] + [col.strftime('%Y-%m-%d') for col in ui_df.columns[1:]]

        # Replace underscores with spaces in Marketing Channels
        ui_df['Marketing Channels'] = ui_df['Marketing Channels'].str.replace('_', ' ')

        # Display the editable table in the UI
        st.subheader("Marketing Spends")
        edited_df = st.data_editor(
            ui_df,
            num_rows="fixed",
            hide_index=True,
            key="custom_spends_editor",
            column_config={
                "Marketing Channels": st.column_config.TextColumn(
                    "Marketing Channels",
                    width="medium",
                    required=True,
                ),
                **{col: st.column_config.NumberColumn(
                    col,
                    min_value=0.0,
                    max_value=1000000.0,
                    step=0.01,
                    format="$%.2f"
                ) for col in ui_df.columns if col != 'Marketing Channels'}
            },
            disabled=["Marketing Channels"]
        )

        # Transform the data for backend processing
        backend_df = edited_df.copy()
        backend_df.columns = ['Channel'] + list(display_df['dynamic_month'])
        backend_df['Channel'] = backend_df['Channel'].str.replace(' ', '_')

        # Print confirmation for backend data to the command prompt
        print("\nBackend Data (for confirmation):")
        print(backend_df.to_string(index=False))

    # Generate Results button
    if st.session_state.run_counter > 0:
        st.warning("⚠️ Warning: Proceeding with generate results operation will overwrite the current results. Please ensure that you have saved the results before continuing.")

    if st.button('Generate Results'):
        with st.spinner('Running predictions... This may take some time.'):
            # Use the backend data for predictions
            numeric_df = backend_df.copy()
            # numeric_df.set_index('Channel', inplace=True)
            # numeric_df = numeric_df.apply(pd.to_numeric, errors='coerce')
            print(numeric_df)
            
            final_preds_df_com_agg, final_preds_df_ama_agg, \
            final_preds_df_dotcom_new, final_preds_df_dotcom_ret, final_preds_df_dotcomsubs_subsplit, \
            final_preds_df_ama_new, final_preds_df_ama_ret, final_preds_df_amasubs_subsplit = load_and_process_data(numeric_df)

            # Store the results in session state
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
        st.rerun()

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
                st.subheader("New Customer Units Sold Predictions")
                display_dataframe_with_totals(st.session_state.final_preds_df_dotcom_new)
                st.subheader("Returning Customer Units Sold Predictions")
                display_dataframe_with_totals(st.session_state.final_preds_df_dotcom_ret)
                st.subheader("Subscription Customer Units Sold Predictions")
                display_dataframe_with_totals(st.session_state.final_preds_df_dotcomsubs_subsplit)

        with tab2:
            if st.session_state.view_option == "Split by Sales Channel":
                display_dataframe_with_totals(st.session_state.final_preds_df_ama_agg)
            else:
                st.subheader("New Customer Units Sold Predictions")
                display_dataframe_with_totals(st.session_state.final_preds_df_ama_new)
                st.subheader("Returning Customer Units Sold Predictions")
                display_dataframe_with_totals(st.session_state.final_preds_df_ama_ret)
                st.subheader("Subscription Customer Units Sold Predictions")
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