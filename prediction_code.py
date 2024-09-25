# 1. Imports and Setup

import os
import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
import pickle
import logging
import warnings
import joblib
import regex as re
import importlib


def get_joblib(version):
    spec = importlib.util.find_spec("joblib")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


joblib_1_2_0 = get_joblib("1.2.0")
joblib_1_4_2 = get_joblib("1.4.2")


# Ignore all warnings
warnings.filterwarnings('ignore')

# Setup logging configuration
logging.basicConfig(filename='preds.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 2. Data Transformation Functions

def create_transformed_df(df, sales_channel):
    """
    Filter the dataframe by the given sales channel and apply the necessary transformations.
    Returns the transformed dataframe.
    """
    
    df_copy = df.copy()
    df_copy['acquisition_month_sku_level'] = pd.to_datetime(df_copy['acquisition_month_sku_level'])
    
    # Drop unnecessary columns
    df_copy = df_copy.drop(['Varient title', 'Amazon_spend_sku_wise'], axis=1)

    # Filter the DataFrame based on the sales_channel
    filtered_df = df_copy[df_copy['sales_channel'] == sales_channel]
    filtered_df.fillna(0,inplace=True)

    
    # Apply transformations for each unique product
    transformed_dfs = []
    unique_product_names = filtered_df['product_name'].unique()
    
    for product_name in unique_product_names:
        if 'Unknown Product' in product_name:
            continue
        
        # Sanitize product name
        sanitized_product_name = str(product_name).replace(" ", "_").replace("/", "_").replace("\\", "_")
        product_df = filtered_df[filtered_df['product_name'] == product_name]
        
        # Rename columns
        product_df = product_df.rename(columns={
            'acquisition_month_sku_level': 'reg_date',
            'Order_Month': 'upgrade_date',
            'mapped_order_interval': 'order_interval',
            'unit_sold_unbundled': 'premiums'
        })
        
        # Modify date columns and filter based on order_interval
        product_df['upgrade_date'] = product_df['upgrade_date'].astype(str) + '-01'
        product_df['reg_date'] = product_df['reg_date'].astype(str)
        product_df = product_df[product_df['order_interval'] != 0]

        # Drop unnecessary columns
        product_df = product_df.drop(columns=['Year', 'sales_channel', 'SKU', 'product_name_priority_sku'], errors='ignore')
        
        transformed_dfs.append(product_df)
    
    if transformed_dfs:
        return pd.concat(transformed_dfs, ignore_index=True)
    else:
        return pd.DataFrame()

# 3. Data Preprocessing Functions

def enhanced_preprocess_data(dataset):
    """
    Preprocess the dataset by converting date columns and extracting date components.
    """
    try:
        logging.info("Starting data preprocessing")

        # Convert date columns
        dataset['reg_date'] = pd.to_datetime(dataset['reg_date'])
        dataset['upgrade_date'] = pd.to_datetime(dataset['upgrade_date'])
        dataset['age'] = dataset['age'].astype(int)

        # Extract date components from 'reg_date'
        dataset['reg_month'] = dataset['reg_date'].dt.month
        dataset['reg_year'] = dataset['reg_date'].dt.year
        dataset['reg_day_of_year'] = dataset['reg_date'].dt.dayofyear
        dataset['reg_week_of_year'] = dataset['reg_date'].dt.isocalendar().week.astype(int, errors='ignore')

        # Convert 'order_interval' to float
        dataset['order_interval'] = dataset['order_interval'].astype('float64')

        logging.info("Data preprocessing completed")
        return dataset

    except Exception as e:
        logging.error(f"Error during data preprocessing. Exception: {e}")
        return None

# 4. Process Products by Name

def process_products_by_name(df_dotcom):
    """
    Process and store data for each unique product in a dictionary.
    """
    try:
        logging.info("Processing all products by unique product names")

        if 'product_name' not in df_dotcom.columns:
            logging.error("'product_name' column is missing in the DataFrame")
            return {}

        product_dict = {}
        unique_product_names = df_dotcom['product_name'].unique()

        for product_name in unique_product_names:
            parts = product_name.split('_')

            if len(parts) < 4:
                logging.warning(f"Skipping product with insufficient parts: {product_name}")
                continue

            serial_number = parts[0]
            sku_id = parts[2]
            identifier = f"df_{serial_number}_{parts[1]}_{sku_id}"

            logging.info(f"Processing dataset with identifier: {identifier}")
            product_data = df_dotcom[df_dotcom['product_name'] == product_name]

            product_data = enhanced_preprocess_data(product_data)
            product_dict[identifier] = product_data

        logging.info("Completed processing and preprocessing all products by unique product names")
        return product_dict

    except Exception as e:
        logging.error(f"Error processing products. Exception: {e}")
        return {}

# 5. Model Prediction



def find_model_file(directory, key):
    """
    Find the appropriate model file based on the key pattern.
    Example of key pattern: 'key_M*.pkl' where * represents any sequence.
    """
    try:
        # Create a regex pattern based on the key (for example, key_M*.pkl)
        pattern = re.compile(f"{key}_M.*\.pkl")

        all_files = os.listdir(directory)
        for file_name in all_files:
            # Match files based on the constructed pattern
            if pattern.match(file_name):
                return os.path.join(directory, file_name)
        return None

    except Exception as e:
        logging.error(f"Error finding model file for key '{key}': {e}")
        return None

def predict_with_saved_models(input_data, directory, keys=None, reconstruct_dates=True):
    """
    Load saved models and make predictions based on the input data.
    """
    logging.info("Starting prediction with saved models")
    results_list = []

    if not os.path.exists(directory):
        logging.error(f"Directory {directory} does not exist.")
        return pd.DataFrame()

    if keys is None:
        keys = input_data.keys()

    for key in keys:
        try:
            logging.info(f"Processing key: {key}")

            if key in input_data:
                data = input_data[key]
                model_path = find_model_file(directory, key)

                if model_path:
                    logging.info(f"Loading model from: {model_path}")
                    model = joblib.load(model_path)
                    logging.info(f"Model loaded successfully for key: {key}")

                    model_features = model.feature_names_in_
                    x_input = data[model_features].copy()

                    predictions = model.predict(x_input)
                    results_df = x_input.copy()
                    results_df['Predicted'] = predictions
                    results_df['SKU'] = key

                    if reconstruct_dates:
                        results_df['reg_date'] = pd.to_datetime(results_df['reg_year'].astype(str) + '-' + results_df['reg_month'].astype(str) + '-01')
                        results_df['upgrade_date'] = results_df.apply(lambda row: row['reg_date'] + relativedelta(months=row['age']), axis=1)

                    results_list.append(results_df)
                    logging.info(f"Results compiled for key: {key}")

                else:
                    logging.warning(f"Model file not found for key: {key}")
            else:
                logging.warning(f"Key '{key}' not found in input data.")
        except Exception as e:
            logging.error(f"An error occurred while processing key '{key}': {e}")

    try:
        final_results_df = pd.concat(results_list, ignore_index=True)
        logging.info("Results successfully compiled into a DataFrame")
    except ValueError as e:
        logging.error(f"An error occurred while concatenating results: {e}")
        final_results_df = pd.DataFrame()

    logging.info("Completed prediction process")
    return final_results_df

# 6. Get Final Results

def get_final_results(best_final_results):
    """
    Filter and pivot the final results DataFrame based on the upgrade_date
    for the current month and until the end of December 2025.
    """
    best_final_results['upgrade_date'] = pd.to_datetime(best_final_results['upgrade_date'])

    current_date = datetime.datetime.now()
    start_date = current_date.replace(day=1).date()

    end_date = datetime.datetime(2025, 12, 31).date()

    best_final_results['upgrade_date'] = best_final_results['upgrade_date'].dt.date
    best_final_results_filtered = best_final_results[
        (best_final_results['upgrade_date'] >= start_date) & 
        (best_final_results['upgrade_date'] <= end_date)
    ]

    best_final_results_grouped = best_final_results_filtered.groupby(['SKU', 'upgrade_date'])['Predicted'].sum().reset_index()
    best_final_results_pivoted = best_final_results_grouped.pivot(index='SKU', columns='upgrade_date', values='Predicted')

    best_final_results_pivoted.columns = best_final_results_pivoted.columns.map(lambda x: x.strftime('%Y-%m-%d'))

    return best_final_results_pivoted.reset_index()

def calculate_total_months(start_date_str, end_date_str):
    """
    Calculate the total number of months between two date strings.
    """
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)
    return (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)

def create_transformed_df_otp(df, sales_channel):
        """
        Filter the dataframe by the given sales channel and apply the necessary transformations.
        Returns the transformed dataframe.
        """
        
        df_copy = df.copy()
        print(df_copy.columns)
        
        # Check if 'order_date' exists, if not, print error
        if 'order_date' not in df_copy.columns:
            print("Error: 'order_date' column is missing from the DataFrame!")
            return pd.DataFrame()  # Return empty DataFrame if 'order_date' is missing
        
        # Convert 'order_date' to datetime
        df_copy['order_date'] = pd.to_datetime(df_copy['order_date'])
        
        # Drop unnecessary columns
        df_copy = df_copy.drop(['Varient title', 'Amazon_spend_sku_wise','product_name_priority_sku'], axis=1, errors='ignore')

        # Filter the DataFrame based on the sales_channel
        filtered_df = df_copy[df_copy['sales_channel'] == sales_channel]

        filtered_df.fillna(0,inplace=True)

        
        # Apply transformations for each unique product
        transformed_dfs = []
        unique_product_names = filtered_df['product_name'].unique()
        
        for product_name in unique_product_names:
            if 'Unknown Product' in product_name:
                continue
            
            # Sanitize product name
            sanitized_product_name = str(product_name).replace(" ", "_").replace("/", "_").replace("\\", "_")
            product_df = filtered_df[filtered_df['product_name'] == product_name]

            # Drop unnecessary columns
            product_df = product_df.drop(columns=['Year', 'sales_channel', 'product_name_priority_sku'], errors='ignore')
            
            transformed_dfs.append(product_df)
        
        if transformed_dfs:
            return pd.concat(transformed_dfs, ignore_index=True)
        else:
            return pd.DataFrame()

def filter_to_current_and_next_months(df, num_months=5):
    """
    Filters the DataFrame to include the current month and the next `num_months` months.

    Parameters:
    df (pd.DataFrame): The DataFrame to filter.
    num_months (int): The number of months to include starting from the current month. Default is 5.

    Returns:
    pd.DataFrame: A filtered DataFrame including the 'SKU' column and the specified months.
    """
    # Get the current date
    today = datetime.datetime.today()

    # Format the current month as 'YYYY-MM-01'
    current_month = today.strftime('%Y-%m-01')

    # Identify columns that are not 'SKU' or 'product_name'
    numeric_columns = [col for col in df.columns if col not in ['SKU', 'product_name']]
    
    # Replace negative values with zero in these columns
    df[numeric_columns] = df[numeric_columns].clip(lower=0)

    # Create a list of all columns
    columns = df.columns

    # Find the index of the current month
    try:
        start_idx = columns.get_loc(current_month)
    except KeyError:
        raise ValueError(f"Column for the current month '{current_month}' not found in DataFrame.")

    # Ensure the index does not exceed the number of columns
    end_idx = min(start_idx + num_months, len(columns))

    # Select the columns for this month and the next `num_months` months
    selected_columns = columns[start_idx:end_idx]

    # Add 'SKU' to the list of columns to keep
    selected_columns = ['SKU'] + list(selected_columns)

    # Filter the DataFrame
    filtered_df = df[selected_columns]

    return filtered_df


def calculate_m0_m1_split(df, sales_channel):
    # Filter data for the given sales channel
    channel_df = df[df['sales_channel'] == sales_channel]

    # Create pivot table
    pivot = pd.pivot_table(channel_df, values='unit_sold_unbundled', 
                           index='SKU', columns='age', aggfunc='sum', fill_value=0)

    # Calculate M0 and M1+
    pivot['M0'] = pivot[0]
    pivot['M1+'] = pivot.drop(columns=[0, 'M0']).sum(axis=1)

    # Calculate percentages
    pivot['Total'] = pivot['M0'] + pivot['M1+']
    pivot['M0%'] = pivot['M0'] / pivot['Total']
    pivot['M1+%'] = pivot['M1+'] / pivot['Total']

    return pivot[['M0%', 'M1+%']]

# 7. Main Implementation

def load_and_process_data(marketing_data):

    # Load the data into a DataFrame
    file_path = 'New Subs Data.xlsx'  # Update this to the actual file path
    sheet_name = 'creatine combined_flag stockout'  # Update this to the actual sheet name

    df_subs = pd.read_excel(file_path, sheet_name=sheet_name)

    # Convert marketing_data to the format expected by the prediction code
    spend_updates = marketing_data.set_index('Channel').to_dict()

    # Initialize the spend data with the provided values
    spends_df_subs = pd.DataFrame(spend_updates)
    print('User provided inputs')
    print(spends_df_subs)

    # # Initialize the spend data with generic month labels
    # spend_updates = {
    #     'current month': [76904.46, 169527.61, 214805.7733, 355941.7067, 449181.4467, 7000],
    #     'month 1': [110000, 175000, 212215, 219228, 306728, 15000],
    #     'month 2': [120000, 225000, 233437, 254928, 364928, 20000],
    #     'month 3': [85000, 175000, 256781, 224228, 331728, 65000],
    #     'month 4': [0, 0, 0, 0, 0, 0]
    # }

    # Create DataFrame
    spends_df_subs = pd.DataFrame(spend_updates, index=['Spends_Meta', 'Spends_Google', 'Spends_Amazon', 'Spends_Audio_SponCon', 'Spends_Partnerships', 'Spends_Others']).T

    # Aggregate spend data in spends_df_subs
    spends_df_subs['platform_spends'] = spends_df_subs['Spends_Meta'] + spends_df_subs['Spends_Google'] + spends_df_subs['Spends_Others']
    spends_df_subs['promotional_spends'] = spends_df_subs['Spends_Audio_SponCon'] + spends_df_subs['Spends_Partnerships']
    spends_df_subs['Amazon_spends'] = spends_df_subs['Spends_Amazon']

    # Function to map generic month labels to actual date strings
    def map_month_labels_to_dates():
        current_date = datetime.datetime.now()  # Using datetime.datetime.now() explicitly
        month_mapping = {}
        for i, label in enumerate(['current month', 'month 1', 'month 2', 'month 3', 'month 4']):
            month_mapping[label] = (current_date + datetime.timedelta(days=30 * i)).strftime('%Y-%m')
        return month_mapping

    # Map the generic month labels to actual month strings
    month_mapping = map_month_labels_to_dates()

    # Ask the user if they want to update the spends data
    # user_response = input("Do you want to update the spends data with new values? (yes/no): ").strip().lower()
    # if user_response == 'yes':
        # Update df_subs with user-provided data
    for generic_label, actual_month in month_mapping.items():
        mask = (df_subs['age'] == 0) & (df_subs['acquisition_month_sku_level'] == actual_month)
        if any(mask):
            df_subs.loc[mask, 'platform_spends'] = spends_df_subs.at[generic_label, 'platform_spends']
            df_subs.loc[mask, 'promotional_spends'] = spends_df_subs.at[generic_label, 'promotional_spends']
            df_subs.loc[mask, 'Amazon_spends'] = spends_df_subs.at[generic_label, 'Amazon_spends']
        
    print("Spends data has been updated successfully.")
        # else:
        #     print("Using default spends data.")

    # Verify the update by printing a summary of the updated data
    print("\nVerification of Updated Spends Data:")
    verification_df = df_subs[df_subs['age'] == 0].groupby('acquisition_month_sku_level').agg({
        'platform_spends': 'first',
        'promotional_spends': 'first',
        'Amazon_spends': 'first'
    }).reset_index()

    # Sort the DataFrame by acquisition_month_sku_level
    verification_df = verification_df.sort_values('acquisition_month_sku_level')

    # Display the verification DataFrame
    print(verification_df.to_string(index=False))
    


    # Create CSVs for Amazon sales channel
    df_dotcomsubs = create_transformed_df(df_subs, '.com')

    dataframes_dotcomsubs = process_products_by_name(df_dotcomsubs)

    # Directory where your models are saved
    directory = 'dot_com/Best Outputs/best_models'

    # Initialize an empty DataFrame to store all predictions
    final_preds_df_dotcomsubs = pd.DataFrame()

    # Loop through each product in the input data dictionary and make predictions
    for product_key, product_data in dataframes_dotcomsubs.items():
        logging.info(f"Making predictions for product: {product_key}")

        # Make predictions for the current product
        predictions_df_dotcomsubs = predict_with_saved_models({product_key: product_data}, directory)

        # Append the predictions for the current product to the final DataFrame
        final_preds_df_dotcomsubs = pd.concat([final_preds_df_dotcomsubs, predictions_df_dotcomsubs], ignore_index=True)


    final_preds_df_dotcomsubs_agg = get_final_results(final_preds_df_dotcomsubs)
    final_preds_df_dotcomsubs_agg.head()

    skus_for_lr_comsubs = [
        'df_03_comsubs_850243008796',
        'df_20_comsubs_850030796011',
        'df_21_comsubs_850243008598',
        'df_23_comsubs_850243008901',
        'df_39_comsubs_850243008970',
        'df_41_comsubs_850030796271',
        'df_43_comsubs_850243008987',
        'df_44_comsubs_3xTongkat',
        'df_47_comsubs_850030796349'
    ]

    final_preds_df_dotcomsubs_agg = final_preds_df_dotcomsubs_agg[~final_preds_df_dotcomsubs_agg['SKU'].isin(skus_for_lr_comsubs)]

    # Directory where the models are saved
    model_save_directory_comsubs = 'dot_com/Best Outputs/best_models'

    # Define the current date and forecast period till the end of next year
    current_date_comsubs = datetime.datetime.now().replace(day=1)  # Current month (1st of the month)
    next_year_end_comsubs = datetime.datetime(current_date_comsubs.year + 1, 12, 31)  # End of next year

    # Calculate the total number of months from the current month to the end of next year
    total_forecast_months_comsubs = (next_year_end_comsubs.year - current_date_comsubs.year) * 12 + (next_year_end_comsubs.month - current_date_comsubs.month + 1)

    # Start dates for each SKU as strings (in 'YYYY-MM-DD' format)
    start_dates_comsubs = {
        'df_03_comsubs_850243008796': '2024-01-01',
        'df_20_comsubs_850030796011': '2024-01-01',
        'df_21_comsubs_850243008598': '2024-01-01',
        'df_23_comsubs_850243008901': '2024-01-01',
        'df_39_comsubs_850243008970': '2024-01-01',
        'df_41_comsubs_850030796271': '2024-01-01',
        'df_43_comsubs_850243008987': '2024-01-01',
        'df_44_comsubs_3xTongkat': '2024-01-01',
        'df_47_comsubs_850030796349': '2024-05-01'  # Ashwagandha only 3 months
    }



    # Initialize a DataFrame to store the results
    predictions_df_lr_comsubs = pd.DataFrame()

    # Loop through each product and make predictions using saved models
    for product_key, start_date_str in start_dates_comsubs.items():
        model_filename_comsubs = os.path.join(model_save_directory_comsubs, f"{product_key}_lm.pkl")
        
        # Check if the model exists for the current SKU
        if os.path.exists(model_filename_comsubs):
            logging.info(f"Loading model for product: {product_key}")
            try:
                # Load the saved model
                with open(model_filename_comsubs, 'rb') as model_file:
                    model = pickle.load(model_file)

                # Calculate the number of months between the start date and current date (full duration)
                total_months_comsubs = calculate_total_months(start_date_str, current_date_comsubs.strftime('%Y-%m-%d'))
                
                # Prepare past dates as integer values (1, 2, 3, ... total_months)
                past_dates_comsubs = np.arange(1, total_months_comsubs + 1).reshape(-1, 1)
                
                # Prepare future dates for the forecast period (until end of next year)
                future_dates_comsubs = np.arange(total_months_comsubs + 1, total_months_comsubs + total_forecast_months_comsubs + 1).reshape(-1, 1)

                # Make predictions for the full duration
                predicted_premiums_comsubs = model.predict(past_dates_comsubs)
                forecasted_premiums_comsubs = model.predict(future_dates_comsubs)

                # Create a combined list of all predictions (past + future)
                all_predictions_comsubs = list(predicted_premiums_comsubs) + list(forecasted_premiums_comsubs)
                
                # Create date labels from the start date for each month (past + future)
                all_dates_comsubs = pd.date_range(start=start_date_str, periods=len(all_predictions_comsubs), freq='MS').strftime('%Y-%m-%d')
                
                # Store the results in the DataFrame for the full duration
                result_row_comsubs = [product_key] + all_predictions_comsubs
                prediction_columns_comsubs = ['SKU'] + list(all_dates_comsubs)
                predictions_df_full_comsubs = pd.DataFrame([result_row_comsubs], columns=prediction_columns_comsubs)
                
                # Concatenate to the main DataFrame
                predictions_df_lr_comsubs = pd.concat([predictions_df_lr_comsubs, predictions_df_full_comsubs], ignore_index=True)

                logging.info(f"Predictions completed for {product_key}")

            except Exception as e:
                logging.error(f"Error making predictions for product {product_key}: {e}")
        else:
            logging.warning(f"No saved model found for product: {product_key}")

    # Reorder the date columns in chronological order
    date_columns_comsubs = sorted([col for col in predictions_df_lr_comsubs.columns if col != 'SKU'], key=lambda x: pd.to_datetime(x))
    predictions_df_lr_comsubs = predictions_df_lr_comsubs[['SKU'] + date_columns_comsubs]

    # Ensure the 'SKU' is set as the index for both DataFrames
    final_preds_df_dotcomsubs_agg.set_index('SKU', inplace=True)
    predictions_df_lr_comsubs.set_index('SKU', inplace=True)

    # Ensure both DataFrames have the same columns
    missing_columns_in_predictions_comsubs = [col for col in final_preds_df_dotcomsubs_agg.columns if col not in predictions_df_lr_comsubs.columns]
    missing_columns_in_best_final_comsubs = [col for col in predictions_df_lr_comsubs.columns if col not in final_preds_df_dotcomsubs_agg.columns]

    # Add missing columns to predictions_df_lr_comsubs with NaN values
    for col in missing_columns_in_predictions_comsubs:
        predictions_df_lr_comsubs[col] = np.nan

    # Add missing columns to final_preds_df_dotcomsubs_agg with NaN values
    for col in missing_columns_in_best_final_comsubs:
        final_preds_df_dotcomsubs_agg[col] = np.nan

    # Reorder columns to make sure they are in the same order
    final_preds_df_dotcomsubs_agg = final_preds_df_dotcomsubs_agg[sorted(final_preds_df_dotcomsubs_agg.columns)]
    predictions_df_lr_comsubs = predictions_df_lr_comsubs[sorted(predictions_df_lr_comsubs.columns)]

    # Now concatenate the two DataFrames
    final_preds_df_dotcomsubs_agg = pd.concat([final_preds_df_dotcomsubs_agg, predictions_df_lr_comsubs])

    # Reset the index if needed
    final_preds_df_dotcomsubs_agg.reset_index(inplace=True)
    final_preds_df_dotcomsubs_agg


    # ### 7.2 Amazonsubs Predictions

    # # Create CSVs for .com sales channel
    df_amasubs = create_transformed_df(df_subs, 'Amazon')

    dataframes_amasubs = process_products_by_name(df_amasubs)

    # Directory where your models are saved
    directory = 'amazon/Best Outputs/best_models'

    # Initialize an empty DataFrame to store all predictions
    final_preds_df_amasubs = pd.DataFrame()

    # Loop through each product in the input data dictionary and make predictions
    for product_key, product_data in dataframes_amasubs.items():
        logging.info(f"Making predictions for product: {product_key}")

        # Make predictions for the current product
        predictions_df_amasubs = predict_with_saved_models({product_key: product_data}, directory)

        # Append the predictions for the current product to the final DataFrame
        final_preds_df_amasubs = pd.concat([final_preds_df_amasubs, predictions_df_amasubs], ignore_index=True)

    final_preds_df_amasubs_agg = get_final_results(final_preds_df_amasubs)
    final_preds_df_amasubs_agg

    skus_for_lr_amasubs = [
        'df_01_amasubs_MOMENTOUS-HUB-MGLT',
        'df_02_amasubs_AMPMCR',
        'df_03_amasubs_ZH-21DU-JGJX',
        'df_15_amasubs_MOMENTOUS-HUB-TYR',
        'df_16_amasubs_8S-5QPR-VPEQ',
        'df_28_amasubs_MOMENTOUS-HUB-TONG',
        'df_30_amasubs_PR300BOTTLE-A-stickerless',
        'df_34_amasubs_MOMENTOUS-HUB-ACAR',
        'df_29_amasubs_MOMENTOUS-HUB-FADO'
    ]

    final_preds_df_amasubs_agg = final_preds_df_amasubs_agg[~final_preds_df_amasubs_agg['SKU'].isin(skus_for_lr_amasubs)]


    # Directory where the models are saved
    model_save_directory_amasubs = 'amazon/Best Outputs/best_models'

    # Define the current date and forecast period until the end of next year
    current_date_amasubs = datetime.datetime.now().replace(day=1)  # Current month (1st of the month)
    end_of_next_year_amasubs = (current_date_amasubs + relativedelta(years=1, month=12, day=31)).replace(day=1)
    forecast_months_amasubs = calculate_total_months(current_date_amasubs.strftime('%Y-%m-%d'), end_of_next_year_amasubs.strftime('%Y-%m-%d'))  # Forecast until the end of next year

    # Initialize a DataFrame to store the results
    predictions_df_lr_amasubs = pd.DataFrame()

    # Start dates for each SKU as strings (in 'YYYY-MM-DD' format)
    start_dates_amasubs = {
        'df_01_amasubs_MOMENTOUS-HUB-MGLT': '2024-04-01',
        'df_02_amasubs_AMPMCR': '2024-04-01',
        'df_03_amasubs_ZH-21DU-JGJX': '2024-01-01',
        'df_15_amasubs_MOMENTOUS-HUB-TYR': '2024-01-01',
        'df_16_amasubs_8S-5QPR-VPEQ': '2024-01-01',
        'df_28_amasubs_MOMENTOUS-HUB-TONG': '2024-01-01',
        'df_30_amasubs_PR300BOTTLE-A-stickerless': '2024-01-01',
        'df_34_amasubs_MOMENTOUS-HUB-ACAR': '2024-01-01',
    }


    # Loop through each product and make predictions using saved models
    for product_key, start_date_str in start_dates_amasubs.items():
        model_filename_amasubs = os.path.join(model_save_directory_amasubs, f"{product_key}_lm.pkl")
        
        # Check if the model exists for the current SKU
        if os.path.exists(model_filename_amasubs):
            logging.info(f"Loading model for product: {product_key}")
            try:
                # Load the saved model
                with open(model_filename_amasubs, 'rb') as model_file:
                    model = pickle.load(model_file)

                # Calculate the number of months between the start date and current date (full duration)
                total_months_amasubs = calculate_total_months(start_date_str, current_date_amasubs.strftime('%Y-%m-%d'))
                
                # Prepare past dates as integer values (1, 2, 3, ... total_months)
                past_dates_amasubs = np.arange(1, total_months_amasubs + 1).reshape(-1, 1)
                
                # Prepare future dates for the forecast period (until end of next year)
                future_dates_amasubs = np.arange(total_months_amasubs + 1, total_months_amasubs + forecast_months_amasubs + 2).reshape(-1, 1)

                # Make predictions for the full duration
                predicted_premiums_amasubs = model.predict(past_dates_amasubs)
                forecasted_premiums_amasubs = model.predict(future_dates_amasubs)

                # Create a combined list of all predictions (past + future)
                all_predictions_amasubs = list(predicted_premiums_amasubs) + list(forecasted_premiums_amasubs)
                
                # Create date labels from the start date for each month (past + future)
                all_dates_amasubs = pd.date_range(start=start_date_str, periods=len(all_predictions_amasubs), freq='MS').strftime('%Y-%m-%d')
                
                # Store the results in the DataFrame for the full duration
                result_row_amasubs = [product_key] + all_predictions_amasubs
                prediction_columns_amasubs = ['SKU'] + list(all_dates_amasubs)
                predictions_df_full_amasubs = pd.DataFrame([result_row_amasubs], columns=prediction_columns_amasubs)
                
                # Concatenate to the main DataFrame
                predictions_df_lr_amasubs = pd.concat([predictions_df_lr_amasubs, predictions_df_full_amasubs], ignore_index=True)

                logging.info(f"Predictions completed for {product_key}")

            except Exception as e:
                logging.error(f"Error making predictions for product {product_key}: {e}")
        else:
            logging.warning(f"No saved model found for product: {product_key}")

    # Reorder the date columns in chronological order
    date_columns_amasubs = sorted([col for col in predictions_df_lr_amasubs.columns if col != 'SKU'], key=lambda x: pd.to_datetime(x))
    predictions_df_lr_amasubs = predictions_df_lr_amasubs[['SKU'] + date_columns_amasubs]

    # Ensure the 'SKU' is set as the index for both DataFrames
    final_preds_df_amasubs_agg.set_index('SKU', inplace=True)
    predictions_df_lr_amasubs.set_index('SKU', inplace=True)

    # Ensure both DataFrames have the same columns
    missing_columns_in_predictions_amasubs = [col for col in final_preds_df_amasubs_agg.columns if col not in predictions_df_lr_amasubs.columns]
    missing_columns_in_best_final_amasubs = [col for col in predictions_df_lr_amasubs.columns if col not in final_preds_df_amasubs_agg.columns]

    # Add missing columns to predictions_df_lr_amasubs with NaN values
    for col in missing_columns_in_predictions_amasubs:
        predictions_df_lr_amasubs[col] = np.nan

    # Add missing columns to final_preds_df_amasubs_agg with NaN values
    for col in missing_columns_in_best_final_amasubs:
        final_preds_df_amasubs_agg[col] = np.nan

    # Reorder columns to make sure they are in the same order
    final_preds_df_amasubs_agg = final_preds_df_amasubs_agg[sorted(final_preds_df_amasubs_agg.columns)]
    predictions_df_lr_amasubs = predictions_df_lr_amasubs[sorted(predictions_df_lr_amasubs.columns)]

    # Now concatenate the two DataFrames
    final_preds_df_amasubs_agg = pd.concat([final_preds_df_amasubs_agg, predictions_df_lr_amasubs])

    # Reset the index if needed
    final_preds_df_amasubs_agg.reset_index(inplace=True)


    # ### 7.3 amaotp Predictions
    
    # Load the data into a DataFrame
    file_path = 'New OTP Data.xlsx' 
    df_otp = pd.read_excel(file_path)


    # Function to determine the number of days in each respective month
    def get_days_in_months():
        current_date = datetime.datetime.now()
        days_in_months = []
        for i in range(5):  # Current month and the next four months
            month_start = current_date.replace(day=1) + datetime.timedelta(days=32 * i)
            next_month_start = month_start.replace(day=28) + datetime.timedelta(days=4)
            last_day_of_month = next_month_start - datetime.timedelta(days=next_month_start.day)
            days_in_months.append(last_day_of_month.day)
        return days_in_months

    # Getting the number of days in each respective month
    days_in_months = get_days_in_months()

    # Create a new dictionary for spends_updates_otp
    spends_updates_otp = {}

    # Create a new dictionary for spends_updates_otp
    spends_updates_otp = {}

    # Iterate over each month and each category, dividing by the number of days in the month
    for i, (month_label, spends) in enumerate(spend_updates.items()):
        spends_updates_otp[month_label] = [float(spend) / days_in_months[i] if days_in_months[i] > 0 else 0.0 for spend in spends.values()]

    spends_df_otp = pd.DataFrame(spends_updates_otp, index=['Spends_Meta', 'Spends_Google', 'Spends_Amazon', 'Spends_Audio_SponCon', 'Spends_partnerships', 'Spends_Others']).T

        # Function to map generic month labels to actual date strings
    def map_month_labels_to_dates():
        current_date = datetime.datetime.now()  # Using datetime.datetime.now() explicitly
        month_mapping = {}
        for i, label in enumerate(['current month', 'month 1', 'month 2', 'month 3', 'month 4']):
            month_mapping[label] = (current_date + datetime.timedelta(days=30 * i)).strftime('%Y-%m')
        return month_mapping

    # Map the generic month labels to actual month strings
    month_mapping = map_month_labels_to_dates()

    # Make sure the 'order_date' is in datetime format
    df_otp['order_date'] = pd.to_datetime(df_otp['order_date'])

    # Create a new column 'order-month' formatted as 'YYYY-MM'
    df_otp['order_month'] = df_otp['order_date'].dt.strftime('%Y-%m')

    # if user_response == 'yes':
        # Update df_otp with user-provided data
    for generic_label, actual_month in month_mapping.items():
        mask = (df_otp['order_month'] == actual_month)
        if any(mask):
            df_otp.loc[mask, 'Spends_Meta'] = spends_df_otp.at[generic_label, 'Spends_Meta']
            df_otp.loc[mask, 'Spends_Google'] = spends_df_otp.at[generic_label, 'Spends_Google']
            df_otp.loc[mask, 'Spends_Amazon'] = spends_df_otp.at[generic_label, 'Spends_Amazon']
            df_otp.loc[mask, 'Spends_Audio_SponCon'] = spends_df_otp.at[generic_label, 'Spends_Audio_SponCon']
            df_otp.loc[mask, 'Spends_partnerships'] = spends_df_otp.at[generic_label, 'Spends_partnerships']
            df_otp.loc[mask, 'Spends_Others'] = spends_df_otp.at[generic_label, 'Spends_Others']
    print("Spends data has been updated successfully.")
    # else:
    #     print("Using default spends data.")

    df_otp.drop(columns=['order_month'], inplace=True)
    


    df_amaotp = create_transformed_df_otp(df_otp, 'Amazon')

    df_amaotp['order_date'] = pd.to_datetime(df_amaotp['order_date'])
    df_amaotp = df_amaotp[df_amaotp['order_date'] >= pd.Timestamp(2023,1,1)]
    df_amaotp['day'] = df_amaotp['order_date'].dt.day
    df_amaotp['is_weekend'] = df_amaotp['order_date'].dt.dayofweek >= 5
    df_amaotp['month'] = df_amaotp['order_date'].dt.month
    df_amaotp['year'] = df_amaotp['order_date'].dt.year
    df_amaotp['week'] = df_amaotp['order_date'].dt.isocalendar().week

    df_amaotp.fillna(0,inplace=True)

    # Define the path to the best models and the forecast data
    best_models_path = "best models amazon OTP"
    forecast_data = df_amaotp[df_amaotp.order_date >= pd.to_datetime("2024-09-01")]

    # Initialize an empty DataFrame to store results
    results_df = pd.DataFrame()

    # Iterate through each model in the specified directory
    for model_file in os.listdir(best_models_path):
        model_path = os.path.join(best_models_path, model_file)
        
        # Extract SKU and product name from the model file name
        model_parts = model_file.split("_")
        sku_id = int(model_parts[2]) if model_parts[2].isdigit() else model_parts[2]
        product_name = model_parts[3].split('.')[0]

        # Skip models for specific products
        if product_name in ["Fadogia Agrestis", "Apigenin"]:
            continue

        # Filter the forecast data for the current SKU
        sku_data = forecast_data[forecast_data.Sku == sku_id].copy()
        sku_data.drop_duplicates(subset=['order_date'], inplace=True)

        try:
            # Load the model
            model = joblib.load(model_path)
            
            # Determine feature columns based on model type
            if 'LGBM' in str(type(model)):
                feature_columns = model.booster_.feature_name()
            elif 'RandomForest' in str(type(model)) or 'XGB' in str(type(model)):
                feature_columns = model.feature_names_in_
            else:
                logging.warning(f"Unknown model type for SKU: {sku_id} and product: {product_name}")
                continue

            # Prepare the data for prediction
            prediction_data = sku_data[feature_columns]

            # Create a temporary DataFrame for storing predictions
            temp_df = pd.DataFrame()
            temp_df['order_date'] = sku_data['order_date']
            temp_df['units_sold'] = model.predict(prediction_data)
            temp_df.set_index('order_date', inplace=True)

            # Resample the data to end of month frequency and reset index
            temp_df = temp_df.resample('M').sum().reset_index()
            temp_df['Sku'] = sku_id
            temp_df['product_name'] = product_name

            # Append the results to the final DataFrame
            results_df = pd.concat([results_df, temp_df], ignore_index=True)

        except Exception as e:
            logging.error(f"Error processing model {model_path}: {e}")

        

    # Pivot the DataFrame to wide format
    result_df1 = pd.pivot_table(data=results_df, index='Sku', columns='order_date', values='units_sold')

    # Flatten the multi-level columns
    result_df1.columns = [f'{col}' for col in result_df1.columns]

    # Reset index to convert the index to a column
    result_df1 = result_df1.reset_index()

    result_df1.columns = [pd.to_datetime(col).strftime('%Y-%m-01') if col != 'Sku' else col for col in result_df1.columns]

    # Show the flattened DataFrame
    final_preds_df_amaotp_agg = result_df1.copy()
    final_preds_df_amaotp_agg

    # ### 7.4 comotp predictions

    df_dotcomotp = create_transformed_df_otp(df_otp, '.com')

    df_dotcomotp['order_date'] = pd.to_datetime(df_dotcomotp['order_date'])
    df_dotcomotp = df_dotcomotp[df_dotcomotp['order_date'] >= pd.Timestamp(2023,1,1)]
    df_dotcomotp['day'] = df_dotcomotp['order_date'].dt.day
    df_dotcomotp['is_weekend'] = df_dotcomotp['order_date'].dt.dayofweek >= 5
    df_dotcomotp['month'] = df_dotcomotp['order_date'].dt.month
    df_dotcomotp['year'] = df_dotcomotp['order_date'].dt.year
    df_dotcomotp['week'] = df_dotcomotp['order_date'].dt.isocalendar().week

    df_dotcomotp.fillna(0,inplace=True)


    
    # Fi# Filter forecast data starting from September 1, 2024
    forecast_data = df_dotcomotp[df_dotcomotp.order_date >= pd.to_datetime("2024-09-01")]

    # Path to the directory containing the best models
    best_models_path = "best models com OTP"

    # Initialize an empty DataFrame to store results
    results_df = pd.DataFrame()

    # Iterate through each model file in the directory
    for model_file in os.listdir(best_models_path):
        model_path = os.path.join(best_models_path, model_file)
        
        # Extract SKU ID and product name from the model file name
        model_parts = model_file.split("_")
        sku_id = model_parts[2]
        if sku_id.isdigit():
            sku_id = int(sku_id)
        product_name = model_parts[3].split('.')[0]

        # Skip models for specific product names
        if product_name in ["Fadogia Agrestis", "Apigenin"]:
            continue

        # Filter the forecast data for the current SKU
        sku_data = forecast_data[forecast_data.Sku == sku_id].copy()
        sku_data.drop_duplicates(subset=['order_date'], inplace=True)

        try:
            # Load the model
            model = joblib.load(model_path)
            
            # Prepare the DataFrame based on the model type
            if 'LGBM' in str(type(model)):
                # LightGBM specific handling
                logging.info(f"Processing LightGBM model for SKU: {sku_id} and product: {product_name}")
                feature_columns = model.booster_.feature_name()
            elif 'RandomForest' in str(type(model)) or 'XGB' in str(type(model)):
                # RandomForest or XGB specific handling
                logging.info(f"Processing RandomForest/XGB model for SKU: {sku_id} and product: {product_name}")
                feature_columns = model.feature_names_in_
            else:
                logging.warning(f"Unknown model type for SKU: {sku_id} and product: {product_name}")
                continue

            # Extract relevant columns from the DataFrame for prediction
            prediction_data = sku_data[feature_columns]

            # Create a DataFrame for storing predictions
            temp_df = pd.DataFrame()
            temp_df['order_date'] = sku_data['order_date']
            temp_df[['Spends_Meta', 'Spends_Google', 'Spends_Amazon', 'Spends_Audio_SponCon', 
                    'Spends_partnerships', 'Spends_Others']] = sku_data[['Spends_Meta', 'Spends_Google', 
                                                                        'Spends_Amazon', 'Spends_Audio_SponCon',
                                                                        'Spends_partnerships', 'Spends_Others']]

            # Make predictions
            result = model.predict(prediction_data)

            # Store results in temp DataFrame
            temp_df['units_sold'] = result
            temp_df.set_index('order_date', inplace=True)

            # Resample the data to end-of-month frequency and reset index
            temp_df = temp_df.resample('M').sum().reset_index()
            temp_df['Sku'] = sku_id
            temp_df['product_name'] = product_name

            # Append the results to the final DataFrame
            results_df = pd.concat([results_df, temp_df], ignore_index=True)

        except AttributeError as e:
            logging.error(f"AttributeError with model {model_path}: {e}")
        except Exception as e:
            logging.error(f"Error processing model {model_path}: {e}")

    # Multiplier adjustments based on specific SKU and dates
    multiplier = {
        850243008918: [{pd.to_datetime("2024-11-01"): 1.56}],
        850030796288: [{pd.to_datetime("2024-11-01"): 1.78}, {pd.to_datetime("2024-06-01"): 1.242}],
        850243008925: [{pd.to_datetime("2024-11-01"): 1.66}],
        850243008956: [{pd.to_datetime("2024-11-01"): 1.96}]
    }

    # Apply multipliers
    for sku_id, date_multiplier_list in multiplier.items():
        for date_multiplier in date_multiplier_list:
            for order_date, multiplier_value in date_multiplier.items():
                condition = (results_df['Sku'] == sku_id) & (results_df['order_date'] == order_date)
                results_df.loc[condition, 'units_sold'] = results_df.loc[condition, 'units_sold'] * multiplier_value
                logging.info(f"Applied multiplier {multiplier_value} for SKU: {sku_id} on {order_date}")

    # Pivot the DataFrame to wide format
    result_df1 = pd.pivot_table(data=results_df, index='Sku', columns='order_date', values='units_sold')

    # Flatten the multi-level columns
    result_df1.columns = [f'{col}' for col in result_df1.columns]

    # Reset index to convert the index to a column
    result_df1 = result_df1.reset_index()

    result_df1.columns = [pd.to_datetime(col).strftime('%Y-%m-01') if col != 'Sku' else col for col in result_df1.columns]

    # Show the flattened DataFrame
    final_preds_df_comotp_agg = result_df1.copy()
    
    # # 8. collating Results
    final_preds_df_dotcomsubs_agg['SKU'] = final_preds_df_dotcomsubs_agg['SKU'].apply(lambda x: x.split('_')[-1])

    final_preds_df_comotp_agg.rename(columns={'Sku': 'SKU'}, inplace=True)

    final_preds_df_amasubs_agg['SKU'] = final_preds_df_amasubs_agg['SKU'].apply(lambda x: x.split('_')[-1])

    final_preds_df_amaotp_agg.rename(columns={'Sku': 'SKU'}, inplace=True)

    final_preds_df_dotcomsubs_agg = filter_to_current_and_next_months(final_preds_df_dotcomsubs_agg)
    final_preds_df_comotp_agg = filter_to_current_and_next_months(final_preds_df_comotp_agg)
    final_preds_df_amasubs_agg = filter_to_current_and_next_months(final_preds_df_amasubs_agg)
    final_preds_df_amaotp_agg = filter_to_current_and_next_months(final_preds_df_amaotp_agg)

    # Ensure SKU column is cleaned (remove any leading/trailing spaces, etc.)
    final_preds_df_dotcomsubs_agg['SKU'] = final_preds_df_dotcomsubs_agg['SKU'].str.strip()
    final_preds_df_comotp_agg['SKU'] = final_preds_df_comotp_agg['SKU'].astype(str)

    # Append the two DataFrames vertically
    combined_df_dotcom = pd.concat([final_preds_df_dotcomsubs_agg, final_preds_df_comotp_agg])

    # Ensure the non-SKU columns are numeric for summing (assuming they are date columns or numeric values)
    # The 'SKU' column will be used as a group key, so the remaining columns must be numeric
    numeric_columns = combined_df_dotcom.columns.drop('SKU')
    combined_df_dotcom[numeric_columns] = combined_df_dotcom[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Group by SKU and sum the numeric columns
    final_summed_df_dotcom = combined_df_dotcom.groupby('SKU', as_index=False).sum()

    # Display the result
    final_preds_df_com_agg = final_summed_df_dotcom.copy()

    skus_to_exclude = ['850030796080', '850030796349', 'PR300BOTTLE-B5']
    final_preds_df_com_agg = final_preds_df_com_agg[~final_preds_df_com_agg['SKU'].isin(skus_to_exclude)]
    final_preds_df_com_agg

    # Ensure SKU column is cleaned (remove any leading/trailing spaces, etc.)
    final_preds_df_amasubs_agg['SKU'] = final_preds_df_amasubs_agg['SKU'].str.strip()
    final_preds_df_amaotp_agg['SKU'] = final_preds_df_amaotp_agg['SKU'].astype(str)

    # Append the two DataFrames
    # combined_df_ama_agg = pd.concat([final_preds_df_amasubs_agg, final_preds_df_amaotp_agg])

    # # Ensure date columns are numeric (convert to numeric if needed)
    # date_columns = combined_df_ama_agg.columns.drop('SKU')  # Assuming all other columns are dates

    # combined_df_ama_agg[date_columns] = combined_df_ama_agg[date_columns].apply(pd.to_numeric, errors='coerce')

    # # Group by SKU and sum all the numeric columns (date columns)
    # final_summed_df = combined_df_ama_agg.groupby('SKU').sum().reset_index()


    # Append the two DataFrames vertically
    combined_df_ama = pd.concat([final_preds_df_amasubs_agg, final_preds_df_amaotp_agg])

    # Ensure the non-SKU columns are numeric for summing (assuming they are date columns or numeric values)
    # The 'SKU' column will be used as a group key, so the remaining columns must be numeric
    numeric_columns = combined_df_ama.columns.drop('SKU')
    combined_df_ama[numeric_columns] = combined_df_ama[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Group by SKU and sum the numeric columns
    final_summed_df_ama = combined_df_ama.groupby('SKU', as_index=False).sum()

    # Display the result
    final_preds_df_ama_agg = final_summed_df_ama.copy()

    final_preds_df_ama_agg = final_preds_df_ama_agg[final_preds_df_ama_agg['SKU'] != 'MOMENTOUS-HUB-FADO']
    
    final_preds_df_com_agg = filter_to_current_and_next_months(final_preds_df_com_agg)
    final_preds_df_ama_agg = filter_to_current_and_next_months(final_preds_df_ama_agg)
    
    # creating new, returning and subscription split
        

    com_splits = calculate_m0_m1_split(df_subs, '.com')
    ama_splits = calculate_m0_m1_split(df_subs, 'Amazon')

    com_splits = com_splits.reset_index()
    ama_splits = ama_splits.reset_index()

    com_splits['SKU'] = com_splits['SKU'].astype(str)

    # Merge the com_splits to final_preds_df_dotcomsubs_agg to align M0% and M1+% with SKUs
    merged_df = pd.merge(final_preds_df_dotcomsubs_agg, com_splits, on='SKU')

    # Multiply forecasts by M0% to get final_preds_df_dotcomsubs_firstsubsplit
    final_preds_df_dotcomsubs_firstsubsplit = merged_df.copy()
    for col in final_preds_df_dotcomsubs_agg.columns[1:]:
        final_preds_df_dotcomsubs_firstsubsplit[col] = merged_df[col] * merged_df['M0%']

    # Multiply forecasts by M1+% to get final_preds_df_dotcomsubs_subsplit
    final_preds_df_dotcomsubs_subsplit = merged_df.copy()
    for col in final_preds_df_dotcomsubs_agg.columns[1:]:
        final_preds_df_dotcomsubs_subsplit[col] = merged_df[col] * merged_df['M1+%']

    # Drop M0% and M1+% columns if no longer needed
    final_preds_df_dotcomsubs_firstsubsplit = final_preds_df_dotcomsubs_firstsubsplit.drop(columns=['M0%', 'M1+%'])
    final_preds_df_dotcomsubs_subsplit = final_preds_df_dotcomsubs_subsplit.drop(columns=['M0%', 'M1+%'])

    final_preds_df_comotp_agg.set_index('SKU', inplace=True)
    final_preds_df_dotcomsubs_firstsubsplit.set_index('SKU', inplace=True)

    # Perform the sum while preserving the SKU index
    final_preds_df_dotcom_new_ret = final_preds_df_comotp_agg.add(final_preds_df_dotcomsubs_firstsubsplit, fill_value=0)

    # Reset the index if you want 'SKU' to be a regular column again
    final_preds_df_dotcom_new_ret.reset_index(inplace=True)

    # Define your variables
    var_comnewsplit = 0.459
    var_comretsplit = 0.541

    # Create a copy to avoid modifying the original DataFrame
    final_preds_df_dotcom_new = final_preds_df_dotcom_new_ret.copy()
    final_preds_df_dotcom_ret = final_preds_df_dotcom_new_ret.copy()

    # Identify numeric columns
    numeric_cols = final_preds_df_dotcom_new_ret.select_dtypes(include='number').columns

    # Multiply numeric columns by the respective variables
    final_preds_df_dotcom_new[numeric_cols] = final_preds_df_dotcom_new_ret[numeric_cols] * var_comnewsplit
    final_preds_df_dotcom_ret[numeric_cols] = final_preds_df_dotcom_new_ret[numeric_cols] * var_comretsplit

    ama_splits['SKU'] = ama_splits['SKU'].astype(str)

    # Merge the ama_splits to final_preds_df_amasubs_agg to align M0% and M1+% with SKUs
    merged_df = pd.merge(final_preds_df_amasubs_agg, ama_splits, on='SKU')

    # Multiply forecasts by M0% to get final_preds_df_amasubs_firstsubsplit
    final_preds_df_amasubs_firstsubsplit = merged_df.copy()
    for col in final_preds_df_amasubs_agg.columns[1:]:
        final_preds_df_amasubs_firstsubsplit[col] = merged_df[col] * merged_df['M0%']

    # Multiply forecasts by M1+% to get final_preds_df_dotcomsubs_subsplit
    final_preds_df_amasubs_subsplit = merged_df.copy()
    for col in final_preds_df_amasubs_agg.columns[1:]:
        final_preds_df_amasubs_subsplit[col] = merged_df[col] * merged_df['M1+%']

    # Drop M0% and M1+% columns if no longer needed
    final_preds_df_amasubs_firstsubsplit = final_preds_df_amasubs_firstsubsplit.drop(columns=['M0%', 'M1+%'])
    final_preds_df_amasubs_subsplit = final_preds_df_amasubs_subsplit.drop(columns=['M0%', 'M1+%'])

    final_preds_df_amaotp_agg.set_index('SKU', inplace=True)
    final_preds_df_amasubs_firstsubsplit.set_index('SKU', inplace=True)

    # Perform the sum while preserving the SKU index
    final_preds_df_ama_new_ret = final_preds_df_amaotp_agg.add(final_preds_df_amasubs_firstsubsplit, fill_value=0)

    # Reset the index if you want 'SKU' to be a regular column again
    final_preds_df_ama_new_ret.reset_index(inplace=True)

    # Define your variables
    var_amanewsplit = 0.459
    var_amaretsplit = 0.541

    # Create a copy to avoid modifying the original DataFrame
    final_preds_df_ama_new = final_preds_df_ama_new_ret.copy()
    final_preds_df_ama_ret = final_preds_df_ama_new_ret.copy()

    # Identify numeric columns
    numeric_cols = final_preds_df_ama_new_ret.select_dtypes(include='number').columns

    # Multiply numeric columns by the respective variables
    final_preds_df_ama_new[numeric_cols] = final_preds_df_ama_new_ret[numeric_cols] * var_amanewsplit
    final_preds_df_ama_ret[numeric_cols] = final_preds_df_ama_new_ret[numeric_cols] * var_amaretsplit



    # Create a DataFrame with SKU ID and Product Name - Final
    comskulist = pd.DataFrame({
    'SKU': [
        '850030796042', '850243008918', '850243008796', '850030796097', '850030796035', 
        '850030796172', '850030796226', '850030796257', '850243008949', '850243008956', 
        '850030796165', '850030796028', '850030796066', '850243008932', '850030796134', 
        '850030796288', '850243008888', '850243008345', '850030796011', '850243008598', 
        '850030796158', '850243008901', '850014080730', '850243008789', '850243008406', 
        'COLL10Bundle', '850030796240', '850030796196', '850030796202', '850243008109', 
        '850030796233', 'CHOCRECOVERY14', '850243008208', '850243008192', 'VANRECOVERY14', 
        '850030796073', '850243008970', '850030796189', '850030796271', '850030796059', 
        '850243008987', '3xTongkat', '850243008925', 'PR300BOTTLE-BCOS', 'PRSTARTER5', 
        'PRPACKET5-DCOS', 'PRLEVELUPWBB', 'COLLPOWSHOTBNDL', 'AMPMAG', 'PRCollagen', 
        'DM2MIX12', 'PR300BOTTLE-B'
    ],
    'product_name': [
        'Magnesium L-Threonate', 'Omega-3', 'Creatine', 'Apigenin', 'L-Theanine', 
        'Zinc', 'Collagen Peptides', 'Inositol', 'Whey Protein Isolate Chocolate - NSF Sport', 
        'Multivitamin', 'Sleep', 'Alpha GPC', 'Tyrosine', 'Whey Protein Isolate Vanilla -  NSF Sport', 
        'Whey Protein Isolate Unflavored', 'Rhodiola Rosea', 'Essential Plant-Based Protein Chocolate', 
        'Brain Drive 60 Capsules', 'L-Glutamine', 'Recovery Chocolate', 'Resveratrol', 
        'Essential Plant-Based Protein Vanilla Chai', 'Collagen Shot', 'Recovery Vanilla', 
        'Whey Protein Isolate Piedmont Chocolate 24 packet bundle', 'Collagen Peptides', 
        'Fuel Cherry Berry / 12 Single Serving Packets', 'Fuel Strawberry Lime', 'Fuel Cherry Berry', 
        'Whey Protein Isolate Veracruz Vanilla 24 packet bundle', 'Fuel Strawberry Lime / 12 Single Serving Packets', 
        'Recovery Chocolate 14 pack bundle', 'Essential Plant-Based Protein 14 S - Piedmont Chocolate', 
        'Essential Plant-Based Protein 14 S - Vanilla Chai', 'Recovery Vanilla 14 pack bundle', 
        'Tongkat Ali', 'Vitamin D 60 serving', 'Magnesium Malate', 'Vital Aminos', 'Acetyl L-Carnitine', 
        'Turmeric 30 serving', 'Tongkat Ali 3 Bottle Value Pack', 'Elite Sleep', 'PR Lotion', 
        'PR Lotion', 'PR Lotion', 'PR Lotion', 'Other', 'Magnesium Malate 120 Capsules, 60 Day Supply', 
        'PR Lotion', 'Fuel 28g of Carbs per Serving, Box of 12 Servings (Mixed Pack:Cherry Berry and Strawberry Lime)', 
        'PR Lotion'
    ]
    })


    # Merge product names into the final_preds_df_com_agg
    # final_preds_df_com_agg = pd.merge(
    #     final_preds_df_com_agg, 
    #     comskulist[['SKU', 'product_name']],  # Only merging SKU and product_name from comskulist
    #     how='left', 
    #     on='SKU'  # Merge on SKU without creating duplicates
    # )

    # columns_order = ['SKU', 'product_name'] + [col for col in final_preds_df_com_agg.columns if col not in ['SKU', 'product_name']]
    # final_preds_df_com_agg = final_preds_df_com_agg[columns_order]

    amaskulist = pd.DataFrame({
    'SKU': [
        'MOMENTOUS-HUB-MGLT', 'AMPMCR', 'ZH-21DU-JGJX', 'AMPMCP', 'MEWPI24CHOC', 
        'MEWPI24VAN', 'MOMENTOUS-HUB-LTHE', 'MHZNPIC', 'MHMINO', 'MHRRE', 
        '850243008956', 'MOMENTOUS-HUB-AGPC', 'I6-8ZCT-FF3T', '1G-SLUS-FC7I', 
        'MOMENTOUS-HUB-TYR', '8S-5QPR-VPEQ', 'MHSLEEP', 'SH-4MF1-GY9W', 
        'MOMENTOUS-HUB-APG', 'MOMENTOUS-HUB-LGLX', '3D-FIWI-VDOX', 'OF-GD7N-LW9Q', 
        'MSRV', 'MDMF15ST', 'MDMF15BE', 'MDMF12ST', 'MDMF12BE', 'MOMENTOUS-HUB-TONG', 
        'MOMENTOUS-HUB-FADO', 'PR300BOTTLE-A-stickerless', '850243008970', 
        'MAMINOS', 'G0-NF7M-4HDP', 'MOMENTOUS-HUB-ACAR', '850243008987', 
        'PRPACKET5-CA', 'MOMENTOUS-APG', 'T4-YDUJ-ZG1F', 'MOMENTOUS-ASHWA'
    ],
    'product_name': [
        'Magnesium L-Threonate 30 Servings', 'Creatine Creatine (5 Grams Per Serving)', 'Omega-3', 
        'Collagen Peptides 30 Servings', 'Whey Protein Isolate 24 Servings (Chocolate)', 
        'Whey Protein Isolate 24 Servings (Vanilla)', 'L-Theanine 60 Servings', 'Zinc 60 Servings', 
        'Inositol 60 Servings', 'Rhodiola Rosea 60 Servings', 'Multivitamin', 'Alpha GPC 60 Servings', 
        'Recovery 15 Servings (Chocolate)', 'Essential Plant-Based Protein', 'Tyrosine 60 Servings', 
        'Essential Plant-Based Protein', 'Sleep 30 Servings', 'Brain Drive 30 Servings/60 Capsules', 
        'Apigenin', 'L-Glutamine 60 Servings', 'Recovery 15 Servings (Vanilla)', 'Collagen Shot 15 Servings', 
        'Resveratrol 30 Servings', 'Fuel 15 Serving Bag, Strawberry Lime', 'Fuel 15 Serving Bag, Cherry Berry', 
        'Fuel 12 Single Serving Packets, Strawberry Lime', 'Fuel 12 Single Serving Packets, Cherry Berry', 
        'Tongkat Ali', 'Fadogia Agrestis 60 Servings', 'PR Lotion Bottle (300g)', 'Vitamin D 60 serving', 
        'Vital Aminos BCAA & EAA, Tropical Punch, 30 Servings', 'Elite Sleep 30 Servings', 
        'Acetyl L-Carnitine 60 Servings', 'Turmeric 30 serving', 'PR Lotion', 'Apigenin', 
        'Whey Protein Isolate 24 Servings Per Pouch (Vanilla)', 'Ashwagandha'
    ]
    })

    # Function to add product names to a DataFrame
    def add_product_names(df, skulist):
        return pd.merge(df, skulist[['SKU', 'product_name']], how='left', on='SKU')


    # Merge product names into the final_preds_df_ama_agg
    # final_preds_df_ama_agg = pd.merge(
    #     final_preds_df_ama_agg, 
    #     amaskulist[['SKU', 'product_name']],  # Only merging SKU and product_name from comskulist
    #     how='left', 
    #     on='SKU'  # Merge on SKU without creating duplicates
    # )

    # columns_order = ['SKU', 'product_name'] + [col for col in final_preds_df_ama_agg.columns if col not in ['SKU', 'product_name']]
    # final_preds_df_ama_agg = final_preds_df_ama_agg[columns_order]

    # Now final_preds_df_ama_agg has the product names included
    # final_preds_df_ama_agg

    final_preds_df_dotcom_new = final_preds_df_dotcom_new[~final_preds_df_dotcom_new['SKU'].isin(skus_to_exclude)]
    final_preds_df_dotcom_ret = final_preds_df_dotcom_ret[~final_preds_df_dotcom_ret['SKU'].isin(skus_to_exclude)]
    final_preds_df_dotcomsubs_subsplit = final_preds_df_dotcomsubs_subsplit[~final_preds_df_dotcomsubs_subsplit['SKU'].isin(skus_to_exclude)]

    # Add product names to all DataFrames
    final_preds_df_com_agg = add_product_names(final_preds_df_com_agg, comskulist)
    final_preds_df_ama_agg = add_product_names(final_preds_df_ama_agg, amaskulist)
    # final_preds_df_dotcomsubs_agg = add_product_names(final_preds_df_dotcomsubs_agg, comskulist)
    # final_preds_df_comotp_agg = add_product_names(final_preds_df_comotp_agg, comskulist)
    # final_preds_df_amasubs_agg = add_product_names(final_preds_df_amasubs_agg, amaskulist)
    # final_preds_df_amaotp_agg = add_product_names(final_preds_df_amaotp_agg, amaskulist)
    final_preds_df_ama_new = add_product_names(final_preds_df_ama_new, amaskulist)
    final_preds_df_ama_ret = add_product_names(final_preds_df_ama_ret, amaskulist)
    final_preds_df_amasubs_subsplit = add_product_names(final_preds_df_amasubs_subsplit, amaskulist)
    final_preds_df_dotcom_new = add_product_names(final_preds_df_dotcom_new, comskulist)
    final_preds_df_dotcom_ret = add_product_names(final_preds_df_dotcom_ret, comskulist)
    final_preds_df_dotcomsubs_subsplit = add_product_names(final_preds_df_dotcomsubs_subsplit, comskulist)


    # Reorder columns to have SKU and product_name first
    column_order = ['SKU', 'product_name'] + [col for col in final_preds_df_com_agg.columns if col not in ['SKU', 'product_name']]
    final_preds_df_com_agg = final_preds_df_com_agg[column_order]
    final_preds_df_ama_agg = final_preds_df_ama_agg[column_order]
    final_preds_df_ama_new = final_preds_df_ama_new[column_order]
    final_preds_df_ama_ret = final_preds_df_ama_ret[column_order]
    final_preds_df_amasubs_subsplit = final_preds_df_amasubs_subsplit[column_order]
    final_preds_df_dotcom_new = final_preds_df_dotcom_new[column_order]
    final_preds_df_dotcom_ret = final_preds_df_dotcom_ret[column_order]
    final_preds_df_dotcomsubs_subsplit = final_preds_df_dotcomsubs_subsplit[column_order]
    # final_preds_df_dotcomsubs_agg = final_preds_df_dotcomsubs_agg[column_order]
    # final_preds_df_comotp_agg = final_preds_df_comotp_agg[column_order]
    # final_preds_df_amasubs_agg = final_preds_df_amasubs_agg[column_order]
    # final_preds_df_amaotp_agg = final_preds_df_amaotp_agg[column_order]

    current_date = datetime.datetime.now().replace(day=1)
    next_4_months = [current_date + relativedelta(months=i) for i in range(1, 5)]
    next_4_months_str = [date.strftime('%Y-%m-01') for date in next_4_months]

    def filter_next_4_months(df):
        columns_to_keep = ['SKU', 'product_name'] + next_4_months_str
        return df[columns_to_keep]

    final_preds_df_com_agg = filter_next_4_months(final_preds_df_com_agg)
    final_preds_df_ama_agg = filter_next_4_months(final_preds_df_ama_agg)
    final_preds_df_dotcom_new = filter_next_4_months(final_preds_df_dotcom_new)
    final_preds_df_dotcom_ret = filter_next_4_months(final_preds_df_dotcom_ret)
    final_preds_df_dotcomsubs_subsplit = filter_next_4_months(final_preds_df_dotcomsubs_subsplit)
    final_preds_df_ama_new = filter_next_4_months(final_preds_df_ama_new)
    final_preds_df_ama_ret = filter_next_4_months(final_preds_df_ama_ret)
    final_preds_df_amasubs_subsplit = filter_next_4_months(final_preds_df_amasubs_subsplit)


    return final_preds_df_com_agg.round(0), final_preds_df_ama_agg.round(0), \
        final_preds_df_dotcom_new.round(0),final_preds_df_dotcom_ret.round(0),final_preds_df_dotcomsubs_subsplit.round(0),\
        final_preds_df_ama_new.round(0),final_preds_df_ama_ret.round(0),final_preds_df_amasubs_subsplit.round(0)    
    
        #    final_preds_df_dotcomsubs_agg.round(0), final_preds_df_comotp_agg.round(0), \
        #    final_preds_df_amasubs_agg.round(0), final_preds_df_amaotp_agg.round(0),\

