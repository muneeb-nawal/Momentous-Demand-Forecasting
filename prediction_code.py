# 1. Imports and Setup

import os
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pickle
import logging
import warnings
import joblib

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
    Find the appropriate model file based on the key.
    """
    try:
        all_files = os.listdir(directory)
        for file_name in all_files:
            if key in file_name and file_name.endswith('.pkl'):
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

    current_date = datetime.now()
    start_date = current_date.replace(day=1).date()

    end_date = datetime(2025, 12, 31).date()

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
    today = datetime.today()

    # Format the current month as 'YYYY-MM-01'
    current_month = today.strftime('%Y-%m-01')

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

# 7. Main Implementation

def load_and_process_data():

    # Load the data into a DataFrame
    file_path = 'New Subs Data.xlsx'  # Update this to the actual file path
    sheet_name = 'creatine combined_flag stockout'  # Update this to the actual sheet name

    df_subs = pd.read_excel(file_path, sheet_name=sheet_name)

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
    current_date_comsubs = datetime.now().replace(day=1)  # Current month (1st of the month)
    next_year_end_comsubs = datetime(current_date_comsubs.year + 1, 12, 31)  # End of next year

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
    current_date_amasubs = datetime.now().replace(day=1)  # Current month (1st of the month)
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
    forecast_data = df_amaotp[df_amaotp.order_date>=pd.to_datetime("2024-09-01")]

    # Initialize an empty DataFrame to store results
    results_df = pd.DataFrame()

    # Iterate through each model in the specified directory
    for model_file in os.listdir(best_models_path):
        model_path = os.path.join(best_models_path, model_file)
        
        # Extract SKU and product name from the model file name
        model_parts = model_file.split("_")
        sku_id = int(model_parts[2]) if model_parts[2].isdigit() else model_parts[2]
        product_name = model_parts[3].split('.')[0]

        # Skip models for a specific product
        if product_name == "Fadogia Agrestis":
            continue

        # Filter the forecast data for the current SKU
        sku_data = forecast_data[forecast_data.Sku == sku_id].copy()

        # Remove duplicate entries based on 'order_date'
        sku_data.drop_duplicates(subset=['order_date'], inplace=True)

        # Load the trained model
        model = joblib.load(model_path)
        feature_columns = model.feature_names_in_

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


    # Filter forecast data starting from September 1, 2024
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

        # Skip models for a specific product name
        if product_name == "Fadogia Agrestis":
            continue

        # Filter the forecast data for the current SKU
        sku_data = forecast_data[forecast_data.Sku == sku_id].copy()
        sku_data.drop_duplicates(subset=['order_date'], inplace=True)

        try:
            # Load the model
            model = joblib.load(model_path)
            
            # Ensure the model has a 'predict' method
            if not hasattr(model, 'predict'):
                print(f"Model at {model_path} does not have a 'predict' method.")
                continue
            
            # Prepare the feature columns for prediction
            feature_columns = [col for col in sku_data.columns if col in model.feature_names_in_]
            prediction_data = sku_data[feature_columns]

            # Create a DataFrame for storing predictions
            temp_df = pd.DataFrame()
            temp_df['order_date'] = sku_data['order_date']
            temp_df['units_sold'] = model.predict(prediction_data)
            temp_df.set_index('order_date', inplace=True)

            # Resample the data to end-of-month frequency and reset index
            temp_df = temp_df.resample('M').sum().reset_index()
            temp_df['Sku'] = sku_id
            temp_df['product_name'] = product_name

            # Append the results to the final DataFrame
            results_df = pd.concat([results_df, temp_df], ignore_index=True)

        except AttributeError as e:
            print(f"AttributeError with model {model_path}: {e}")
        except Exception as e:
            print(f"Error processing model {model_path}: {e}")

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

    # Ensure SKU column is cleaned (remove any leading/trailing spaces, etc.)
    final_preds_df_dotcomsubs_agg['SKU'] = final_preds_df_dotcomsubs_agg['SKU'].str.strip()
    final_preds_df_comotp_agg['SKU'] = final_preds_df_comotp_agg['SKU'].str.strip()

    # Append the two DataFrames
    combined_df_otp_agg = pd.concat([final_preds_df_dotcomsubs_agg, final_preds_df_comotp_agg])

    # Ensure date columns are numeric (convert to numeric if needed)
    date_columns = combined_df_otp_agg.columns.drop('SKU')  # Assuming all other columns are dates

    combined_df_otp_agg[date_columns] = combined_df_otp_agg[date_columns].apply(pd.to_numeric, errors='coerce')

    # Group by SKU and sum all the numeric columns (date columns)
    final_summed_df = combined_df_otp_agg.groupby('SKU').sum().reset_index()

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

    final_preds_df_com_agg = final_preds_df_com_agg[final_preds_df_com_agg['SKU'] != '850030796080']
    final_preds_df_com_agg

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

    return final_preds_df_com_agg , final_preds_df_ama_agg

