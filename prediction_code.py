import os
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import logging
import warnings
import joblib
# Ignore all warnings
warnings.filterwarnings('ignore')
# Setup logging configuration
logging.basicConfig(filename='preds.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')



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




def get_final_results(best_final_results, months_ahead=4):
    """
    Filter and pivot the final results DataFrame based on the upgrade_date
    for the current month and the next specified number of months.
    """
    best_final_results['upgrade_date'] = pd.to_datetime(best_final_results['upgrade_date'])
    current_date = datetime.now()
    start_date = current_date.replace(day=1).date()
    end_date = (start_date + pd.DateOffset(months=months_ahead)).date()
    best_final_results['upgrade_date'] = best_final_results['upgrade_date'].dt.date
    best_final_results_filtered = best_final_results[
        (best_final_results['upgrade_date'] >= start_date) & 
        (best_final_results['upgrade_date'] <= end_date)
    ]
    best_final_results_grouped = best_final_results_filtered.groupby(['SKU', 'upgrade_date'])['Predicted'].sum().reset_index()
    best_final_results_pivoted = best_final_results_grouped.pivot(index='SKU', columns='upgrade_date', values='Predicted')
    best_final_results_pivoted.columns = best_final_results_pivoted.columns.map(lambda x: x.strftime('%Y-%m-%d'))
    return best_final_results_pivoted.reset_index()



def create_transformed_df_otp(df, sales_channel):
    """
    Filter the dataframe by the given sales channel and apply the necessary transformations.
    Returns the transformed dataframe.
    """
    
    df_copy = df.copy()
    df_copy['order_date'] = pd.to_datetime(df_copy['order_date'])
    
    # Drop unnecessary columns
    df_copy = df_copy.drop(['Varient title', 'Amazon_spend_sku_wise','product_name_priority_sku'], axis=1)
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


    final_preds_df_dotcomsubs_agg = get_final_results(final_preds_df_dotcomsubs, months_ahead=4)

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

    final_preds_df_amasubs_agg = get_final_results(final_preds_df_amasubs, months_ahead=4)

    # Load the data into a DataFrame
    file_path = 'New OTP Data.xlsx'  # Update this to the actual file path
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

    final_preds_df_dotcomsubs_agg['SKU'] = final_preds_df_dotcomsubs_agg['SKU'].apply(lambda x: x.split('_')[-1])
    final_preds_df_comotp_agg.rename(columns={'Sku': 'SKU'}, inplace=True)
    final_preds_df_amasubs_agg['SKU'] = final_preds_df_amasubs_agg['SKU'].apply(lambda x: x.split('_')[-1])
    final_preds_df_amaotp_agg.rename(columns={'Sku': 'SKU'}, inplace=True)


    # Ensure SKU column is cleaned (remove any leading/trailing spaces, etc.)
    final_preds_df_dotcomsubs_agg['SKU'] = final_preds_df_dotcomsubs_agg['SKU'].str.strip()
    final_preds_df_comotp_agg['SKU'] = final_preds_df_comotp_agg['SKU'].str.strip()


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
    
    return final_preds_df_com_agg.round(0) , final_preds_df_ama_agg.round(0)



