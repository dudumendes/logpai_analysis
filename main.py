import sys
import os
import time
import pandas as pd
import matplotlib.pyplot as plt # Import matplotlib for plots generated in evaluation

# Add the directory containing the local modules to sys.path
# This is necessary when running the script from a different directory
# or in environments like some notebooks where the current directory isn't
# automatically in the path for imports.
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
if current_dir not in sys.path:
    sys.path.append(current_dir)

# print("Current sys.path:", sys.path) # Uncomment for debugging import issues

# Import modules after modifying sys.path
try:
    from data_acquisition import acquire_data, FILE_LIST
    from data_processing import extract_data_log_lines, create_dataframe
    from feature_engineering import apply_feature_engineering
    from model_training import train_model
    from model_evaluation import evaluate_model
except ImportError as e:
    print(f"Failed to import a required module. Please ensure all .py files are in the same directory.")
    print(f"ImportError: {e}")
    # print("Current sys.path after import attempt:", sys.path) # Uncomment for debugging
    sys.exit(1) # Exit if imports fail


if __name__ == "__main__":
    start_time = time.time()
    print("Starting the full data processing and model training pipeline...")

    # Step 1: Data Acquisition
    print("\n--- Step 1: Data Acquisition ---")
    acquisition_start_time = time.time()
    log_lines = acquire_data(FILE_LIST)
    acquisition_end_time = time.time()
    print(f"\nAcquired {len(log_lines)} log lines.")
    print(f"Data Acquisition completed in {acquisition_end_time - acquisition_start_time:.2f} seconds.")

    # Step 2: Data Processing
    print("\n--- Step 2: Data Processing ---")
    processing_start_time = time.time()
    df = pd.DataFrame() # Initialize df as empty DataFrame
    if log_lines:
        general_information, messages, messages_length, level_names, _ = extract_data_log_lines(log_lines)
        if general_information and messages and level_names: # Check if extraction returned data
            df = create_dataframe(general_information, level_names, messages)
            print(f"Created DataFrame with {len(df)} rows.")
        else:
            print("No data extracted after processing log lines.")
    else:
        print("No log lines acquired, skipping data processing.")
    processing_end_time = time.time()
    print(f"Data Processing completed in {processing_end_time - processing_start_time:.2f} seconds.")

    # Step 3: Feature Engineering
    print("\n--- Step 3: Feature Engineering ---")
    feature_engineering_start_time = time.time()
    data_random_forest = pd.DataFrame() # Initialize data_random_forest
    if not df.empty:
        data_random_forest = apply_feature_engineering(df)
        print(f"Applied feature engineering. Resulting DataFrame shape: {data_random_forest.shape}")
    else:
        print("DataFrame is empty, skipping feature engineering.")
    feature_engineering_end_time = time.time()
    print(f"Feature Engineering completed in {feature_engineering_end_time - feature_engineering_start_time:.2f} seconds.")

    # Define Features and Target for the model - ensure these match the column names
    Features = ['Tokens', 'Verbs', 'Preps', 'Nouns', 'Length', 'Tokens_By_Length', 'Nouns_By_Length']
    Target_column_name = 'severity_level_original'

    # Step 4: Model Training
    print("\n--- Step 4: Model Training ---")
    model_training_start_time = time.time()
    # Initialize outputs to None
    rf_model, train_features, train_labels, test_features, test_labels, label_names = None, None, None, None, None, None
    if not data_random_forest.empty and all(f in data_random_forest.columns for f in Features) and Target_column_name in data_random_forest.columns:
        rf_model, train_features, train_labels, test_features, test_labels, label_names = train_model(
            data_random_forest, Features, Target_column_name)
        if rf_model:
             print("Model training completed.")
        else:
             print("Model training failed or skipped.")
    else:
        print("DataFrame is empty or missing required columns for training, skipping model training.")
    model_training_end_time = time.time()
    print(f"Model Training completed in {model_training_end_time - model_training_start_time:.2f} seconds.")

    # Step 5: Model Evaluation
    print("\n--- Step 5: Model Evaluation ---")
    model_evaluation_start_time = time.time()
    # Check if all necessary components for evaluation are available
    if rf_model and test_features is not None and test_labels is not None and train_features is not None and train_labels is not None and label_names:
        print("Starting model evaluation...")
        evaluate_model(rf_model, test_features, test_labels, label_names, train_features, train_labels)
        print("Model evaluation completed.")
    else:
        print("Model or necessary data for evaluation not available, skipping model evaluation.")
    model_evaluation_end_time = time.time()
    print(f"Model Evaluation completed in {model_evaluation_end_time - model_evaluation_start_time:.2f} seconds.")

    end_time = time.time()
    print("\nFull pipeline execution finished.")
    print(f"Total execution time: {end_time - start_time:.2f} seconds.")

    # Keep plots displayed if running in a notebook environment
    # plt.show() # This might block execution in a script, but is useful in notebooks