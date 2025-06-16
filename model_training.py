import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import time

def train_model(data_random_forest, features, target_column_name='severity_level_original'):
    """
    Trains a RandomForestClassifier model on the provided data.

    Args:
        data_random_forest (pd.DataFrame): DataFrame containing features and target.
        features (list): List of feature column names.
        target_column_name (str): Name of the target column.

    Returns:
        tuple: A tuple containing the trained model, train_features, train_labels,
               test_features, test_labels, and label_names (original string labels).
               Returns None for all elements if training fails or data is insufficient.
    """
    if not isinstance(data_random_forest, pd.DataFrame) or data_random_forest.empty:
        print("Error: Input to train_model must be a non-empty pandas DataFrame.")
        return None, None, None, None, None, None

    if target_column_name not in data_random_forest.columns:
         print(f"Error: Target column '{target_column_name}' not found in the DataFrame.")
         return None, None, None, None, None, None

    # Handle potential missing values in the target column
    if data_random_forest[target_column_name].isnull().any():
        print(f"Warning: Target column '{target_column_name}' contains missing values. Dropping rows with missing target.")
        data_random_forest = data_random_forest.dropna(subset=[target_column_name]).copy()
        if data_random_forest.empty:
            print("DataFrame is empty after dropping rows with missing target.")
            return None, None, None, None, None, None


    # Factorize the target variable if it's not already numerical
    # Store original label names and create numerical target
    if data_random_forest[target_column_name].dtype == 'object':
        data_random_forest['severity'], label_names_arr = pd.factorize(data_random_forest[target_column_name])
        label_names = label_names_arr.tolist() # Convert numpy array to list for easier handling
        Target = 'severity'
    else:
        Target = target_column_name
        # Assuming numerical target means labels are the unique values sorted
        label_names = sorted(data_random_forest[Target].unique().tolist())


    Features = features
    # Check if all feature columns exist
    if not all(f in data_random_forest.columns for f in Features):
        missing_features = [f for f in Features if f not in data_random_forest.columns]
        print(f"Error: Missing feature columns: {missing_features}")
        return None, None, None, None, None, None

    print("Splitting data into training and testing sets...")
    start_time_split = time.time()
    try:
        # Check if stratification is possible (at least 2 samples per class)
        min_samples_per_class = data_random_forest[Target].value_counts().min()
        if min_samples_per_class >= 2 and len(data_random_forest[Target].unique()) > 1:
             x, y = train_test_split(data_random_forest,
                                     test_size=0.25,
                                     train_size=0.75,
                                     random_state=42,
                                     stratify=data_random_forest[Target]) # Use numerical target for stratify
             print(f"Using stratified split.")
        elif len(data_random_forest) > 1:
             # Fallback to non-stratified split if stratification is not possible
             x, y = train_test_split(data_random_forest,
                                     test_size=0.25,
                                     train_size=0.75,
                                     random_state=42)
             print(f"Using non-stratified split due to insufficient samples per class for stratification.")
        else:
             print("Not enough data for splitting.")
             return None, None, None, None, None, None


        end_time_split = time.time()
        print(f"Data splitting completed in {end_time_split - start_time_split:.2f} seconds.")

        train_features = x[Features]
        train_labels = x[Target]
        test_features = y[Features]
        test_labels = y[Target]

        print('Finished data split.')
        print('Feature Set Used    : ' + str(Features))
        print('Target Class        : ' + str(Target))
        print('Training Set Size   : ' + str(x.shape))
        print('Test Set Size       : ' + str(y.shape))

    except ValueError as e:
        print(f"Error during data splitting: {e}")
        print("Please check if the number of samples per class is sufficient for splitting with the chosen test size.")
        return None, None, None, None, None, None


    print("Initializing and training RandomForestClassifier model...")
    start_time_train = time.time()
    rf_model = RandomForestClassifier(n_estimators=100,
                                      min_samples_split = 30,
                                      bootstrap = True,
                                      min_samples_leaf = 25,
                                      max_features = 'sqrt',
                                      random_state=42,
                                      n_jobs=-1)

    # Check if there is more than one unique class in the training data before training
    if len(np.unique(train_labels)) > 1:
        rf_model.fit(X=train_features, y=train_labels)
        end_time_train = time.time()
        print(f"Model training completed in {end_time_train - start_time_train:.2f} seconds.")
        return rf_model, train_features, train_labels, test_features, test_labels, label_names
    else:
        print("Skipping model training: Not enough unique classes in training data.")
        return None, train_features, train_labels, test_features, test_labels, label_names


if __name__ == '__main__':
    # Example usage when run as a script
    start_time = time.time()
    print("Using dummy DataFrame for model training test...")

    # Define the number of samples
    n_samples = 310

    # Create a list of severity levels with approximate counts
    # Ensure enough samples per class for stratification with 0.25 test size (at least 2 samples per class in minority)
    severity_levels = (['INFO'] * 100 + ['ERROR'] * 50 + ['WARN'] * 70 + ['DEBUG'] * 30 +
                       ['FATAL'] * 20 + ['NOTICE'] * 25 + ['VERBOSE'] * 10 + ['SEVERE'] * 5) # Adjusted counts to ensure min samples >= 2

    # Ensure the list has exactly n_samples, truncate or extend if necessary
    # Shuffle the list to mix severity levels
    import random
    random.seed(42) # for reproducibility
    severity_levels = (severity_levels * ((n_samples // len(severity_levels)) + 1))[:n_samples]
    random.shuffle(severity_levels)


    data = {'severity_level_original': severity_levels,
            'message': [f'msg{i}' for i in range(n_samples)],
            'Tokens': np.random.randint(2, 50, n_samples),
            'Verbs': np.random.randint(0, 10, n_samples),
            'Preps': np.random.randint(0, 5, n_samples),
            'Nouns': np.random.randint(0, 20, n_samples),
            'Length': np.random.randint(10, 200, n_samples),
            'Tokens_By_Length': np.random.rand(n_samples) * 0.2,
            'Nouns_By_Length': np.random.rand(n_samples) * 0.2}
    dummy_data_random_forest = pd.DataFrame(data)

    features = ['Tokens', 'Verbs', 'Preps', 'Nouns', 'Length', 'Tokens_By_Length', 'Nouns_By_Length']
    target_column = 'severity_level_original'

    end_time = time.time()
    print(f"Dummy DataFrame prepared in {end_time - start_time:.2f} seconds.")

    start_time = time.time()
    print("Training model...")
    rf_model, train_features, train_labels, test_features, test_labels, label_names = train_model(
        dummy_data_random_forest, features, target_column)
    end_time = time.time()
    print(f"Total model training execution time: {end_time - start_time:.2f} seconds.")

    if rf_model:
        print("\nModel training test completed successfully.")
        print("Trained model:", rf_model)
        print("Train features shape:", train_features.shape)
        print("Test features shape:", test_features.shape)
        print("Label names:", label_names)
        print("\nValue counts for training labels:")
        print(pd.Series(train_labels).value_counts().sort_index())
        print("\nValue counts for test labels:")
        print(pd.Series(test_labels).value_counts().sort_index())

    else:
        print("\nModel training test failed or was skipped.")
