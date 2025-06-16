import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    roc_auc_score)
import matplotlib.pyplot as plt
import seaborn as sns
from yellowbrick.classifier import ClassificationReport
# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
import time
import re # Import re for the dummy extract_data_log_lines

def evaluate_model(rf_model, test_features, test_labels, label_names, train_features, train_labels):
    """
    Evaluates the trained RandomForestClassifier model and prints various metrics.

    Args:
        rf_model: The trained RandomForestClassifier model.
        test_features (pd.DataFrame): DataFrame of test features.
        test_labels (pd.Series): Series of true test labels (numerical).
        label_names (list): List of original string label names, where index corresponds to numerical labels.
        train_features (pd.DataFrame): DataFrame of training features (needed for Yellowbrick).
        train_labels (pd.Series): Series of true training labels (numerical) (needed for Yellowbrick).
    """
    if rf_model is None or test_features is None or test_labels is None or train_features is None or train_labels is None or label_names is None:
        print("Error: Invalid input provided to evaluate_model. One or more inputs are None.")
        return

    print("Evaluating model...")
    start_time_eval = time.time()

    try:
        y_pred_test = rf_model.predict(test_features)
    except Exception as e:
        print(f"Error during model prediction: {e}")
        return


    # Accuracy score
    accuracy = accuracy_score(test_labels, y_pred_test)
    print(f"Accuracy score: {accuracy:.4f}")

    # Confusion Matrix
    print("Generating confusion matrix...")
    try:
        # Ensure test_labels and y_pred_test are compatible
        # Get all unique labels present in both test_labels and y_pred_test
        unique_labels_numeric = np.unique(np.concatenate((test_labels, y_pred_test)))

        # Map these numerical labels back to original names using the provided label_names
        # Ensure index is within the bounds of label_names
        matrix_display_labels = [label_names[i] for i in sorted(unique_labels_numeric) if i < len(label_names) and i >= 0]

        # Generate confusion matrix using only the unique labels found in test_labels and y_pred_test
        matrix = confusion_matrix(test_labels, y_pred_test, labels=sorted(unique_labels_numeric))

        # Handle cases where a class has no true samples for normalization
        row_sums = matrix.sum(axis=1)
        # Avoid division by zero for classes with no true samples
        matrix_normalized = np.divide(matrix.astype('float'), row_sums[:, np.newaxis], out=np.zeros_like(matrix.astype('float')), where=row_sums[:, np.newaxis]!=0)

        # Adjust figure size and font scale dynamically based on the number of unique labels
        num_labels = len(matrix_display_labels)
        plt.figure(figsize=(max(6, num_labels * 1.2), max(5, num_labels * 0.9)))
        sns.set(font_scale=min(1.2, 30 / num_labels) if num_labels > 0 else 1) # Adjust font scale based on number of labels
        sns.heatmap(matrix_normalized, annot=True, annot_kws={'size':min(10, 250 / num_labels) if num_labels > 0 else 10},
                    cmap=plt.cm.Greens, linewidths=0.2, fmt='.2f',
                    xticklabels=matrix_display_labels, yticklabels=matrix_display_labels) # Set tick labels

        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion Matrix for Random Forest Model')
        plt.tight_layout() # Adjust layout to prevent labels overlapping
        plt.show()
    except Exception as e:
        print(f"Error generating confusion matrix: {e}")


    # Random Forest Results (Standard Metrics)
    print("\n========== Random Forest Results ==========")
    try:
        ac_sc = accuracy_score(test_labels, y_pred_test)
        # Use the sorted unique labels from the model's classes for weighted averages
        # This avoids errors if a class is in label_names but not in test_labels/y_pred_test
        # Ensure rf_model.classes_ exists and is not empty
        if hasattr(rf_model, 'classes_') and len(rf_model.classes_) > 0:
             metrics_labels = sorted(rf_model.classes_)
             rc_sc = recall_score(test_labels, y_pred_test, labels=metrics_labels, average="weighted", zero_division=0) # Added zero_division and labels
             pr_sc = precision_score(test_labels, y_pred_test, labels=metrics_labels, average="weighted", zero_division=0) # Added zero_division and labels
        else:
             # Fallback if model classes are not available (e.g., model not trained)
             rc_sc = recall_score(test_labels, y_pred_test, average="weighted", zero_division=0)
             pr_sc = precision_score(test_labels, y_pred_test, average="weighted", zero_division=0)


        f1_sc = f1_score(test_labels, y_pred_test, average='micro') # Micro average is not affected by missing classes

        confusion_m = confusion_matrix(test_labels, y_pred_test)

        print(f"Accuracy    : {ac_sc:.4f}")
        print(f"Recall (weighted): {rc_sc:.4f}")
        print(f"Precision (weighted): {pr_sc:.4f}")
        print(f"F1 Score (micro) : {f1_sc:.4f}")
        print("Confusion Matrix: ")
        print(confusion_m)

    except Exception as e:
         print(f"Error calculating standard metrics: {e}")


    # Classification Report (Yellowbrick)
    print('\n================= Classification Report =================')
    try:
        # Ensure the classes argument for Yellowbrick matches the labels the model was trained on
        # rf_model.classes_ contains the sorted unique class labels from the training data (numerical)
        # Map these back to original label names
        if hasattr(rf_model, 'classes_') and len(rf_model.classes_) > 0:
             yb_classes_names = [label_names[i] for i in sorted(rf_model.classes_) if i < len(label_names) and i >= 0]
             visualizer = ClassificationReport(rf_model, classes=yb_classes_names, support=True) # Added support=True
             visualizer.fit(X=train_features, y=train_labels)     # Fit the training data to the visualizer
             visualizer.score(test_features, test_labels)       # Evaluate the model on the test data
             visualizer.poof() # Draw/show/poof the data
             print("\nYellowbrick Classification Report generated.")
        else:
            print("Skipping Yellowbrick Classification Report: Model classes not available.")


    except Exception as e:
        print(f"Error generating Yellowbrick classification report: {e}")

    # Classification Report (sklearn)
    print('\n================= Sklearn Classification Report =================')
    try:
        # Use the sorted unique labels present in the test set for the classification report
        unique_test_labels_numeric = np.unique(test_labels)
        # Map numeric labels back to names
        label_map = {i: name for i, name in enumerate(label_names)}
        # Ensure that we only get names for labels actually present in the test set
        target_names_subset = [label_map[i] for i in sorted(unique_test_labels_numeric) if i in label_map]

        # Use the unique labels present in the test set and corresponding names
        print(classification_report(test_labels, y_pred_test, target_names=target_names_subset, labels=sorted(unique_test_labels_numeric), zero_division=0))
        print("\nSklearn Classification Report generated.")

    except Exception as e:
        print(f"Error generating sklearn classification report: {e}")


    # Tree Visualization (Export to a file)
    print("\nExporting decision tree visualization...")
    try:
        # Pull out one tree from the forest
        if hasattr(rf_model, 'estimators_') and len(rf_model.estimators_) > 0:
            tree = rf_model.estimators_[0]

            # Define the path for the dot file
            dot_file_path = 'tree.dot'
            png_file_path = 'tree.png'

            # Export the image to a dot file
            export_graphviz(tree, out_file = dot_file_path,
                            feature_names = test_features.columns.tolist(), # Use test_features.columns as list for feature names
                            class_names=[label_names[i] for i in sorted(rf_model.classes_)], # Use the classes the model was trained on, mapped to names
                            rounded = True,
                            filled = True,
                            precision = 1,
                            proportion=False) # Set proportion to False

            # Use dot file to create a graph
            try:
                (graph, ) = pydot.graph_from_dot_file(dot_file_path)
                # Write graph to a png file
                graph.write_png(png_file_path)
                print(f"Decision tree exported to {png_file_path}")
                # Display the image path
                print(f"Tree visualization image saved at: {png_file_path}")
            except ImportError:
                 print("pydot not found. Cannot generate tree visualization PNG.")
                 print("Please install pydot: pip install pydot")
            except Exception as e:
                print(f"Could not generate tree visualization PNG: {e}")
                print("Please ensure graphviz is installed and configured correctly (e.g., by installing graphviz system package).")
        else:
            print("Skipping tree visualization: Model estimators not available (model might not have been trained or has no trees).")


    except Exception as e:
        print(f"Error during tree visualization export: {e}")

    end_time_eval = time.time()
    print(f"\nModel evaluation completed in {end_time_eval - start_time_eval:.2f} seconds.")


if __name__ == '__main__':
    # Example usage when run as a script
    start_time = time.time()
    print("Running full pipeline for model evaluation test...")

    # --- Dummy Data Generation ---
    print("\n--- Generating Dummy Data ---")
    # Create a larger dummy DataFrame to allow for splitting and evaluation
    n_samples = 500
    # Ensure enough samples per class for stratified splitting and evaluation metrics
    severity_levels = (['INFO'] * 150 + ['ERROR'] * 80 + ['WARN'] * 100 + ['DEBUG'] * 50 +
                       ['FATAL'] * 30 + ['NOTICE'] * 40 + ['VERBOSE'] * 30 + ['SEVERE'] * 20)
    # Ensure the list has exactly n_samples, truncate or extend if necessary
    import random
    random.seed(42)
    severity_levels = (severity_levels * ((n_samples // len(severity_levels)) + 1))[:n_samples]
    random.shuffle(severity_levels)

    data = {'severity_level_original': severity_levels,
            'message_component': [f'[comp{i%5}] message content {i}' for i in range(n_samples)], # Simulate message_component
            'general_information': [f'info {i}' for i in range(n_samples)]} # Simulate general_information
    dummy_df = pd.DataFrame(data)

    print(f"Generated dummy DataFrame with {len(dummy_df)} rows.")


    # --- Dummy Feature Engineering ---
    print("\n--- Dummy Feature Engineering ---")
    # Use a simplified dummy feature engineering function for this test
    def dummy_apply_feature_engineering(df):
        df = df.copy()
        df['component'] = df['message_component'].str.extract(r'\[(.*?)\]').iloc[:, 0].apply(lambda x: f'[{x}]' if pd.notnull(x) else None)
        df['message'] = df.apply(lambda row: str(row['message_component']).replace(str(row['component']), '', 1).strip() if pd.notnull(row['component']) else str(row['message_component']).strip(), axis=1)

        # Create dummy features based on message length and random values
        df['Length'] = df['message'].str.len().fillna(0).astype(int)
        df['Tokens'] = (df['Length'] / 5).astype(int) + 2 # Simple token approx
        df['Verbs'] = np.random.randint(0, 5, len(df))
        df['Preps'] = np.random.randint(0, 3, len(df))
        df['Nouns'] = np.random.randint(0, 10, len(df))

        # Calculate ratio features, handling division by zero
        df['Tokens_By_Length'] = df.apply(lambda row: row['Tokens'] / (row['Length'] if row['Length'] > 0 else 1), axis=1)
        df['Nouns_By_Length'] = df.apply(lambda row: row['Nouns'] / (row['Length'] if row['Length'] > 0 else 1), axis=1)

        # Select final columns
        data_random_forest = df[['severity_level_original', 'message', 'Tokens', 'Verbs', 'Preps', 'Nouns', 'Length', 'Tokens_By_Length', 'Nouns_By_Length']].copy()
        return data_random_forest

    data_random_forest = dummy_apply_feature_engineering(dummy_df)
    print(f"Applied dummy feature engineering. Resulting DataFrame shape: {data_random_forest.shape}")
    print("Dummy feature DataFrame head:")
    print(data_random_forest.head())


    # --- Dummy Model Training ---
    print("\n--- Dummy Model Training and Split ---")
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    # Need factorized target for model training and evaluation
    data_random_forest['severity'], label_names = pd.factorize(data_random_forest['severity_level_original'])
    Target = 'severity'
    Features = ['Tokens', 'Verbs', 'Preps', 'Nouns', 'Length', 'Tokens_By_Length', 'Nouns_By_Length']

    try:
        # Ensure enough samples for stratification
        min_samples_per_class = data_random_forest[Target].value_counts().min()
        if min_samples_per_class >= 2 and len(data_random_forest[Target].unique()) > 1:
             x, y = train_test_split(data_random_forest,
                                     test_size=0.25,
                                     train_size=0.75,
                                     random_state=42,
                                     stratify=data_random_forest[Target])
             print(f"Using stratified split ({len(x)} train, {len(y)} test).")
        elif len(data_random_forest) > 1:
             x, y = train_test_split(data_random_forest,
                                     test_size=0.25,
                                     train_size=0.75,
                                     random_state=42)
             print(f"Using non-stratified split ({len(x)} train, {len(y)} test) due to insufficient samples per class for stratification.")
        else:
             print("Not enough data for splitting.")
             rf_model, train_features, train_labels, test_features, test_labels = None, None, None, None, None
             print("Skipping model training and evaluation due to insufficient data.")


        if 'x' in locals() and 'y' in locals(): # Check if split was successful
            train_features = x[Features]
            train_labels = x[Target]
            test_features = y[Features]
            test_labels = y[Target]

            # Dummy model training
            dummy_rf_model = RandomForestClassifier(n_estimators=10, random_state=42) # Fewer estimators for speed
            # Handle case where train_labels might be empty or have only one class
            if len(np.unique(train_labels)) > 1:
                dummy_rf_model.fit(train_features, train_labels)
                rf_model = dummy_rf_model
                print("Dummy model trained.")
            else:
                 print("Skipping dummy model training: Not enough unique classes in training data.")
                 rf_model = None # Set model to None if training skipped

    except Exception as e:
         print(f"Error during dummy model training or split: {e}")
         rf_model, train_features, train_labels, test_features, test_labels = None, None, None, None, None


    if rf_model:
        print(f"Train data shape: {train_features.shape}, Test data shape: {test_features.shape}")
        print(f"Label names used for training/evaluation: {label_names}")

        # --- Model Evaluation ---
        print("\n--- Model Evaluation ---")
        evaluate_model(rf_model, test_features, test_labels, label_names, train_features, train_labels)
    else:
        print("\nSkipping model evaluation as model training failed or was skipped.")


    end_time = time.time()
    print(f"\nTotal script execution time: {end_time - start_time:.2f} seconds.")