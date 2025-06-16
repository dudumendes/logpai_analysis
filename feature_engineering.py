import pandas as pd
import re
import spacy
import json
import time
import numpy as np

# Load spacy model only once
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    from spacy.cli import download
    download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')


def get_component(obj):
    """Extracts the component string (e.g., [component]) from the message_component."""
    regex = r'\[([^\[\]]+)\]' # Use raw string and capture group
    message_component = str(obj['message_component']).strip() if pd.notnull(obj['message_component']) else ""
    match = re.search(regex, message_component) # No need for IGNORECASE unless component names are case-insensitive
    if match:
        return match.group(0)
    return None

def get_message(obj):
    """Extracts the message content after removing the component."""
    message_component = str(obj['message_component']).strip() if pd.notnull(obj['message_component']) else ""
    component = str(obj['component']).strip() if pd.notnull(obj['component']) else ""

    if component and component in message_component:
        # Replace the component only if it's found at the beginning or surrounded by space/punctuation
        # A more robust approach might be needed depending on component placement
        # For now, replace the first occurrence anywhere
        return message_component.replace(component, "", 1).strip()
    else:
        return message_component.strip()

def get_tokens(message):
    """Tokenizes a message using spaCy and returns a JSON string of tokens with POS and DEP."""
    message = str(message) if pd.notnull(message) else ""
    tokens = nlp(message)
    token_list = []
    for a in tokens:
        token_dict = {
            "token": str(a),
            "pos": a.pos_,
            "dep": a.dep_,
            "is_alpha": a.is_alpha, # Add more useful token attributes
            "is_stop": a.is_stop,
            "is_punct": a.is_punct,
            "is_digit": a.is_digit
        }
        token_list.append(token_dict)
    return json.dumps(token_list)

def get_message_tokens(token_list_json):
    """Gets the number of tokens from the JSON token list."""
    try:
        doc = json.loads(token_list_json)
        return len(doc)
    except (json.JSONDecodeError, TypeError):
        return 0

def get_message_verbs(token_list_json):
    """Counts verbs and auxiliary verbs from the JSON token list."""
    try:
        doc = json.loads(token_list_json)
        message_verbs = sum((token.get('pos') == 'AUX' or token.get('pos') == 'VERB') for token in doc)
        return message_verbs
    except (json.JSONDecodeError, TypeError):
        return 0

def get_message_preps(token_list_json):
    """Counts prepositions from the JSON token list."""
    try:
        doc = json.loads(token_list_json)
        message_preps = sum(token.get('dep') == 'prep' for token in doc)
        return message_preps
    except (json.JSONDecodeError, TypeError):
        return 0

def get_message_nouns(token_list_json):
    """Counts nouns from the JSON token list."""
    try:
        doc = json.loads(token_list_json)
        message_nouns = sum(token.get('pos') == 'NOUN' for token in doc)
        return message_nouns
    except (json.JSONDecodeError, TypeError):
        return 0


def get_message_length(message):
    """Gets the length of the raw message string."""
    message = str(message) if pd.notnull(message) else ""
    return len(message)


def get_message_length_by_nouns(token_list_json, message):
    """Calculates the ratio of nouns to message length."""
    message = str(message) if pd.notnull(message) else ""
    if len(message) == 0:
        return 0.0
    try:
        doc = json.loads(token_list_json)
        message_nouns = sum(token.get('pos') == 'NOUN' for token in doc)
        return message_nouns / len(message)
    except (json.JSONDecodeError, TypeError):
        return 0.0


def get_message_length_by_tokens(token_list_json, message):
    """Calculates the ratio of tokens to message length."""
    message = str(message) if pd.notnull(message) else ""
    if len(message) == 0:
        return 0.0
    try:
        doc = json.loads(token_list_json)
        message_tokens_count = len(doc) # Using total tokens, not just preps as in previous version
        return message_tokens_count / len(message)
    except (json.JSONDecodeError, TypeError):
        return 0.0


def apply_feature_engineering(df):
    """
    Applies feature engineering steps to the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame with 'severity_level_original'
                          and 'message_component' columns.

    Returns:
        pd.DataFrame: A new DataFrame with extracted features.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        print("Error: Input to apply_feature_engineering must be a non-empty pandas DataFrame.")
        return pd.DataFrame()

    print("Applying component and message extraction...")
    df['component'] = df.apply(get_component, axis=1)
    df['message'] = df.apply(get_message, axis=1)

    # Select relevant columns and create a copy to avoid SettingWithCopyWarning
    # Ensure 'message' column is treated as string
    data_random_forest = df[['severity_level_original', 'message']].copy()
    data_random_forest['message'] = data_random_forest['message'].astype(str).fillna('')

    # Reset index and drop the old one
    data_random_forest.reset_index(drop=True, inplace=True)

    print("Applying tokenization...")
    start_time_tokens = time.time()
    data_random_forest['token_list'] = data_random_forest['message'].apply(get_tokens)
    end_time_tokens = time.time()
    print(f"Tokenization completed in {end_time_tokens - start_time_tokens:.2f} seconds.")

    print("Applying feature extraction...")
    start_time_features = time.time()
    # Apply functions row-wise using apply with axis=1
    data_random_forest['Tokens'] = data_random_forest['token_list'].apply(get_message_tokens)
    data_random_forest['Verbs'] = data_random_forest['token_list'].apply(get_message_verbs)
    data_random_forest['Preps'] = data_random_forest['token_list'].apply(get_message_preps)
    data_random_forest['Nouns'] = data_random_forest['token_list'].apply(get_message_nouns)
    data_random_forest['Length'] = data_random_forest['message'].apply(get_message_length)
    # For functions requiring both token_list and message, continue using apply with axis=1
    data_random_forest['Nouns_By_Length'] = data_random_forest.apply(lambda row: get_message_length_by_nouns(row['token_list'], row['message']), axis=1)
    data_random_forest['Tokens_By_Length'] = data_random_forest.apply(lambda row: get_message_length_by_tokens(row['token_list'], row['message']), axis=1)
    end_time_features = time.time()
    print(f"Feature extraction completed in {end_time_features - start_time_features:.2f} seconds.")

    # Drop the intermediate 'token_list' column
    data_random_forest = data_random_forest.drop(columns=['token_list'])

    return data_random_forest

if __name__ == '__main__':
    # Example usage when run as a script
    start_time = time.time()
    print("Using dummy DataFrame for feature engineering test...")
    # Create a dummy DataFrame that mimics the output of data_processing.create_dataframe
    data = {'general_information': ['info1', 'info2', 'info3', 'info4', 'info5', 'info6'],
            'severity_level_original': ['INFO', 'ERROR', 'WARN', 'DEBUG', 'INFO', 'FATAL'],
            'message_component': ['[component1] message1 with nouns and verbs.',
                                  'message2 without component.',
                                  '[component2] another message, maybe with preps.',
                                  'debug message with verbs and nouns.',
                                  'message without component or specific features.',
                                  '[critical] Fatal error occurred.']}
    dummy_df = pd.DataFrame(data)

    end_time = time.time()
    print(f"Dummy DataFrame prepared in {end_time - start_time:.2f} seconds.")

    start_time = time.time()
    print("Applying feature engineering...")
    processed_df = apply_feature_engineering(dummy_df)
    end_time = time.time()
    print(f"Feature engineering completed in {end_time - start_time:.2f} seconds.")

    print("\nProcessed DataFrame:")
    print(processed_df)
    print("\nDataFrame Info:")
    processed_df.info()
    print("\nDataFrame Describe:")
    print(processed_df.describe())
