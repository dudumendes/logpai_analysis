import pandas as pd
import re
import time

def extract_data_log_lines(log_lines) :
    """
    Gets a list of log lines and returns extracted data.

    Args:
        log_lines (list): A list of raw log line strings.

    Returns:
        tuple: A tuple containing lists for general_information, messages,
               messages_length, level_names, and original logs.
    """

    general_information = []
    messages = []
    messages_length = []
    level_names = []
    logs = []

    log_levels = {"INFO": [" I ", " [info] ", " INFO "],
                  "DEBUG": [ " D ", " [debug] ", " DEBUG "],
                  "ERROR": [" E ", " [error] ", " ERROR "],
                  "WARN": [" W ", " [warn] ", " WARN ", " WARNING "],
                  "ASSERT": [ " ASSERT " ],
                  "NOTICE": [ " [notice] "],
                  "VERBOSE": [" V "],
                  "EMERG": [" [emerg] "],
                  "ALERT": [ " [alert] "],
                  "CRITICAL": [" [crit] ", " CRITICAL "],
                  "FATAL": [" FATAL "],
                  "FAILURE": [" FAILURE "],
                  "TRACE": [" TRACE "],
                  "AUDIT": [" AUDIT "],
                  "SEVERE" : [" SEVERE "]
                  }
    levels_regex = []
    reverse_dict = {}
    for ll in log_levels:
        regex = [re.escape(a) for a in log_levels[ll]]
        levels_regex.extend(regex)
        for regx in log_levels[ll]:
            reverse_dict[regx] = ll

    regex_pattern = '|'.join(levels_regex)

    for line in log_lines:
        if line:
            line = line.strip()
            levelname_match = re.search(regex_pattern, line)
            if levelname_match:
                levelname_str = levelname_match.group(0)
                try:
                    level = reverse_dict[levelname_str]
                    # Find the start of the message part after the level
                    split_point = levelname_match.span()[1] # End position of the matched level string
                    general_info = line[:split_point].strip()
                    message_content = line[split_point:].strip()

                    general_information.append(general_info)
                    level_names.append(level)
                    messages.append(message_content)
                    messages_length.append(len(message_content))
                    logs.append(line)
                except KeyError:
                     # This should ideally not happen if regex_pattern is built correctly from reverse_dict keys
                     print(f"Warning: Internal error - Matched levelname string '{levelname_str}' not found in reverse_dict for line: {line}")
                     pass
            else:
                # Optionally handle lines that do not match any severity level regex
                # For now, we skip them as in the original code structure
                pass


    return general_information, messages, messages_length, level_names, logs

def create_dataframe(general_information, level_names, messages):
    """
    Creates a pandas DataFrame from extracted log data lists.

    Args:
        general_information (list): List of general information strings.
        level_names (list): List of severity level names.
        messages (list): List of message content strings.

    Returns:
        pd.DataFrame: A DataFrame containing the log data.
    """
    min_len = min(len(general_information), len(level_names), len(messages))
    if not (len(general_information) == len(level_names) == len(messages)):
        print(f"Warning: Mismatched list lengths during DataFrame creation. Truncating to minimum length: {min_len}")
        general_information = general_information[:min_len]
        level_names = level_names[:min_len]
        messages = messages[:min_len]

    df = pd.DataFrame({
        'general_information': pd.Series(general_information, dtype='object'),
        'severity_level_original': pd.Series(level_names, dtype='object'),
        'message_component': pd.Series(messages, dtype='object') # Storing the message part here as per original notebook flow
    })

    df['index'] = df.index
    return df

if __name__ == '__main__':
    # Example usage when run as a script
    start_time = time.time()
    print("Using dummy log data for processing test...")
    dummy_log_lines = [
        '03-17 16:13:38.811  1702  2395 D WindowManager: printFreezingDisplayLogsopening app wtoken = AppWindowToken{9f4ef63 token=Token{a64f992 ActivityRecord{de9231d u0 com.tencent.qt.qtl/.activity.info.NewsDetailXmlActivity t761}}}, allDrawn= false, startingDisplayed =  false, startingMoved =  false, isRelaunching =  false',
        '[Sun Dec 04 04:47:44 2005] [error] mod_jk child workerEnv in error state 6\r',
        '2015-08-10 17:52:39,698 - INFO  [zkClient-EventThread-16] - Client attempting to establish new session to /10.10.34.13:2181',
        'This line has no level indicator',
        '03-17 16:13:38.819  1702  8671 D PowerManagerService: acquire lock=233570404, flags=0x1, tag="View Lock", name=com.android.systemui, ws=null, uid=10037, pid=2227',
        'Some text here with WARNING in it but not as a level WARN this is not a level',
        '2015-08-10 17:53:14,914 - INFO  - Closed socket connection for client /10.10.34.13:42917',
        '2015-08-10 18:12:34,001 - ERROR - Expiring session 0x14f05578bd8000f, timeout of 6000ms expired',
        '2015-08-10 18:12:34,004 - NOTICE - Processed session termination for sessionid: 0x14f05578bd8000f',
        'A fatal error FATAL occurred unexpectedly', # Added FATAL
        'SEVERE - A severe issue was detected', # Added SEVERE
        'AUDIT - An audit trail entry', # Added AUDIT
        'CRITICAL - A critical system failure', # Added CRITICAL
        'ALERT - System alert notification', # Added ALERT
        'EMERG - Emergency system shutdown imminent', # Added EMERG
        'FAILURE - Operation failed', # Added FAILURE
        'TRACE - Function call tracing' # Added TRACE
    ]
    end_time = time.time()
    print(f"Dummy data prepared in {end_time - start_time:.2f} seconds.")

    start_time = time.time()
    print("Extracting data from dummy log lines...")
    general_information, messages, messages_length, level_names, logs = extract_data_log_lines(dummy_log_lines)
    end_time = time.time()
    print(f"Extraction completed in {end_time - start_time:.2f} seconds.")
    print(f"Extracted {len(general_information)} records.")

    start_time = time.time()
    print("Creating DataFrame...")
    df = create_dataframe(general_information, level_names, messages)
    end_time = time.time()
    print(f"DataFrame created in {end_time - start_time:.2f} seconds.")

    print("\nFirst 5 rows of the created DataFrame:")
    print(df.head())
    print("\nDataFrame Info:")
    df.info()
    print("\nValue counts for severity_level_original:")
    print(df['severity_level_original'].value_counts())
