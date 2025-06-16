from urllib3 import PoolManager
import time

FILE_LIST = [
    # "https://raw.githubusercontent.com/logpai/loghub/master/Andriod/Android_2k.log",
    "https://raw.githubusercontent.com/logpai/loghub/refs/heads/master/Android/Android_2k.log",
    "https://raw.githubusercontent.com/logpai/loghub/refs/heads/master/Apache/Apache_2k.log",
    "https://raw.githubusercontent.com/logpai/loghub/refs/heads/master/BGL/BGL_2k.log",
    "https://raw.githubusercontent.com/logpai/loghub/refs/heads/master/HDFS/HDFS_2k.log",
    "https://raw.githubusercontent.com/logpai/loghub/refs/heads/master/Hadoop/Hadoop_2k.log",
    "https://raw.githubusercontent.com/logpai/loghub/refs/heads/master/OpenStack/OpenStack_2k.log",
    "https://raw.githubusercontent.com/logpai/loghub/refs/heads/master/Zookeeper/Zookeeper_2k.log",
]

def acquire_data(file_list):
    """
    Acquires log data from a list of URLs.

    Args:
        file_list (list): A list of URLs to fetch log data from.

    Returns:
        list: A list of log lines as strings.
    """
    log_lines = []
    http = PoolManager()
    for file_url in file_list:
        try:
            print(f"Acquiring data from {file_url}...")
            resp = http.request("GET", file_url).data.decode('utf-8', errors='ignore').split('\n') # Use utf-8 and ignore errors
            log_lines.extend([line for line in resp if line.strip()]) # Extend and filter empty lines
        except Exception as e:
            print(f"Error acquiring data from {file_url}: {e}")
    return log_lines

if __name__ == '__main__':
    # Example usage when run as a script
    start_time = time.time()
    log_lines = acquire_data(FILE_LIST)
    end_time = time.time()
    print(f"\nAcquired {len(log_lines)} non-empty log lines in {end_time - start_time:.2f} seconds.")
    print("First 10 log lines:")
    for line in log_lines[:10]:
        print(line)
