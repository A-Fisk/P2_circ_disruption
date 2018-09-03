# Script to define functions for analysing sleep

def sleep_process(data, window=4):
    """
    Function to score the PIR based activity data as sleep

    :param data: pandas dataframe to be sleep scored
    :param window: length of window, default =4 which is 40 seconds
    :return: dataframe of sleep
    """
    # if >40 seconds (4 rows) of inactivity score as 1

    rolling_sum_data = data.rolling(window).sum()

    scored_data = rolling_sum_data == 0

    return scored_data.astype(int)
