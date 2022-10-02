import numpy as np

"""For evaluation metrics"""


def train_test_divide(data_x, data_x_hat, data_t, data_t_hat, train_rate=0.8):
    """Divide train and test data for both original and synthetic data.
    Args:
        - data_x: original data
        - data_x_hat: generated data
        - data_t: original time
        - data_t_hat: generated time
        - train_rate: ratio of training data from the original data"""
    # Divide train/test index (original data)
    # permute the indexies and split the first 0.8 percent to be training data
    no = len(data_x)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no * train_rate)]
    test_idx = idx[int(no * train_rate):]

    train_x = [data_x[i] for i in train_idx]
    test_x = [data_x[i] for i in test_idx]
    train_t = [data_t[i] for i in train_idx]
    test_t = [data_t[i] for i in test_idx]

    # Divide train/test index (synthetic data)
    # Repeat it again for the synthetic data
    no = len(data_x_hat)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no * train_rate)]
    test_idx = idx[int(no * train_rate):]

    train_x_hat = [data_x_hat[i] for i in train_idx]
    test_x_hat = [data_x_hat[i] for i in test_idx]
    train_t_hat = [data_t_hat[i] for i in train_idx]
    test_t_hat = [data_t_hat[i] for i in test_idx]

    return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat


def batch_generator(data, time, batch_size):
    """Mini-batch generator. Slice the original data to the size a batch.

    Args:
        - data: time-series data
        - time: time series length for each sample
        - batch_size: the number of samples in each batch

    Returns:
        - X_mb: time-series data in each batch (bs, seq_len, dim)
        - T_mb: time series length of samples in that batch (bs, len of the sample)"""
    # randomly select a batch of idx
    no = len(data)
    idx = np.random.permutation(no)
    train_idx = idx[:batch_size]

    # picked the selected samples and their corresponding series length
    X_mb = list(data[i] for i in train_idx)
    T_mb = list(time[i] for i in train_idx)

    return X_mb, T_mb


def extract_time(data):
    """Returns Maximum sequence length and each sequence length.

    Args:
    - data: original data (no, seq_len, dim)

    Returns:
    - time: a list for each sequence length
    - max_seq_len: maximum sequence length
    """
    time = list()
    max_seq_len = 0
    for i in range(len(data)):
        max_seq_len = max(max_seq_len, len(data[i][:, 0]))
        time.append(len(data[i][:, 0]))
    return time, max_seq_len


def extract_factors(n):
    if (n == 0) or (n == 1):
        return [n]

    factor_list = []
    i = 2
    while i < n:
        if n % i == 0:
            factor_list.append(i)
        i += 1

    return factor_list