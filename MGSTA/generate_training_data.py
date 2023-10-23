from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import h5py


def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False, scaler=None
):
    """
    Generate samples from
    :param df: T*N*2
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes, features = df.shape

    data = df
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):  # t is the index of the last observation.
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def generate_train_val_test(args):
    # 后面的sequence_length指序长度 x是-12 y是12 过去12步预测未来12步
    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    df = h5py.File(args.traffic_df_filename,'r')
    a = np.array(df['/data'][()], np.float32)  # date hour x y 2
    d, h, x, y, f = a.shape
    a = np.reshape(a, (d * h, x * y, -1))  # T*N*2
    # 0 is the latest observed sample.
    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    # Predict the next one hour
    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(
        a,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=args.dow,
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)
    # Write the data into npz file.
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.1)
    num_train = round(num_samples * 0.8)
    num_val = num_samples - num_test - num_train
    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    x_test, y_test = x[-num_test:], y[-num_test:]

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, f"{cat}.npz"),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )
    df.close()


def load_h5(filename, keywords):
    f = h5py.File(filename, 'r')
    data = []
    for name in keywords:
        data.append(np.array(f[name]))
    f.close()
    if len(data) == 1:
        return data[0]
    return data


def generate_x_y(data, input_len, output_len):
    mask = np.sum(data, axis=(1, 2)) > 5000

    n_timestamp, num_nodes, _ = data.shape

    flow = data

    feature, data, label = [], [], []
    for i in range(n_timestamp - input_len - output_len + 1):
        if mask[i + input_len: i + input_len + output_len].sum() != output_len:
            continue

        data.append(flow[i: i + input_len])
        label.append(flow[i + input_len: i + input_len + output_len])

    data = np.stack(data)  # [B, T, N, D]
    label = np.stack(label)  # [B, T, N, D]
    return data, label


def generate_data(args):
    input_len = args.seq_length_x
    output_len = args.seq_length_y
    data = load_h5(args.traffic_df_filename, ['data'])
    days, hours, rows, cols, _ = data.shape

    data = np.reshape(data, (days * hours, rows * cols, -1))

    n_timestamp = data.shape[0]
    num_train = int(n_timestamp * 0.7)
    num_test = int(n_timestamp * 0.2)
    num_eval = n_timestamp - num_train - num_test

    train = data[:num_train]
    valid = data[num_train: num_train + num_eval]
    test = data[-num_test:]

    x_train, y_train = generate_x_y(data=train, input_len=input_len, output_len=output_len)
    x_val, y_val = generate_x_y(data=valid, input_len=input_len, output_len=output_len)
    x_test, y_test = generate_x_y(data=test, input_len=input_len, output_len=output_len)

    x_offsets = np.sort(np.concatenate((np.arange(-(input_len - 1), 1, 1),)))
    # Predict the next one hour
    y_offsets = np.sort(np.arange(args.y_start, (output_len + 1), 1))

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, f"{cat}.npz"),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/BJ-2-7", help="Output directory.")
    parser.add_argument("--traffic_df_filename", type=str, default="data/BJ_FLOW.h5", help="Raw traffic readings.", )
    parser.add_argument("--seq_length_x", type=int, default=12, help="Sequence Length.", )
    parser.add_argument("--seq_length_y", type=int, default=3, help="Sequence Length.", )
    parser.add_argument("--y_start", type=int, default=1, help="Y pred start", )
    parser.add_argument("--dow", action='store_true', )

    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        reply = str(input(f'{args.output_dir} exists. Do you want to overwrite it? (y/n)')).lower().strip()
        if reply[0] != 'y': exit
    else:
        os.makedirs(args.output_dir)
    generate_data(args)
