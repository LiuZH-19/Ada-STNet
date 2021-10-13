import os
import h5py
import numpy as np
import pandas as pd

DATA_PATH = r'data/METR-LA'


class Scaler:
    def __init__(self, data):
        self.mean = np.mean(data.values)
        self.std = np.std(data.values)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean


def sensor_index():
    with open(os.path.join(DATA_PATH, 'sensor_graph/graph_sensor_ids.txt')) as f:
        sensor_ids = f.read().strip().split(',')
    sensor_idx = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_idx[sensor_id] = i
    return sensor_idx


def distant_matrix(n_neighbors):
    filename = os.path.join(DATA_PATH, 'stmetanet/adjacent_matrix_%d.h5' % n_neighbors)
    # if not os.path.exists(filename):
    sensor_idx = sensor_index()
    graph = pd.read_csv(os.path.join(DATA_PATH, 'sensor_graph/distances_la_2012.csv'),
                        dtype={'from': 'str', 'to': 'str'})

    n = len(sensor_idx)
    dist = np.zeros((n, n))
    dist[:] = np.inf

    cnt = 0
    for row in graph.values:
        if row[0] in sensor_idx and row[1] in sensor_idx:
            dist[sensor_idx[row[0]], sensor_idx[row[1]]] = row[2]
            cnt += 1
    # print('# roads', cnt)

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i, k] + dist[k, j] < dist[i, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]

    e_in, e_out = [], []
    for i in range(n):
        e_in.append(np.argsort(dist[:, i])[:n_neighbors + 1])
        e_out.append(np.argsort(dist[i, :])[:n_neighbors + 1])
    e_in = np.array(e_in, dtype=np.int32)
    e_out = np.array(e_out, dtype=np.int32)

    f = h5py.File(filename, 'w')
    f.create_dataset('dist', data=dist)
    f.create_dataset('e_in', data=e_in)
    f.create_dataset('e_out', data=e_out)
    f.flush()
    f.close()

    f = h5py.File(filename, 'r')
    adj_mat = np.array(f['dist'])
    e_in = np.array(f['e_in'])
    e_out = np.array(f['e_out'])
    f.close()
    return adj_mat, e_in, e_out


def sensor_location():
    sensor_idx = sensor_index()
    sensor_locs = np.loadtxt(os.path.join(DATA_PATH, 'sensor_graph/graph_sensor_locations.csv'), delimiter=',',
                             skiprows=1)

    n = len(sensor_idx)
    loc = np.zeros((n, 2))
    for i in range(n):
        loc[sensor_idx[str(int(sensor_locs[i, 1]))], :] = sensor_locs[i, 2:4]
    return loc


def fill_missing(data):
    data = data.copy()
    data[data < 1e-5] = float('nan')
    data = data.fillna(method='pad')
    data = data.fillna(method='bfill')
    return data


def get_geo_feature(n_neighbors: int):
    # get locations
    loc = sensor_location()
    loc = (loc - np.mean(loc, axis=0)) / np.std(loc, axis=0)

    # get distance matrix
    dist, e_in, e_out = distant_matrix(n_neighbors)

    # normalize distance matrix
    n = loc.shape[0]
    edge = np.zeros((n, n))
    for i in range(n):
        for j in range(n_neighbors):
            edge[e_in[i][j], i] = edge[i, e_out[i][j]] = 1
    dist[edge == 0] = np.inf

    values = dist.flatten()
    values = values[values != np.inf]
    dist_mean = np.mean(values)
    dist_std = np.std(values)
    dist = np.exp(-(dist - dist_mean) / dist_std)

    # merge features
    features = []
    for i in range(n):
        f = np.concatenate([loc[i], dist[e_in[i], i], dist[i, e_out[i]]])
        features.append(f)
    features = np.stack(features)
    return features, (dist, e_in, e_out)
