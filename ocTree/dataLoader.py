import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TrajDataset(Dataset):
    def __init__(self, traj_data, dis_matrix, edgs_adj, phase, sample_num):
        self.traj_data = traj_data
        self.dis_matrix = dis_matrix
        self.phase = phase
        self.sample_num = sample_num
        self.edgs_adj = edgs_adj
        self.sorted_indices = np.argsort(dis_matrix, axis=1)

    def __len__(self):
        return len(self.traj_data)

    def __getitem__(self, idx):
        traj_list = []
        dis_list = []

        top_indices = np.argsort(-self.edgs_adj[idx])[1:11]
        id_most_sim = np.random.choice(top_indices)
        sim_traj = self.traj_data[id_most_sim]

        if self.phase == "train":
            id_list = self.sorted_indices[idx]

            sample_index = []
            sample_index.extend(id_list[: self.sample_num // 2])
            sample_index.extend(id_list[len(id_list) - self.sample_num // 2 :])

            for i in sample_index:
                traj_list.append(self.traj_data[i])
                dis_list.append(self.dis_matrix[sample_index[0], i])

        elif self.phase == "val" or "test":
            traj_list.append(self.traj_data[idx])
            dis_list = None
            sample_index = None

        return traj_list, dis_list, idx, sample_index, sim_traj


class TrajTokenDataLoader:
    def __init__(self, traj_data, dis_matrix, edgs_adj, phase, train_batch_size, eval_batch_size, sample_num, data_features, num_workers, x_range, y_range, z_range, treeid_list_list, treeid_range):
        self.traj_data = traj_data
        self.dis_matrix = dis_matrix / dis_matrix.max()
        self.edgs_adj = edgs_adj
        self.phase = "val" if phase in ["val", "embed"] else phase
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.sample_num = sample_num
        self.data_features = data_features
        self.num_workers = num_workers

        tr = treeid_range
        self._norm_min = torch.tensor(
            [x_range[0], y_range[0], z_range[0], tr[0], tr[2], tr[4], tr[6]],
            dtype=torch.float32,
        )
        self._norm_range = torch.tensor(
            [
                x_range[1] - x_range[0],
                y_range[1] - y_range[0],
                z_range[1] - z_range[0],
                tr[1] - tr[0],
                tr[3] - tr[2],
                tr[5] - tr[4],
                tr[7] - tr[6],
            ],
            dtype=torch.float32,
        )

    def get_data_loader(self):
        self.dataset = TrajDataset(
            traj_data=self.traj_data,
            dis_matrix=self.dis_matrix,
            edgs_adj=self.edgs_adj,
            phase=self.phase,
            sample_num=self.sample_num,
        )

        if self.phase == "train":
            is_shuffle = False
            batch_size = self.train_batch_size
        elif self.phase == "val" or "test":
            is_shuffle = False
            batch_size = self.eval_batch_size

        data_loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=is_shuffle,
            num_workers=self.num_workers,
            collate_fn=self._collate_func,
        )

        return data_loader

    def _collate_func(self, samples):
        traj_list_list, dis_list_list, idx, sample_index, sim_traj = map(list, zip(*samples))
        traj_feature_list_list = self._prepare(traj_list_list)
        sim_traj_norm = self._prepare(sim_traj)
        return traj_feature_list_list, dis_list_list, idx, sample_index, sim_traj_norm

    def _prepare(self, traj_l_l):
        result = []
        for traj_l in traj_l_l:
            arr = np.stack(traj_l)  # (N, 200, 7)
            t = torch.tensor(arr, dtype=torch.float32)
            t = (t - self._norm_min) / self._norm_range
            result.append(list(t))
        return result
