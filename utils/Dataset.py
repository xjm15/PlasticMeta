import os
import pickle
import torch
from torch.utils.data import Dataset
import numpy as np

class SequenceDataset(Dataset):
    def __init__(self, root_dir, length=5, transform=None):
        super(SequenceDataset, self).__init__()
        self.root_dir = root_dir
        self.length = length
        self.transform = transform
        self.data_list = []
        self.file_start_indices = []
        self.file_index = 0
        self.file_index_list = []
        self.file_lengths = {}

        c_dirs = os.listdir(root_dir)
        if not c_dirs:
            return

        current_index = 0
        for c_dir in c_dirs:
            c_path = os.path.join(root_dir, c_dir)
            if not os.path.isdir(c_path):
                continue

            pkl_files = [f for f in os.listdir(c_path) if f.endswith('.pkl')]
            if not pkl_files:
                continue

            for pkl_file in pkl_files:
                pkl_path = os.path.join(c_path, pkl_file)
                try:
                    with open(pkl_path, 'rb') as f:
                        data = pickle.load(f)
                except (pickle.UnpicklingError, EOFError, FileNotFoundError) as e:
                    continue

                if data:
                    file_length = len(data)
                    self.file_lengths[self.file_index] = file_length
                    self.file_start_indices.append(current_index)
                    self.data_list.extend(data)
                    current_index += len(data)
                    for sample in data:
                        self.file_index_list.append(self.file_index)
                self.file_index += 1
        self.file_index = len(self.file_start_indices)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if idx >= len(self.data_list):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data_list)}")

        file_index = max(i for i in range(len(self.file_start_indices)) if self.file_start_indices[i] <= idx)
        feature_dim = self.data_list[idx][0].shape[0]
        zero_padding = np.zeros((self.length - 1, feature_dim))
        sequence_buffer = []

        for i in range(max(0, idx - self.length + 1), idx + 1):
            sequence_buffer.append(self.data_list[i][0])

        if len(sequence_buffer) < self.length:
            padded_sequence = np.vstack([zero_padding[-(self.length - len(sequence_buffer)):], sequence_buffer])
        else:
            padded_sequence = np.vstack(sequence_buffer)

        if len(self.data_list[idx]) == 3:
            _, _, label = self.data_list[idx]
        elif len(self.data_list[idx]) == 5:
            _, _, label, _, _ = self.data_list[idx]
        elif len(self.data_list[idx]) == 4:
            _, _, label, _ = self.data_list[idx]
        else:
            raise ValueError(f"Unexpected sample format: {self.data_list[idx]}")

        feature = torch.tensor(padded_sequence, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            feature = self.transform(feature)

        return feature, label, file_index

    def get_data_from_file(self, folder_name, pkl_file_name, idx):
        folder_path = os.path.join(self.root_dir, folder_name)
        pkl_path = os.path.join(folder_path, pkl_file_name)

        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
        except (pickle.UnpicklingError, EOFError, FileNotFoundError) as e:
            return None, None

        if idx >= len(data):
            return None, None

        sample = data[idx]
        feature_dim = sample[0].shape[0]
        zero_padding = np.zeros((self.length - 1, feature_dim))
        sequence_buffer = []

        for i in range(max(0, idx - self.length + 1), idx + 1):
            sequence_buffer.append(data[i][0])

        if len(sequence_buffer) < self.length:
            padded_sequence = np.vstack([zero_padding[-(self.length - len(sequence_buffer)):], sequence_buffer])
        else:
            padded_sequence = np.vstack(sequence_buffer)

        if len(sample) == 3:
            _, _, label = sample
        elif len(sample) == 5:
            _, _, label, _, _ = sample
        elif len(sample) == 4:
            _, _, label, _ = sample
        else:
            raise ValueError(f"Unexpected sample format: {sample}")

        feature = torch.tensor(padded_sequence, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            feature = self.transform(feature)

        return feature, label

