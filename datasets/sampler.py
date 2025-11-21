import os
import random


class TaskSampler:
    def __init__(self, base_folder, file_tuples, dataset, m_tasks, n_samples):
        self.base_folder = base_folder
        self.file_tuples = file_tuples
        self.dataset = dataset
        self.m_tasks = m_tasks
        self.n_samples = n_samples
        self.file_lengths = dataset.file_lengths
        self.file_index_map = self._create_index_map(dataset)

    def _create_index_map(self, dataset):
        index_map = {}
        current_file_index = 0

        c_dirs = os.listdir(dataset.root_dir)
        for c_dir in c_dirs:
            c_path = os.path.join(dataset.root_dir, c_dir)
            if not os.path.isdir(c_path):
                continue

            pkl_files = [f for f in os.listdir(c_path) if f.endswith('.pkl')]
            for pkl_file in pkl_files:
                key = (c_dir, pkl_file)
                index_map[key] = current_file_index
                current_file_index += 1

        return index_map

    def _get_file_index(self, folder_name, file_name):
        return self.file_index_map.get((folder_name, file_name), -1)

    def sample(self):
        if len(self.file_tuples) < self.m_tasks:
            sampled_files = random.sample(self.file_tuples, len(self.file_tuples))
        else:
            sampled_files = random.sample(self.file_tuples, self.m_tasks)

        sampled_data = []

        for folder_name, file_name in sampled_files:

            file_index = self._get_file_index(folder_name, file_name)
            L = self.file_lengths.get(file_index, 0)

            if L < self.n_samples or L == 0:
                continue

            max_start_index = L - self.n_samples
            if max_start_index < 0:
                continue

            start_index = random.randint(0, max_start_index)
            sample_indices = list(range(start_index, start_index + self.n_samples))
            sampled_data.append((folder_name, file_name, sample_indices))

        return sampled_data