import torch
import numpy as np
from torch.utils.data import Dataset

class PrecomputedDataset(Dataset):
    def __init__(self, precomputed_file, multi_label=True, with_index=False):
        if not multi_label:
            precomp = np.load(precomputed_file)
            self.features = precomp[:, :-1]
            self.labels = precomp[:, -1]
            self.indexes = None
            if with_index:
                raise ValueError("Index retrieval is not support with single label!")
        else:
            precomp = np.load(precomputed_file)
            self.features = precomp['features']
            self.labels = precomp['labels']
            self.indexes = torch.as_tensor(precomp['indexes']).long()
        self.with_index = with_index
        self.features, self.labels = torch.as_tensor(self.features), torch.as_tensor(self.labels).long()
        super(PrecomputedDataset, self).__init__()

    def __getitem__(self, index):
        if self.with_index:
            return self.features[index], self.labels[index], self.indexes[index]
        else:
            return self.features[index], self.labels[index]

    def __len__(self):
        return self.features.size(0)

if __name__ == '__main__':
    d = PrecomputedDataset('cifar_feat.test.npz')
    f, l = d[-3:]
    print(f.size(), l)