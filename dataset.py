import os
import random
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

def create_train_val_test_paths(partition_path='./PR_Project_Git/list_eval_partition.txt'):
    train_paths = []
    val_paths = []
    test_paths = []
    with open(partition_path) as f:
        samples = f.readlines()

    for sample in samples:
        sample = sample.strip().split()
        if int(sample[1]) == 0:
            train_paths.append(sample[0])
        elif int(sample[1]) == 1:
            val_paths.append(sample[0])
        else:
            test_paths.append(sample[0])

    return train_paths, val_paths, test_paths


def create_train_val_test_dct(paths, dataset_dir = './PR_project/img_align_celeba/', identity_path='./PR_Project_Git/identity_CelebA.txt', num_id=1000):
    train_dct = {}
    val_dct = {}
    test_dct = {}
    lst_identities = []
    with open(identity_path) as f:
        identities = f.readlines()

    for identity in identities:
        img_file, id_ = identity.rstrip().split()
        if img_file in paths['train'] and int(id_) <= num_id:
            if id_ in train_dct.keys():
                train_dct[id_].append(os.path.join(dataset_dir, img_file))
            else:
                train_dct[id_] = []
                train_dct[id_].append(os.path.join(dataset_dir, img_file))
        elif img_file in paths['val']:
            if id_ in val_dct.keys():
                val_dct[id_].append(os.path.join(dataset_dir, img_file))
            else:
                val_dct[id_] = []
                val_dct[id_].append(os.path.join(dataset_dir, img_file))

        elif img_file in paths['test']:
            if id_ in test_dct.keys():
                test_dct[id_].append(os.path.join(dataset_dir, img_file))
            else:
                test_dct[id_] = []
                test_dct[id_].append(os.path.join(dataset_dir, img_file))
        
    return train_dct, val_dct, test_dct



class MyDataset(Dataset):
    def __init__(self, dct, split='train'):
        super().__init__()
        self.dct = dct
        self.split = split
        self.data = []
        self._prepare_data()



    def _prepare_data(self):
        if self.split == 'train':
            for identity in self.dct['train'].keys():
                for img_path in self.dct['train'][identity]:
                    self.data.append((img_path, int(identity)))


        elif self.split == 'val':
            for identity in self.dct['val'].keys():
                for img_path in self.dct['val'][identity]:
                    self.data.append((img_path, int(identity)))

        else:
            for identity in self.dct['test'].keys():
                for img_path in self.dct['test'][identity]:
                    self.data.append((img_path, int(identity)))



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img2_path = random.choice(self.dct[self.split][str(label)])
        merge_img_path = img_path + '\t' + img2_path
        return merge_img_path, label
    




if __name__ == "__main__":
    partition_path = './PR_Project_Git/list_eval_partition.txt'
    train_paths, val_paths, test_paths = create_train_val_test_paths(partition_path=partition_path)
    paths = {
        'train': train_paths,
        'val': val_paths,
        'test': test_paths
    }
    print(len(train_paths), len(val_paths), len(test_paths))
    train_dct, val_dct, test_dct = create_train_val_test_dct(paths = paths)
    print(len(train_dct.keys()), len(val_dct.keys()), len(test_dct.keys()))
