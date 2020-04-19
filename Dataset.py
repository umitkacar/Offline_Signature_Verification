from torch.utils.data import Dataset
import pickle
from utils import convert_to_image_tensor, invert_image_path

class TrainDataset(Dataset):

    def __init__(self):
        with open('./Data/train_index.pkl', 'rb') as train_index_file:
            self.pairs = pickle.load(train_index_file)

    def __getitem__(self, index):
        item = self.pairs[index]
        X = convert_to_image_tensor(invert_image_path(item[0]))
        Y = convert_to_image_tensor(invert_image_path(item[1]))
        return [X, Y, item[2]]

    def __len__(self):
        return len(self.pairs)

class TestDataset(Dataset):

    def __init__(self):
        with open('./Data/test_index.pkl', 'rb') as test_index_file:
            self.pairs = pickle.load(test_index_file)

    def __getitem__(self, index):
        item = self.pairs[index]
        X = convert_to_image_tensor(invert_image_path(item[0]))
        Y = convert_to_image_tensor(invert_image_path(item[1]))
        return [X, Y, item[2]]

    def __len__(self):
        return len(self.pairs)
