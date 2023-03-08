import pandas as pd
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
import os


class CarsDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        super(CarsDataset, self).__init__()
        self.root_dir = root_dir
        self.data = pd.read_csv(os.path.join(self.root_dir, csv_file))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        filename = self.data.iloc[index, 0]
        folder = self.data.iloc[index, 1]
        image_dir = folder + "/" +filename
        img_path = os.path.join(self.root_dir, image_dir)
        img = Image.open(img_path).convert('RGB')
        label = float(self.data.iloc[index, -1])

        if self.transform:
            img = self.transform(img)

        return img, label
