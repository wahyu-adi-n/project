from data.cars_dataloader import CarsDataModule
from utils.utils import read_cfg

cfg = read_cfg('configs/base_config.yaml')

dataset = CarsDataModule(cfg)
train_dl = dataset.train_dataloader()
print(len(dataset.train_set))
print(len(train_dl))

# count=0
# for imgs, labels in train_dl:
#     count +=1
#     print('Total bs')
#     print(len(labels))

# print('Total perulangan')
# print(count)
