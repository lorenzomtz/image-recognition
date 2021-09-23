from PIL import Image
from torch.utils import data

from torch.utils.data import Dataset, DataLoader, dataset
from torchvision import transforms

import csv

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']

class CustomDataset(Dataset):
    def __init__(self, dataset_path):

        # intialize list of (image, label) pairs
        self.labels = []

        # concatenate path to csv label file
        # TODO: find correct path
        labels_path = dataset_path + '/labels.csv'

        # create image to tensor transformation
        trans = transforms.Compose([transforms.ToTensor()])

        # open csv file
        with open(labels_path, newline='') as csvfile:
            # create reader and skip header file
            reader = csv.reader(csvfile)
            next(reader)
            
            # loop through file
            for row in reader:
                # retrieve label from list, use int index
                label = LABEL_NAMES.index(row[1])
                
                # concatenate path to image file 
                img_path = dataset_path + '/' + row[0]
                
                # open image
                image = Image.open(img_path)

                # transform image into tensor and add pair into list
                self.labels.append((trans(image), label))

    def __len__(self):
        # return length of (image, label) list
        return len(self.labels)

    def __getitem__(self, idx):
        # return (image, label) pair at specified index
        return self.labels[idx]


def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = CustomDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)

def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()