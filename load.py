from torchvision.transforms import ToTensor
from torchvision import datasets
from torch.utils.data import DataLoader

training_data = datasets.MNIST(root="numbers",
                               download=True,
                               train=True,
                               transform=ToTensor,
                               target_transform=None)

testing_data = datasets.MNIST(root="numbers",
                              download=True,
                              train=False,
                              transform=ToTensor,
                              target_transform=None)

number_labels = training_data.classes

#into batches rather than each image
BATCH_SIZE = 32
train_dataLoader = DataLoader(dataset=training_data,
                              shuffle=True,
                              batch_size=BATCH_SIZE,
                              )

test_dataLoader = DataLoader(dataset=testing_data,
                             shuffle=False,
                             batch_size=BATCH_SIZE)
