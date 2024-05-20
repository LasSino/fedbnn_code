'''
This module exports the base datasets used in the project.

Each dataset is exported as a function with corresponding name.
The function has parameters to resize samples into certain sizes.
For character recognizing datasets, a default size of 28x28 is used following
  MNIST dataset.
The function returns a `dict` with two entries:
  - "train" : an `ImageDataset` pointing to the train set.
  - "test" : a `ImageDataset` pointing to the test set.

Datasets including:
  - MNIST
  - CIFAR10
  - CIFAR10_gray
  - CIFAR100
  - CIFAR100_gray
  - EMNIST
  - EMNIST_digits
  - SVHN
  - SVHN_gray
  - USPS
  - Food101
  - Flowers102
  - STL10
  - Stanford Cars
  - SUN397
  - Places365
  - Tiny-ImageNet
'''

from torchvision import datasets as tvds, transforms as tvtransform
from .dataset_utils import ImageDataset,create_sampled_dataset
import random

def MNIST(padding_to_32=False):
    train=tvds.MNIST(root="./data/mnist",train=True,download=True)
    test=tvds.MNIST(root="./data/mnist",train=False,download=True)
    
    augmentation=None

    if padding_to_32:
        augmentation=tvtransform.Pad(2,padding_mode="edge")

    train=ImageDataset(train,augmentation,True)
    train.classes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    test=ImageDataset(test,augmentation,True)
    test.classes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    return {"train":train,"test":test}

def CIFAR10_gray(resize_to=28):
    train=tvds.CIFAR10(root="./data/cifar10",train=True,download=True)
    test=tvds.CIFAR10(root="./data/cifar10",train=False,download=True)

    if resize_to>32:
        tr_augmentation = tvtransform.Compose([
            tvtransform.Resize(resize_to),
            tvtransform.RandomHorizontalFlip(),
            tvtransform.Grayscale(num_output_channels=1),
        ])
        test_augmentation = tvtransform.Compose([
            tvtransform.Resize(resize_to),
            tvtransform.Grayscale(num_output_channels=1),
        ])
    elif resize_to<32:
        tr_augmentation = tvtransform.Compose([
            tvtransform.RandomCrop(resize_to),
            tvtransform.RandomHorizontalFlip(),
            tvtransform.Grayscale(num_output_channels=1),
        ])
        test_augmentation = tvtransform.Compose([
            tvtransform.CenterCrop(resize_to),
            tvtransform.Grayscale(num_output_channels=1),
        ])
    else:
        tr_augmentation = tvtransform.Compose([
            tvtransform.RandomHorizontalFlip(),
            tvtransform.Grayscale(num_output_channels=1),
        ])
        test_augmentation = tvtransform.Compose([
            tvtransform.Grayscale(num_output_channels=1),
        ])

    train=ImageDataset(train,tr_augmentation,True)
    test=ImageDataset(test,test_augmentation,True)

    return {"train":train,"test":test}

def CIFAR10(resize_to=28):
    train=tvds.CIFAR10(root="./data/cifar10",train=True,download=True)
    test=tvds.CIFAR10(root="./data/cifar10",train=False,download=True)

    if resize_to>32:
        tr_augmentation = tvtransform.Compose([
            tvtransform.Resize(resize_to),
            tvtransform.RandomHorizontalFlip(),
            tvtransform.ColorJitter(brightness=(0.5,1.5), contrast=(0.5,1.5)),
        ])
        test_augmentation = tvtransform.Resize(resize_to)

    elif resize_to<32:
        tr_augmentation = tvtransform.Compose([
            tvtransform.RandomCrop(resize_to),
            tvtransform.RandomHorizontalFlip(),
            tvtransform.ColorJitter(brightness=(0.5,1.5), contrast=(0.5,1.5)),
        ])
        test_augmentation = tvtransform.CenterCrop(resize_to)

    else:
        tr_augmentation=tvtransform.Compose([
            tvtransform.RandomHorizontalFlip(),
            tvtransform.ColorJitter(brightness=(0.5,1.5), contrast=(0.5,1.5)),
        ])
        test_augmentation=None

    train=ImageDataset(train,tr_augmentation,False)
    test=ImageDataset(test,test_augmentation,False)

    return {"train":train,"test":test}

def CIFAR100_gray(resize_to=28):
    train=tvds.CIFAR100(root="./data/cifar100",train=True,download=True)
    test=tvds.CIFAR100(root="./data/cifar100",train=False,download=True)

    if resize_to>32:
        tr_augmentation = tvtransform.Compose([
            tvtransform.Resize(resize_to),
            tvtransform.RandomHorizontalFlip(),
            tvtransform.Grayscale(num_output_channels=1),
        ])
        test_augmentation = tvtransform.Compose([
            tvtransform.Resize(resize_to),
            tvtransform.Grayscale(num_output_channels=1),
        ])
    elif resize_to<32:
        tr_augmentation = tvtransform.Compose([
            tvtransform.RandomCrop(resize_to),
            tvtransform.RandomHorizontalFlip(),
            tvtransform.Grayscale(num_output_channels=1),
        ])
        test_augmentation = tvtransform.Compose([
            tvtransform.CenterCrop(resize_to),
            tvtransform.Grayscale(num_output_channels=1),
        ])
    else:
        tr_augmentation = tvtransform.Compose([
            tvtransform.RandomHorizontalFlip(),
            tvtransform.Grayscale(num_output_channels=1),
        ])
        test_augmentation = tvtransform.Compose([
            tvtransform.Grayscale(num_output_channels=1),
        ])

    train=ImageDataset(train,tr_augmentation,True)
    test=ImageDataset(test,test_augmentation,True)

    return {"train":train,"test":test}

def CIFAR100(resize_to=28):
    train=tvds.CIFAR100(root="./data/cifar100",train=True,download=True)
    test=tvds.CIFAR100(root="./data/cifar100",train=False,download=True)

    if resize_to>32:
        tr_augmentation = tvtransform.Compose([
            tvtransform.Resize(resize_to),
            tvtransform.RandomHorizontalFlip(),
            tvtransform.ColorJitter(brightness=(0.5,1.5), contrast=(0.5,1.5)),
        ])
        test_augmentation = tvtransform.Resize(resize_to)

    elif resize_to<32:
        tr_augmentation = tvtransform.Compose([
            tvtransform.RandomCrop(resize_to),
            tvtransform.RandomHorizontalFlip(),
            tvtransform.ColorJitter(brightness=(0.5,1.5), contrast=(0.5,1.5)),
        ])
        test_augmentation = tvtransform.CenterCrop(resize_to)

    else:
        tr_augmentation=tvtransform.Compose([
            tvtransform.RandomHorizontalFlip(),
            tvtransform.ColorJitter(brightness=(0.5,1.5), contrast=(0.5,1.5)),
        ])
        test_augmentation=None

    train=ImageDataset(train,tr_augmentation,False)
    test=ImageDataset(test,test_augmentation,False)

    return {"train":train,"test":test}

def EMNIST_digits(padding_to_32=False):
    train=tvds.EMNIST(root="./data/emnist",split="digits",train=True,download=True)
    test=tvds.EMNIST(root="./data/emnist",split="digits",train=False,download=True)
    
    augmentation=None

    if padding_to_32:
        augmentation=tvtransform.Pad(2,padding_mode="edge")

    train=ImageDataset(train,augmentation,True)
    test=ImageDataset(test,augmentation,True)

    return {"train":train,"test":test}

def EMNIST(padding_to_32=False):
    train=tvds.EMNIST(root="./data/emnist",split="balanced",train=True,download=True)
    test=tvds.EMNIST(root="./data/emnist",split="balanced",train=False,download=True)
    
    augmentation=None

    if padding_to_32:
        augmentation=tvtransform.Pad(2,padding_mode="edge")

    train=ImageDataset(train,augmentation,True)
    test=ImageDataset(test,augmentation,True)

    return {"train":train,"test":test}

def SVHN_gray(crop_to_28=True):
    train=tvds.SVHN(root="./data/svhn",split="train",download=True)
    test=tvds.SVHN(root="./data/svhn",split="test",download=True)

    if crop_to_28:
        tr_augmentation = tvtransform.Compose([
            tvtransform.RandomCrop(28),
            tvtransform.RandomHorizontalFlip(),
            tvtransform.Grayscale(num_output_channels=1),
        ])
        test_augmentation = tvtransform.Compose([
            tvtransform.CenterCrop(28),
            tvtransform.Grayscale(num_output_channels=1),
        ])
    else:
        tr_augmentation = tvtransform.Compose([
            tvtransform.RandomHorizontalFlip(),
            tvtransform.Grayscale(num_output_channels=1),
        ])
        test_augmentation = tvtransform.Compose([
            tvtransform.Grayscale(num_output_channels=1),
        ])

    train=ImageDataset(train,tr_augmentation,True)
    test=ImageDataset(test,test_augmentation,True)

    return {"train":train,"test":test}

def SVHN(crop_to_28=True):
    train=tvds.SVHN(root="./data/svhn",split="train",download=True)
    test=tvds.SVHN(root="./data/svhn",split="test",download=True)

    if crop_to_28:
        tr_augmentation = tvtransform.Compose([
            tvtransform.RandomCrop(28),
            tvtransform.RandomHorizontalFlip(),
            tvtransform.ColorJitter(brightness=(0.5,1.5), contrast=(0.5,1.5)),
        ])
        test_augmentation = tvtransform.CenterCrop(28)

    else:
        tr_augmentation=tvtransform.Compose([
            tvtransform.RandomHorizontalFlip(),
            tvtransform.ColorJitter(brightness=(0.5,1.5), contrast=(0.5,1.5)),
        ])
        test_augmentation=None

    train=ImageDataset(train,tr_augmentation,False)
    test=ImageDataset(test,test_augmentation,False)

    return {"train":train,"test":test}

def USPS(resize_to=28):
    train=tvds.USPS(root="./data/usps",train=True,download=True)
    test=tvds.USPS(root="./data/usps",train=False,download=True)

    if resize_to==16:
        augmentation=None
    else:
        augmentation=tvtransform.Resize(resize_to)

    train=ImageDataset(train,augmentation,True)
    train.classes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    test=ImageDataset(test,augmentation,True)
    test.classes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    return {"train":train,"test":test}

def PLACES365(resize_to=256):
    if resize_to<=256:
        train=tvds.Places365(root="./data/places365",split="train-standard",
                    small=True,download=True)
        test=tvds.Places365(root="./data/places365",split="val",
                    small=True,download=True)                    
        augmentation=None if resize_to==256 else tvtransform.Resize(resize_to)
    else:
        train=tvds.Places365(root="./data/places365",split="train-standard",
                    small=False,download=True)
        test=tvds.Places365(root="./data/places365",split="val",
                    small=False,download=True)                    
        augmentation=tvtransform.Resize(resize_to)

    train=ImageDataset(train,augmentation,False)
    test=ImageDataset(test,augmentation,False)

    return {"train":train,"test":test}

def FOOD101(resize_to=256):
    train=tvds.Food101(root="./data/food101",split="train",download=True)
    test=tvds.Food101(root="./data/food101",split="test",download=True)

    augmentation=tvtransform.Compose([
                                tvtransform.Resize(resize_to),
                                tvtransform.RandomCrop(resize_to)
                                ])

    train=ImageDataset(train,augmentation,False)
    test=ImageDataset(test,augmentation,False)

    return {"train":train,"test":test}

def FLOWERS102(resize_to=256):
    # The flowers102 comes with testset larger than trainset. 
    # We use them reversely.
    train=tvds.Flowers102(root="./data/flowers102",split="test",download=True)
    test=tvds.Flowers102(root="./data/flowers102",split="train",download=True)

    augmentation=tvtransform.Compose([
                                tvtransform.Resize(resize_to),
                                tvtransform.RandomCrop(resize_to)
                                ])

    train=ImageDataset(train,augmentation,False)
    train.classes=list(range(110))
    test=ImageDataset(test,augmentation,False)
    test.classes=list(range(110))

    return {"train":train,"test":test}

def STANFORD_CARS(resize_to=256):
    train=tvds.StanfordCars(root="./data/stanford_cars",
                split="train",
                download=True)
    test=tvds.StanfordCars(root="./data/stanford_cars",
                split="test",
                download=True)

    augmentation=tvtransform.Compose([
                                tvtransform.Resize(resize_to),
                                tvtransform.RandomCrop(resize_to)
                                ])

    train=ImageDataset(train,augmentation,False)
    test=ImageDataset(test,augmentation,False)

    return {"train":train,"test":test}

def STL10(resize_to=96):
    train=tvds.STL10(root="./data/stl10",split="train",download=True)
    test=tvds.STL10(root="./data/stl10",split="test",download=True)

    augmentation=None if resize_to==96 else tvtransform.Resize(resize_to)

    train=ImageDataset(train,augmentation,False)
    test=ImageDataset(test,augmentation,False)

    return {"train":train,"test":test}

def SUN397(resize_to=256):
    dataset=tvds.SUN397(root="./data/sun397",download=True)

    augmentation=tvtransform.Compose([
                                tvtransform.Resize(resize_to),
                                tvtransform.RandomCrop(resize_to)
                                ])
    
    whole_dataset=ImageDataset(dataset,augmentation,False)

    whole_dataset_size=len(whole_dataset)
    whole_dataset_idx=list(range(whole_dataset_size))
    random.shuffle(whole_dataset_idx)

    trainset_size=int(whole_dataset_size*0.7)
    train_idx=whole_dataset_idx[0:trainset_size]
    test_idx=whole_dataset_idx[trainset_size:]
    
    train=create_sampled_dataset(whole_dataset, train_idx,False)

    trainset_classes=train.classes
    test_idx_filtered=[index for index in test_idx \
                        if whole_dataset.classes[whole_dataset[index][1]] \
                            in trainset_classes]
    test=create_sampled_dataset(whole_dataset, test_idx_filtered,False)

    return {"train":train,"test":test}

def TINY_IMAGENET(resize_to=32):
    '''
    This is the Tiny-ImageNet dataset. It is required to manually download the
    dataset, and do some directory organization, following
    [this tutorial](https://www.cnblogs.com/liuyangcode/p/14689893.html).
    '''
    train=tvds.ImageFolder(root='./data/tiny-imagenet-200/train/')
    test=tvds.ImageFolder(root='./data/tiny-imagenet-200/val/')

    if resize_to==64:
        tr_augmentation = None
        test_augmentation = None
    else:
        tr_augmentation = tvtransform.Resize(resize_to)
        test_augmentation = tvtransform.Resize(resize_to)

    train=ImageDataset(train,tr_augmentation,False)
    test=ImageDataset(test,test_augmentation,False)

    return {"train":train,"test":test}