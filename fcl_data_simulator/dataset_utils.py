import torch
import random
import torchvision.transforms as tvtransform

'''
This file exports some utility functions and classes used to process the datasets.
'''

def _get_mean_std(dataset, batch_size):
    '''
    This function returns the standard deviation and mean, used for
        normalization.
    '''
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                shuffle=False)
    total = 0
    mean = 0.
    var = 0.

    for examples, _ in loader:
        # Rearrange batch to be the shape of [B, C, W * H]
        examples = examples.view(examples.size(0), examples.size(1), -1)
        # Update total number of images
        total += examples.size(0)
        # Compute mean and var here
        mean += examples.mean(2).sum(0)
        var += examples.var(2).sum(0)

    # Final step
    mean /= total
    var /= total

    return mean.tolist(), torch.sqrt(var).tolist()

class ImageDataset(torch.utils.data.Dataset):
    '''
    ImageDataset serves as a wrapper for some simple image datasets, mainly does
     some preprocessings for the sample.

    When retrieving sample from this dataset, it fetches the original sample from
      the underlying dataset, and preprocesses it before returning the sample.
    '''
    def __init__(self,underlying_dataset,augmentation=None,normalization=False):
        self.underlying_dataset=underlying_dataset
        # Copy some necessary members from the underlying dataset.
        if hasattr(underlying_dataset,"classes"):
            self.classes=underlying_dataset.classes
        if hasattr(underlying_dataset,"targets"):
            self.targets=underlying_dataset.targets

        # Set the augmentation and normalizations.
        self.augmentation=augmentation
        self.totensor=tvtransform.ToTensor()

        self.normalization=False
        if normalization:
            self._set_normalization()

    def _set_normalization(self):
        mean,std=_get_mean_std(self,50)
        self.normalization=True
        self.normalize=tvtransform.Normalize(mean,std)

    def __len__(self) -> int:
        return len(self.underlying_dataset)

    def __getitem__(self,idx):
        example,target=self.underlying_dataset[idx]
        if self.augmentation!=None:
            example=self.augmentation(example)
        example=self.totensor(example)
        if self.normalization:
            example=self.normalize(example)
        return example,target

class SampledDataset(torch.utils.data.Dataset):
    '''
    SampledDataset wraps a dataset with the sampled indices in the dataset.

    If reassign_label is set to true, SampledDataset will find the classes of
     the sampled examples, and assign the examples with a new label.
    '''
    def __init__(self,underlying_dataset,idx,reassign_label=False):

        self.underlying_dataset=underlying_dataset
        self.idx=[]
        self.idx.extend(idx) # Directly assigning the idx is a shallow copy.
        self.reassign_label=reassign_label

        if self.reassign_label:
            assert hasattr(underlying_dataset,"classes"),\
                "Reassigning labels requires the dataset to have member \
                    variable `classes`."
            # If reassign_label is true, we first find the remaining classes.
            result_class_idx=[] # The remaining unique class indices (`y`).
            for index in idx:
                _,y = self.underlying_dataset[index]
                if y not in result_class_idx:
                    result_class_idx.append(y)
            result_class_idx.sort()

            # Get the remaining class names with the class indices.
            self.classes=[self.underlying_dataset.classes[index]\
                            for index in result_class_idx]

            self.label_mapping=[]
            for class_name in underlying_dataset.classes:
                if class_name in self.classes:
                    self.label_mapping.append(self.classes.index(class_name))
                else:
                    self.label_mapping.append(-1)
        else:
            if hasattr(underlying_dataset,"classes"):
                self.classes=underlying_dataset.classes


    def __len__(self):
        return len(self.idx)

    def __getitem__(self,index):
        original_index=self.idx[index]
        x,original_y=self.underlying_dataset[original_index]
        if self.reassign_label:
            y=self.label_mapping[original_y]
            if y==-1:
                raise Exception("Unexpectedly accessed an example \
                                    with unassigned class label")
            else:
                return (x,y)
        else:
            return (x,original_y)

    def sampleFrom(self,idx,reassign_label=True):
        '''
        If we do sample on SampledDataset, we construct a new SampledDataset,
         rather than wrap it with another SampledDataset.
        '''
        original_idx=[]
        for index in idx:
            original_idx.append(self.idx[index])
        return SampledDataset(self.underlying_dataset,
                original_idx,reassign_label)

class CombinedDataset(torch.utils.data.Dataset):
    '''
    The CombinedDataset is a combination of different datasets.
    It maps the input index to respective datasets, and also does label
     reassignment if necessary.

    When two datasets have classes with same name, `aggregate_classes` can be
     used to specify whether to view them as a same class. If so, then the class
     will be aggregated, and the label will be reassigned.
    '''
    def __init__(self,underlying_datasets,aggregate_classes=False):
        self.underlying_datasets=underlying_datasets
        self.num_of_datasets=len(self.underlying_datasets)
        self.dataset_sizes=[]
        self.n_classes_of_each_set=[]
        self.aggregate_classes=aggregate_classes

        classes=[]
        for dataset in self.underlying_datasets:
            self.dataset_sizes.append(len(dataset))
            classes+=dataset.classes
            self.n_classes_of_each_set.append(len(dataset.classes))

        self.total_size=sum(self.dataset_sizes)

        # Get the index limit for each of the dataset.
        self.index_limit=[0]
        # We have to reassign the class labels, therefore, we find the label
        # offset for each dataset. And the converted label is
        # dataset_label_offset+original_label.
        self.class_label_offset=[0]
        for dataset_index in range(self.num_of_datasets):
            self.index_limit.append(self.index_limit[-1]+\
                                        self.dataset_sizes[dataset_index])
            self.class_label_offset.append(
                                        self.class_label_offset[-1]+\
                                        self.n_classes_of_each_set[dataset_index]
                                    )

        if self.aggregate_classes:
            # First find the unique classes in the classes list.
            self.classes=[]
            for class_name in classes:
                if class_name not in self.classes:
                    self.classes.append(class_name)

            # Now create a mapping for each of the datasets.
            self.mappings=[]
            for dataset in self.underlying_datasets:
                mapping=[]
                for class_name in dataset.classes:
                    if class_name in self.classes:
                        mapping.append(self.classes.index(class_name))
                    else:
                        raise Exception("Unexpected class occured.")
                self.mappings.append(mapping)
        else:
            self.classes=classes

    def __len__(self):
        return self.total_size

    def __getitem__(self,index):
        # First decide which dataset the index belongs to.
        if index>=self.total_size:
            raise IndexError()
        for dataset_index in range(self.num_of_datasets):
            if (index>=self.index_limit[dataset_index]) \
                and (index<self.index_limit[dataset_index+1]):
                break

        # Then decide the index in the selected dataset.
        internal_index=index-self.index_limit[dataset_index]
        x,original_y=self.underlying_datasets[dataset_index][internal_index]

        # Convert the original_y.
        if self.aggregate_classes:
            # If classes are aggregated, then use the mapping to find the
            # converted y.
            y=self.mappings[dataset_index][original_y]
        else:
            # The converted y is the dataset label offset plus original y.
            y=original_y+self.class_label_offset[dataset_index]

        return (x,y)

def get_index_by_class(dataset):
    '''Get the indices of each class in the dataset.

    Arguments:
        dataset -- A dataset to be divided by class. The dataset should have an
                     attribute `classes`.

    Returns:
        An list whose each element is a list of indices that belongs to a same
          class. The `i`-th element of the returned list is the `i`-th class in
          `dataset.classes`.
    '''
    assert hasattr(dataset,"classes"),\
        "The dataset is expected to have member `classes`."
    class_idxes=[]
    for _ in range(len(dataset.classes)):
        class_idxes.append([])
    for index in range(len(dataset)):
        _,y=dataset[index]
        class_idxes[y].append(index)
    return class_idxes

def create_sampled_dataset(dataset,idx,reassign_label=True):
    if isinstance(dataset,SampledDataset):
        return dataset.sampleFrom(idx,reassign_label)
    else:
        return SampledDataset(dataset,idx,reassign_label)

def sample_slice(dataset,number_of_sample,batch_size):
    size_of_dataset=len(dataset)
    idx=random.sample(range(size_of_dataset),number_of_sample)
    subset=create_sampled_dataset(dataset, idx,reassign_label=True)
    return (torch.utils.data.DataLoader(subset,batch_size),subset.classes)
