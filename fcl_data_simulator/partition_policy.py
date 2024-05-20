'''
This file implements some federated learning's data partitioning policies.

The policies generates sub-datasets used for each client.
'''

import math
import random
import numpy

from .dataset_utils import CombinedDataset, create_sampled_dataset, get_index_by_class

class PartitioningPolicies:
    @staticmethod
    def IID_partitioning(dataset,number_of_clients,client_data_size=None,**args):
        '''
        Partition the dataset i.i.d-ly to the clients.

        The client_data_size is an upper bound of the partitioned dataset size. 
         If the total size of the dataset is not enough, then the partitioned
         dataset size is bounded by the total dataset size.
        '''
        dataset_size=len(dataset)
        partitioned_size=math.floor(len(dataset)/number_of_clients)
        if client_data_size!=None and partitioned_size>client_data_size:
            partitioned_size=client_data_size
        
        # Create samples based on the whole dataset.
        idxs=random.sample(range(dataset_size),
                        partitioned_size*number_of_clients)
        partitions=[]
        for i in range(number_of_clients):
            partition=create_sampled_dataset(dataset,
                        idxs[i*partitioned_size:(i+1)*partitioned_size],
                        reassign_label=args.get("reassign_label",True))
            partitions.append(partition)
        
        return partitions

    @staticmethod
    def pathological_nonIID_partitioning(dataset,number_of_clients,
            client_dataset_size=None,class_per_client=2,**args):
        '''
        Partition the dataset in the scheme often referred to as "pathological
         non-IID". In the scheme, each client only has several classes in the
         whole dataset.
        '''
        # Assign each client with `class_per_client` classes and make sure that
        # the classes are evenly shared.
        number_of_classes=len(dataset.classes)
        class_for_clients=[]
        class_occurence=[0]*number_of_classes
        for client in range(number_of_clients):
            # Each client has a class that is associated with the client id.
            main_class=client%number_of_classes
            class_for_each_client=[main_class]
            # Then the rest classes are chosen from the other classes.
            class_pool=list(range(number_of_classes))
            class_pool.remove(main_class)
            class_for_each_client+=random.sample(class_pool,class_per_client-1)

            class_for_clients.append(class_for_each_client)
            for class_id in class_for_each_client: class_occurence[class_id]+=1

        class_idx_offset=[0]*number_of_classes
        class_idx=get_index_by_class(dataset)
        for idx in class_idx: random.shuffle(idx)
        client_datasets=[]

        for client in class_for_clients:
            if client_dataset_size!=None:
                per_class_size=math.floor(client_dataset_size/class_per_client)
            else:
                per_class_size=None

            idx=[]

            for client_class_id in client:
                class_data_size=per_class_size or \
                                math.floor(len(class_idx[client_class_id])/\
                                            class_occurence[client_class_id])
                idx_offset=class_idx_offset[client_class_id]
                idx+=class_idx[client_class_id]\
                        [idx_offset:idx_offset+class_data_size]
                class_idx_offset[client_class_id]+=class_data_size
            
            client_dataset=create_sampled_dataset(dataset,idx,
                            reassign_label=args.get("reassign_label",True))
            client_datasets.append(client_dataset)

        return client_datasets

    @staticmethod
    def dirichlet_nonIID_partitioning(dataset,number_of_clients,
            non_iid_ness=1,client_dataset_size=None,**args):
        '''Partition the client dataset using the Dirichlet distribution.

        Arguments:
            dataset -- The input dataset, should be subscriptable and has 
                       attribute `classes`.
            number_of_clients -- The number of clients.

        Keyword Arguments:
            non_iid_ness -- The alpha argument of Dirichlet distribution. 
                            (default: {1})
            client_dataset_size -- Limit for client dataset size. (default: {None})
            reassign_label: Passed to `create_sampled_dataset`. (default: True)

        Returns:
            A list of datasets, each for a client.
        '''
        number_of_classes=len(dataset.classes)

        label_distribution=numpy.random.dirichlet(
                                            [non_iid_ness]*number_of_clients,
                                            number_of_classes)
        
        classes_idx=get_index_by_class(dataset)

        client_idx=[list() for _ in range(number_of_clients)]

        for class_idx,fraction in zip(classes_idx,label_distribution):
            random.shuffle(class_idx)
            split_point=numpy.cumsum(fraction*len(class_idx)).round().astype(int)
            class_splits=numpy.split(class_idx, split_point[:-1])
            for client in range(number_of_clients):
                client_idx[client].extend(class_splits[client])

        # If there is a limit for local dataset, then resample the indices.
        if not (client_dataset_size is None) :
            for client in range(number_of_clients):
                if len(client_idx[client])>=client_dataset_size:
                    sampled=random.sample(client_idx[client], client_dataset_size)
                    client_idx[client]=sampled

        client_datasets=[]
        for client in range(number_of_clients):
            client_ds=create_sampled_dataset(dataset, client_idx[client],
                        reassign_label=args.get("reassign_label",True))
            client_datasets.append(client_ds)
        
        return client_datasets

    @staticmethod
    def multitask_nonIID_partitioning(dataset:CombinedDataset,number_of_clients,
            client_dataset_size=None,**args):
        '''
        This partitioning scheme is based on the tasks in the continual learning
         settings. Some client will switch to the latter tasks while some will 
         remain in previous tasks.
        '''
        assert isinstance(dataset,CombinedDataset),\
            "The dataset used for this method should be an `CombinedDataset`."
        
        # Number of clients for each task is proportional to the task dataset 
        # size.
        whole_size=len(dataset)
        client_for_each_task=[math.ceil(number_of_clients*task_size/whole_size)
                                for task_size in dataset.dataset_sizes]
        client_datasets=[]
        for task_index,task in enumerate(dataset.underlying_datasets):
            index_perm=list(range(dataset.dataset_sizes[task_index]))
            random.shuffle(index_perm)

            num_of_client=client_for_each_task[task_index]
            client_size=\
                client_dataset_size or math.floor(len(index_perm)/num_of_client)
            
            for i in range(num_of_client):
                idx=index_perm[i*client_size:(i+1)*client_size]
                client_ds=create_sampled_dataset(
                            dataset.underlying_datasets[task_index],idx,True)
                client_datasets.append(client_ds)
            
        return client_datasets[0:number_of_clients]
        