from .continual_policy import ContinualPolicyFinished
from torch.utils.data import DataLoader

class DataManager:
    '''
    This class manages the data generation process. 
    In accordance with the setting of FCL, the data distribution may change over
      time, and also differ among clients.
    '''
    def __init__(self,batch_size,continual_policy,
            partition_policy,partiton_policy_args):
        self.batch_size=batch_size
        
        self.continual_policy=continual_policy
        self.partition_policy=partition_policy
        self.partiton_policy_args=partiton_policy_args
        self.next_round()

    def next_round(self) -> None:
        '''
        Let the DataManger step to the next round, possibly changing the
          temporal data distribution.
        '''
        self.past_tasks=self.continual_policy.get_past_tasks()
        self.global_data=self.continual_policy.get_data_and_step()
        if self.partition_policy!=None:
            sliced_dataset=self.partition_policy(self.global_data,
                    **self.partiton_policy_args)
        else:
            sliced_dataset=[self.global_data]
        
        self.slices=[(DataLoader(dataset,
                        self.batch_size),dataset.classes)
                     for dataset in sliced_dataset]

    def get_past_tasks(self):
        return self.past_tasks

    def get_slice(self,slice_number):
        '''
        Get a slice for a certain client.
        '''
        return self.slices[slice_number]

    def get_slices(self):
        return self.slices

    def __iter__(self):
        return self

    def __next__(self):
        try:
            slices=self.slices
            self.next_round()
            return slices
        except ContinualPolicyFinished:
            raise StopIteration()