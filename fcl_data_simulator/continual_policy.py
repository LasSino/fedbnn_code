'''
This file implements some continual learning's data generation policies.

The policies generates sub-datasets used for each round of learning, and manages
 task switch during the process.
'''
import math
from .dataset_utils import create_sampled_dataset,CombinedDataset
import random

class ContinualPolicyFinished(Exception):
    pass

class StaticContinualPolicy:
    def __init__(self,dataset,
            maximum_dataset_size=None,replacement_between_rounds=True):
        self.dataset=dataset
        self.dataset_size=len(dataset)
        self.maximum_dataset_size=maximum_dataset_size
        self.replacement_between_rounds=replacement_between_rounds

        if self.maximum_dataset_size and not self.replacement_between_rounds:
            self.idx=list(range(self.dataset_size))
            random.shuffle(self.idx)
            self.idx_offset=0

        self._sample_dataset()

    def _sample_dataset(self):
        if self.maximum_dataset_size==None:
            self.data=self.dataset
        else:
            if self.replacement_between_rounds:
                idx=random.sample(range(self.dataset_size),
                            self.maximum_dataset_size)
            else:
                remaining_idx=self.dataset_size-self.idx
                if remaining_idx>self.maximum_dataset_size:
                    idx=self.idx[self.idx_offset:\
                                    self.idx_offset+self.maximum_dataset_size]
                    self.idx_offset+=self.maximum_dataset_size
                else:
                    idx=self.idx[self.idx_offset:self.dataset_size]+\
                        self.idx[0:self.maximum_dataset_size-remaining_idx]
                    self.idx_offset=0
                    random.shuffle(self.idx)

            self.data=create_sampled_dataset(self.dataset,idx,True)


    def get_data(self):
        return self.data

    def step(self):
        self._sample_dataset()

    def get_data_and_step(self):
        data=self.get_data()
        self.step()
        return data

    def get_past_tasks(self):
        return [self.dataset]

class TaskSeparateContiunalPolicy:
    '''
    This is the task separate continual learning policy. In this policy,
      when switching between tasks, the accessible data is instantly changed
      to the new task's data, and data from previous task is instantly absent.

    `maximum_dataset_size`: The dataset size returned for each round, if this is
     not set, the whole dataset is returned.
    `replacement_between_rounds`: Used when `maximum_dataset_size` is set, to
     decide whether the sampling of each round is with or without replacement.
     Note if replacement is set to False, the dataset might be exhausted and an
     empty set will be returned.
    '''
    def __init__(self,tasks,task_duration,
            maximum_dataset_size=None,replacement_between_rounds=True):
        self.tasks=tasks
        if isinstance(task_duration,int):
            self.task_duration=[task_duration]*len(self.tasks)
        else:
            assert len(task_duration)==tasks,\
                "The length of duration list must equal number of tasks."
            self.task_duration=task_duration

        # Get the cumsum of the task duration, which marks the round to switch
        # tasks.
        self.task_switch_round=[0]
        for duration in self.task_duration:
            self.task_switch_round.append(self.task_switch_round[-1]+duration)

        self.current_round=0
        self.current_task_index=0
        self.maximum_dataset_size=maximum_dataset_size
        self.replacement_between_rounds=replacement_between_rounds
        self.stopped=False

        if self.maximum_dataset_size!=None and \
            not self.replacement_between_rounds:
            self._prepare_index_permutation()

    def _prepare_index_permutation(self):
        current_task_set_size=len(self.tasks[self.current_task_index])
        self.index_permutation=list(range(current_task_set_size))
        random.shuffle(self.index_permutation)
        self.offset_in_index_perm=0

    def get_data(self):
        if self.stopped:
            raise ContinualPolicyFinished()

        current_task=self.tasks[self.current_task_index]

        if self.maximum_dataset_size==None:
            # Directly return the whole task dataset.
            return current_task
        else:
            if self.replacement_between_rounds:
                # Sample some indices from the current task dataset.
                idx=random.sample(range(len(current_task)),
                            self.maximum_dataset_size)
                return create_sampled_dataset(current_task,idx,True)
            else:
                # Get the next group of indices from the index permutation.
                idx=self.index_permutation[\
                        self.offset_in_index_perm:\
                        self.offset_in_index_perm+self.maximum_dataset_size]
                return create_sampled_dataset(current_task,idx,True)

    def step(self):
        if self.stopped:
            raise ContinualPolicyFinished()

        self.current_round+=1
        if self.current_round>=self.task_switch_round[self.current_task_index+1]:
            self.current_task_index+=1
            if self.current_task_index>=len(self.tasks):
                # The designed process has stopped.
                self.stopped=True
                return
        if self.maximum_dataset_size!=None and \
            not self.replacement_between_rounds:
            self._prepare_index_permutation()

    def get_data_and_step(self):
        data=self.get_data()
        self.step()
        return data

    def get_past_tasks(self):
        if self.stopped:
            return self.tasks
        else:
            return [self.tasks[task_index]
                        for task_index in range(self.current_task_index+1)]

class GradualContiunalPolicy:
    '''
    This is the gradual continual learning policy. In this policy,
      when switching between tasks, the ratio of data from each task gradually
      changes.
    '''
    @staticmethod
    def create_by_task_durations_linear(tasks,task_start_round,task_end_round,
            dataset_size,replacement_between_rounds=True):
        total_rounds=max(task_end_round)
        num_of_tasks=len(tasks)
        ratio=[]
        # First fill the ratio list with zeros.
        for _ in range(total_rounds): ratio.append([0]*num_of_tasks)
        for task_index in range(num_of_tasks):
            start_round,end_round=task_start_round[task_index],\
                                    task_end_round[task_index]
            middle_round=math.floor((start_round+end_round-1)/2)
            for n_round in range(start_round,end_round):
                if n_round<=middle_round:
                    task_ratio=\
                        (1+n_round-start_round)/(middle_round-start_round+1)
                else:
                    task_ratio=(end_round-n_round)/(end_round-middle_round)
                ratio[n_round][task_index]=task_ratio

        return GradualContiunalPolicy(tasks,ratio,
                dataset_size,replacement_between_rounds)

    def __init__(self,tasks,task_ratio,
            dataset_size,replacement_between_rounds=True) -> None:

        self.tasks=tasks
        self.task_ratio=task_ratio
        self.dataset_size=dataset_size
        self.num_of_task=len(self.tasks)
        self.task_sizes=[len(task) for task in self.tasks]
        self.task_accessed=[0]*self.num_of_task

        self.rounds=len(task_ratio)
        self.current_round=0

        self.replacement_between_rounds=replacement_between_rounds

        if not self.replacement_between_rounds:
            self.permutation_offset=[0]*self.num_of_task
            self.permutations=[list(range(size)) for size in self.task_sizes]
            for _ in map(random.shuffle,self.permutations): pass

        self.stopped=False
        self._set_number_of_samples()

    def _set_number_of_samples(self):
        ratio=self.task_ratio[self.current_round]
        # Normalize the ratio
        norm_coeff=sum(ratio)
        ratio=[r/norm_coeff for r in ratio]

        self.number_of_samples=[math.floor(r*self.dataset_size) for r in ratio]

        for task_index in range(self.num_of_task):
            if self.number_of_samples[task_index]>0:
                self.task_accessed[task_index]=1

    def get_data(self):
        if self.stopped:
            raise ContinualPolicyFinished()

        subsets=[]

        for dataset_index,sample_number in enumerate(self.number_of_samples):
            # Ignore the datasets where no sample is needed.
            if sample_number!=0:
                dataset=self.tasks[dataset_index]
                if self.replacement_between_rounds:
                    # Sample from the dataset.
                    idx=random.sample(range(len(dataset)),sample_number)
                else:
                    perm_offset=self.permutation_offset[dataset_index]
                    idx=self.permutations[dataset_index]\
                            [perm_offset:perm_offset+sample_number]
                subsets.append(create_sampled_dataset(dataset,idx,True))

        return CombinedDataset(subsets,True)

    def step(self):
        if self.stopped:
            raise ContinualPolicyFinished()

        # Switch to the next round.
        self.current_round+=1
        # Move the offset if permutation exists.
        if not self.replacement_between_rounds:
            self.permutation_offset=[offset+sample_number \
                                        for offset,sample_number \
                                        in zip(self.permutation_offset,
                                            self.number_of_samples)]

        if self.current_round>=self.rounds:
            self.stopped=True
            return
        else:
            self._set_number_of_samples()

    def get_data_and_step(self):
        data=self.get_data()
        self.step()
        return data

    def get_past_tasks(self):
        return [self.tasks[task_index]
                    for task_index in range(self.num_of_task)
                    if self.task_accessed[task_index]==1]
