import copy
import math
import torch
from configs import configs
from evaluation import accuracy

class fcl_fedavg_client:
    '''
    This class simulates the behaviour of a client.
    The client will train the model upon recieving the global 
      model and next piece of data.
    Since the FCL model is usually multi-headed to cope with different classes,
      we store the header params by heads in a hash table.
    '''
    def __init__(self) -> None:
        self.heads={}
        self.shared_model=None
        self.shared_model_output_dim=None
        self.head_need_ft=False
    
    def update_model(self,shared_model,heads=None):
        # Sanity check for the output dimension of the shared network.
        if self.shared_model_output_dim==None:
            self.shared_model_output_dim=shared_model[-1].out_features
        else:
            assert self.shared_model_output_dim==shared_model[-1].out_features,\
                "Shared model output dimension mismatch."
        # Deep copy the shared model.
        self.shared_model=copy.deepcopy(shared_model)
        # Update the heads if needed.
        # The heads are stacked into the fc, so deepcopy is not needed.
        if heads!=None:
            self.heads.update(heads)
        self.head_need_ft=True
        
    def assemble_head(self,class_list):
        head_params=([],[])
        for data_class in class_list:
            if data_class in self.heads:
                # Load the head parameters from the storage.
                weight,bias=self.heads[data_class]
            else:
                # Initialize the new parameters for new classes.
                weight=torch.Tensor(self.shared_model_output_dim).normal_()
                bias=torch.tensor(0.0).normal_()
                self.head_need_ft=True
            head_params[0].append(weight)
            head_params[1].append(bias)
        
        head=torch.nn.Linear(in_features=self.shared_model_output_dim,
                        out_features=len(class_list))
        head.weight=torch.nn.Parameter(torch.stack(head_params[0]))
        head.bias=torch.nn.Parameter(torch.stack(head_params[1]))
        
        return head

    def update_head(self,head,class_list):
        for class_idx,class_name in enumerate(class_list):
            # The entry in the table is a tuple:
            # (weight_mu,weight_log_sigma,bias_mu,bias_log_sigma)
            class_head=(
                        head.weight[class_idx].clone().detach(),
                        head.bias[class_idx].clone().detach(),
                    )
            self.heads[class_name]=class_head

    def finetune_head(self,head,data_split):

        full_model=torch.nn.Sequential(
                                self.shared_model,
                                torch.nn.ReLU(),
                                head,
                                torch.nn.Softmax(dim=-1))
        
        # Build optimizer and loss functions.
        ce_loss=torch.nn.CrossEntropyLoss()
        head_ft_optimizer=torch.optim.Adam(head.parameters(),
                                lr=configs["head_ft_learn_rate"])

        # Finetune the head if needed.
        full_model[0].requires_grad_(False)
        full_model[1].requires_grad_(False)
        full_model[2].requires_grad_(True)
        full_model[3].requires_grad_(True)

        for ft_epoch in range(configs["head_finetune_epoch"]):
            for ft_batch,batch_data in enumerate(data_split):
                data_x,data_y=batch_data
                nn_output=full_model(data_x)
                cost=ce_loss(nn_output,data_y)

                head_ft_optimizer.zero_grad()
                cost.backward()
                head_ft_optimizer.step()

        #Now recover the require_grad_.
        full_model[0].requires_grad_(True)
        full_model[1].requires_grad_(True)
        full_model[2].requires_grad_(True)
        full_model[3].requires_grad_(True)
        
        return full_model[2]

    def train_model(self,data_split,split_classes,head_ft=True):
        assert self.shared_model!=None,"The shared model has not been set."
        
        # First we have to assmeble the head used for the data.
        head=self.assemble_head(class_list=split_classes)

        if head_ft and self.head_need_ft:
            head=self.finetune_head(head, data_split)

        self.head_need_ft=False    

        full_model=torch.nn.Sequential(
                                self.shared_model,
                                torch.nn.ReLU(),
                                head,
                                torch.nn.Softmax(dim=-1))

        # Set the objective loss function.
        ce_loss=torch.nn.CrossEntropyLoss()
        full_optimizer=torch.optim.Adam(full_model.parameters(),
                                lr=configs["learn_rate"])

        acc_after_headft=accuracy(data_split,full_model)

        #Now train the full model.
        full_model[0].requires_grad_(True)
        full_model[1].requires_grad_(True)
        full_model[2].requires_grad_(True)
        full_model[3].requires_grad_(True)

        for train_epoch in range(configs["train_epoch"]):
            for train_batch,batch_data in enumerate(data_split):
                data_x,data_y=batch_data
                nn_output=full_model(data_x)
                cost=ce_loss(nn_output,data_y)

                full_optimizer.zero_grad()
                cost.backward()
                full_optimizer.step()

        self.update_head(head,split_classes)

        acc_final=accuracy(data_split,full_model)

        return (acc_after_headft,acc_final)
    
    def test_model(self,data_split,split_classes):
        assert self.shared_model!=None,"The shared model has not been set."
        
        # First we have to assmeble the head used for the data.
        head=self.assemble_head(class_list=split_classes)

        full_model=torch.nn.Sequential(
                                self.shared_model,
                                torch.nn.ReLU(),
                                head,
                                torch.nn.Softmax(dim=-1))

        return accuracy(data_split,full_model)

    def get_model(self):
        return (self.shared_model,self.heads)