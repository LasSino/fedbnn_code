import copy
import math
import torch
import torchbnn
from configs import configs
from evaluation import accuracy

class fcl_client:
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
        head_params=([],[],[],[])
        for data_class in class_list:
            if data_class in self.heads:
                # Load the head parameters from the storage.
                weight_mu,weight_log_sigma,bias_mu,bias_log_sigma=\
                    self.heads[data_class]
            else:
                # Initialize the new parameters for new classes.
                weight_mu=torch.Tensor(self.shared_model_output_dim)\
                                    .fill_(configs["prior_mu"])
                weight_log_sigma=torch.Tensor(self.shared_model_output_dim)\
                                        .fill_(math.log(configs["prior_sigma"]))
                bias_mu=torch.tensor(configs["prior_mu"])
                bias_log_sigma=torch.tensor(math.log(configs["prior_sigma"]))
                self.head_need_ft=True
            head_params[0].append(weight_mu)
            head_params[1].append(weight_log_sigma)
            head_params[2].append(bias_mu)
            head_params[3].append(bias_log_sigma)

        head=torchbnn.BayesLinear(prior_mu=configs["prior_mu"],
                            prior_sigma=configs["prior_sigma"],
                            in_features=self.shared_model_output_dim,
                            out_features=len(class_list))
        head.weight_mu=torch.nn.Parameter(torch.stack(head_params[0]))
        head.weight_log_sigma=torch.nn.Parameter(torch.stack(head_params[1]))
        head.bias_mu=torch.nn.Parameter(torch.stack(head_params[2]))
        head.bias_log_sigma=torch.nn.Parameter(torch.stack(head_params[3]))

        head.prior_weight_mu=torch.stack(head_params[0])
        head.prior_weight_log_sigma=torch.stack(head_params[1])
        head.prior_bias_mu=torch.stack(head_params[2])
        head.prior_bias_log_sigma=torch.stack(head_params[3])

        return head

    def update_head(self,head,class_list):
        for class_idx,class_name in enumerate(class_list):
            # The entry in the table is a tuple:
            # (weight_mu,weight_log_sigma,bias_mu,bias_log_sigma)
            class_head=(
                        head.weight_mu[class_idx].clone().detach(),
                        head.weight_log_sigma[class_idx].clone().detach(),
                        head.bias_mu[class_idx].clone().detach(),
                        head.bias_log_sigma[class_idx].clone().detach(),
                    )
            self.heads[class_name]=class_head

    def finetune_head(self,head,data_split,
            grad_mc_times=configs["grad_mc_times"],
            optimizer=None,optimizer_args={}):

        full_model=torch.nn.Sequential(
                                self.shared_model,
                                torch.nn.ReLU(),
                                head,
                                torch.nn.Softmax(dim=-1))

        if configs["gpu"]==True:
            full_model.to(device="cuda")
        
        # Build optimizer and loss functions.
        kl_weight=configs["kl_weight"]
        ce_loss=torch.nn.CrossEntropyLoss()
        kl_loss=torchbnn.BKLLoss(reduction="mean",last_layer_only=False)
        if optimizer is None:
            optimizer=torch.optim.Adam
        head_ft_optimizer=optimizer(head.parameters(),
                                        lr=configs["head_ft_learn_rate"],
                                        **optimizer_args)

        # Finetune the head if needed.
        full_model[0].requires_grad_(False)
        full_model[1].requires_grad_(False)
        full_model[2].requires_grad_(True)
        full_model[3].requires_grad_(True)

        for ft_epoch in range(configs["head_finetune_epoch"]):
            for ft_batch,batch_data in enumerate(data_split):
                data_x,data_y=batch_data
                if configs["gpu"]==True:
                    data_x=data_x.to(device='cuda')
                    data_y=data_y.to(device='cuda')

                head_ft_optimizer.zero_grad()

                # Calculate gradient with monte carlo.
                # With monte carlo, the cost is:
                # cost=sum_of_ce/grad_mc_times+kl_weight*kl
                # Note the KL Loss does not need monte carlo.
                # This can be accumulated via cost.backward.
                for _ in range(grad_mc_times):
                    nn_output=full_model(data_x)
                    ce=ce_loss(nn_output,data_y)/float(grad_mc_times)
                    ce.backward()

                kl=kl_loss(full_model)*kl_weight
                kl.backward()

                head_ft_optimizer.step()

        #Now recover the require_grad_.
        full_model[0].requires_grad_(True)
        full_model[1].requires_grad_(True)
        full_model[2].requires_grad_(True)
        full_model[3].requires_grad_(True)

        if configs["gpu"]==True:
            full_model.to(device="cpu")
        
        return full_model[2]

    def train_model(self,data_split,split_classes,
            head_ft=True,grad_mc_times=configs["grad_mc_times"],
            optimizer=None,optimizer_args={}):
        assert self.shared_model!=None,"The shared model has not been set."

        # First we have to assmeble the head used for the data.
        head=self.assemble_head(class_list=split_classes)

        if head_ft and self.head_need_ft:
            head=self.finetune_head(head, data_split,grad_mc_times,
                        optimizer, optimizer_args)

        self.head_need_ft=False

        full_model=torch.nn.Sequential(
                                self.shared_model,
                                torch.nn.ReLU(),
                                head,
                                torch.nn.Softmax(dim=-1))

        if configs["gpu"]==True:
            full_model.to(device="cuda")

        # Set the objective loss function.
        kl_weight=configs["kl_weight"]
        ce_loss=torch.nn.CrossEntropyLoss()
        kl_loss=torchbnn.BKLLoss(reduction="mean",last_layer_only=False)
        if optimizer is None:
            optimizer=torch.optim.Adam
        train_optimizer=optimizer(full_model.parameters(),
                                        lr=configs["learn_rate"],
                                        **optimizer_args)

        acc_after_headft=accuracy(data_split,full_model
                            ,configs["monte_carlo_times"],cuda=configs["gpu"])

        #Now train the full model.
        full_model[0].requires_grad_(True)
        full_model[1].requires_grad_(True)
        full_model[2].requires_grad_(True)
        full_model[3].requires_grad_(True)

        for train_epoch in range(configs["train_epoch"]):
            for train_batch,batch_data in enumerate(data_split):
                data_x,data_y=batch_data
                if configs["gpu"]==True:
                    data_x=data_x.to(device='cuda')
                    data_y=data_y.to(device='cuda')

                train_optimizer.zero_grad()

                for _ in range(grad_mc_times):
                    nn_output=full_model(data_x)
                    ce=ce_loss(nn_output,data_y)/float(grad_mc_times)
                    ce.backward()

                kl=kl_loss(full_model)*kl_weight
                kl.backward()

                train_optimizer.step()


        if configs["gpu"]==True:
            full_model.to(device="cpu")

        self.update_head(head,split_classes)

        acc_final=accuracy(data_split,full_model,configs["monte_carlo_times"])

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

        return accuracy(data_split,full_model,configs["monte_carlo_times"])

    def get_model(self):
        return (self.shared_model,self.heads)
