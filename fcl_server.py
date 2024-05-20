from typing import Dict, List
from torchbnn import BayesLinear,BayesBatchNorm2d,BayesConv2d
import torch.nn as snn
from configs import configs
import torch
import copy

def _normalize_aggregate_weight(aggregate_weight):
    if not torch.is_tensor(aggregate_weight):
        aggregate_weight=torch.tensor(aggregate_weight)
    aggregate_weight=aggregate_weight.reshape(1,aggregate_weight.shape.numel())
    normalized_weight=aggregate_weight/sum(aggregate_weight.flatten())
    return normalized_weight

def _aggregate_gaussian_parameters(mus,log_sigmas,aggregate_weights):
    '''
    Aggregate the gaussian distribution parameters by moment matching or mean,
    the default is by moment matching. The behaviour can be changed in 
    `configs`.

    `mus` and `log_sigmas` are list of tensors for mu and **log_sigma**.
    '''
    num_components=len(mus)
    # Normalize the weights.
    normalized_weight=_normalize_aggregate_weight(aggregate_weights)

    # The mu can be aggregated by weighted sum.
    param_mu=torch.stack([mu.flatten() for mu in mus])
    aggregated_mu=normalized_weight.mm(param_mu.float())

    # Aggregate the sigma.
    
    if configs['gaussian_aggregation']=='mean':
        # If designated, the sigma will be aggregated by mean.
        param_sigma=torch.stack([sigma.flatten() for sigma in log_sigmas])
        aggregated_sigma=normalized_weight.mm(param_sigma.float())
    else:
        # By default, the sigma param is aggregated by moment matching.
        param_sigma=torch.stack([sigma.flatten() for sigma in log_sigmas]).exp()
        square_sum=param_mu.square()+param_sigma.square()
        aggregated_sigma=normalized_weight.mm(square_sum.float())-\
                            aggregated_mu.square()
        aggregated_sigma.sqrt_()
        aggregated_sigma.log_()

    return (aggregated_mu.reshape_as(mus[0]),
            aggregated_sigma.reshape_as(log_sigmas[0]))

    
def _aggregate_heads(heads:List,aggregate_weights):
    '''
    Aggregate a list of heads. The weights can be unnormalized.
    '''
    # Normalize the weights.
    normalized_weight=_normalize_aggregate_weight(aggregate_weights)
    # Check the first element of the head to determine what type of head it is.
    if len(heads[0])==2:
        # If a 2-element tuple, then it is a standard NN FC head.
        param_weights=[]
        param_biases=[]
        for head in heads:
            head_weight,head_bias=head
            param_weights.append(head_weight.flatten())
            param_biases.append(head_bias.flatten())
        # Stack the parameters into 2-D matrices.
        param_weights=torch.stack(param_weights)
        param_biases=torch.stack(param_biases)

        # The parameters can be aggregated with matrix multiplication.
        aggregated_weight=normalized_weight.mm(param_weights.float())
        aggregated_bias=normalized_weight.mm(param_biases.float())
        return (aggregated_weight.reshape_as(heads[0][0]),
                aggregated_bias.reshape_as(heads[0][1]))
    elif len(heads[0])==4:
        # If a 4-element tuple, then it is a bayesian NN FC head.
        # The heads is a list of tuples like
        # [(weight_mu,weight_log_sigma,bias_mu,bias_log_sigma)].
        weight_mus=[]
        weight_log_sigmas=[]
        bias_mus=[]
        bias_log_sigmas=[]
        for head in heads:
            weight_mu,weight_log_sigma,bias_mu,bias_log_sigma=head
            weight_mus.append(weight_mu)
            weight_log_sigmas.append(weight_log_sigma)
            bias_mus.append(bias_mu)
            bias_log_sigmas.append(bias_log_sigma)
        agg_w_mu,agg_w_sigma=_aggregate_gaussian_parameters(weight_mus,
                                        weight_log_sigmas,normalized_weight)
        agg_b_mu,agg_b_sigma=_aggregate_gaussian_parameters(bias_mus,
                                bias_log_sigmas,normalized_weight)
        return (agg_w_mu,agg_w_sigma,agg_b_mu,agg_b_sigma)
    else:
        raise AssertionError("The length of the head parameter tuple \
                should be 2 (standard NN FC) or 4 (bayesian NN FC).")

def _aggregate_bayesian_linear(layers:List[BayesLinear],aggregate_weights,
        update_prior=True,**kwargs):
    # Create a new layer with the same structure as the input layers.
    aggregated_layer=BayesLinear(configs["prior_mu"],configs["prior_sigma"],
                        layers[0].in_features,
                        layers[0].out_features,layers[0].bias)

    # Collect the parameters.
    weight_mus=[]
    weight_log_sigmas=[]
    bias_mus=[]
    bias_log_sigmas=[]
    for layer in layers:
        weight_mus.append(layer.weight_mu.data)
        weight_log_sigmas.append(layer.weight_log_sigma.data)
        if aggregated_layer.bias:
            bias_mus.append(layer.bias_mu.data)
            bias_log_sigmas.append(layer.bias_log_sigma.data)
    
    # Aggregate the weights.
    agg_w_mu,agg_w_sigma=_aggregate_gaussian_parameters(weight_mus,
                            weight_log_sigmas,aggregate_weights)
    # Set the priors and the current parameters.
    aggregated_layer.weight_mu.data=agg_w_mu
    aggregated_layer.weight_log_sigma.data=agg_w_sigma

    if update_prior:
        aggregated_layer.prior_weight_log_sigma=agg_w_sigma
        aggregated_layer.prior_weight_mu=agg_w_mu

    # Aggregate the biases if present.
    if aggregated_layer.bias:
        agg_b_mu,agg_b_sigma=_aggregate_gaussian_parameters(bias_mus,
                                bias_log_sigmas,aggregate_weights)
        
        # Set the priors and the current parameters.
        aggregated_layer.bias_mu.data=agg_b_mu
        aggregated_layer.bias_log_sigma.data=agg_b_sigma

        if update_prior:
            aggregated_layer.prior_bias_mu=agg_b_mu
            aggregated_layer.prior_bias_log_sigma=agg_b_sigma
    
    return aggregated_layer

def _aggregate_bayesian_conv(layers:List[BayesConv2d],aggregate_weights,
        update_prior=True,**kwargs):
    # Create a new layer with the same structure as the input layers.
    example_layer=layers[0]
    aggregated_layer=BayesConv2d(configs["prior_mu"],configs["prior_sigma"],
                        example_layer.in_channels,example_layer.out_channels,
                        example_layer.kernel_size,example_layer.stride,
                        example_layer.padding,example_layer.dilation,
                        example_layer.groups,example_layer.bias,
                        example_layer.padding_mode)

    # Collect the parameters.
    weight_mus=[]
    weight_log_sigmas=[]
    bias_mus=[]
    bias_log_sigmas=[]
    for layer in layers:
        weight_mus.append(layer.weight_mu.data)
        weight_log_sigmas.append(layer.weight_log_sigma.data)
        if aggregated_layer.bias:
            bias_mus.append(layer.bias_mu.data)
            bias_log_sigmas.append(layer.bias_log_sigma.data)
    
    # Aggregate the weights.
    agg_w_mu,agg_w_sigma=_aggregate_gaussian_parameters(weight_mus,
                            weight_log_sigmas,aggregate_weights)
    # Set the priors and the current parameters.
    aggregated_layer.weight_mu.data=agg_w_mu
    aggregated_layer.weight_log_sigma.data=agg_w_sigma

    if update_prior:
        aggregated_layer.prior_weight_log_sigma=agg_w_sigma
        aggregated_layer.prior_weight_mu=agg_w_mu

    # Aggregate the biases if present.
    if aggregated_layer.bias:
        agg_b_mu,agg_b_sigma=_aggregate_gaussian_parameters(bias_mus,
                                bias_log_sigmas,aggregate_weights)
        
        # Set the priors and the current parameters.
        aggregated_layer.bias_mu.data=agg_b_mu
        aggregated_layer.bias_log_sigma.data=agg_b_sigma
        if update_prior:
            aggregated_layer.prior_bias_mu=agg_b_mu
            aggregated_layer.prior_bias_log_sigma=agg_b_sigma
    
    return aggregated_layer

def _aggregate_bayesian_batchnorm(layers:List[BayesBatchNorm2d],
        aggregate_weights,update_prior=True,**kwargs):
    # Create a new layer with the same structure as the input layers.
    example_layer=layers[0]
    aggregated_layer=BayesBatchNorm2d(configs["prior_mu"],configs["prior_sigma"],
                        example_layer.num_features,example_layer.eps,
                        example_layer.momentum,example_layer.affine,
                        example_layer.track_running_stats)

    # The parameters is required to be aggregated only if the `affine` is True.
    if aggregated_layer.affine:
        # Collect the parameters.
        weight_mus=[]
        weight_log_sigmas=[]
        bias_mus=[]
        bias_log_sigmas=[]
        for layer in layers:
            weight_mus.append(layer.weight_mu.data)
            weight_log_sigmas.append(layer.weight_log_sigma.data)

            bias_mus.append(layer.bias_mu.data)
            bias_log_sigmas.append(layer.bias_log_sigma.data)
        
        # Aggregate the weights.
        agg_w_mu,agg_w_sigma=_aggregate_gaussian_parameters(weight_mus,
                                weight_log_sigmas,aggregate_weights)

        # Set the priors and the current parameters.
        aggregated_layer.weight_mu.data=agg_w_mu
        aggregated_layer.weight_log_sigma.data=agg_w_sigma

        # Aggregate the biases.
        agg_b_mu,agg_b_sigma=_aggregate_gaussian_parameters(bias_mus,
                                bias_log_sigmas,aggregate_weights)
        
        # Set the priors and the current parameters.
        aggregated_layer.bias_mu.data=agg_b_mu
        aggregated_layer.bias_log_sigma.data=agg_b_sigma

        if update_prior:
            aggregated_layer.prior_bias_mu=agg_b_mu
            aggregated_layer.prior_bias_log_sigma=agg_b_sigma
            aggregated_layer.prior_weight_log_sigma=agg_w_sigma
            aggregated_layer.prior_weight_mu=agg_w_mu

    
    return aggregated_layer

def _aggregate_scalar_parameters(parameters,aggregate_weights,**kwargs):
    normalized_weight=_normalize_aggregate_weight(aggregate_weights)

    param=torch.stack([parameter.flatten() for parameter in parameters])
    aggregated_param=normalized_weight.mm(param.float())

    return aggregated_param.reshape_as(parameters[0])

def _aggregate_snn_linear(layers:List[snn.Linear],aggregate_weights,**kwargs):
    # Create a new layer with the same structure as the input layers.
    aggregated_layer=snn.Linear(layers[0].in_features,layers[0].out_features,
                            (layers[0].bias is not None))

    # Collect the parameters.
    weight=[]
    bias=[]
    for layer in layers:
        weight.append(layer.weight.data)
        if aggregated_layer.bias is not None:
            bias.append(layer.bias.data)
    
    # Aggregate the weights.
    agg_weight=_aggregate_scalar_parameters(weight,aggregate_weights)
    aggregated_layer.weight.data=agg_weight

    # Set the priors and the current parameters.
    if aggregated_layer.bias is not None:
        agg_bias=_aggregate_scalar_parameters(bias,aggregate_weights)
        aggregated_layer.bias.data=agg_bias
    
    return aggregated_layer

def _aggregate_snn_conv2d(layers:List[snn.Conv2d],aggregate_weights,**kwargs):
    # Create a new layer with the same structure as the input layers.
    aggregated_layer=snn.Conv2d(layers[0].in_channels, layers[0].out_channels,
                            layers[0].kernel_size,layers[0].stride,
                            layers[0].padding,layers[0].dilation,
                            layers[0].groups,(layers[0].bias is not None),
                            layers[0].padding_mode)

    # Collect the parameters.
    weight=[]
    bias=[]
    for layer in layers:
        weight.append(layer.weight.data)
        if aggregated_layer.bias is not None:
            bias.append(layer.bias.data)
    
    # Aggregate the weights.
    agg_weight=_aggregate_scalar_parameters(weight,aggregate_weights)
    aggregated_layer.weight.data=agg_weight

    # Set the priors and the current parameters.
    if aggregated_layer.bias is not None:
        agg_bias=_aggregate_scalar_parameters(bias,aggregate_weights)
        aggregated_layer.bias.data=agg_bias
    
    return aggregated_layer

def _aggregate_non_parametric_layer(layers,aggregate_weights,**kwargs):
    '''
    This function copies the non-parametric modules in a model, such as ReLU.
    '''
    return copy.deepcopy(layers[0])


class fcl_server:
    '''
    The server manages the aggregation of the client models.
    Since the server is state-less, class methods are used.

    It is **assumed** that the network structures of all clients are identical,
     therefore the functions generally do not do checks on layer structures.
    '''
    LAYER_AGGREGATE_MAP={
        BayesLinear:_aggregate_bayesian_linear,
        BayesConv2d:_aggregate_bayesian_conv,
        BayesBatchNorm2d:_aggregate_bayesian_batchnorm,
        torch.nn.ReLU:_aggregate_non_parametric_layer,
        torch.nn.Flatten:_aggregate_non_parametric_layer,
        torch.nn.MaxPool2d:_aggregate_non_parametric_layer,
        torch.nn.Linear:_aggregate_snn_linear,
        torch.nn.Conv2d:_aggregate_snn_conv2d,
    }

    @classmethod
    def aggregate_shared_model(cls,shared_models,aggregate_weights=None,
            update_prior=True):
        '''
        Aggregate a list of shared models into a single model.
        '''
        # If the weights is not present, assume equal weights.
        if aggregate_weights==None:
            aggregate_weights=[1.0]*len(shared_models)

        aggregated_model=torch.nn.Sequential()
        for index,layer in enumerate(shared_models[0]):
            layers=[model[index] for model in shared_models]
            aggregate_func=cls.LAYER_AGGREGATE_MAP.get(type(layer),None)
            assert aggregate_func!=None,\
                "Aggregation function for {} is not present, \
                    may be not implemented yet.".format(type(layer))
            aggregated_layer=aggregate_func(layers,aggregate_weights
                                ,update_prior=update_prior)
            aggregated_model.append(aggregated_layer)
        
        aggregated_model.zero_grad()
        return aggregated_model
    
    @classmethod
    def aggregate_heads(cls,head_bases : List[Dict],aggregate_weights=None):
        '''
        This fuction gets the head bases of all clients, as a list of dicts 
         {class_name -> head_parameter}, and returns an aggregated head base as
         a dict.
        '''
        head_list={}
        for client_idx,client_head_base in enumerate(head_bases):
            for class_name,head_parameters in client_head_base.items():
                if class_name not in head_list:
                    head_list[class_name]=([],[])
                head_list[class_name][0].append(head_parameters)
                # Assign the weight of the head.
                # If no weight submitted, assume equal weights.
                head_list[class_name][1].append(
                                            aggregate_weights[client_idx] \
                                            if aggregate_weights else 1.0)
        
        aggregated_head_base={}
        for class_name, head_parameter_list in head_list.items():
            aggregated_head_base[class_name]=_aggregate_heads(
                                                head_parameter_list[0],
                                                head_parameter_list[1])
        
        return aggregated_head_base
        