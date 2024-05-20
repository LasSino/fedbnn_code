import torch
import torchbnn
import math
from configs import configs

def head_library_transform(snn_head_lib,bnn_head_lib):
    '''
    This function transforms an SNN head library into the BNN head librar,
    filling the given `bnn_head_lib`.
    The mean (mu) of BNN parameter will be the parameter of SNN, while the std
    (sigma) of BNN will be filled by `log(configs['prior_sigma'])`.


    Arguments:
        snn_head_lib -- Input, the SNN head library.
        bnn_head_lib -- Output, the BNN head library, entries of it will be
                        updated by the transformed SNN heads.
    '''
    prior_log_sigma=math.log(configs['prior_sigma'])
    for class_name,snn_head_param in snn_head_lib.items():
        weight,bias=snn_head_param

        weight_log_sigma=torch.full_like(weight, prior_log_sigma)
        bias_log_sigma=torch.full_like(bias, prior_log_sigma)

        bnn_head_param=(weight.clone().detach(),
                        weight_log_sigma,
                        bias.clone().detach(),
                        bias_log_sigma)
        
        bnn_head_lib[class_name]=bnn_head_param

    return bnn_head_lib

def _transform_linear(snn_layer,bnn_layer):
    '''Transform an SNN Linear layer to BayesLinear.
    Note the dimension is not checked.

    Arguments:
        snn_layer -- The input SNN layer.
        bnn_layer -- The output BNN layer to be filled.
    '''
    # Basic type check.
    assert (isinstance(snn_layer, torch.nn.Linear) and 
            isinstance(bnn_layer, torchbnn.BayesLinear)), \
            "Can not transform from {} to {}".format(type(snn_layer),
                                                     type(bnn_layer))

    prior_log_sigma=math.log(configs['prior_sigma'])
    
    bnn_layer.prior_weight_mu=snn_layer.weight.clone().detach()
    bnn_layer.prior_weight_log_sigma=torch.full_like(bnn_layer.prior_weight_mu,
                                                     prior_log_sigma)
    bnn_layer.weight_mu.data=snn_layer.weight.clone().detach()
    bnn_layer.weight_log_sigma.data=torch.full_like(bnn_layer.weight_mu.data,
                                                     prior_log_sigma)
    
    if snn_layer.bias is not None:
        bnn_layer.prior_bias_mu=snn_layer.bias.clone().detach()
        bnn_layer.prior_bias_log_sigma=torch.full_like(bnn_layer.prior_bias_mu,
                                                       prior_log_sigma)
        bnn_layer.bias_mu.data=snn_layer.bias.clone().detach()
        bnn_layer.bias_log_sigma.data=torch.full_like(bnn_layer.bias_mu.data,
                                                      prior_log_sigma)

    return bnn_layer

def _transform_batchnorm2d(snn_layer,bnn_layer):
    '''Transform an SNN BatchNorm2d layer to BayesBatchNorm2d.
    Note the dimension is not checked.

    Arguments:
        snn_layer -- The input SNN layer.
        bnn_layer -- The output BNN layer to be filled.
    '''
    # Basic type check.
    assert (isinstance(snn_layer, torch.nn.BatchNorm2d) and 
            isinstance(bnn_layer, torchbnn.BayesBatchNorm2d)), \
            "Can not transform from {} to {}".format(type(snn_layer),
                                                     type(bnn_layer))
    
    if snn_layer.affine:

        prior_log_sigma=math.log(configs['prior_sigma'])
    
        bnn_layer.prior_weight_mu=snn_layer.weight.clone().detach()
        bnn_layer.prior_weight_log_sigma=torch.full_like(bnn_layer.prior_weight_mu,
                                                        prior_log_sigma)
        bnn_layer.weight_mu.data=snn_layer.weight.clone().detach()
        bnn_layer.weight_log_sigma.data=torch.full_like(bnn_layer.weight_mu.data,
                                                        prior_log_sigma)
    
        bnn_layer.prior_bias_mu=snn_layer.bias.clone().detach()
        bnn_layer.prior_bias_log_sigma=torch.full_like(bnn_layer.prior_bias_mu,
                                                       prior_log_sigma)
        bnn_layer.bias_mu.data=snn_layer.bias.clone().detach()
        bnn_layer.bias_log_sigma.data=torch.full_like(bnn_layer.bias_mu.data,
                                                      prior_log_sigma)

    return bnn_layer

def _transform_conv2d(snn_layer,bnn_layer):
    '''Transform an SNN Conv2d layer to BayesConv2d.
    Note the dimension is not checked.

    Arguments:
        snn_layer -- The input SNN layer.
        bnn_layer -- The output BNN layer to be filled.
    '''
    # Basic type check.
    assert (isinstance(snn_layer, torch.nn.Conv2d) and 
            isinstance(bnn_layer, torchbnn.BayesConv2d)), \
            "Can not transform from {} to {}".format(type(snn_layer),
                                                     type(bnn_layer))

    prior_log_sigma=math.log(configs['prior_sigma'])
    
    bnn_layer.prior_weight_mu=snn_layer.weight.clone().detach()
    bnn_layer.prior_weight_log_sigma=torch.full_like(bnn_layer.prior_weight_mu,
                                                     prior_log_sigma)
    bnn_layer.weight_mu.data=snn_layer.weight.clone().detach()
    bnn_layer.weight_log_sigma.data=torch.full_like(bnn_layer.weight_mu.data,
                                                     prior_log_sigma)
    
    if snn_layer.bias is not None:
        bnn_layer.prior_bias_mu=snn_layer.bias.clone().detach()
        bnn_layer.prior_bias_log_sigma=torch.full_like(bnn_layer.prior_bias_mu,
                                                       prior_log_sigma)
        bnn_layer.bias_mu.data=snn_layer.bias.clone().detach()
        bnn_layer.bias_log_sigma.data=torch.full_like(bnn_layer.bias_mu.data,
                                                      prior_log_sigma)

    return bnn_layer

MODEL_TRANSFORM_MAPPER={
    torch.nn.Linear:_transform_linear,
    torch.nn.BatchNorm2d:_transform_batchnorm2d,
    torch.nn.Conv2d:_transform_conv2d
}

def model_transform(snn_model,bnn_model):
    '''This function transforms a given SNN model into the BNN model.
    The parameters of the given BNN model will be filled by the transformed SNN
    model parameters.

    Arguments:
        snn_model -- Input, the SNN model.
        bnn_model -- Output, the BNN model. The parameter of it will be filled.
    '''
    for snn_layer,bnn_layer in zip(snn_model,bnn_model):
        transform_func=MODEL_TRANSFORM_MAPPER.get(type(snn_layer),None)
        if not transform_func is None:
            transform_func(snn_layer,bnn_layer)

    return bnn_model

# Adding this for nested networks.
MODEL_TRANSFORM_MAPPER[torch.nn.Sequential]=model_transform