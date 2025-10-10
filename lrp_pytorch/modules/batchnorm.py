import torch
import torch.nn as nn
# import torch_geometric.nn as geom_nn
from lrp_pytorch.modules.base import lrp_backward, cam_backward, eb_backward, guided_lrp_backward


class LRPBatchNorm1d(nn.Module):
    def __init__(self, module, autograd_fn, **kwargs):
        super(LRPBatchNorm1d, self).__init__()
        self.module = module
        self.autograd_fn = autograd_fn
        self.params = kwargs['params']

    def forward(self, x):
        return self.autograd_fn.apply(x, self.module, self.params)


class BatchNorm1d_Autograd_Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, module, params):
        eps = params.get('bn1d', 1e-12)

        def config_values_to_tensors(module):
            if isinstance(module, nn.BatchNorm1d):
                property_names = ['num_features', 'eps', 'momentum']
            else:
                print('Error: module not BatchNorm1d layer')
                raise Exception
            values = []
            for attr in property_names:
                value = getattr(module, attr)
                if isinstance(value, int):
                    value = torch.tensor([value], dtype=torch.int32, device=module.weight.device)
                elif isinstance(value, float):
                    value = torch.tensor([value], dtype=torch.float, device=module.weight.device)
                elif isinstance(value, tuple):
                    value = torch.tensor(value, dtype=torch.int32, device=module.weight.device)
                else:
                    print('error: property value is neither int nor tuple')
                    exit()
                values.append(value)
            return property_names, values
        property_names, values = config_values_to_tensors(module)

        if module.bias is None:
            bias = None
        else:
            bias = module.bias.data.clone()
        weight = module.weight.data.clone()
        running_var = module.running_var.data.clone()
        running_mean = module.running_mean.data.clone()
        eps_tensor = torch.tensor([eps], dtype=torch.float, device=x.device)

        ctx.save_for_backward(x, weight, bias, running_mean, running_var, eps_tensor, *values)

        # print('BatchNorm1d ctx.needs_input_grad', ctx.needs_input_grad)

        return module(x)

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight, bias, running_mean, running_var, eps_tensor, *values = ctx.saved_tensors
        # print('retrieved', len(values))

        def tensors_to_dict(values):
            property_names = property_names = ['num_features', 'eps', 'momentum']
            params_dict = {}

            for i, property_name in enumerate(property_names):
                value = values[i]
                if value.numel == 1:
                    params_dict[property_name] = value.item()
                else:
                    value_list = value.tolist()
                    if len(value_list) == 1:
                        params_dict[property_name] = value_list[0]
                    else:
                        params_dict[property_name] = tuple(value_list)
            return params_dict

        params_dict = tensors_to_dict(values)

        eps = eps_tensor.item()
        device = eps_tensor.device

        if bias is None:
            module = nn.BatchNorm1d(**params_dict)
        else:
            module = nn.BatchNorm1d(**params_dict)
            module.bias = nn.Parameter(bias)
        module.weight = nn.Parameter(weight)
        module.running_var = running_var
        module.running_mean = running_mean

        var_bn = (running_var + module.eps) ** .5
        w_bn = weight
        bias_bn = bias
        mu_bn = running_mean

        threshold = -bias_bn * var_bn / w_bn + mu_bn
        module = ClampLayer(threshold, torch.sign(w_bn))

        # print('BatchNorm1d custom input_.shape', input_.shape)

        X = input_.clone().detach().requires_grad_(True)
        R = lrp_backward(input_=X,
                         layer=module,
                         relevance_output=grad_output,
                         eps0=1e-12,
                         eps=1e-12)
        # print('BatchNorm1d custom R', R.shape)
        return R, None, None


class LRP_Zero_BatchNorm1d_Autograd_Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, module, params):
        eps = params.get('bn1d', 1e-12)

        def config_values_to_tensors(module):
            if isinstance(module, nn.BatchNorm1d):
                property_names = ['num_features', 'eps', 'momentum']
            else:
                print('Error: module not BatchNorm1d layer')
                raise Exception
            values = []
            for attr in property_names:
                value = getattr(module, attr)
                if isinstance(value, int):
                    value = torch.tensor([value], dtype=torch.int32, device=module.weight.device)
                elif isinstance(value, float):
                    value = torch.tensor([value], dtype=torch.float, device=module.weight.device)
                elif isinstance(value, tuple):
                    value = torch.tensor(value, dtype=torch.int32, device=module.weight.device)
                else:
                    print('error: property value is neither int nor tuple')
                    exit()
                values.append(value)
            return property_names, values
        property_names, values = config_values_to_tensors(module)

        if module.bias is None:
            bias = None
        else:
            bias = module.bias.data.clone()
        weight = module.weight.data.clone()
        running_var = module.running_var.data.clone()
        running_mean = module.running_mean.data.clone()
        eps_tensor = torch.tensor([eps], dtype=torch.float, device=x.device)

        ctx.save_for_backward(x, weight, bias, running_mean, running_var, eps_tensor, *values)

        # print('BatchNorm1d ctx.needs_input_grad', ctx.needs_input_grad)

        return module(x)

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight, bias, running_mean, running_var, eps_tensor, *values = ctx.saved_tensors
        # print('retrieved', len(values))

        def tensors_to_dict(values):
            property_names = property_names = ['num_features', 'eps', 'momentum']
            params_dict = {}

            for i, property_name in enumerate(property_names):
                value = values[i]
                if value.numel == 1:
                    params_dict[property_name] = value.item()
                else:
                    value_list = value.tolist()
                    if len(value_list) == 1:
                        params_dict[property_name] = value_list[0]
                    else:
                        params_dict[property_name] = tuple(value_list)
            return params_dict

        params_dict = tensors_to_dict(values)

        eps = eps_tensor.item()
        device = eps_tensor.device

        if bias is None:
            module = nn.BatchNorm1d(**params_dict)
        else:
            module = nn.BatchNorm1d(**params_dict)
            module.bias = nn.Parameter(bias)
        module.weight = nn.Parameter(weight)
        module.running_var = running_var
        module.running_mean = running_mean

        var_bn = (running_var + module.eps) ** .5
        w_bn = weight
        bias_bn = bias
        mu_bn = running_mean

        threshold = -bias_bn * var_bn / w_bn + mu_bn
        module = ClampLayer(threshold, torch.sign(w_bn))

        # print('BatchNorm1d custom input_.shape', input_.shape)

        X = input_.clone().detach().requires_grad_(True)
        R = lrp_backward(input_=X,
                         layer=module,
                         relevance_output=grad_output,
                         eps0=0,
                         eps=0)
        # print('BatchNorm1d custom R', R.shape)
        return R, None, None


class Guided_LRPBatchNorm1d_Autograd_Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, module, params):
        eps = params.get('bn1d', 1e-12)

        def config_values_to_tensors(module):
            if isinstance(module, nn.BatchNorm1d):
                property_names = ['num_features', 'eps', 'momentum']
            else:
                print('Error: module not BatchNorm1d layer')
                raise Exception
            values = []
            for attr in property_names:
                value = getattr(module, attr)
                if isinstance(value, int):
                    value = torch.tensor([value], dtype=torch.int32, device=module.weight.device)
                elif isinstance(value, float):
                    value = torch.tensor([value], dtype=torch.float, device=module.weight.device)
                elif isinstance(value, tuple):
                    value = torch.tensor(value, dtype=torch.int32, device=module.weight.device)
                else:
                    print('error: property value is neither int nor tuple')
                    exit()
                values.append(value)
            return property_names, values
        property_names, values = config_values_to_tensors(module)

        if module.bias is None:
            bias = None
        else:
            bias = module.bias.data.clone()
        weight = module.weight.data.clone()
        running_var = module.running_var.data.clone()
        running_mean = module.running_mean.data.clone()
        eps_tensor = torch.tensor([eps], dtype=torch.float, device=x.device)

        ctx.save_for_backward(x, weight, bias, running_mean, running_var, eps_tensor, *values)

        # print('BatchNorm1d ctx.needs_input_grad', ctx.needs_input_grad)

        return module(x)

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight, bias, running_mean, running_var, eps_tensor, *values = ctx.saved_tensors
        # print('retrieved', len(values))

        def tensors_to_dict(values):
            property_names = property_names = ['num_features', 'eps', 'momentum']
            params_dict = {}

            for i, property_name in enumerate(property_names):
                value = values[i]
                if value.numel == 1:
                    params_dict[property_name] = value.item()
                else:
                    value_list = value.tolist()
                    if len(value_list) == 1:
                        params_dict[property_name] = value_list[0]
                    else:
                        params_dict[property_name] = tuple(value_list)
            return params_dict

        params_dict = tensors_to_dict(values)

        eps = eps_tensor.item()
        device = eps_tensor.device

        if bias is None:
            module = nn.BatchNorm1d(**params_dict)
        else:
            module = nn.BatchNorm1d(**params_dict)
            module.bias = nn.Parameter(bias)
        module.weight = nn.Parameter(weight)
        module.running_var = running_var
        module.running_mean = running_mean

        var_bn = (running_var + module.eps) ** .5
        w_bn = weight
        bias_bn = bias
        mu_bn = running_mean

        threshold = -bias_bn * var_bn / w_bn + mu_bn
        module = ClampLayer(threshold, torch.sign(w_bn))

        # print('BatchNorm1d custom input_.shape', input_.shape)

        X = input_.clone().detach().requires_grad_(True)
        R = guided_lrp_backward(input_=X,
                                layer=module,
                                relevance_output=grad_output)
        # print('BatchNorm1d custom R', R.shape)
        return R, None, None


class CAM_BatchNorm1d_Autograd_Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, module, params):
        eps = params.get('bn1d', 1e-12)

        def config_values_to_tensors(module):
            if isinstance(module, nn.BatchNorm1d):
                property_names = ['num_features', 'eps', 'momentum']
            else:
                print('Error: module not BatchNorm1d layer')
                raise Exception
            values = []
            for attr in property_names:
                value = getattr(module, attr)
                if isinstance(value, int):
                    value = torch.tensor([value], dtype=torch.int32, device=module.weight.device)
                elif isinstance(value, float):
                    value = torch.tensor([value], dtype=torch.float, device=module.weight.device)
                elif isinstance(value, tuple):
                    value = torch.tensor(value, dtype=torch.int32, device=module.weight.device)
                else:
                    print('error: property value is neither int nor tuple')
                    exit()
                values.append(value)
            return property_names, values
        property_names, values = config_values_to_tensors(module)

        if module.bias is None:
            bias = None
        else:
            bias = module.bias.data.clone()
        weight = module.weight.data.clone()
        running_var = module.running_var.data.clone()
        running_mean = module.running_mean.data.clone()
        eps_tensor = torch.tensor([eps], dtype=torch.float, device=x.device)

        ctx.save_for_backward(x, weight, bias, running_mean, running_var, eps_tensor, *values)

        # print('BatchNorm1d ctx.needs_input_grad', ctx.needs_input_grad)

        return module(x)

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight, bias, running_mean, running_var, eps_tensor, *values = ctx.saved_tensors
        # print('retrieved', len(values))

        def tensors_to_dict(values):
            property_names = property_names = ['num_features', 'eps', 'momentum']
            params_dict = {}

            for i, property_name in enumerate(property_names):
                value = values[i]
                if value.numel == 1:
                    params_dict[property_name] = value.item()
                else:
                    value_list = value.tolist()
                    if len(value_list) == 1:
                        params_dict[property_name] = value_list[0]
                    else:
                        params_dict[property_name] = tuple(value_list)
            return params_dict

        params_dict = tensors_to_dict(values)

        eps = eps_tensor.item()
        device = eps_tensor.device

        if bias is None:
            module = nn.BatchNorm1d(**params_dict)
        else:
            module = nn.BatchNorm1d(**params_dict)
            module.bias = nn.Parameter(bias)
        module.weight = nn.Parameter(weight)
        module.running_var = running_var
        module.running_mean = running_mean

        var_bn = (running_var + module.eps) ** .5
        w_bn = weight
        bias_bn = bias
        mu_bn = running_mean

        threshold = -bias_bn * var_bn / w_bn + mu_bn
        module = ClampLayer(threshold, torch.sign(w_bn))

        # print('BatchNorm1d custom input_.shape', input_.shape)

        X = input_.clone().detach().requires_grad_(True)
        R = cam_backward(input_=X,
                         layer=module,
                         relevance_output=grad_output)
        # print('BatchNorm1d custom R', R.shape)
        return R, None, None


class EB_BatchNorm1d_Autograd_Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, module, params):
        eps = params.get('bn1d', 1e-12)

        def config_values_to_tensors(module):
            if isinstance(module, nn.BatchNorm1d):
                property_names = ['num_features', 'eps', 'momentum']
            else:
                print('Error: module not BatchNorm1d layer')
                raise Exception
            values = []
            for attr in property_names:
                value = getattr(module, attr)
                if isinstance(value, int):
                    value = torch.tensor([value], dtype=torch.int32, device=module.weight.device)
                elif isinstance(value, float):
                    value = torch.tensor([value], dtype=torch.float, device=module.weight.device)
                elif isinstance(value, tuple):
                    value = torch.tensor(value, dtype=torch.int32, device=module.weight.device)
                else:
                    print('error: property value is neither int nor tuple')
                    exit()
                values.append(value)
            return property_names, values
        property_names, values = config_values_to_tensors(module)

        if module.bias is None:
            bias = None
        else:
            bias = module.bias.data.clone()
        weight = module.weight.data.clone()
        running_var = module.running_var.data.clone()
        running_mean = module.running_mean.data.clone()
        eps_tensor = torch.tensor([eps], dtype=torch.float, device=x.device)

        ctx.save_for_backward(x, weight, bias, running_mean, running_var, eps_tensor, *values)

        # print('BatchNorm1d ctx.needs_input_grad', ctx.needs_input_grad)

        return module(x)

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight, bias, running_mean, running_var, eps_tensor, *values = ctx.saved_tensors
        # print('retrieved', len(values))

        def tensors_to_dict(values):
            property_names = property_names = ['num_features', 'eps', 'momentum']
            params_dict = {}

            for i, property_name in enumerate(property_names):
                value = values[i]
                if value.numel == 1:
                    params_dict[property_name] = value.item()
                else:
                    value_list = value.tolist()
                    if len(value_list) == 1:
                        params_dict[property_name] = value_list[0]
                    else:
                        params_dict[property_name] = tuple(value_list)
            return params_dict

        params_dict = tensors_to_dict(values)

        eps = eps_tensor.item()
        device = eps_tensor.device

        if bias is None:
            module = nn.BatchNorm1d(**params_dict)
        else:
            module = nn.BatchNorm1d(**params_dict)
            module.bias = nn.Parameter(bias)
        module.weight = nn.Parameter(weight)
        module.running_var = running_var
        module.running_mean = running_mean

        var_bn = (running_var + module.eps) ** .5
        w_bn = weight
        bias_bn = bias
        mu_bn = running_mean

        threshold = -bias_bn * var_bn / w_bn + mu_bn
        module = ClampLayer(threshold, torch.sign(w_bn))

        # print('BatchNorm1d custom input_.shape', input_.shape)

        X = input_.clone().detach().requires_grad_(True)
        R = eb_backward(input_=X,
                         layer=module,
                         relevance_output=grad_output)
        # print('BatchNorm1d custom R', R.shape)
        return R, None, None


class ClampLayer(nn.Module):
    def __init__(self, threshold, w_bn_sign):
        super(ClampLayer, self).__init__()

        self.threshold = threshold
        self.w_bn_sign = w_bn_sign

    def forward(self, x):
        return (x - self.threshold) * \
               ((x > self.threshold) * (self.w_bn_sign > 0) +
                (x < self.threshold) * (self.w_bn_sign < 0)) + self.threshold