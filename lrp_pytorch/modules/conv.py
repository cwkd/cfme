import torch
import torch.nn as nn
from lrp_pytorch.modules.base import lrp_backward, cam_backward, eb_backward


class PosNegConv(nn.Module):
    def _clone_module(self, module):
        clone = nn.Conv1d(module.in_channels, module.out_channels, module.kernel_size,
                          **{attr: getattr(module, attr) for attr in ['stride', 'padding', 'dilation', 'groups']})
        return clone.to(module.weight.device)

    def __init__(self, conv, ignorebias):
        super(PosNegConv, self).__init__()

        self.posconv = self._clone_module(conv)
        self.posconv.weight = torch.nn.Parameter(conv.weight.data.clone().clamp(min=0)).to(conv.weight.device)

        self.negconv = self._clone_module(conv)
        self.negconv.weight = torch.nn.Parameter(conv.weight.data.clone().clamp(max=0)).to(conv.weight.device)

        if ignorebias:
            self.posconv.bias = None
            self.negconv.bias = None
        else:
            if conv.bias is not None:
                self.posconv.bias = torch.nn.Parameter(conv.bias.data.clone().clamp(min=0))
                self.negconv.bias = torch.nn.Parameter(conv.bias.data.clone().clamp(max=0))

    def forward(self, x):
        vp = self.posconv(torch.clamp(x, min=0))
        vn = self.negconv(torch.clamp(x, max=0))
        return vp + vn


class LRPConv1d(nn.Module):
    def __init__(self, module, autograd_fn, **kwargs):
        super(LRPConv1d, self).__init__()
        self.module = module
        self.autograd_fn = autograd_fn
        self.params = kwargs['params']

    def forward(self, x):
        return self.autograd_fn.apply(x, self.module, self.params)


class Conv1d_Autograd_Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, module, params):
        eps = params.get('conv1d_eps', 1e-12)
        ignore_bias = params.get('ignore_conv_bias', False)

        def config_values_to_tensors(module):
            if isinstance(module, nn.Conv1d):
                property_names = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups']
            else:
                print('Error: module not Conv1d layer')
                raise Exception
            values = []
            for attr in property_names:
                value = getattr(module, attr)
                if isinstance(value, int):
                    value = torch.tensor([value], dtype=torch.int32, device=module.weight.device)
                elif isinstance(value, tuple):
                    value = torch.tensor(value, dtype=torch.int32, device=module.weight.device)
                else:
                    print('error: property value is neither int nor tuple')
                    exit()
                values.append(value)
            return property_names, values
        property_names, values = config_values_to_tensors(module)
        eps_tensor = torch.tensor([eps], dtype=torch.float, device=x.device)
        ignore_bias = torch.tensor([ignore_bias], dtype=torch.bool, device=module.weight.device)

        if module.bias is None:
            bias = None
        else:
            bias = module.bias.data.clone()

        ctx.save_for_backward(x, module.weight.data.clone(), bias, eps_tensor, ignore_bias, *values)

        # print('linear ctx.needs_input_grad', ctx.needs_input_grad)

        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight, bias, eps_tensor, ignore_bias, *values = ctx.saved_tensors
        # print('retrieved', len(values))

        def tensors_to_dict(values):
            property_names = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups']
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

        if bias is None:
            module = nn.Conv1d(**params_dict, bias=False)
        else:
            module = nn.Conv1d(**params_dict, bias=True)
            module.bias = nn.Parameter(bias)

        module.weight = nn.Parameter(weight)
        # print(module.weight.device)

        module = PosNegConv(module, ignorebias=ignore_bias.item())

        # print('Conv1d custom input_.shape', input_.shape)
        eps = eps_tensor.item()

        X = input_.clone().detach().requires_grad_(True)
        R = lrp_backward(input_=X,
                         layer=module,
                         relevance_output=grad_output,
                         eps0=eps,
                         eps=eps)
        # print('Conv1d custom R', R.shape)
        return R, None, None


class Skip_Layer_Conv1d_Autograd_Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, module, params):
        eps = params.get('conv1d_eps', 1e-12)
        ignore_bias = params.get('ignore_conv_bias', False)

        def config_values_to_tensors(module):
            if isinstance(module, nn.Conv1d):
                property_names = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups']
            else:
                print('Error: module not Conv1d layer')
                raise Exception
            values = []
            for attr in property_names:
                value = getattr(module, attr)
                if isinstance(value, int):
                    value = torch.tensor([value], dtype=torch.int32, device=module.weight.device)
                elif isinstance(value, tuple):
                    value = torch.tensor(value, dtype=torch.int32, device=module.weight.device)
                else:
                    print('error: property value is neither int nor tuple')
                    exit()
                values.append(value)
            return property_names, values
        property_names, values = config_values_to_tensors(module)
        eps_tensor = torch.tensor([eps], dtype=torch.float, device=x.device)
        ignore_bias = torch.tensor([ignore_bias], dtype=torch.bool, device=module.weight.device)

        if module.bias is None:
            bias = None
        else:
            bias = module.bias.data.clone()

        ctx.save_for_backward(x, module.weight.data.clone(), bias, eps_tensor, ignore_bias, *values)

        # print('linear ctx.needs_input_grad', ctx.needs_input_grad)

        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight, bias, eps_tensor, ignore_bias, *values = ctx.saved_tensors
        # print('retrieved', len(values))

        def tensors_to_dict(values):
            property_names = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups']
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

        if bias is None:
            module = nn.Conv1d(**params_dict, bias=False)
        else:
            module = nn.Conv1d(**params_dict, bias=True)
            module.bias = nn.Parameter(bias)

        module.weight = nn.Parameter(weight)
        # print(module.weight.device)

        module = PosNegConv(module, ignorebias=ignore_bias.item())

        # print('Conv1d custom input_.shape', input_.shape)
        eps = eps_tensor.item()

        X = input_.clone().detach().requires_grad_(True)
        # R = guided_lrp_backward(input_=X,
        #                         layer=module,
        #                         relevance_output=grad_output)
        R = torch.zeros_like(grad_output, device=input_.device)
        # print('Conv1d custom R', R.shape)
        return R, None, None


class CAM_Conv1d_Autograd_Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, module, params):
        eps = params.get('conv1d_eps', 1e-12)
        ignore_bias = params.get('ignore_conv_bias', True)

        def config_values_to_tensors(module):
            if isinstance(module, nn.Conv1d):
                property_names = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups']
            else:
                print('Error: module not Conv1d layer')
                raise Exception
            values = []
            for attr in property_names:
                value = getattr(module, attr)
                if isinstance(value, int):
                    value = torch.tensor([value], dtype=torch.int32, device=module.weight.device)
                elif isinstance(value, tuple):
                    value = torch.tensor(value, dtype=torch.int32, device=module.weight.device)
                else:
                    print('error: property value is neither int nor tuple')
                    exit()
                values.append(value)
            return property_names, values
        property_names, values = config_values_to_tensors(module)
        eps_tensor = torch.tensor([eps], dtype=torch.float, device=x.device)
        ignore_bias = torch.tensor([ignore_bias], dtype=torch.bool, device=module.weight.device)

        if module.bias is None:
            bias = None
        else:
            bias = module.bias.data.clone()

        ctx.save_for_backward(x, module.weight.data.clone(), bias, eps_tensor, ignore_bias, *values)

        # print('linear ctx.needs_input_grad', ctx.needs_input_grad)

        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight, bias, eps_tensor, ignore_bias, *values = ctx.saved_tensors
        # print('retrieved', len(values))

        def tensors_to_dict(values):
            property_names = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups']
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

        if bias is None:
            module = nn.Conv1d(**params_dict, bias=False)
        else:
            module = nn.Conv1d(**params_dict, bias=True)
            module.bias = nn.Parameter(bias)

        module.weight = nn.Parameter(weight)
        # print(module.weight.device)

        module = PosNegConv(module, ignorebias=ignore_bias.item())

        # print('Conv1d custom input_.shape', input_.shape)
        eps = eps_tensor.item()

        X = input_.clone().detach().requires_grad_(True)
        R = cam_backward(input_=X,
                         layer=module,
                         relevance_output=grad_output)
        # print('Conv1d custom R', R.shape)
        return R, None, None


class EB_Conv1d_Autograd_Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, module, params):
        eps = params.get('conv1d_eps', 1e-12)
        ignore_bias = params.get('ignore_conv_bias', True)

        def config_values_to_tensors(module):
            if isinstance(module, nn.Conv1d):
                property_names = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups']
            else:
                print('Error: module not Conv1d layer')
                raise Exception
            values = []
            for attr in property_names:
                value = getattr(module, attr)
                if isinstance(value, int):
                    value = torch.tensor([value], dtype=torch.int32, device=module.weight.device)
                elif isinstance(value, tuple):
                    value = torch.tensor(value, dtype=torch.int32, device=module.weight.device)
                else:
                    print('error: property value is neither int nor tuple')
                    exit()
                values.append(value)
            return property_names, values
        property_names, values = config_values_to_tensors(module)
        eps_tensor = torch.tensor([eps], dtype=torch.float, device=x.device)
        ignore_bias = torch.tensor([ignore_bias], dtype=torch.bool, device=module.weight.device)

        if module.bias is None:
            bias = None
        else:
            bias = module.bias.data.clone()

        ctx.save_for_backward(x, module.weight.data.clone(), bias, eps_tensor, ignore_bias, *values)

        # print('linear ctx.needs_input_grad', ctx.needs_input_grad)

        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight, bias, eps_tensor, ignore_bias, *values = ctx.saved_tensors
        # print('retrieved', len(values))

        def tensors_to_dict(values):
            property_names = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups']
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

        if bias is None:
            module = nn.Conv1d(**params_dict, bias=False)
        else:
            module = nn.Conv1d(**params_dict, bias=True)
            module.bias = nn.Parameter(bias)

        module.weight = nn.Parameter(weight)
        # print(module.weight.device)

        module = PosNegConv(module, ignorebias=ignore_bias.item())

        # print('Conv1d custom input_.shape', input_.shape)
        eps = eps_tensor.item()

        X = input_.clone().detach().requires_grad_(True)
        R = eb_backward(input_=X,
                        layer=module,
                        relevance_output=grad_output)
        # print('Conv1d custom R', R.shape)
        return R, None, None
