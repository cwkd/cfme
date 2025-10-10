import torch
import torch.nn as nn
import torch_geometric.nn as geom_nn
from lrp_pytorch.modules.base import lrp_backward, cam_backward, eb_backward


class LRPLinear(nn.Module):
    def __init__(self, module, autograd_fn, **kwargs):
        super(LRPLinear, self).__init__()
        self.module = module
        self.autograd_fn = autograd_fn
        self.params = kwargs['params']
        self.module.saved_rels = kwargs.get('saved_rels', None)
        self.ref_name = kwargs.get('ref_name', None)
        self.module.ref_name = self.ref_name
        self.module.x_in = []

    def forward(self, x):
        return self.autograd_fn.apply(x, self.module, self.params)


class Linear_Epsilon_Autograd_Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, module, params):
        eps = params.get('linear_eps', 1e-9)

        def config_values_to_tensors(module):
            if isinstance(module, nn.Linear):
                property_names = ['in_features', 'out_features']
            elif isinstance(module, geom_nn.Linear):
                property_names = ['in_channels', 'out_channels']
            else:
                print('Error: module not linear layer')
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

        if module.bias is None:
            bias = None
        else:
            bias = module.bias.data.clone()

        ctx.save_for_backward(x, module.weight.data.clone(), bias, eps_tensor, *values)
        ctx.saved_rels = module.saved_rels
        ctx.ref_name = module.ref_name

        # print('linear ctx.needs_input_grad', ctx.needs_input_grad)

        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight, bias, eps_tensor, *values = ctx.saved_tensors
        saved_rels = ctx.saved_rels
        ref_name = ctx.ref_name
        # print('retrieved', len(values))

        def tensors_to_dict(values):
            property_names = ['in_features', 'out_features']
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
            module = nn.Linear(**params_dict, bias=False)
        else:
            module = nn.Linear(**params_dict, bias=True)
            module.bias = nn.Parameter(bias)

        module.weight = nn.Parameter(weight)
        # print(module.weight.device)

        # print('linear custom input_.shape', input_.shape)
        eps = eps_tensor.item()

        X = input_.clone().detach().requires_grad_(True)
        # R = lrp_backward(input_=X,
        #                  layer=module,
        #                  relevance_output=grad_output,
        #                  eps0=eps,
        #                  eps=eps)
        R = lrp_backward(input_=X,
                         layer=module,
                         relevance_output=grad_output,
                         eps0=eps,
                         eps=0)

        if ref_name is not None:
            saved_rels[ref_name] = module.saved_relevance
        else:
            pass
            # print(ref_name, saved_rels)
            # print(module.__dict__)
        # print('linear custom R', R.shape)
        return R, None, None


class CAM_Linear_Autograd_Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, module, params):
        # eps = params.get('linear_eps', 1e-6)

        def config_values_to_tensors(module):
            if isinstance(module, nn.Linear):
                property_names = ['in_features', 'out_features']
            elif isinstance(module, geom_nn.Linear):
                property_names = ['in_channels', 'out_channels']
            else:
                print('Error: module not linear layer')
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
        # eps_tensor = torch.tensor([eps], dtype=torch.float, device=x.device)

        if module.bias is None:
            bias = None
        else:
            bias = module.bias.data.clone()

        ctx.save_for_backward(x, module.weight.data.clone(), bias, *values)
        ctx.saved_rels = module.saved_rels
        ctx.ref_name = module.ref_name

        # print('linear ctx.needs_input_grad', ctx.needs_input_grad)

        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight, bias, *values = ctx.saved_tensors
        saved_rels = ctx.saved_rels
        ref_name = ctx.ref_name
        # print('retrieved', len(values))

        def tensors_to_dict(values):
            property_names = ['in_features', 'out_features']
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
            module = nn.Linear(**params_dict, bias=False)
        else:
            module = nn.Linear(**params_dict, bias=True)
            module.bias = nn.Parameter(bias)

        module.weight = nn.Parameter(weight)
        # print(module.weight.device)

        # print('linear custom input_.shape', input_.shape)
        # eps = eps_tensor.item()

        X = input_.clone().detach().requires_grad_(True)
        R = cam_backward(input_=X,
                         layer=module,
                         relevance_output=grad_output)
        if ref_name is not None:
            saved_rels[ref_name] = module.saved_relevance
        else:
            pass
        # print(ref_name, list(saved_rels.keys()))
            # print(module.__dict__)
        # print('linear custom R', R.shape)
        return R, None, None


class EB_Linear_Autograd_Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, module, params):
        # eps = params.get('linear_eps', 1e-6)

        def config_values_to_tensors(module):
            if isinstance(module, nn.Linear):
                property_names = ['in_features', 'out_features']
            elif isinstance(module, geom_nn.Linear):
                property_names = ['in_channels', 'out_channels']
            else:
                print('Error: module not linear layer')
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
        # eps_tensor = torch.tensor([eps], dtype=torch.float, device=x.device)

        if module.bias is None:
            bias = None
        else:
            bias = module.bias.data.clone()

        ctx.save_for_backward(x, module.weight.data.clone(), bias, *values)
        ctx.saved_rels = module.saved_rels
        ctx.ref_name = module.ref_name

        # print('linear ctx.needs_input_grad', ctx.needs_input_grad)
        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight, bias, *values = ctx.saved_tensors
        saved_rels = ctx.saved_rels
        ref_name = ctx.ref_name
        # print('retrieved', len(values))

        def tensors_to_dict(values):
            property_names = ['in_features', 'out_features']
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
            module = nn.Linear(**params_dict, bias=False)
        else:
            module = nn.Linear(**params_dict, bias=True)
            module.bias = nn.Parameter(bias)

        module.weight = nn.Parameter(weight)
        # print(module.weight.device)

        # print('linear custom input_.shape', input_.shape)
        # eps = eps_tensor.item()

        X = input_.clone().detach().requires_grad_(True)
        R = eb_backward(input_=X,
                        layer=module,
                        relevance_output=grad_output)
        if ref_name is not None:
            saved_rels[ref_name] = module.saved_relevance
        else:
            pass
        # print(ref_name, list(saved_rels.keys()))
        # print(module.__dict__)
        # print(f'linear custom R {ref_name}', R.shape, X.grad.shape)
        return R, None, None


class C_EB_Linear_Autograd_Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, module, params):
        # eps = params.get('linear_eps', 1e-6)

        def config_values_to_tensors(module):
            if isinstance(module, nn.Linear):
                property_names = ['in_features', 'out_features']
            elif isinstance(module, geom_nn.Linear):
                property_names = ['in_channels', 'out_channels']
            else:
                print('Error: module not linear layer')
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
        # eps_tensor = torch.tensor([eps], dtype=torch.float, device=x.device)

        if module.bias is None:
            bias = None
        else:
            bias = module.bias.data.clone()

        ctx.save_for_backward(x, module.weight.data.clone(), bias, *values)
        ctx.saved_rels = module.saved_rels
        ctx.ref_name = module.ref_name

        # print('linear ctx.needs_input_grad', ctx.needs_input_grad)
        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight, bias, *values = ctx.saved_tensors
        saved_rels = ctx.saved_rels
        ref_name = ctx.ref_name
        # print('retrieved', len(values))

        def tensors_to_dict(values):
            property_names = ['in_features', 'out_features']
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
            module = nn.Linear(**params_dict, bias=False)
            contrastive_module = nn.Linear(**params_dict, bias=False)
        else:
            module = nn.Linear(**params_dict, bias=True)
            contrastive_module = nn.Linear(**params_dict, bias=True)
            module.bias = nn.Parameter(bias)
            contrastive_module.bias = nn.Parameter(bias.clone().detach().requires_grad_(True))

        module.weight = nn.Parameter(weight)
        contrastive_module.weight = nn.Parameter(-weight.clone().detach().requires_grad_(True))
        # print(module.weight.device)

        # print('linear custom input_.shape', input_.shape)
        # eps = eps_tensor.item()


        X = input_.clone().detach().requires_grad_(True)
        contrastive_X = input_.clone().detach().requires_grad_(True)
        R = eb_backward(input_=X,
                        layer=module,
                        relevance_output=grad_output)
        # with torch.no_grad():
        #     module.weight.copy_(-module.weight.float())
        # print('contrastive_module', contrastive_module, flush=True)
        R_contrastive = eb_backward(input_=contrastive_X,
                                    layer=contrastive_module,
                                    relevance_output=grad_output)
        R = R - R_contrastive
        if ref_name is not None:
            saved_rels[ref_name] = module.saved_relevance
        else:
            pass
        # print(ref_name, list(saved_rels.keys()))
        # print(module.__dict__)
        # print(f'linear custom R {ref_name}', R.shape, X.grad.shape)
        return R, None, None