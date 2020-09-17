import sys, math
from argparse import Namespace
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.nn.utils import weight_norm
from torch.autograd import Variable


""" Constants """
temp_arch = """ 
c1(och 10, ker 3, str 1, pad 1, bias, bn, pt max, pk 2, ps 1, actv relu); 
fc(out 100, bias, actv relu);
fc(out 1, bias, bn, actv relu)
"""

CONV1D_TABLE = {
    'och':  {'key': 'out_channels', 'default': 'must specify'},
    'ker':  {'key': 'kernel_size', 'default': 5},
    'str':  {'key': 'stride', 'default': 1},
    'pad':  {'key': 'padding', 'default': 0},
    'bias': {'key': 'bias', 'default': False},
    'bn':   {'key': 'use_bn', 'default': False},
    'pt':   {'key': 'pool_type', 'default': None},
    'pk':   {'key': 'pool_kernel_size', 'default': None},
    'ps':   {'key': 'pool_stride', 'default': 1},
    'pp':   {'key': 'pool_padding', 'default': 0},
    'actv': {'key': 'actv_type', 'default': None},

}

FC_TABLE = {
    'out':  {'key': 'out_size', 'default': 'must specify'},
    'bias': {'key': 'bias', 'default': False},
    'bn':   {'key': 'use_bn', 'default': False},
    'actv': {'key': 'actv_type', 'default': None},
}


""" Utilities """

def decode_architecture(arch_str):
    decoded_list = []
    layers = [x.strip() for x in arch_str.split(';')]
    for item in layers:
        l = Namespace()

        options_dic = {}
        p_beg, p_end = item.index('('), item.index(')')

        if item[:p_beg] == 'c1':
            l.type = 'conv1d'
            lookup_table = CONV1D_TABLE
        elif item[:p_beg] == 'fc':
            l.type = 'fc'
            lookup_table = FC_TABLE

        options_list = (item[p_beg+1:p_end]).split(',')
        for o in options_list:
            parsed_o = o.split()
            if len(parsed_o) == 1:
                options_dic[parsed_o[0]] = True
            else:
                if parsed_o[1].isnumeric():
                    parsed_o[1] = int(parsed_o[1])
                options_dic[parsed_o[0]] = parsed_o[1]

        for k,v in lookup_table.items():
            if k in options_dic.keys():
                assign_value = options_dic[k]
            else:
                assign_value = v['default']
            if assign_value == 'must specify':
                raise ValueError('Must assign a value for {} in layer type {}'.format(k, l.type))
            assign_key = v['key']
            vars(l)[assign_key] = assign_value


        #for k,v in options_dic.items():
        #    assign_value = lookup_table[k]['default'] if v is None else v
        #    if assign_value is None:
        #        raise ValueError('Must assign a value for {} in layer type {}'.format(k, l.type))
        #    assign_key = lookup_table[k]['key']
        #    vars(l)[assign_key] = assign_value
        decoded_list.append(l)
    return decoded_list


""" Implementation of layer types """

class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features,
                 bias=True, use_bn=True,
                 actv_type='relu'):
        super(LinearLayer, self).__init__()

        """ linear layer """
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        """ batch normalization """
        if use_bn:
            self.bn = nn.BatchNorm1d(self.out_features)
        else:
            self.bn = None

        """ activation """
        if actv_type is None:
            self.activation = None
        elif actv_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif actv_type == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif actv_type == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif actv_type == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError

    def reset_parameters(self):
        # # init.kaiming_uniform_(self.weight, a=math.sqrt(0)) # kaiming init
        # if (reset_indv_bias is None) or (reset_indv_bias is False):
        #     init.xavier_uniform_(self.weight, gain=1.0)  # xavier init
        # if (reset_indv_bias is None) or ((self.bias is not None) and reset_indv_bias is True):
        #     init.constant_(self.bias, 0)
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # concat channels and length of signal if input is from conv layer
        if len(input.shape) > 2:
            batch_size = input.shape[0]
            input = input.view(batch_size, -1)

        out = F.linear(input, self.weight, self.bias)
        #print('after matmul\n', out)

        if self.bn:
            out = self.bn(out)
        #print('after bn\n', out)
        if self.activation is not None:
            out = self.activation(out)
        #print('after linear layer\n', out)

        return out


class Conv1DLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding,
                 bias, use_bn,
                 pool_type, pool_kernel_size, pool_stride, pool_padding,
                 actv_type):

        super(Conv1DLayer, self).__init__()

        """ conv1d layer """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = Parameter(torch.Tensor(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        """ batch normalization """
        if use_bn:
            self.bn = nn.BatchNorm1d(self.out_channels)
        else:
            self.bn = None

        """ activation """
        if actv_type is None:
            self.activation = None
        elif actv_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif actv_type == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif actv_type == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif actv_type == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError('actv_type not valid')

        """ pool """
        if pool_type is None:
            # not setting pool
            self.pool = None
        else:
            # if setting pool
            self.pool_kernel_size = pool_kernel_size
            self.pool_stride = pool_stride
            self.pool_padding = pool_padding

            if pool_type == 'max':
                # TODO: missing dilation and ceil_mode options
                self.pool = nn.MaxPool1d(kernel_size=self.pool_kernel_size,
                                         stride=self.pool_stride,
                                         padding=self.pool_padding)
            elif pool_type == 'avg':
                # TODO: missing ceil_mode and count_include_pad options
                self.pool = nn.AvgPool1d(kernel_size=self.pool_kernel_size,
                                         stride=self.pool_stride,
                                         padding=self.pool_padding)
            else:
                raise ValueError('pool_type not valid')

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):

        # 1. convolution
        out = F.conv1d(input=input, weight=self.weight, bias=self.bias,
                       stride=self.stride, padding=self.padding)
        #print('after conv kernel\n', out)
        # 2. pooling
        if self.pool is not None:
            out = self.pool(out)
        #print('after pool\n', out)
        # 3. batch normalization
        if self.bn:
            out = self.bn(out)
        #print('after bn\n', out)
        # 4. activation
        if self.activation is not None:
            out = self.activation(out)
        #print('after conv layer\n', out)
        return out


""" Implementation of network architectures """

class vanilla_nn(nn.Module):
    def __init__(self, input_size=1, output_size=1, bias=True,
                 hidden_size=400, num_layers=4,
                 use_bn=False, actv_type='relu',
                 softmax=False):

        super(vanilla_nn, self).__init__()
        self.softmax = softmax
        # TODO: make loss an option
        self.loss = nn.MSELoss()

        self.fcs = nn.ModuleList()
        """ input layer """
        self.fcs.append(LinearLayer(input_size, hidden_size, bias,
                                    use_bn=use_bn, actv_type=actv_type))
        for _ in range(num_layers-1):
            self.fcs.append(LinearLayer(hidden_size, hidden_size, bias,
                                        use_bn=use_bn, actv_type=actv_type))
        self.fcs.append(LinearLayer(hidden_size, output_size, bias,
                                    use_bn=False, actv_type=None))

    def forward(self, X):
        for layer in self.fcs:
            X = layer(X)

        if self.softmax:
            out = F.softmax(X, dim=1)
        else:
            out = X

        return out


class prob_nn(nn.Module):
    def __init__(self, input_size=1, output_size=2, bias=True,
                 hidden_size=400, num_layers=4,
                 adversarial_eps_percent = 1,
                 use_bn=True, actv_type='relu',
                 softmax=False):

        super(prob_nn, self).__init__()
        self.softmax = softmax
        # self.loss = nn.MSELoss()
        self.mean_dim = 1

        self.fcs = nn.ModuleList()
        self.fcs.append(LinearLayer(input_size, hidden_size, bias,
                                    use_bn=use_bn, actv_type=actv_type))
        for _ in range(num_layers-1):
            self.fcs.append(LinearLayer(hidden_size, hidden_size, bias,
                                        use_bn=use_bn, actv_type=actv_type))
        self.fcs.append(LinearLayer(hidden_size, output_size, bias,
                                    use_bn=False, actv_type=None))
        self.max_var = 1e6
        self.min_var = 0.0
        self.adversarial_eps_percent = adversarial_eps_percent
        self.adversarial_eps = None

    def softplus(self, x):
        softplus = torch.log(1+torch.exp(x))
        #softplus = torch.where(softplus == float('inf'), x, softplus)
        return softplus

    def determine_adversarial_eps(self, full_train_X):
        num_features = full_train_X.shape[1]
        feat_eps_tensor = torch.empty(num_features)
        for feat_idx in range(num_features):
            feat_min = torch.min(full_train_X[:,feat_idx], dim=1)
            feat_max = torch.max(full_train_X[:,feat_idx], dim=1)
            feat_range = feat_max - feat_min
            feat_eps = feat_range * self.adversarial_eps_percent / 100.
            feat_eps_tensor[feat_idx] = feat_eps
        self.adversarial_eps = feat_eps_tensor

    def loss(self, batch_pred, batch_y):
        pred_mean, pred_var = torch.split(batch_pred, self.mean_dim, dim=1)
        # pred_mean = batch_pred[:,:self.mean_dim]
        # pred_var = batch_pred[:,self.mean_dim:]

        diff = torch.sub(batch_y, pred_mean)
        for v in pred_var:
            if v == float('inf'):
                raise ValueError('infinite variance')
            if v > self.max_var:
                self.max_var = v
            if v < self.min_var:
                self.min_var = v
        loss = torch.mean(torch.div(diff**2, 2*pred_var))
        loss += torch.mean(torch.log(pred_var)/2)

        # pred_var = torch.clamp(pred_var, min=1e-10)
        # term_1 = torch.log(pred_var)/2
        # term_2 = (batch_y - pred_mean)**2/(2*pred_var)
        # loss = torch.mean(term_1 + term_2, dim=0)

        return loss

    def gen_adversarial_example(self, batch_X, grad_X):
        with torch.no_grad():
            grad_sign = torch.sign(grad_X)
            adv_ex = batch_X + (self.adversarial_eps * grad_sign)
        return adv_ex

    def nll_adversarial_loss(self, batch_X, batch_y, optimizer):
        if self.adversarial_eps is None:
            raise RuntimeError("Must run 'determine_adversarial_eps' to set eps" )

        # 1. calculate nll loss of original batch_X
        #    make it a Variable so we can get grad of loss wrt batch_X
        batch_X = Variable(batch_X)
        batch_X.requires_grad = True

        # 2. zero out gradients before calculating grad of batch_X
        optimizer.zero_grad()
        batch_pred = self.forward(batch_X)
        nll_loss = self.loss(batch_pred, batch_y)
        nll_loss.backward()
        grad_X = batch_X.grad

        # 3. make the adversarial example from batch_X
        batch_adversarial_example = self.gen_adversarial_example(batch_X, grad_X)

        # 4. no longer need to calculate gradients for batch_X and adversarial batch_X
        batch_adversarial_example.requires_grad = False
        batch_X.requires_grad = False

        # 5. calculate nll loss of adversarial batch_X
        adversarial_batch_pred = self.forward(batch_adversarial_example)
        adv_nll_loss = self.loss(adversarial_batch_pred, batch_y)

        # 6. zero out gradient before calculating final loss
        optimizer.zero_grad()

        # final loss
        batch_loss = nll_loss + adv_nll_loss

        return batch_loss

    def forward(self, X):
        for layer in self.fcs:
            X = layer(X)

        if self.softmax:
            out = F.softmax(X, dim=1)
        else:
            out = X

        means = out[:,:self.mean_dim]
        variances = F.softplus(out[:,self.mean_dim:]) + 1e-8
        pnn_out = torch.cat([means, variances], dim=1)
        #pnn_out = torch.cat([out[:,:self.mean_dim], F.softplus(out[:,self.mean_dim:])], dim=1)
        return pnn_out


class pnn(nn.Module):
    def __init__(self, input_size=1, output_size=1, bias=True,
                 hidden_size=400, num_layers=4,
                 adversarial_eps_percent = 1,
                 use_bn=True, actv_type='relu'):

        super(pnn, self).__init__()

        # create the mean network
        self.mean_fcs = nn.ModuleList()
        self.mean_fcs.append(LinearLayer(input_size, hidden_size, bias,
                                    use_bn=use_bn, actv_type=actv_type))
        for _ in range(num_layers-1):
            self.mean_fcs.append(LinearLayer(hidden_size, hidden_size, bias,
                                        use_bn=use_bn, actv_type=actv_type))
        self.mean_fcs.append(LinearLayer(hidden_size, output_size, bias,
                                    use_bn=False, actv_type=None))

        # create the variance network
        self.var_fcs = nn.ModuleList()
        self.var_fcs.append(LinearLayer(input_size, hidden_size, bias,
                                         use_bn=use_bn, actv_type=actv_type))
        for _ in range(num_layers - 1):
            self.var_fcs.append(LinearLayer(hidden_size, hidden_size, bias,
                                             use_bn=use_bn, actv_type=actv_type))
        self.var_fcs.append(LinearLayer(hidden_size, output_size**2, bias,
                                         use_bn=False, actv_type=None))

        # set adversarial example parameters
        self.adversarial_eps_percent = adversarial_eps_percent
        self.adversarial_eps = None

    def determine_adversarial_eps(self, full_train_X):
        num_features = full_train_X.shape[1]
        feat_eps_tensor = torch.empty(num_features)
        for feat_idx in range(num_features):
            feat_min = torch.min(full_train_X[:,feat_idx], dim=1)
            feat_max = torch.max(full_train_X[:,feat_idx], dim=1)
            feat_range = feat_max - feat_min
            feat_eps = feat_range * self.adversarial_eps_percent / 100.
            feat_eps_tensor[feat_idx] = feat_eps
        self.adversarial_eps = feat_eps_tensor

    def loss(self, pred_mean, pred_var, batch_y):
        diff = torch.sub(batch_y, pred_mean)
        for v in pred_var:
            if v == float('inf'):
                raise ValueError('infinite variance')
        loss = torch.mean(torch.div(diff**2, 2*pred_var))
        loss += torch.mean(torch.log(pred_var)/2)

        return loss

    def gen_adversarial_example(self, batch_X, grad_X):
        with torch.no_grad():
            grad_sign = torch.sign(grad_X)
            adv_ex = batch_X + (self.adversarial_eps * grad_sign)
        return adv_ex

    def nll_adversarial_loss(self, batch_X, batch_y, optimizer):
        if self.adversarial_eps is None:
            raise RuntimeError("Must run 'determine_adversarial_eps' to set eps" )

        # 1. calculate nll loss of original batch_X
        #    make it a Variable so we can get grad of loss wrt batch_X
        batch_X = Variable(batch_X)
        batch_X.requires_grad = True

        # 2. zero out gradients before calculating grad of batch_X
        optimizer.zero_grad()
        batch_pred = self.forward(batch_X)
        nll_loss = self.loss(batch_pred, batch_y)
        nll_loss.backward()
        grad_X = batch_X.grad

        # 3. make the adversarial example from batch_X
        batch_adversarial_example = self.gen_adversarial_example(batch_X, grad_X)

        # 4. no longer need to calculate gradients for batch_X and adversarial batch_X
        batch_adversarial_example.requires_grad = False
        batch_X.requires_grad = False

        # 5. calculate nll loss of adversarial batch_X
        adversarial_batch_pred = self.forward(batch_adversarial_example)
        adv_nll_loss = self.loss(adversarial_batch_pred, batch_y)

        # 6. zero out gradient before calculating final loss
        optimizer.zero_grad()

        # final loss
        batch_loss = nll_loss + adv_nll_loss

        return batch_loss

    def forward(self, X):
        mean_X = X
        var_X = deepcopy(X)

        for layer in self.mean_fcs:
            mean_X = layer(mean_X)

        for layer in self.var_fcs:
            var_X = layer(var_X)

        mean_out = mean_X
        var_out = F.softplus(var_X) + 1e-8

        return mean_out, var_out


class cnn(nn.Module):
    def __init__(self, arch_str, in_channels, in_length, softmax=False):
        super(cnn, self).__init__()

        self.softmax = softmax
        # TODO: make loss an option
        self.loss = nn.MSELoss()

        last_out_channels, last_out_length = in_channels, in_length
        last_out_size = last_out_channels * last_out_length
        print('input to conv1D of {} channels * {} signal length ({} values)'.format(
              in_channels, in_length, last_out_size))

        layer_list = decode_architecture(arch_str)
        self.layers = nn.ModuleList()

        DUMMY_BATCH = 3
        dummy_tensor = torch.empty(DUMMY_BATCH, in_channels, in_length)
        for l in layer_list:
            # adding conv1d layer
            if l.type == 'conv1d':
                assert(last_out_channels is not None and last_out_length is not None)
                conv_layer = Conv1DLayer(in_channels=last_out_channels,
                                         out_channels=l.out_channels,
                                         kernel_size=l.kernel_size, stride=l.stride,
                                         padding=l.padding,
                                         bias=l.bias, use_bn=l.use_bn, pool_type=l.pool_type,
                                         pool_kernel_size=l.pool_kernel_size,
                                         pool_stride=l.pool_stride,
                                         pool_padding=l.pool_padding,
                                         actv_type=l.actv_type)

                self.layers.append(conv_layer)
                dummy_tensor = conv_layer(dummy_tensor)
                out_shape = dummy_tensor.shape
                assert out_shape[0] == DUMMY_BATCH
                last_out_channels, last_out_length = out_shape[1], out_shape[2]
                last_out_size = last_out_channels*last_out_length
                print('output of conv1D of {} channels * {} signal length ({} values), actv {}'.format(
                      last_out_channels, last_out_length, last_out_size, l.actv_type))


            # adding a linear layer
            elif l.type == 'fc':
                dummy_tensor = dummy_tensor.view(DUMMY_BATCH, -1)
                lin_layer = LinearLayer(in_features=last_out_size, out_features=l.out_size,
                                        bias=l.bias, use_bn=l.use_bn, actv_type=l.actv_type)
                self.layers.append(lin_layer)
                dummy_tensor = lin_layer(dummy_tensor)
                out_shape = dummy_tensor.shape
                assert l.out_size == out_shape[1]
                last_out_channels, last_out_length = None, None
                last_out_size = l.out_size
                print('output of linear layer of {} values, actv {}'.format(last_out_size, l.actv_type))

        """ check network dimensions """
        del dummy_tensor
        print(self.layers)

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)

        if self.softmax:
            out = F.softmax(X, dim=1)
        else:
            out = X

        return out


if __name__=='__main__':
    print(decode_architecture(temp_arch))


