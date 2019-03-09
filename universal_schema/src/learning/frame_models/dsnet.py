"""
Implements basic elements of deep sets which can then be used in any way
one likes.
TODO: The relnet.py file and this can share a whole lot of code.
    Design classes better. --low-pri
"""
from __future__ import print_function
import sys
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import functional


class ElmNet(torch.nn.Module):
    """
    Function giving a representation for every element of the set. This is a simple
    feedforward network.
    """
    def __init__(self, in_elm_dim, out_elm_dim, non_linearity='tanh'):
        """
        :param in_elm_dim: int; the input embedding dimensions.
        :param out_elm_dim: int; the outpur dimensions of the embeddings.
        :param non_linearity: string; fixed to tanh.
        """
        torch.nn.Module.__init__(self)
        self.in_elm_dim = in_elm_dim
        self.out_elm_dim = out_elm_dim
        # Network for complete set of arguments.
        layers = OrderedDict()
        layers['lin_0'] = torch.nn.Linear(in_features=self.in_elm_dim,
                                          out_features=self.out_elm_dim, bias=False)
        if non_linearity == 'tanh':
            layers['nonlin_0'] = torch.nn.Tanh()
        else:
            sys.stderr.write('Unknown non-linearity: {:s}\n'.format(non_linearity))
        self.elm_ffn = torch.nn.Sequential(layers)

    def forward(self, elms):
        """
        Return representations for the "roles". Embeddings obtained by combining
        individual arguments with the predicates.
        :param elms: Variable(torch.Tensor); batch_size x max_set_len x arg_dim
        :return: elm_reps: Variable(torch.Tensor); batch_size x max_set_len x out_elm_dim
        """
        # Pass through the ffn layer.
        elm_reps = self.elm_ffn(elms)
        return elm_reps


class ElmNetBiSI(torch.nn.Module):
    """
    Function giving a contextualized representation for every element of the set
    the context information for every element is passed as "side-information".
    The network used to combine the context info with the raw element embedding
    is a bilinear network so this is most suited for side-info which has operator
    like semantics for the set element.
    """

    def __init__(self, in_elm_dim, elm_si_dim, out_elm_dim, non_linearity='tanh'):
        """
        :param in_elm_dim: int; input element embedding dimensions.
        :param elm_si_dim: int; dimension of the side information embedding.
        :param out_elm_dim: int; dimension of contextualized element embedding.
        :param non_linearity: string. fixed to tanh.
        """
        torch.nn.Module.__init__(self)
        self.in_elm_dim = in_elm_dim
        self.elm_si_dim = elm_si_dim
        self.out_elm_dim = out_elm_dim
        # Setting the bias to true screws up some of padding of role_reps.
        # Fix that before you set bias to true.
        self.bilinear = torch.nn.Bilinear(in1_features=self.elm_si_dim,
                                          in2_features=self.in_elm_dim,
                                          out_features=self.out_elm_dim, bias=False)
        if non_linearity == 'tanh':
            self.non_linear = torch.nn.Tanh()
        else:
            sys.stderr.write('Unknown non-linearity: {:s}\n'.format(non_linearity))

    def forward(self, elms, side_info):
        """
        Return representations for the "roles". Embeddings obtained by combining
        individual arguments with the predicates.
        :param elms: Variable(torch.Tensor); batch_size x max_set_len x arg_dim
        :param side_info: Variable(torch.Tensor); batch_size x max_set_len x si_dim
        :return: si_elm_reps: Variable(torch.Tensor); batch_size x max_set_len x out_elm_dim
        """
        batch_size, max_set_len, _ = elms.size()
        # Pass through the bilinear layer.
        si_elm_reps = self.bilinear(side_info.view(batch_size*max_set_len, self.elm_si_dim),
                                    elms.view(batch_size*max_set_len, self.in_elm_dim))
        si_elm_reps = self.non_linear(si_elm_reps.view(batch_size, max_set_len,
                                                       self.out_elm_dim))
        return si_elm_reps


class ElmNetCatSI(torch.nn.Module):
    """
    Function giving a contextualized representation for every element of the set
    the context information for every element is passed as "side-information".
    The network used to combine the side info with the element representation is
    a feedforward network which transforms the element concated with the side info
    embedding. This is more generic than the ElmNetBiSI network.
    """

    def __init__(self, in_elm_dim, elm_si_dim, out_elm_dim, non_linearity='tanh'):
        """
        :param in_elm_dim: int; input element embedding dimensions.
        :param elm_si_dim: int; dimension of the side information embedding.
        :param out_elm_dim: int; dimension of contextualized element embedding.
        :param non_linearity: string. fixed to tanh.
        """
        torch.nn.Module.__init__(self)
        self.in_elm_dim = in_elm_dim
        self.elm_si_dim = elm_si_dim
        self.out_elm_dim = out_elm_dim
        # Add the feedforward net which combines the side information into the
        # relation embedding.
        # Network for complete set of arguments.
        layers = OrderedDict()
        layers['lin_0'] = torch.nn.Linear(in_features=self.in_elm_dim+self.elm_si_dim,
                                          out_features=self.out_elm_dim, bias=False)
        if non_linearity == 'tanh':
            layers['nonlin_0'] = torch.nn.Tanh()
        else:
            sys.stderr.write('Unknown non-linearity: {:s}\n'.format(non_linearity))
        self.cat_si_ffn = torch.nn.Sequential(layers)

    def forward(self, elms, side_info):
        """
        Return representations for the "roles". Embeddings obtained by combining
        individual arguments with the predicates.
        :param elms: Variable(torch.Tensor); batch_size x max_set_len x arg_dim
        :param side_info: Variable(torch.Tensor); batch_size x max_set_len x si_dim
        :return: si_elm_reps: Variable(torch.Tensor); batch_size x seq_len x role_dim
        """
        # Pass the pure role reps through a ffn infusing the additional side information into
        # the relations.
        concated_elms_si = torch.cat([elms, side_info], dim=2)
        si_elm_reps = self.cat_si_ffn(concated_elms_si)
        # Reshape the result to be of the same dimension as the arg input.
        return si_elm_reps


class SetNet(torch.nn.Module):
    """
    Function modeling interactions between all the elements of a set to give an
    final set representation. The set is being modeled as an unordered collection
    of all the elements of the set.
    """
    def __init__(self, in_elm_dim, out_elm_dim, composition_dims, elm_si_comp=None,
                 elm_si_dim=None, non_linearity='tanh', use_bias=True):
        """
        :param in_elm_dim: int; input element embedding dimensions.
        :param out_elm_dim: int; dimension of contextualized element embedding.
        :param composition_dims: tuple(int); says what the dimensions of the ffn
            composing the summed role reps should be.
        :param elm_si_dim: int; dimension of the side information embedding.
        :param elm_si_comp: string; says how side info and element embeddings should
            be combined. {bi/cat} bilinear combination or concat and pass through
            a linear layer.
        :param use_bias: bool; Says whether the outer set net gets to use a bias.
            The element nets dont. Fixing that is a todo.
        """
        torch.nn.Module.__init__(self)
        self.set_dim = composition_dims[-1]
        self.out_elm_dim = out_elm_dim
        self.elm_si_dim = elm_si_dim
        # Define model components.
        # Network for individual element embeddings. If there are elm_si_dim
        # passed, then use the side-info network.
        if self.elm_si_dim and elm_si_comp:
            if elm_si_comp == 'bilinear':
                self.elm_ffn = ElmNetBiSI(in_elm_dim=in_elm_dim, elm_si_dim=elm_si_dim,
                                          out_elm_dim=self.out_elm_dim, non_linearity=non_linearity)
            elif elm_si_comp == 'cat':
                self.elm_ffn = ElmNetCatSI(in_elm_dim=in_elm_dim, elm_si_dim=elm_si_dim,
                                           out_elm_dim=self.out_elm_dim, non_linearity=non_linearity)
            else:
                raise ValueError("Unknow composition {:s} for elm and elm side info comp.".
                                 format(elm_si_comp))
        elif elm_si_comp:
            raise ValueError("Unknow side info dim.")
        else:
            self.elm_ffn = ElmNet(in_elm_dim=in_elm_dim, out_elm_dim=out_elm_dim,
                                  non_linearity=non_linearity)
        # Network for complete set of arguments.
        layers = OrderedDict()
        layers['lin_0'] = torch.nn.Linear(in_features=self.out_elm_dim,
                                          out_features=composition_dims[0], bias=False)
        layers['nonlin_0'] = torch.nn.Tanh()
        # Next append remaining hidden layers.
        if len(composition_dims) > 1:
            for layer_i in range(len(composition_dims) - 1):
                layers['lin_{:d}'.format(layer_i+1)] = \
                    torch.nn.Linear(in_features=composition_dims[layer_i],
                                    out_features=composition_dims[layer_i+1],
                                    bias=use_bias)
                layers['nonlin_{:d}'.format(layer_i + 1)] = torch.nn.Tanh()
        self.set_ffn = torch.nn.Sequential(layers)

    def forward(self, elms, side_info=None):
        """
        Compute event and role representations.
        :param elms: Variable(torch.Tensor); batch_size x max_set_len x arg_dim
        :param side_info: Variable(torch.Tensor); batch_size x max_set_len x si_dim
        :return:
            set_rep: Variable(torch.Tensor); batch_size x event_dim
            elm_reps: Variable(torch.Tensor); batch_size x max_set_len x out_elm_rep
        """
        # Pass through the predicate network to obtain reps for acted upon
        # arguments of size: batch_size x max_seq_len x final_role_dim
        if self.elm_si_dim:
            if isinstance(side_info, type(None)):
                raise ValueError("side_info should be a valid tensor when using side info"
                                 " element network.")
            else:
                elm_reps = self.elm_ffn(elms=elms, side_info=side_info)
        else:
            elm_reps = self.elm_ffn(elms=elms)

        # Sum across arguments to get unordered rep.
        summed_elms = torch.sum(elm_reps, dim=1)
        # Pass through the event_ffn to obtain an event embedding.
        set_rep = self.set_ffn(summed_elms)
        # Maybe it might be better to return role_reps as:
        # batch_size x role_dim x max_seq_len
        return set_rep, elm_reps


class ScoreFFN(torch.nn.Module):
    """
    Given an input pair of features condense the features down with a nn, multiply
    with a set of parameters and yield a scalar score for compatibility of features
    in the pair.
    """
    def __init__(self, x1_dim, x2_dim, non_linearity='tanh'):
        """
        :param x1_dim: int; Dimension of first feature in pair to be scored.
        :param x2_dim: int; Dimension of first feature in pair to be scored.
        :param non_linearity: string; Says what non-linearity to use. {'tanh'}
        """
        torch.nn.Module.__init__(self)
        self.x1_dim = x1_dim
        self.x2_dim = x2_dim
        layers = OrderedDict()
        layers['lin_0'] = torch.nn.Linear(in_features=x1_dim+x2_dim,
                                          out_features=(x1_dim+x2_dim)//2, bias=True)
        if non_linearity == 'tanh':
            layers['nonlin_0'] = torch.nn.Tanh()
        layers['lin_1'] = torch.nn.Linear(in_features=(x1_dim+x2_dim)//2,
                                          out_features=1, bias=True)
        # Define a network which predicts a single score.
        self.score_ffn = torch.nn.Sequential(layers)

    def forward(self, x1, x2):
        """
        :param x1: torch Tensor; [batch_size x x1_sim]
        :param x2: torch Tensor; [batch_size x x1_sim]
        :return: scores: torch Tensor; [batch_size x 1]
        """
        concated = torch.cat([x1, x2], dim=1)
        return self.score_ffn.forward(concated)


if __name__ == '__main__':
    a = np.random.randint(0, 5, (5, 4, 3))
    si = np.random.randint(0, 5, (5, 4, 2))
    le = [1, 2, 1, 2, 4]
    for i, l in enumerate(le):
        a[i, l:, :] = 0.0
        si[i, l:, :] = 0.0
    a = torch.FloatTensor(a)
    si = torch.FloatTensor(si)
    print(a.size())
    print(si.size())
    # Predicate and predicate+side_info nets
    if sys.argv[1] == 'test_elmnet':
        pf = ElmNet(in_elm_dim=3, out_elm_dim=3)
        print(pf)
        print(pf.forward(Variable(a)))
    elif sys.argv[1] == 'test_elmnetbi':
        pfsi = ElmNetBiSI(in_elm_dim=3, elm_si_dim=2, out_elm_dim=3)
        print(pfsi)
        print(pfsi.forward(elms=Variable(a), side_info=Variable(si)))
    elif sys.argv[1] == 'test_elmnetcat':
        pfsi = ElmNetCatSI(in_elm_dim=3, elm_si_dim=2, out_elm_dim=3)
        print(pfsi)
        print(pfsi.forward(elms=Variable(a), side_info=Variable(si)))
    elif sys.argv[1] == 'test_dsnet':
        ds = SetNet(in_elm_dim=3, out_elm_dim=3, composition_dims=(3, 4))
        print(ds)
        print(ds.forward(Variable(a)))
    elif sys.argv[1] == 'test_dsnetcat':
        ds = SetNet(in_elm_dim=3, out_elm_dim=3, elm_si_dim=2,
                    composition_dims=(3, 4), elm_si_comp='cat')
        print(ds)
        print(ds.forward(elms=Variable(a), side_info=Variable(si)))
    elif sys.argv[1] == 'test_dsnetbi':
        ds = SetNet(in_elm_dim=3, out_elm_dim=3, elm_si_dim=2,
                    composition_dims=(3, 4), elm_si_comp='bilinear')
        print(ds)
        print(ds.forward(elms=Variable(a), side_info=Variable(si)))
    else:
        print('Unknown action: {:s}'.format(sys.argv[1]))
