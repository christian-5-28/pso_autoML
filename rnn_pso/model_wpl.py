import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from genotypes import STEPS
import logging
from utils import mask2d
from utils import LockedDropout
from utils import embedded_dropout
from torch.autograd import Variable
import numpy as np

INITRANGE = 0.04

model_logger = logging.getLogger("train_search.model")


class DARTSCell(nn.Module):
    """
    torch module class representing the DARTS cell
    """

    def __init__(self, ninp, nhid, dropouth, dropoutx, genotype, use_edge_matrices, use_glorot,
                 steps,
                 handle_hidden_mode=None):
        """
        :param ninp: input size (word embedding size)
        :param nhid: hidden state size (number of hidden units of the FC layer in a rnn cell)
        :param dropouth: dropout rate for hidden
        :param dropoutx: dropout rate for input
        :param genotype: string representation of the cell (description of the edges involved)
        """
        super(DARTSCell, self).__init__()
        self.nhid = nhid
        self.dropouth = dropouth
        self.dropoutx = dropoutx
        self.genotype = genotype
        self.handle_hidden_mode = handle_hidden_mode
        self.use_edge_matrices = use_edge_matrices

        model_logger.info("CELL INITIALIZED ")

        # IMPORTANT: genotype is None when doing arch search
        # the steps are equal to the number of intermediate nodes
        # in the cell (default 4 nodes for cnns, 8 nodes for rnn)
        self.steps = len(self.genotype.recurrent) if self.genotype is not None else steps

        # initializing the first weight matrix between input x and the hidden layer
        self._W0 = nn.Parameter(torch.Tensor(ninp + nhid, 2 * nhid).uniform_(-INITRANGE, INITRANGE))

        # initializing the weight matrices used towards the intermediate nodes (range of steps)
        # TODO(ASK): vedi nota quaderno riguardo inizializzazione delle matrici

        if not self.use_edge_matrices:

            self._Ws = nn.ParameterList([
                nn.Parameter(torch.Tensor(nhid, 2 * nhid).uniform_(-INITRANGE, INITRANGE)) for _ in range(self.steps)
                                        ])
        else:
            self.max_num_edges = sum(num for num in range(1, self.steps + 1))
            self._Ws = nn.ParameterList([
                nn.Parameter(torch.Tensor(nhid, 2 * nhid).uniform_(-INITRANGE, INITRANGE)) for _ in range(self.max_num_edges)
                                        ])
        # WPL PARAMS
        self._Ws_opt = [
            Variable(torch.Tensor(nhid, 2 * nhid).uniform_(-INITRANGE, INITRANGE), requires_grad=False)
            for _ in range(self.steps)]
        self._Ws_fisher = [Variable(torch.Tensor(nhid, 2 * nhid).zero_(), requires_grad=False)
                           for _ in range(self.steps)]

        if use_glorot:
            for param in self._Ws:
                nn.init.xavier_normal(param)

    # #### WPL METHODS ####

    def _apply(self, fn):
        super(DARTSCell, self)._apply(fn)
        self._Ws_opt = [fn(opt) for opt in self._Ws_opt]
        self._Ws_fisher = [fn(fisher) for fisher in self._Ws_fisher]
        return self

    def fisher_alpha(self, epoch, params):
        alpha = params.alpha_fisher

        if epoch >= params.alpha_decay_after:
            degree = max(epoch - params.alpha_decay_after + 1, 0)
            alpha = max(params.alpha_fisher - degree * params.alpha_decay, 0)
        return alpha

    def update_fisher(self, genotype, epoch, params):
        def _update_fisher(fisher, p):
            """ Update fisher information logic """

            alpha = self.fisher_alpha(epoch, params)
            # fisher = fisher.cuda() + params['lambda_fisher'] * p.data.clone() ** 2
            fisher.data += alpha * params.lambda_fisher * p.data.clone() ** 2
            if fisher.data.norm() > params.fisher_clip_by_norm:
                fisher /= fisher.norm() * params.fisher_clip_by_norm
            return fisher

        for i in range(self.steps):
            # pred, act = genotype.recurrent[i]
            ws_grad = self._Ws[i].grad
            self._Ws_fisher[i] = _update_fisher(self._Ws_fisher[i], ws_grad)

    def compute_weight_plastic_loss_with_update_fisher(self, genotype, params=None):
        """
        Return the weight layer (can be freely accessed)

        based on dag figure
        - Update the gradient as fisher information
        - return loss

        :param dag: list of dags
        :return: loss function term. with Fisher information.
        """
        loss = 0
        for i in range(self.steps):

            # IPython.embed()
            ws = self._Ws[i]
            ws_opt = self._Ws_opt[i]
            ws_fisher = self._Ws_fisher[i]

            ws_diff = (ws - ws_opt) ** 2

            try:
                # loss += (ws_fisher.cuda() * ws_diff).sum()
                loss += (ws_fisher * ws_diff).sum()
                # print(f"Fisher loss {loss} with fisher norm {ws_fisher.norm(2)[0]} and diff norm {ws_diff.norm(2)}")
            except RuntimeError as e:
                print("Got error {e}")
                self.cuda()
                # Run self.cuda() to move all the values to GPU, even again.
        return loss

    def update_optimal_weights(self):
        """ Update the weights with optimal """
        for i in range(self.steps):
            self._Ws_opt[i].data = self._Ws[i].data.clone()

    def set_fisher_zero(self):
        """ Sets the fisher information to zero """
        self._Ws_fisher = [fisher.zero_() for fisher in self._Ws_fisher]

    def fisher_norm(self):
        n = 0
        for i, fisher in enumerate(self._Ws_fisher):
            norm = fisher.data.norm()
            model_logger.info('fisher norm of node {}: {}'.format(i+1, norm))
            n += norm
        return n

    def forward(self, inputs, hidden):
        T, B = inputs.size(0), inputs.size(1)

        if self.training:
            x_mask = mask2d(B, inputs.size(2), keep_prob=1. - self.dropoutx)
            h_mask = mask2d(B, hidden.size(2), keep_prob=1. - self.dropouth)
        else:
            x_mask = h_mask = None

        hidden = hidden[0]
        hiddens = []
        clipped_num = 0
        max_clipped_norm = 0

        # forward pass through time in the cell, T is the sequence length
        for t in range(T):

            """
            if hidden.max().data[0] > 1000 or inputs.max().data[0] > 1000:
                model_logger.info("FORWARD IN CELL CLASS, EXPLODING IN step " + str(t))
                model_logger.info("hidden max: " + str(hidden.max().data[0]))
                model_logger.info("input max: " + str(inputs.max().data[0]))
            """
            # main forward inside the cell
            hidden = self.cell(inputs[t], hidden, x_mask, h_mask)

            if self.handle_hidden_mode == 'NORM':
                hidden_norms = hidden.norm(dim=-1)
                max_norm = 25.0
                if hidden_norms.data.max() > max_norm:
                    # Just directly use the torch slice operations
                    # in PyTorch v0.4.
                    #
                    # This workaround for PyTorch v0.3.1 does everything in numpy,
                    # because the PyTorch slicing and slice assignment is too
                    # flaky.
                    hidden_norms = hidden_norms.data.cpu().numpy()

                    clipped_num += 1
                    if hidden_norms.max() > max_clipped_norm:
                        max_clipped_norm = hidden_norms.max()

                    clip_select = hidden_norms > max_norm
                    clip_norms = hidden_norms[clip_select]

                    mask = np.ones(hidden.size())
                    normalizer = max_norm / clip_norms
                    normalizer = normalizer[:, np.newaxis]

                    mask[clip_select] = normalizer
                    hidden *= torch.autograd.Variable(
                        torch.FloatTensor(mask).cuda(), requires_grad=False)

            # saving the hidden output for each step
            hiddens.append(hidden)

        if clipped_num > 0:
            model_logger.info('clipped {} hidden states in one forward '.format(clipped_num))
            model_logger.info('max clipped hidden state norm: {}'.format(max_clipped_norm))

        # creating a tensor from the list of hiddens in order to have the elements stacked
        hiddens = torch.stack(hiddens)

        # return the stack of hidden outputs and the hidden output of the last step
        return hiddens, hiddens[-1].unsqueeze(0)

    def _compute_init_state(self, x, h_prev, x_mask, h_mask):

        if self.training:
            try:
                xh_prev = torch.cat([x * x_mask, h_prev * h_mask], dim=-1)
            except Exception as e:
                model_logger.info('x size: {}'.format(x.size()))
                model_logger.info('x_mask size: {}'.format(x_mask.size()))
                model_logger.info('h size: {}'.format(h_prev.size()))
                model_logger.info('h_mask size: {}'.format(h_mask.size()))
                raise

        else:
            xh_prev = torch.cat([x, h_prev], dim=-1)
        c0, h0 = torch.split(xh_prev.mm(self._W0), self.nhid, dim=-1)
        c0 = c0.sigmoid()
        h0 = h0.tanh()
        s0 = h_prev + c0 * (h0 - h_prev)

        # state could explode, clipping using its norm
        '''
        norm = s0.norm().data[0]
        if norm > 10:
            coeff = 10 / norm
            s0 = s0 * coeff
        '''

        """
        if s0.max().data[0] > 1000:
            model_logger.info("COMPUTED INITIAL DAG STATE, EXPLODING VALUE:")
            model_logger.info("input X max value: " + str(x.max().data[0]))
            model_logger.info("first hidden state max value: " + str(h_prev.max().data[0]))
            model_logger.info("concat XH max value: " + str(xh_prev.max().data[0]))
            model_logger.info("concat c0 max value: " + str(c0.max().data[0]))
            model_logger.info("concat h0 max value: " + str(h0.max().data[0]))
            model_logger.info("x_mask max value: " + str(x_mask.max().data[0]))
            model_logger.info("x * x_mask max value: " + str((x * x_mask).max().data[0]))
            model_logger.info("h_mask max value: " + str(h_mask.max().data[0]))
            model_logger.info("h_prev * h_mask max value: " + str((h_prev * h_mask).max().data[0]))
            model_logger.info("xh mul w0 max value: " + str((xh_prev.mm(self._W0)).max().data[0]))
            model_logger.info("W0 max value: " + str(self._W0.max().data[0]))
        """

        return s0

    def _get_activation(self, name):
        if name == 'tanh':
            f = F.tanh
        elif name == 'relu':
            f = F.relu
        elif name == 'sigmoid':
            f = F.sigmoid
        elif name == 'identity':
            f = lambda x: x
        elif name == 'selu':
            f = torch.nn.functional.selu
        else:
            raise NotImplementedError
        return f

    def cell(self, x, h_prev, x_mask, h_mask):
        """
        forwards inside the cell of our model
        :param x: input
        :param h_prev: hidden of previous step
        :param x_mask: mask for input dropout
        :param h_mask: mask for hidden dropout
        :return: the hidden output of the current step
        """

        s0 = self._compute_init_state(x, h_prev, x_mask, h_mask)

        # states contains the nodes computed as described in the paper,
        # that means, the sum of output of the operations of the incoming
        # edges. If genotype defined, there is only one incoming edge
        # as a constraint described in the paper.
        states = [s0]

        # IMPORTANT: genotype is None when doing arch search
        # "i" is the index of the next intermediate node,
        # "name" is the label of the activation function,
        # "pred" is the index of the previous node, so the edge will be pred -->name--->i
        for i, (name, pred) in enumerate(self.genotype.recurrent):

            # taking the previous state using its index
            s_prev = states[pred]

            if self.use_edge_matrices:
                # sum of first (i-1) natural numbers plus the previous node index
                edge_weights_id = sum(num for num in range(1, i)) + pred

            else:
                edge_weights_id = i

            # applying dropout masks if training.
            # computing the matrix mul between the previous output
            # and the weights of the current node "i" (FC layer)
            if self.training:
                ch = (s_prev * h_mask).mm(self._Ws[edge_weights_id])
            else:
                ch = s_prev.mm(self._Ws[edge_weights_id])
            c, h = torch.split(ch, self.nhid, dim=-1)
            c = c.sigmoid()

            # getting the chosen activation function
            fn = self._get_activation(name)

            # activation function on hidden
            h = fn(h)

            s = s_prev + c * (h - s_prev)

            states += [s]

        # computing the output as the mean of the output of
        # the INTERMEDIATE nodes, where their index are
        # defined by the "concat" list in the genotype
        output = torch.mean(torch.stack([states[i] for i in self.genotype.concat], -1), -1)

        if self.handle_hidden_mode == 'ACTIVATION':
            # avoid the explosion of the hidden state by forcing the value in [-1, 1]
            output = F.tanh(output)
        return output

    def set_genotype(self, genotype):
        """
        setting a new genotype for the DAG
        :param genotype: new genotype to be used in the forward of the cell
        """
        self.genotype = genotype


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self,
                 ntoken,
                 args,
                 cell_cls=DARTSCell,
                 genotype=None):

        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.encoder = nn.Embedding(ntoken, args.emsize)
        self.nlayers = args.nlayers
        self.rnns = []

        assert args.emsize == args.nhid == args.nhidlast

        for layers in range(args.nlayers):
            if cell_cls == DARTSCell:
                assert genotype is not None
                self.rnns.append(cell_cls(args.emsize,
                                          args.nhid,
                                          args.dropouth,
                                          args.dropoutx,
                                          genotype,
                                          args.use_matrices_on_edge,
                                          args.use_glorot,
                                          args.num_intermediate_nodes,
                                          args.handle_hidden_mode))
            else:
                assert genotype is None
                # TODO: for the init of DartsCell there is no default value for genotype,
                # next line will raise error in theory
                self.rnns = [cell_cls(args.emsize, args.nhid, args.dropouth, args.dropoutx)]

        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(args.emsize, ntoken)
        self.decoder.weight = self.encoder.weight
        self.init_weights()

        self.ninp = args.emsize
        self.nhid = args.nhid
        self.nhidlast = args.nhidlast
        self.dropout = args.dropout
        self.dropouti = args.dropouti
        self.dropoute = args.dropoute
        self.ntoken = ntoken
        self.cell_cls = cell_cls

        model_logger.info("MODEL INITIALIZED ")

    def init_weights(self):
        self.encoder.weight.data.uniform_(-INITRANGE, INITRANGE)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-INITRANGE, INITRANGE)

    def forward(self, input, hidden, return_h=False):
        batch_size = input.size(1)

        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        logit = self.decoder(output.view(-1, self.ninp))
        log_prob = nn.functional.log_softmax(logit, dim=-1)
        model_output = log_prob
        model_output = model_output.view(-1, batch_size, self.ntoken)

        if return_h:
            return model_output, hidden, raw_outputs, outputs
        return model_output, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        init_hidden_list = list()
        for layer in range(self.nlayers):
            init_h = Variable(weight.new(1, bsz, self.nhid).zero_())
            '''
            if layer == self.nlayers - 1 and self.tie_weights:
                init_h = Variable(weight.new(1, bsz, self.ninp).zero_())
            '''
            init_hidden_list.append(init_h)
        return init_hidden_list

    def change_genotype(self, genotype):

        for rnn in self.rnns:
            rnn.set_genotype(genotype=genotype)

    def genotype(self):
        return self.rnns[0].genotype


class CustomDataParallel(nn.DataParallel):

    def __init__(self, model):
        super(CustomDataParallel, self).__init__(model)
        self.model = model

    def init_hidden(self, bsz):
        return self.model.init_hidden(bsz=bsz)

    def change_genotype(self, genotype):
        self.model.change_genotype(genotype=genotype)

    def genotype(self):
        return self.model.genotype()
