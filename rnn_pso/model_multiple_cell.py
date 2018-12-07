import torch.nn as nn
import torch
from torch.autograd import Variable


class DagCellTorch(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DagCellTorch, self).__init__()

        self.ninp = input_dim
        self.nhid = hidden_dim

        # Shared Weight matrices
        self.w_xc = nn.Linear(self.ninp, self.nhid)
        self.w_xh = nn.Linear(self.ninp, self.nhid)

        self.w_hc = nn.Parameter(torch.Tensor(self.nhid, self.nhid))
        self.w_hh = nn.Parameter(torch.Tensor(self.nhid, self.nhid))

        self.w_h = {}
        self.w_c = {}
        self.connections = {}
        self._connections = nn.ModuleList()

        self.dag = load_dag()
        self.prepare_shared_weights()
        self.init_weights(0.1)
        self.handle_hidden_mode = 'NORM'

    def forward(self, inputs, hidden):
        """
        forward pass of a RNN cell
        """

        # sequence length init
        time_steps = inputs.size(0)

        # list to collect the outputs
        outputs = list()
        clipped_num = 0
        max_clipped_norm = 0

        # main part of the forward pass (unrolling)
        for step in range(time_steps):
            x_t = inputs[step]
            output, hidden = self.cell(x_t, hidden)
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
                    output *= torch.autograd.Variable(
                        torch.FloatTensor(mask).cuda(), requires_grad=False)

            if clipped_num > 0:
                print('clipped {} hidden states in one forward '.format(clipped_num))
                print('max clipped hidden state norm: {}'.format(max_clipped_norm))

            outputs.append(output)
            # print('OUTPUT SHAPE:', output.size())

        outputs = torch.cat(outputs)
        return outputs, hidden

    def cell(self, x_t, h_prev):
        """Computes a single pass through the discovered RNN cell."""
        c = {}
        h = {}
        f = {}

        f[0] = self.get_f(self.dag[-1][0].name)
        c[0] = F.sigmoid(self.w_xc(x_t) + F.linear(h_prev, self.w_hc, None))
        h[0] = (c[0] * f[0](self.w_xh(x_t) + F.linear(h_prev, self.w_hh, None)) +
                (1 - c[0]) * h_prev)

        leaf_node_ids = []
        q = collections.deque()
        q.append(0)

        while True:
            if len(q) == 0:
                break

            node_id = q.popleft()
            nodes = self.dag[node_id]

            for next_node in nodes:
                next_id = next_node.id
                if next_id == config["num_nodes"]:
                    leaf_node_ids.append(node_id)
                    assert len(nodes) == 1, ('parent of leaf node should have '
                                             'only one child')
                    continue

                w_h = self.w_h[node_id][next_id]
                w_c = self.w_c[node_id][next_id]

                f[next_id] = self.get_f(next_node.name)
                c[next_id] = F.sigmoid(w_c(h[node_id]))
                h[next_id] = (c[next_id] * f[next_id](w_h(h[node_id])) +
                              (1 - c[next_id]) * h[node_id])

                q.append(next_id)

        # average all the loose ends
        leaf_nodes = [h[node_id] for node_id in leaf_node_ids]
        output = torch.mean(torch.stack(leaf_nodes, 2), 2)

        return output, output

    def get_f(self, name):
        name = name.lower()
        if name == 'relu':
            f = F.relu
        elif name == 'tanh':
            f = F.tanh
        elif name == 'identity':
            f = lambda x: x
        elif name == 'sigmoid':
            f = F.sigmoid

        return f

    def prepare_shared_weights(self):

        q = collections.deque()
        q.append(0)
        while True:
            if len(q) == 0:
                break

            node_id = q.popleft()
            nodes = self.dag[node_id]  # TODO What is dag actually?

            for next_node in nodes:
                next_id = next_node.id
                if next_id == config['num_nodes']:
                    # leaf_node_ids.append(node_id)
                    assert len(nodes) == 1, ('parent of leaf node should have '
                                             'only one child')
                    continue

                key_h = node_to_key((node_id, next_id, 'h'))
                key_c = node_to_key((node_id, next_id, 'c'))

                if node_id not in self.w_c:
                    self.w_c[node_id] = {}
                    self.w_h[node_id] = {}
                    self.w_c[node_id][next_id] = self.connections[key_c] = nn.Linear(self.nhid, self.nhid, bias=False).cuda()
                    self._connections.append(self.w_c[node_id][next_id])
                    self.w_h[node_id][next_id] = self.connections[key_h] = nn.Linear(self.nhid, self.nhid, bias=False).cuda()
                    self._connections.append(self.w_h[node_id][next_id])

                else:
                    self.w_c[node_id][next_id] = self.connections[key_c] = nn.Linear(self.nhid, self.nhid, bias=False).cuda()
                    self._connections.append(self.w_c[node_id][next_id])
                    self.w_h[node_id][next_id] = self.connections[key_h] = nn.Linear(self.nhid, self.nhid, bias=False).cuda()
                    self._connections.append(self.w_h[node_id][next_id])

                q.append(next_id)

    def init_weights(self, init_range):
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        for param in self._parameters.values():
            if param is not None:
                param.data = fn(param.data)
                if param._grad is not None:
                    param._grad.data = fn(param._grad.data)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        return self

# DARTS CELL #
class DARTSCell(nn.Module):
    """
    torch module class representing the DARTS cell
    """

    def __init__(self, ninp, nhid, dropouth, dropoutx, genotype, handle_hidden_mode=None):
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

        model_logger.info("CELL INITIALIZED ")

        # IMPORTANT: genotype is None when doing arch search
        # the steps are equal to the number of intermediate nodes
        # in the cell (default 4 nodes for cnns, 8 nodes for rnn)
        steps = len(self.genotype.recurrent) if self.genotype is not None else STEPS

        # initializing the first weight matrix between input x and the hidden layer
        self._W0 = nn.Parameter(torch.Tensor(ninp + nhid, 2 * nhid).uniform_(-INITRANGE, INITRANGE))

        # initializing the weight matrices used towards the intermediate nodes (range of steps)
        # TODO(ASK): vedi nota quaderno riguardo inizializzazione delle matrici
        self._Ws = nn.ParameterList([
            nn.Parameter(torch.Tensor(nhid, 2 * nhid).uniform_(-INITRANGE, INITRANGE)) for _ in range(steps)
                                    ])

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
            xh_prev = torch.cat([x * x_mask, h_prev * h_mask], dim=-1)
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

            # applying dropout masks if training.
            # computing the matrix mul between the previous output
            # and the weights of the current node "i" (FC layer)
            if self.training:
                ch = (s_prev * h_mask).mm(self._Ws[i])
            else:
                ch = s_prev.mm(self._Ws[i])
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

    def __init__(self, args, ntoken):
        """
        :param cell_type: type string (LSTM, DAGCELL)
        :param ntoken: Size of the vocabulary
        :param ninp: Embedding dimension
        :param nhid: Hidden unit dimension
        :param nlayers: Number of layers
        :param dropout: output dropout rate
        :param dropouth: hidden dropout rate
        :param dropouti: input dropout rate
        :param dropoute: embedding dropout rate
        :param tie_weights: Weight tying
        """

        super(RNNModel, self).__init__()
        self.cell_type = args.cell_type
        if args.cell_type in ['DAGCELL', 'ENASCELL']:
            assert args.nlayers == 1, 'For DAGCELL and ENASCELL only one layer'

        self.encoder = nn.Embedding(ntoken, args.emsize)  # Embedding layer n_token x n_input

        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(args.dropouti)
        self.hdrop = nn.Dropout(args.dropouth)
        self.drop = nn.Dropout(args.dropout)

        self.rnns = list()

        for layer in range(args.nlayers):
            cell_in_dim = args.nhid
            if layer == 0:
                cell_in_dim = args.emsize

            cell_h_dim = args.nhid
            if layer == args.nlayers - 1 and args.tied:
                cell_h_dim = args.emsize

            if self.cell_type == 'LSTM':
                print('USING LSTM!')
                self.rnns.append(nn.LSTM(cell_in_dim, cell_h_dim))
            if self.cell_type == 'GRU':
                self.rnns.append(nn.GRU(cell_in_dim, cell_h_dim, 1, dropout=0))
            elif self.cell_type == 'DAGCELL':
                self.rnns.append(DagCellTorch(cell_in_dim, cell_h_dim))
            elif self.cell_type == 'ENASCELL':
                self.rnns.append(EnasCell(cell_in_dim, cell_h_dim))

        self.rnns = nn.ModuleList(self.rnns)

        self.decoder = nn.Linear(args.nhid, ntoken)
        if args.tied:
            # will tie the weight now will be ntoken x ninp and equal to the encoder weights
            self.decoder.weight = self.encoder.weight

        self.init_weights(0.04)

        self.ninp = args.emsize
        self.nhid = args.nhid
        self.nlayers = args.nlayers
        self.dropout = args.dropout
        self.dropouti = args.dropouti
        self.dropouth = args.dropouth
        self.dropoute = args.dropoute
        self.tie_weights = args.tied

    def init_weights(self, initrange):
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_h=False):
        # Embedding with embedding dropout (drop full word)
        # emb dimension is sequence_length x batch x ninp
        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        # Embeddings with classical dropout
        emb = self.lockdrop(emb, self.dropouti)

        # Multilayer RNN forward

        new_hidden = list()
        outputs = list()
        raw_outputs = list()

        # loop on different layers of RNN cells, like stack of LSTMs
        current_input = emb
        for layer, rnn in enumerate(self.rnns):
            raw_output, new_h = rnn(current_input, hidden[layer])
            new_hidden.append(new_h)

            raw_outputs.append(raw_output)

            # clean the outputs between one RNN cell adn the next one
            if layer != self.nlayers - 1:
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)

            current_input = raw_output

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        # forward in decoder to project in the original dimension of the tokens
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        result = decoded.view(output.size(0), output.size(1), decoded.size(1))

        if return_h:
            return result, new_hidden, raw_outputs, outputs
        return result, new_hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data  # only here to get the type of the weight tensor

        if self.cell_type == 'LSTM':

            init_hidden_list = list()  # nlayer tuples (c0, h0)
            for layer in range(self.nlayers):
                init_h = Variable(weight.new(1, batch_size, self.nhid).zero_())
                init_c = Variable(weight.new(1, batch_size, self.nhid).zero_())
                if layer == self.nlayers - 1 and self.tie_weights:
                    init_h = Variable(weight.new(1, batch_size, self.ninp).zero_())
                    init_c = Variable(weight.new(1, batch_size, self.ninp).zero_())
                init_hidden_list.append((init_h, init_c))

        elif self.cell_type == 'QRNN' or self.cell_type == 'GRU':
            return [weight.new(1, batch_size, self.nhid if l != self.nlayers - 1 else (
                self.ninp if self.tie_weights else self.nhid)).zero_()
                    for l in range(self.nlayers)]

        elif self.cell_type in ['DAGCELL', 'ENASCELL']:
            init_hidden_list = list()
            for layer in range(self.nlayers):
                init_h = Variable(weight.new(1, batch_size, self.nhid).zero_())
                if layer == self.nlayers - 1 and self.tie_weights:
                    init_h = Variable(weight.new(1, batch_size, self.ninp).zero_())
                init_hidden_list.append(init_h)

        return init_hidden_list


def embedded_dropout(embed, words, dropout=0.1, scale=None):
    if dropout:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(
            embed.weight) / (1 - dropout)
        mask = mask
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight
    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight
    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1
    X = torch.nn.functional.embedding(words, masked_embed_weight, padding_idx,
                                      embed.max_norm, embed.norm_type,
                                      embed.scale_grad_by_freq, embed.sparse)
    return X


class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):

        # print("DROPOUT FORWARD X SIZE", x.size())
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x