# -*- coding: utf-8 -*-

r"""
HIDE
################################################

Reference:
    Yinfeng Li et al. "Enhancing Hypergraph Neural Networks with Intent Disentanglement for Session-based Recommendation." in SIGIR 22.

Reference code:
    https://github.com/yf-li15/HIDE

"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss


import math
import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
import scipy.sparse as sp

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss


class DisentangleGraph(nn.Module):
    def __init__(self, dim, alpha, e=0.3, t=10.0):
        super(DisentangleGraph, self).__init__()
        self.latent_dim = dim
        self.e = e
        self.t = t
        self.w = nn.Parameter(torch.Tensor(self.latent_dim, self.latent_dim))
        self.w1 = nn.Parameter(torch.Tensor(self.latent_dim, 1))
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, hidden, H, int_emb, mask):
        node_num = torch.sum(mask, dim=1, keepdim=True).unsqueeze(-1)
        select_k = self.e * node_num
        select_k = select_k.floor()
        mask = mask.float().unsqueeze(-1)
        h = hidden
        batch_size = h.shape[0]
        N = H.shape[1]
        k = int_emb.shape[0]
        select_k = select_k.repeat(1, N, k)
        int_emb = int_emb.unsqueeze(0).repeat(batch_size, 1, 1)
        int_emb = int_emb.unsqueeze(1).repeat(1, N, 1, 1)
        hs = h.unsqueeze(2).repeat(1, 1, k, 1)
        cos = nn.CosineSimilarity(dim=-1)
        sim_val = self.t * cos(hs, int_emb)
        sim_val = sim_val * mask
        _, indices = torch.sort(sim_val, dim=1, descending=True)
        _, idx = torch.sort(indices, dim=1)
        judge_vec = idx - select_k
        ones_vec = 3 * torch.ones_like(sim_val)
        zeros_vec = torch.zeros_like(sim_val)
        int_H = torch.where(judge_vec <= 0, ones_vec, zeros_vec)
        H_out = torch.cat([int_H, H], dim=-1)
        return H_out


class LocalHyperGATlayer(nn.Module):
    def __init__(self, dim, layer, alpha, dropout=0., bias=False, act=True):
        super(LocalHyperGATlayer, self).__init__()
        self.dim = dim
        self.layer = layer
        self.alpha = alpha
        self.dropout = dropout
        self.bias = bias
        self.act = act
        if self.act:
            self.acf = torch.relu
        self.w1 = Parameter(torch.Tensor(self.dim, self.dim))
        self.w2 = Parameter(torch.Tensor(self.dim, self.dim))
        self.a10 = nn.Parameter(torch.Tensor(size=(self.dim, 1)))
        self.a11 = nn.Parameter(torch.Tensor(size=(self.dim, 1)))
        self.a12 = nn.Parameter(torch.Tensor(size=(self.dim, 1)))
        self.a20 = nn.Parameter(torch.Tensor(size=(self.dim, 1)))
        self.a21 = nn.Parameter(torch.Tensor(size=(self.dim, 1)))
        self.a22 = nn.Parameter(torch.Tensor(size=(self.dim, 1)))
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, hidden, H, s_c):
        batch_size = hidden.shape[0]
        N = H.shape[1]
        edge_num = H.shape[2]
        H_adj = torch.ones_like(H)
        mask = torch.zeros_like(H)
        H_adj = torch.where(H > 0, H_adj, mask)
        s_c = s_c.expand(-1, N, -1)
        h_emb = hidden
        h_embs = []
        for i in range(self.layer):
            edge_cluster = torch.matmul(H_adj.transpose(1, 2), h_emb)
            h_t_cluster = h_emb + s_c
            edge_c_in = edge_cluster.unsqueeze(1).expand(-1, N, -1, -1)
            h_4att0 = h_emb.unsqueeze(2).expand(-1, -1, edge_num, -1)
            feat = edge_c_in * h_4att0
            atts10 = self.leakyrelu(torch.matmul(feat, self.a10).squeeze(-1))
            atts11 = self.leakyrelu(torch.matmul(feat, self.a11).squeeze(-1))
            atts12 = self.leakyrelu(torch.matmul(feat, self.a12).squeeze(-1))
            zero_vec = -9e15 * torch.ones_like(H)
            alpha1 = torch.where(H.eq(1), atts10, zero_vec)
            alpha1 = torch.where(H.eq(2), atts11, alpha1)
            alpha1 = torch.where(H.eq(3), atts12, alpha1)
            alpha1 = F.softmax(alpha1, dim=1)
            edge = torch.matmul(alpha1.transpose(1, 2), h_emb)
            edge_in = edge.unsqueeze(1).expand(-1, N, -1, -1)
            h_4att1 = h_t_cluster.unsqueeze(2).expand(-1, -1, edge_num, -1)
            feat_e2n = edge_in * h_4att1
            atts20 = self.leakyrelu(torch.matmul(feat_e2n, self.a20).squeeze(-1))
            atts21 = self.leakyrelu(torch.matmul(feat_e2n, self.a21).squeeze(-1))
            atts22 = self.leakyrelu(torch.matmul(feat_e2n, self.a22).squeeze(-1))
            alpha2 = torch.where(H.eq(1), atts20, zero_vec)
            alpha2 = torch.where(H.eq(2), atts21, alpha2)
            alpha2 = torch.where(H.eq(3), atts22, alpha2)
            alpha2 = F.softmax(alpha2, dim=2)
            h_emb = torch.matmul(alpha2, edge)
            h_embs.append(h_emb)
        h_embs = torch.stack(h_embs, dim=1)
        h_out = torch.sum(h_embs, dim=1)
        return h_out


class HIDE(SequentialRecommender):
    def __init__(self, config, dataset):
        super(HIDE, self).__init__(config, dataset)
        self.dim = config["embedding_size"]
        self.step = config["step"]
        self.device = config["device"]
        self.loss_type = config["loss_type"]

        self.n_layer = config["n_layer"]
        self.dropout_gcn = config["dropout_gcn"]
        self.n_factor = config["n_factor"]
        self.sparsity = config["sparsity"]

        self.alpha = config["alpha"]
        self.intent_aware_emb = config["intent_aware_emb"]
        self.intent_loss_weight = config["intent_loss_weight"]
        self.w_k = config["w_k"]
        self.sw = [2]
        self.max_edge_num = 100 # TODO

        self.item_embedding = nn.Embedding(
            self.n_items, self.dim, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.dim)

        if self.intent_aware_emb:
            self.feat_latent_dim = self.dim // self.n_factor
            self.split_sections = [self.feat_latent_dim] * self.n_factor
        else:
            self.feat_latent_dim = self.dim

        if self.intent_aware_emb:
            self.disen_graph = DisentangleGraph(dim=self.feat_latent_dim, alpha=config["alpha"], e=self.sparsity)
            self.disen_aggs = nn.ModuleList([LocalHyperGATlayer(self.feat_latent_dim, self.n_layer, config["alpha"], config["dropout_gcn"]) for _ in range(self.n_factor)])
        else:
            self.local_agg = LocalHyperGATlayer(self.dim, self.n_layer, config["alpha"], config["dropout_gcn"])

        self.w_1 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(3 * self.dim, 1))
        self.w_s = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.glu1 = nn.Linear(self.dim, self.dim, bias=True)
        self.glu2 = nn.Linear(self.dim, self.dim, bias=True)
        self.glu3 = nn.Linear(self.dim, self.dim, bias=True)

        self.leakyrelu = nn.LeakyReLU(config["alpha"])

        if self.intent_aware_emb:
            self.classifier = nn.Linear(self.feat_latent_dim, self.n_factor)
            self.loss_aux = nn.CrossEntropyLoss()
            self.intent_loss = 0

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self._reset_parameters()

    def _reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def _get_slice(self, item_seq):
        max_n_node = item_seq.size(1)
        max_n_edge = self.max_edge_num

        item_seq = item_seq.cpu().numpy()

        all_alias_inputs, all_Hs, all_items, all_mask = [], [], [], []

        for u_input in item_seq:
            node = np.unique(u_input)
            items = node.tolist() + (max_n_node - len(node)) * [0]
            alias_inputs = [np.where(node == i)[0][0] for i in u_input]
            mask = [1] * len(node) + [0] * (max_n_node - len(node))

            rows, cols, vals = [], [], []
            edge_idx = 0

            # generate slide window hyperedge
            for win in self.sw:
                for i in range(len(u_input) - win + 1):
                    if i + win <= len(u_input):
                        if u_input[i + win - 1] == 0:
                            break
                        for j in range(i, i + win):
                            rows.append(np.where(node == u_input[j])[0][0])
                            cols.append(edge_idx)
                            vals.append(1.0)
                        edge_idx += 1

            # generate in-item hyperedge, ignore 0
            for item in node:
                if item != 0:
                    for i in range(len(u_input)):
                        if u_input[i] == item and i > 0:
                            rows.append(np.where(node == u_input[i - 1])[0][0])
                            cols.append(edge_idx)
                            vals.append(2.0)
                    rows.append(np.where(node == item)[0][0])
                    cols.append(edge_idx)
                    vals.append(2.0)
                    edge_idx += 1

            u_Hs = sp.coo_matrix((vals, (rows, cols)), shape=(max_n_node, max_n_edge))
            Hs = np.asarray(u_Hs.todense())

            alias_inputs = torch.LongTensor(alias_inputs).to(self.device)
            Hs = torch.FloatTensor(Hs).to(self.device)
            items = torch.LongTensor(items).to(self.device)
            mask = torch.BoolTensor(mask).to(self.device)

            all_alias_inputs.append(alias_inputs)
            all_Hs.append(Hs)
            all_items.append(items)
            all_mask.append(mask)

        all_alias_inputs = torch.stack(all_alias_inputs)
        all_Hs = torch.stack(all_Hs)
        all_items = torch.stack(all_items)
        all_mask = torch.stack(all_mask)

        return all_alias_inputs, all_Hs, all_items, all_mask

    def compute_disentangle_loss(self, intents_feat):
        labels = [torch.ones(f.shape[0]) * i for i, f in enumerate(intents_feat)]
        labels = torch.cat(labels, 0).long().to(self.device)
        intents_feat = torch.cat(intents_feat, 0)
        pred = self.classifier(intents_feat)
        discrimination_loss = self.loss_aux(pred, labels)
        return discrimination_loss

    def compute_scores(self, hidden, mask, item_embeddings):
        mask = mask.float().unsqueeze(-1)
        batch_size = hidden.shape[0]
        len = hidden.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)
        hs = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        ht = hidden[:, 0, :]
        ht = ht.unsqueeze(-2).repeat(1, len, 1)
        nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        nh = torch.tanh(nh)
        hs = torch.cat([hs, ht], -1).matmul(self.w_s)
        feat = hs * hidden
        nh = torch.sigmoid(torch.cat([self.glu1(nh), self.glu2(hs), self.glu3(feat)], -1))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        if self.disen:
            select = torch.sum(beta * hidden, 1)
            score_all = []
            select_split = torch.split(select, self.split_sections, dim=-1)
            b = torch.split(item_embeddings[1:], self.split_sections, dim=-1)
            for i in range(self.n_factor):
                sess_emb_int = self.w_k * select_split[i]
                item_embeddings_int = b[i]
                scores_int = torch.mm(sess_emb_int, torch.transpose(item_embeddings_int, 1, 0))
                score_all.append(scores_int)
            score = torch.stack(score_all, dim=1)
            scores = score.sum(1)
        else:
            select = torch.sum(beta * hidden, 1)
            b = item_embeddings[1:]
            scores = torch.matmul(select, b.transpose(1, 0))
        return scores

    def forward(self, inputs, Hs, mask_item, item):
        batch_size = inputs.shape[0]
        seqs_len = inputs.shape[1]
        item_embeddings = self.item_embedding.weight
        zeros = torch.zeros(1, self.dim).to(self.device)
        item_embeddings = torch.cat([zeros, item_embeddings], 0)
        h = item_embeddings[inputs]
        item_emb = item_embeddings[item] * mask_item.float().unsqueeze(-1)
        session_c = torch.sum(item_emb, 1) / torch.sum(mask_item.float(), -1).unsqueeze(-1)
        session_c = session_c.unsqueeze(1)
        if self.intent_aware_emb:
            all_items = item_embeddings[1:]
            intents_cat = torch.mean(all_items, dim=0, keepdim=True)
            mask_node = torch.ones_like(inputs)
            zero_vec = torch.zeros_like(inputs)
            mask_node = torch.where(inputs.eq(0), zero_vec, mask_node)
            h_split = torch.split(h, self.split_sections, dim=-1)
            s_split = torch.split(session_c, self.split_sections, dim=-1)
            intent_split = torch.split(intents_cat, self.split_sections, dim=-1)
            h_ints = []
            intents_feat = []
            for i in range(self.n_factor):
                h_int = h_split[i]
                Hs = self.disen_graph(h_int, Hs, intent_split[i], mask_node)
                h_int = self.disen_aggs[i](h_int, Hs, s_split[i])

                intent_p = intent_split[i].unsqueeze(0).repeat(batch_size, seqs_len, 1)
                sim_val = h_int * intent_p
                cor_att = torch.sigmoid(sim_val)
                h_int = h_int * cor_att + h_int

                h_ints.append(h_int)
                intents_feat.append(torch.mean(h_int, dim=1))   # (b ,latent_dim)

            h_stack = torch.stack(h_ints, dim=2)
            h_local = h_stack.reshape(batch_size, seqs_len, self.dim)
            self.intent_loss = self.compute_disentangle_loss(intents_feat)
        else:
            h_local = self.local_agg(h, Hs, session_c)
        output = h_local
        return output, item_embeddings

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        alias_inputs, Hs, items, mask = self._get_slice(item_seq)
        hidden, item_embeddings = self.forward(items, Hs, mask, alias_inputs)
        seq_hidden = torch.stack([hidden[i][alias_inputs[i]] for i in range(len(alias_inputs))])
        pos_items = interaction[self.POS_ITEM_ID]

        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_hidden * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_hidden * neg_items_emb, dim=-1)
            loss = self.loss_fct(pos_score, neg_score)
        else:
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_hidden, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)

        if self.disen:
            loss += self.intent_loss_weight * self.intent_loss

        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        alias_inputs, Hs, items, mask = self._get_slice(item_seq)
        hidden, item_embeddings = self.forward(items, Hs, mask, alias_inputs)
        seq_hidden = torch.stack([hidden[i][alias_inputs[i]] for i in range(len(alias_inputs))])
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_hidden, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        alias_inputs, Hs, items, mask = self._get_slice(item_seq)
        hidden, item_embeddings = self.forward(items, Hs, mask, alias_inputs)
        seq_hidden = torch.stack([hidden[i][alias_inputs[i]] for i in range(len(alias_inputs))])
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_hidden, test_items_emb.transpose(0, 1))
        return scores