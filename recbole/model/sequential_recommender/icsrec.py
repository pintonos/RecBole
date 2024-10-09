# -*- coding: utf-8 -*-
# @Time    : 2024/10/09 14:10
# @Author  : Andreas Peintner
# @Email   : a.peintner@gmx.net

"""
Reference:
    Xiuyuan Qin et al. "Intent Contrastive Learning with Cross Subsequences for Sequential Recommendation." in WSDM 2024.

Reference:
    https://github.com/QinHsiu/ICSRec

"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import KMeans, TransformerEncoder
from recbole.model.loss import BPRLoss


class ICSRec(SequentialRecommender):
    def __init__(self, config, dataset):
        super(ICSRec, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.n_clusters = config['n_clusters']
        self.f_neg = config['f_neg']
        self.sim = config['sim']
        self.temperature = config['temperature']

        self.TARGET_ITEM = "Target_" + self.ITEM_ID
        self.TARGET_ITEM_SEQ_1 = "Target_1_" + self.ITEM_SEQ
        self.TARGET_ITEM_SEQ_2 = "Target_2_" + self.ITEM_SEQ
        self.TARGET_ITEM_SEQ_LEN_1 = self.TARGET_ITEM_SEQ_1 + config["ITEM_LIST_LENGTH_FIELD"]
        self.TARGET_ITEM_SEQ_LEN_2 = self.TARGET_ITEM_SEQ_2 + config["ITEM_LIST_LENGTH_FIELD"]

        self.cicl_loss_weight = config['cicl_loss_weight']
        self.ficl_loss_weight = config['ficl_loss_weight']

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        self.cluster = KMeans(
            num_cluster=self.n_clusters,
            seed=config["seed"],
            hidden_size=self.hidden_size,
            device=self.device,
        )

        self.clusters = [self.cluster]
        self.clusters_t = [self.clusters]

        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def cluster_intention(self, train_data):
        kmeans_training_data = []
        for batch_idx, interaction in enumerate(train_data):
            item_seq = interaction[self.ITEM_SEQ]
            item_seq_len = interaction[self.ITEM_SEQ_LEN]
            seq_output = self.forward(item_seq, item_seq_len)
            kmeans_training_data.append(seq_output.detach().cpu().numpy())

        kmeans_training_data = np.concatenate(kmeans_training_data, axis=0)
        kmeans_training_data_t = [kmeans_training_data]

        for i, clusters in enumerate(self.clusters_t):
            for j, cluster in enumerate(clusters):
                cluster.train(kmeans_training_data_t[i])
                self.clusters_t[i][j] = cluster


    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]

        output = self.gather_indexes(output, item_seq_len - 1)

        return output  # [B H]

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]

        seq_output = self.forward(item_seq, item_seq_len)

        # build intent clusters
        if self.cluster.trainable:
            self.cluster.train(self.item_embedding.weight)
            self.cluster.trainable = False

        if self.loss_type == "BPR" or self.loss_type == "BCE":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight[:self.n_items]  # unpad the augmentation mask
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)

        target_item = interaction[self.TARGET_ITEM]
        target_item_seq_1 = interaction[self.TARGET_ITEM_SEQ_1]
        target_item_seq_len_1 = interaction[self.TARGET_ITEM_SEQ_LEN_1]

        coarse_intent_1 = self.forward(item_seq, item_seq_len)
        coarse_intent_2 = self.forward(target_item_seq_1, target_item_seq_len_1)

        cicl_loss = self.cicl_loss(coarse_intent_1, coarse_intent_2, pos_items)
        ficl_loss = self.ficl_loss(coarse_intent_1, coarse_intent_2, self.clusters_t[0])

        return loss + (self.cicl_loss_weight * cicl_loss) + (self.ficl_loss_weight * ficl_loss)

    def cicl_loss(self, coarse_intent_1, coarse_intent_2, target_item):
        sem_nce_logits, sem_nce_labels = self.info_nce(coarse_intent_1, coarse_intent_2,
                                                       self.temperature, coarse_intent_1.shape[0], self.sim,
                                                       target_item)
        cicl_loss = nn.CrossEntropyLoss()(sem_nce_logits, sem_nce_labels)
        return cicl_loss


    def ficl_loss(self, coarse_intent_1, coarse_intent_2, clusters_t):
        intent_id, seq_to_v = clusters_t[0].query(coarse_intent_1)
        a, b = self.info_nce(coarse_intent_1, seq_to_v, self.temperature, coarse_intent_1.shape[0], sim=self.sim, intent_id=intent_id)
        loss_n_0 = nn.CrossEntropyLoss()(a, b)

        intent_id, seq_to_v_1 = clusters_t[0].query(coarse_intent_2)  # [BxH]
        a, b = self.info_nce(coarse_intent_2, seq_to_v_1, self.temperature, coarse_intent_2.shape[0], sim=self.sim, intent_id=intent_id)
        loss_n_1 = nn.CrossEntropyLoss()(a, b)

        ficl_loss = loss_n_0 + loss_n_1

        return ficl_loss

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=torch.bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    # False Negative Mask
    def mask_correlated_samples_(self, label):
        label = label.view(1, -1)
        label = label.expand((2, label.shape[-1])).reshape(1, -1)
        label = label.contiguous().view(-1, 1)
        mask = torch.eq(label, label.t())
        return mask == 0

    def info_nce(self, z_i, z_j, temp, batch_size, sim='dot', intent_id=None):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)
        if sim == 'cos':
            sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim = torch.mm(z, z.t()) / temp

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        if self.f_neg:
            mask = self.mask_correlated_samples_(intent_id)
            negative_samples = sim
            negative_samples[mask == 0] = float("-inf")
        else:
            mask = self.mask_correlated_samples(batch_size)
            negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
