from typing import List, Tuple, Dict
from math import sqrt
import torch
from torch import Tensor
import torch.nn as nn
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.nn import MessagePassing
from torch_geometric.utils.loop import add_self_loops
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
# from attack.models import subgraph
# from rdkit import Chem
from matplotlib.axes import Axes
from matplotlib.patches import Path, PathPatch
import collections
import numpy as np
from attack.models import GNNPool
import random

EPS = 1e-15


class AttackBase(nn.Module):
    """
    for target attack
    indirect_level = 0: no constrain, newly added nodes can be directly connected to any nodes
    indirect_level = 1: with constrain that, no directly connected with target node itself
    indirect_level = 2: with constrain that, no directly connected with target node 1-hop neighbors
    indirect_level = 3: with constrain that, no directly connected with target node 2-hop neighbors
    """

    def __init__(self, model: nn.Module, new_node_num=0, epochs=0, lr=0, attack_graph=True, mask_features=False, mask_structure=True, indirect_level=0, random_structure=False, random_feature=False, molecule=False):
        super().__init__()
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.attack_graph = attack_graph
        self.mask_features = mask_features
        self.mask_structure = mask_structure
        # self.random_feature = random_feature
        self.random_mask_structure = random_structure
        self.random_mask_feature = random_feature
        self.molecule = molecule
        self.mp_layers = [module for module in self.model.modules() if isinstance(module, MessagePassing)]
        self.num_layers = len(self.mp_layers)

        self.ori_pred = None
        self.ex_labels = None
        self.edge_mask = None
        self.updated_edge_mask = None
        self.fixed_edge_mask = None

        self.hard_edge_mask = None

        self.added_node_num = new_node_num
        self.indirect_level = indirect_level

        self.num_edges = None
        self.num_nodes = None
        self.device = None
        # self.table = Chem.GetPeriodicTable().GetElementSymbol

    def __construct_mask__(self, edge_index):
        mask_size = edge_index.shape[1]
        # print(" edge index is", edge_index)
        # print(" edge index [0][0] is", edge_index[0][0])
        # print(" edge index [0][1] is", edge_index[0][1])
        # print(" edge index [1][0] is", edge_index[1][0])
        """
        start add indirectly target attak
        """

        # TODO
        # node_idx = 341
        # banned_nodes = []
        # def ___get_hop_k_neighbors___(edge_index, node_idx, k):
        #     node_list = collections.deque([node_idx])
        #     while node_list and k:
        #         for i in range(len(node_list)):
        #             node_list.popleft()
        #     return node_list
        #
        # def __get_neighbors__(edge_index, node_idx):
        #     neighbors = []
        #     cord_x, cord_y = edge_index[0], edge_index[1]
        #     idx = (cord_x == node_idx).nonzero(as_tuple=True)[0]
        #     idy = (cord_y == node_idx).nonzero(as_tuple=True)[0]
        #     for item in idx:
        #         neighbors.append(edge_index[1][item].item())
        #         # print(" idx cor: ", edge_index[1][item])
        #     for item in idy:
        #         neighbors.append(edge_index[0][item].item())
        #         # print(" idy cor ", edge_index[0][item])
        #     # print("type ",type(neighbors))
        #     # print("neigh", neighbors)
        #     # print(" set is", set(neighbors))
        #     return list(set(neighbors))
        # __get_neighbors__(edge_index, node_idx)
        # exit(-2)
        # not directly connect to target node
        # if self.indirect_level == 1:
        #     banned_nodes.append(node_idx)
        # # not directly connect to target node and target node 1-hop neighbors
        # elif self.indirect_level == 2:
        #     banned_nodes.append()
        # # not directly connect to target node and target node 1-hop and 1-hop neighbors
        # elif self.indirect_level == 3:
        #     banned_nodes.append()
        """
        end add indirectly target attak
        """
        cord_x = edge_index[0]
        cord_y = edge_index[1]
        updated_mask = None
        print(" num_node =",self.num_nodes)
        print(" new added ", self.added_node_num)
        for i in range(self.num_nodes-self.added_node_num, self.num_nodes):
            idx = (cord_x == i).nonzero(as_tuple=True)[0]
            idy = (cord_y == i).nonzero(as_tuple=True)[0]
            if updated_mask == None:
                updated_mask = torch.cat((idx, idy))
                # updated_mask = idx
            else:
                updated_mask = torch.cat((updated_mask, idx))
                updated_mask = torch.cat((updated_mask, idy))
        # updated_mask = torch.unique(ids, sorted=True)
        # print("size of update = ",len(torch.unique(updated_mask)))
        updated_mask = torch.unique(updated_mask)
        updated_mask_size = updated_mask.shape[0]
        # edge_mask = torch.randn(mask_size)
        edge_mask = torch.randn(mask_size)
        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * self.num_nodes))
        initialize_update_mask = torch.randn(updated_mask_size, requires_grad=True)*std
        initialize_fixed_mask = torch.ones(mask_size - updated_mask_size, requires_grad=False)
        edge_mask[updated_mask] = initialize_update_mask
        fixed_mask = []
        for i in range(mask_size):
            if i not in updated_mask:
                fixed_mask.append(i)
        print(" len(fixed)", len(fixed_mask), "updated = ", updated_mask_size, "mask size =", mask_size)
        assert len(fixed_mask) + updated_mask_size == mask_size
        fixed_mask = torch.LongTensor(fixed_mask)
        edge_mask[fixed_mask] = initialize_fixed_mask

        self.updated_edge_mask = updated_mask
        self.fixed_edge_mask = fixed_mask

        print(" total mask size ", mask_size, "fixed mask size :", fixed_mask.shape[0], " updated mask size: ", updated_mask_size)
        return edge_mask

    def __set_masks__(self, x, edge_index, init="normal"):
        (N, F), E = x.size(), edge_index.size(1)

        print(" node = ", N, " feature dim = ", F, " edge size = ", E)

        edge_mask = self.__construct_mask__(edge_index)

        std = 0.1
        self.node_feat_mask = torch.nn.Parameter(torch.randn((self.added_node_num,F), requires_grad=True, device=self.device) * 0.1)
        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        # self.edge_mask = torch.nn.Parameter(torch.randn(E, requires_grad=True, device=self.device) * std)
        # self.edge_mask = torch.nn.Parameter(100 * torch.ones(E, requires_grad=True))
        self.edge_mask = torch.nn.Parameter(edge_mask)

        for module in self.model.modules():
            # print(" module is", module)
            if isinstance(module, MessagePassing):
                # print(" instance of MP ", module)
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask

    def __clear_masks__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
        self.node_feat_masks = None
        self.edge_mask = None

    @property
    def __num_hops__(self):
        if self.attack_graph:
            return -1
        else:
            return self.num_layers

    def __flow__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                return module.flow
        return 'source_to_target'


    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                **kwargs
                ):
        self.num_edges = edge_index.shape[1]
        self.num_nodes = x.shape[0]
        self.device = x.device


    def control_sparsity(self, feat_mask, mask, fix_sparsity=None, sparsity=None, feat_sparsity=None, **kwargs):
        r"""

        :param mask: mask that need to transform
        :param sparsity: sparsity we need to control i.e. 0.7, 0.5
        :return: transformed mask where top 1 - sparsity values are set to inf.
        """
        if sparsity is None:
            sparsity = 0.7
        if feat_sparsity is None:
            feat_sparsity = 0.7

        print(" structure sparsity is ", sparsity, " feature sparsity is ", feat_sparsity)

        mask_len = self.updated_edge_mask.shape[0]
        split_point = int((1 - sparsity) * mask_len)
        trans_mask = mask.clone()

        structure_sp = None
        feature_sp = None
        print(" total mask len is", len(trans_mask))

        # For structure mask
        trans_mask[:] = -float('inf')
        print(" updated mask len = ", mask_len, "split poitn = ", split_point)
        if self.random_mask_structure:
            for idx in self.fixed_edge_mask:
                trans_mask[idx] = float('inf')
            # random set split_point each time
            split_point = int(random.uniform(0,1) * mask_len)
            indices = torch.randperm(mask_len)[:split_point]
            for idx in self.updated_edge_mask[indices]:
                trans_mask[idx] = float('inf')
            structure_sp = sparsity
        else:
            sorted_mask, indices = torch.sort(mask, descending=True)
            print(" sorted mask is", sorted_mask)

            cnt = 0
            print(" fix sparisty is", fix_sparsity)
            if not fix_sparsity:
                for idx in indices:
                    if idx in self.fixed_edge_mask:
                        trans_mask[idx] = float('inf')
                    elif idx in self.updated_edge_mask:
                        if mask[idx] > 0:
                            trans_mask[idx] = float('inf')
                            cnt+=1
            else:
                print(" fixed sparsity-----------------")
                for idx in indices:
                    if idx in self.fixed_edge_mask.to(idx.device):
                        trans_mask[idx] = float('inf')
                    elif idx in self.updated_edge_mask.to(idx.device) and cnt <= split_point:
                        if mask[idx] > 0:
                            trans_mask[idx] = float('inf')
                            cnt+=1

            structure_sp = cnt/mask_len
            print(" non zero cnt is ", cnt, "structure sparsity = ", structure_sp)

        # For feature mask
        trans_feat_mask = None
        if self.mask_features:
            rows, cols = feat_mask.shape[0], feat_mask.shape[1]
            trans_feat_mask = feat_mask.clone()
            split_point = int((1 - feat_sparsity) * cols)
            print(" rows, cols", rows, cols)
            print(" split point = ", split_point, trans_feat_mask.shape)
            if self.random_mask_feature:
                print("------------------------- random feature mask-------------------------")
                for i in range(rows):
                    split_point = int(random.uniform(0,1) * mask_len)
                    indices = torch.randperm(cols)[:split_point]
                    for idx in range(cols):
                        if idx in indices:
                            trans_feat_mask[i][idx] = 1
                        else:
                            trans_feat_mask[i][idx] = 0
                feature_sp = feat_sparsity
            else:
                sorted_feat_mask, indices = torch.sort(feat_mask, descending=True)
                print(" sorted feature mask",sorted_feat_mask)
                non_zero_cnt = 0
                if not fix_sparsity:
                    for i in range(rows):
                        for j in range(cols):
                            if trans_feat_mask[i][indices[i][j]] > 0:
                                trans_feat_mask[i][indices[i][j]] = 1
                                non_zero_cnt += 1
                            else:
                                trans_feat_mask[i][indices[i][j]] = 0
                    feature_sp = non_zero_cnt/(rows*cols)
                else:
                    for i in range(rows):
                        for j in range(cols):
                            if j <= split_point and trans_feat_mask[i][indices[i][j]] > 0:
                                trans_feat_mask[i][indices[i][j]] = 1
                                non_zero_cnt += 1
                            else:
                                trans_feat_mask[i][indices[i][j]] = 0
                    feature_sp = non_zero_cnt / (rows*cols)
                print(" non zero cnt is ", non_zero_cnt, "feature sparsity = ", feature_sp)
        else:
            feature_sp = feat_sparsity

        return trans_feat_mask, trans_mask, structure_sp, feature_sp