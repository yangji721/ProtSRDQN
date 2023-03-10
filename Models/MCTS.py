from pyexpat import EXPAT_VERSION
import tqdm
import tensorflow as tf
import numpy as np
import math
from Utils.Utils import sparse_to_dense, read_pkl_data, dense_to_sparse

MCTS_MIN = 4
MCTS_MAX = 7
EXPAND_ATOMS = 10
n_s = 974

class MCTS_Node():

    def __init__(self, coalition, data, lambda_exp = 0.7, sum = 0.0, counts = 0, reward = 0.0):
        self.data = data
        self.coalition = coalition
        self.lambda_exp = lambda_exp

        self.children = []
        # Total Rewards
        self.sum = sum
        # Counts for selection
        self.counts = counts
        # Immediate reward
        self.reward = reward

    def calculate_q(self):
        return self.sum / self.counts if self.counts > 0 else 0
    
    def calculate_u(self, n):
        return self.lambda_exp * self.reward * math.sqrt(n) / (1 + self.counts)

def MCTS_Rollout(tree_node, data, score_func):
    current_state_view = tree_node.coalition

    if len(current_state_view) <= MCTS_MIN:
        return tree_node.reward
    
    # Expand if the node has never been visited
    if len(tree_node.children) == 0:

        if len(current_state_view) < EXPAND_ATOMS:
            expand_nodes = current_state_view
        else:
            expand_nodes = current_state_view[:EXPAND_ATOMS]
        
        for each_node in expand_nodes:
            # for each node, pruning it and get the remaining sub-set
            subset_step = [node for node in current_state_view if node != each_node]
            new_node = MCTS_Node(coalition = subset_step, data=data)
            tree_node.children.append(new_node)
        
        scores = compute_scores(score_func, tree_node.children)

        for child, score in zip(tree_node.children, scores):
            child.reward = score
        
    sum_count = sum([child.counts for child in tree_node.children])
    selected_node = max(tree_node.children, key = lambda x: x.calculate_q() + x.calculate_u(sum_count))
    next_iter = MCTS_Rollout(selected_node, data, score_func)
    selected_node.sum += next_iter
    selected_node.counts += 1

    return next_iter

def compute_scores(score_func, children):
    results = []
    for child in children:
        if child.reward == 0:
            score = score_func(child.coalition)
        else:
            score = child.reward
        results.append(score)
    return results
        