from pyexpat import EXPAT_VERSION
from functools import partial
from matplotlib import testing
import tensorflow as tf
import numpy as np
import random
import math
from tqdm import tqdm
import os
import sys
sys.path.append("../../")
sys.path.append("../")
import pickle
import itertools
from collections import Counter
from Models.ProtSRDQN_v3 import ProtSRDQN_dec_v3
from pprint import pprint
from CONFIG import HOME_PATH
import timeit


from Utils.JobReader import n_skill, get_frequent_itemset, skill_lst
from Utils.Utils import sparse_to_dense, dense_to_sparse, dense_to_sparse_prob

MCTS_MIN: int = 2
MCTS_MAX: int = 9
ROLLOUT: int = int(sys.argv[13]) # the rollout number
# EXPAND_ATOMS: int = int(sys.argv[3]) # of atoms to expand children
n_s: int = 987
# FILE_PATH : str = sys.argv[3]


class MCTS_Node():

    def __init__(self, coalition, alpha: float=0.0, lambda_exp: float = 5.0, sum: float = 0.000, counts: int = 0, reward: float = 0.000):
        self.coalition = coalition
        self.lambda_exp = lambda_exp
        self.alpha = alpha

        self.children = []
        # Total Rewards
        self.sum = sum
        # Counts for selection
        self.counts = counts
        # Immediate reward
        self.reward = reward

    def calculate_alpha(self):
        return self.alpha

    def calculate_q(self):
        return self.sum / self.counts if self.counts > 0 else 0
    
    def calculate_u(self, n):
        return self.lambda_exp * self.reward * math.sqrt(n) / (1 + self.counts)

def MCTS_Expansion(decoder_result):
    count = set()
    frequency_set = get_frequent_itemset(2, 9)
    #if the decoder result is empty
    if len(decoder_result) == 0:
        for freq_set in frequency_set:
            count.update(freq_set)
    else:
        for freq_set in frequency_set:
            indicator = 0
            for res in decoder_result:
                if res in freq_set:
                   indicator += 1
                if indicator >= 2:
                   count.update(freq_set)
    # Choose a lower threshold if the set is too small
    if len(count) <= 20:
        for freq_set in frequency_set:
            indicator = 0
            for res in decoder_result:
                if res in freq_set:
                    indicator += 1
                if indicator >= 1:
                    count.update(freq_set)
    return count

def get_remaining_set(decoder_result, arr):
    root_coalition = MCTS_Expansion(decoder_result)
    result = root_coalition.difference(set(arr))
    return list(result)


def MCTS_Rollout(decoder_result, tree_node, state_map, score_func, flag):
    current_state_view = tree_node.coalition
    remaining_state_view = get_remaining_set(decoder_result, current_state_view)

    if len(current_state_view) >= MCTS_MAX:
        return tree_node.reward
    
    # Expand if the node has never been visited
    if len(tree_node.children) == 0:
        
        # follow the simpliest expansion rule
        '''
        if len(remaining_state_view) < EXPAND_ATOMS:
            expand_nodes = remaining_state_view
        else:
            expand_nodes = remaining_state_view[:EXPAND_ATOMS]
        '''
        expand_nodes = remaining_state_view
        
        for each_node in expand_nodes:
            # for each node, adding this node to the subset
            subset_step = [node for node in current_state_view]
            subset_step.append(each_node)
            subset_step = sorted(subset_step)

            # check the same subset and merge them 
            # for the state map
            find_same = False
            for old_node in state_map.values():
                if Counter(old_node.coalition) == Counter(subset_step):
                    new_node = old_node
                    find_same = True

            if find_same == False:
                # set biased selection
                if set(subset_step).issubset(set(decoder_result)):
                   new_node = MCTS_Node(coalition = subset_step, alpha=0.4)
                elif each_node in decoder_result:
                   new_node = MCTS_Node(coalition = subset_step, alpha=0.2)
                else:
                   new_node = MCTS_Node(coalition = subset_step)
                state_map[str(new_node.coalition)] = new_node     

            # for the tree-node children
            find_same_child = False
            for cur_child in tree_node.children:
                if Counter(cur_child.coalition) == Counter(subset_step):
                    find_same_child = True

            if find_same_child == False:
                tree_node.children.append(new_node)
        
        scores = compute_scores(score_func, tree_node.children, flag)

        for child, score in zip(tree_node.children, scores):
            child.reward = score
        
    sum_count = sum([child.counts for child in tree_node.children])
    selected_node = max(tree_node.children, key = lambda x: x.calculate_alpha() + x.calculate_q() + x.calculate_u(sum_count))
    next_iter = MCTS_Rollout(decoder_result, selected_node, state_map, score_func, flag)
    selected_node.sum += next_iter
    selected_node.counts += 1

    return next_iter

def compute_scores(score_func, children, flag):
    results = []
    for child in children:
        if child.reward == 0.0:
            score = score_func(data = child.coalition, flag=flag)
        else:
            score = child.reward
        results.append(score)
    return results
        
def compute_similarities(data, Qnet, prototype_embeddings, flag):
    # calculate the similarites between the input coalition with the selected prototype embeddings
    # here, we simply take the prototype embeddings as the original input set TODO: transformed into the embedding
    emb_coalition = Qnet.get_embeddings(data, flag)

    # prototype = Qnet.get_embeddings(data)
    # using L2 distance as our Q-Network
    norm1 = np.linalg.norm(prototype_embeddings) 
    norm2 = np.linalg.norm(emb_coalition)
    cosine_similarity = np.sum(np.multiply(prototype_embeddings, emb_coalition))
    return cosine_similarity / (norm1 * norm2)

def MCTS(decoder_result, Qnet, prototype_embeddings, flag):
    root_coalition = list()
    root = MCTS_Node(root_coalition)
    state_map = {str(root.coalition): root}
    score_func = partial(compute_similarities, Qnet=Qnet, prototype_embeddings=prototype_embeddings)

    for i in range(ROLLOUT):
        print("-------The %s-th Rollout:" %i)
        MCTS_Rollout(decoder_result, root, state_map, score_func, flag)

        explanations = [node for _, node in state_map.items()]
        # reward, descending 
        explanations = sorted(explanations, key=lambda x: x.reward, reverse=True)
        # length, increasing
        explanations = sorted(explanations, key=lambda x: len(x.coalition))

        result_node = explanations[0]
        for result_idx in range(len(explanations)):
            x = explanations[result_idx]
            if len(x.coalition) >= MCTS_MIN and x.reward > result_node.reward:
               result_node = x
    
        embeddings = Qnet.get_embeddings(result_node.coalition, flag)

        score = compute_similarities(result_node.coalition, Qnet, embeddings, flag)

        # early stopping
        if score >= 0.99999:
            break
        
    print("Rollout is over")


    return result_node.coalition, result_node.reward, embeddings, state_map

def write_data(names, file):
    with open("mcts/" + file, 'a') as fp:
        fp.write("%s\n" % names)

if __name__ == '__main__':

    direct_name = "resume"
    file = sys.argv[9]
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
    salary_prototype, difficulty_prototype = int(sys.argv[2]), int(sys.argv[3])
    lambda_freq, lambda_div, lambda_enc = float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])
    diversity1 = eval(sys.argv[7])
    if not isinstance(diversity1, bool):
        raise TypeError('diversity1 should be a bool')
    diversity2 = eval(sys.argv[8])
    if not isinstance(diversity2, bool):
        raise TypeError('diversity2 should be a bool')

    lambda_d, beta, pool_size = float(sys.argv[10]), float(sys.argv[11]), int(sys.argv[12])
    env_params = {"lambda_d": lambda_d, "beta": beta, 'pool_size': pool_size}
    # ----------------- 初始模型 ---------------------
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    init_salary_q_value = 60
    init_salary_d_value = 30

    model =  ProtSRDQN_dec_v3(n_skill, lambda_div= lambda_div, lambda_enc = lambda_enc, lambda_freq = lambda_freq, diversity1 = diversity1, diversity2 = diversity2, verbose=1, learning_rate=0.00025, lambda_d=env_params['lambda_d'], embsize=256, 
               pool_size=env_params['pool_size'], salary_prototype = salary_prototype, difficulty_prototype = difficulty_prototype,
               salary_encoder_layers=[256, 256], salary_decoder_layers=[256, 256], dropout_salary=[1.0, 1.0, 1.0], difficulty_decoder_layers=[256, 256],
               difficulty_encoder_layers=[256, 256], dropout_difficulty=[1.0, 1.0, 1.0],
               salary_q_value_max = init_salary_q_value, difficulty_q_value_max = init_salary_d_value,
               activation=tf.nn.leaky_relu, random_seed=2022, l2_reg=0.001, name="qnet", sess=sess)

    sess.run(tf.global_variables_initializer())

    # ----------------- 模型读取 ---------------------
    # model.load(HOME_PATH + "data/model/%s_ProtSRDQN_dec_v3_Prot_sal_%s_dif_%s_freq_%s_div_%s_enc_%s_bool_%s" %(direct_name, salary_prototype, difficulty_prototype, lambda_freq, lambda_div, lambda_enc, diversity1))
    model.load(HOME_PATH + "data/model/%s_ProtSRDQN_dec_v3_Prot_sal_%s_dif_%s_freq_%s_div_%s_enc_%s_bool_%s_%s" %(direct_name, salary_prototype, difficulty_prototype, lambda_freq, lambda_div, lambda_enc, diversity1, env_params))

    # ----------------- 开始搜索 ---------------------
    print("---------------Starting Searching for Salary---------------")
    decoder_results, salary_prototypes = model.get_salary_prototypes()
    # Salary 
    flag = True
    salary_time = []
    salary_mcts_similarity = []
    salary_dec_similarity = []
    for it in tqdm(range(salary_prototype)):
    # for it in tqdm(range(19, 0, -1)):

        embeddings = salary_prototypes[it]

        # add testcase
        threshold = 0.95
        decoder_result = dense_to_sparse_prob(decoder_results[it], threshold)

        # Choose the best decoder result
        while len(decoder_result) < 2:
            if threshold == 0:
                break
            threshold = round(threshold - 0.01, 2)
            decoder_result = dense_to_sparse_prob(decoder_results[it], threshold)

        while len(decoder_result) > 10:
            if threshold == 0.99:
                break
            threshold = round(threshold + 0.01, 2)
            decoder_result = dense_to_sparse_prob(decoder_results[it], threshold)
        
        if len(decoder_result) < 2 and threshold != 0:
            threshold = round(threshold - 0.01, 2)
        
        decoder_result = dense_to_sparse_prob(decoder_results[it], threshold)
        print("threshold is: ", threshold)
        print([skill_lst[u] for u in decoder_result])

        start = timeit.default_timer()
        coalition, reward, final_embeddings, state_map = MCTS(decoder_result, model, embeddings, flag)
        stop = timeit.default_timer() 
        salary_time.append(float(stop - start))

        print("%s-th pair is" %it, coalition)

        print("the length of state_map is", len(state_map))
        example = [[key, node.reward, node.counts] for key, node in state_map.items() if key == str(coalition)]
        print(str(coalition), example)

        score1 = compute_similarities(coalition, model, embeddings, flag)
        print([skill_lst[u] for u in coalition], str(coalition), score1)
        names = str(str([skill_lst[u] for u in coalition]) + ' ' + str(score1))
        file1 = str(file + '(1)')
        write_data(names, file1)

        score2 = compute_similarities(decoder_result, model, embeddings, flag)
        print([skill_lst[u] for u in decoder_result], decoder_result, score2)

        salary_mcts_similarity.append(float(score1))
        salary_dec_similarity.append(float(score2))

    print("Salary time is:", salary_time)
    print("Avg Salary time is:", sum(salary_time)/len(salary_time))
    print("Salary mcts score is:", salary_mcts_similarity)
    print("Avg Salary mcts is:", sum(salary_mcts_similarity)/len(salary_mcts_similarity))
    print("Salary dec is:", salary_dec_similarity)
    print("Avg Salary dec is:", sum(salary_dec_similarity)/len(salary_dec_similarity))
         

    print("---------------Starting Searching for Difficulty---------------")
    decoder_results, difficulty_prototypes = model.get_difficulty_prototypes()
    # Difficulty 
    difficulty_time = []
    difficulty_mcts_similarity = []
    difficulty_dec_similarity = []
    flag = False
    for it in tqdm(range(difficulty_prototype)):
        embeddings = difficulty_prototypes[it]

        # Choose the best decoder result
        threshold = 0.95
        decoder_result = dense_to_sparse_prob(decoder_results[it], threshold)

        while len(decoder_result) < 2:
            if threshold == 0:
                break
            threshold = round(threshold - 0.01, 2)
            decoder_result = dense_to_sparse_prob(decoder_results[it], threshold)

        while len(decoder_result) > 10:
            if threshold == 0.99:
                break
            threshold = round(threshold + 0.01, 2)
            decoder_result = dense_to_sparse_prob(decoder_results[it], threshold)
        
        if len(decoder_result) < 2:
            threshold = round(threshold - 0.01, 2)

        # add testcase
        decoder_result = dense_to_sparse_prob(decoder_results[it], threshold)
        print("threshold is: ", threshold)
        print([skill_lst[u] for u in decoder_result])

        start = timeit.default_timer()
        coalition, reward, final_embeddings, state_map = MCTS(decoder_result, model, embeddings, flag)
        stop = timeit.default_timer() 
        difficulty_time.append(float(stop - start))

        print("%s-th pair is" %it, coalition)

        print("the length of state_map is", len(state_map))
        example = [[key, node.reward, node.counts] for key, node in state_map.items() if key == str(coalition)]
        print(str(coalition), example)

        score1 = compute_similarities(coalition, model, embeddings, flag)
        print([skill_lst[u] for u in coalition], str(coalition), score1)
        names = str(str([skill_lst[u] for u in coalition]) + ' ' + str(score1))
        file2 = str(file + '(2)')
        write_data(names, file2)

        score2 = compute_similarities(decoder_result, model, embeddings, flag)
        print([skill_lst[u] for u in decoder_result], decoder_result, score2)

        difficulty_mcts_similarity.append(float(score1))
        difficulty_dec_similarity.append(float(score2))

    print("Difficulty time is:", difficulty_time)
    print("Avg Difficulty time is:", sum(difficulty_time)/len(difficulty_time))
    print("Difficulty mcts score is:", difficulty_mcts_similarity)
    print("Avg Difficulty mcts is:", sum(difficulty_mcts_similarity)/len(difficulty_mcts_similarity))
    print("Difficulty dec is:", difficulty_dec_similarity)
    print("Avg Difficulty dec is:", sum(difficulty_dec_similarity)/len(difficulty_dec_similarity))




    