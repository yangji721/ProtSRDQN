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
from Models.ProtSRDQN_dec import ProtSRDQN_dec
from pprint import pprint
from CONFIG import HOME_PATH

from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from Utils.JobReader import n_skill, get_frequent_itemset, skill_lst
from Utils.Utils import sparse_to_dense, dense_to_sparse, dense_to_sparse_prob

n_s: int = 987
# FILE_PATH : str = sys.argv[3]

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

if __name__ == '__main__':

    direct_name = "resume"
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
    salary_prototype, difficulty_prototype = int(sys.argv[2]), int(sys.argv[3])
    lambda_freq, lambda_div, lambda_enc = float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])
    diversity1 = eval(sys.argv[7])
    if not isinstance(diversity1, bool):
        raise TypeError('diversity1 should be a bool')
    diversity2 = eval(sys.argv[8])
    if not isinstance(diversity2, bool):
        raise TypeError('diversity2 should be a bool')
    env_params = {"lambda_d": 0.1, "beta": 0.2, 'pool_size': 100}

    # ----------------- 初始模型 ---------------------
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    init_salary_q_value = 40
    init_salary_d_value = 20

    model = ProtSRDQN_dec_v3(n_skill, lambda_div= lambda_div, lambda_enc = lambda_enc, lambda_freq = lambda_freq, diversity1 = diversity1, diversity2 = diversity2, verbose=1, learning_rate=0.001, lambda_d=env_params['lambda_d'], embsize=256, 
               pool_size=env_params['pool_size'], salary_prototype = salary_prototype, difficulty_prototype = difficulty_prototype,
               salary_encoder_layers=[256, 256], salary_decoder_layers=[256, 256], dropout_salary=[1.0, 1.0, 1.0], difficulty_decoder_layers=[256, 256],
               difficulty_encoder_layers=[256, 256], dropout_difficulty=[1.0, 1.0, 1.0],
               salary_q_value_max = init_salary_q_value, difficulty_q_value_max = init_salary_d_value,
               activation=tf.nn.leaky_relu, random_seed=2022, l2_reg=0.001, name="qnet", sess=sess)


    sess.run(tf.global_variables_initializer())

    # ----------------- 模型读取 ---------------------
    model.load(HOME_PATH + "data/model/%s_ProtSRDQN_dec_v3_Prot_sal_%s_dif_%s_freq_%s_div_%s_enc_%s_bool_%s" %(direct_name, salary_prototype, difficulty_prototype, lambda_freq, lambda_div, lambda_enc, diversity1))

    frequents = get_frequent_itemset(2, 9)
    index = np.random.randint(0, len(frequents), 10000)
    frequent_set = []
    for item in index:
         frequent_set.append(frequents[item])
    

    # ----------------- 读取原型 ---------------------
    print("---------------Starting Searching for Salary---------------")
    decoder_results, salary_prototypes = model.get_salary_prototypes()
    print(salary_prototypes.shape)    

    tsne = TSNE(n_components=2, verbose=1, random_state=1234, metric="precomputed")
    # y= np.arange(1, 41)
    y = np.zeros(40)

    for item in frequent_set:
        embedding = model.get_embeddings(item, True)
        salary_prototypes = np.append(salary_prototypes, [embedding], axis = 0)
        y = np.append(y, 1)
    
    y = np.flip(y, 0)
    salary_prototypes = np.flip(salary_prototypes, 0)
    distance_matrix = pairwise_distances(salary_prototypes, salary_prototypes, metric='cosine', n_jobs=-1)
    z = tsne.fit_transform(distance_matrix)

    df1 = pd.DataFrame()
    df1["y"] = y
    df1["First Dimension"] = z[:,0]
    df1["Second Dimension"] = z[:,1] 

    sns.scatterplot(x="First Dimension", y="Second Dimension", hue=df1.y.tolist(),
                palette=sns.color_palette("husl", 2), 
                data=df1).set(title="Salary Prototype T-SNE projection")
    
    plt.legend(markerscale=2)
    plt.savefig("Salary.png")
    plt.clf()
  
    print("---------------Starting Searching for Difficulty---------------")
    decoder_results, difficulty_prototypes = model.get_difficulty_prototypes()

    print(difficulty_prototypes.shape)

    tsne = TSNE(n_components=2, verbose=1, random_state=1234, metric="precomputed")
    # y= np.arange(1, 41)
    y = np.zeros(40)

    for item in frequent_set:
        embedding = model.get_embeddings(item, False)
        difficulty_prototypes = np.append(difficulty_prototypes, [embedding], axis = 0)
        y = np.append(y, 1)

    y = np.flip(y, 0)
    difficulty_prototypes = np.flip(difficulty_prototypes, 0)
    distance_matrix = pairwise_distances(difficulty_prototypes, difficulty_prototypes, metric='cosine', n_jobs=-1)
    z = tsne.fit_transform(distance_matrix)
    # y= np.arange(1, 41)
    
    df2 = pd.DataFrame()
    df2["y"] = y
    df2["First Dimension"] = z[:,0]
    df2["Second Dimension"] = z[:,1] 

    sns.scatterplot(x="First Dimension", y="Second Dimension", hue=df2.y.tolist(),
                palette=sns.color_palette("hls", 2),
                data=df2).set(title="Difficulty Prototype T-SNE projection")
    plt.savefig("Difficulty.png")



