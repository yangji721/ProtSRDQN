# -*- coding: utf-8 -*-
from turtle import distance, shape
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import numpy as np
from Utils.Utils import sparse_to_dense

class ProtSRDQN_dec_v3_tow(object):
    def __init__(self, n_s, lambda_d = 0.01, lambda_div = 0.001, lambda_enc = 0.001, lambda_freq = 0.01, diversity1 = False, diversity2 = True, verbose =1,learning_rate = 0.1, embsize =16, pool_size =100, salary_encoder_layers=[24, 16], salary_decoder_layers = [24, 16], dropout_salary = [1.0, 1.0, 1.0], salary_prototype = 21, difficulty_encoder_layers = [24, 16], difficulty_decoder_layers = [24, 16], dropout_difficulty = [1.0, 1.0, 1.0], difficulty_prototype = 21, activation = tf.nn.relu, random_seed = 2022, salary_q_value_max = 21, difficulty_q_value_max = 21, l2_reg = 0.0, name="", sess=None):
        
        self.n_s = n_s
        self.lambda_d = lambda_d
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.embsize = embsize
        self.pool_size = pool_size
        self.lambda_div = lambda_div
        self.lambda_enc = lambda_enc
        self.lambda_freq = lambda_freq
        self.diversity1 = diversity1
        self.diversity2 = diversity2

        self.salary_encoder_layers = salary_encoder_layers
        self.salary_decoder_layers = salary_decoder_layers
        self.dropout_salary_feed = dropout_salary
        self.salary_prototype = salary_prototype
        self.difficulty_encoder_layers = difficulty_encoder_layers
        self.difficulty_decoder_layers = difficulty_decoder_layers
        self.dropout_difficulty_feed = dropout_difficulty
        self.difficulty_prototype = difficulty_prototype

        self.activation = activation
        self.random_seed = random_seed
        self.l2_reg = l2_reg
        self.name = name 
        self.sess = sess

        self.salary_q_value = salary_q_value_max
        self.difficulty_q_value = difficulty_q_value_max
        self.weights = {}
        # self.attention = {}
        self.dropouts = {}
        self._init_graph()

    def _init_graph(self):
        tf.set_random_seed(self.random_seed)
        self._build_network()
        #with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate = self.learning_rate).minimize(self.loss)
        # init 
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.saver = tf.train.Saver(max_to_keep=1)
        init = tf.global_variables_initializer()
        if self.sess is None:
            self.sess = tf.Session(config=config)
            self.sess.run(init)

        print("#params: %d" % self._count_parameters())

    def run(self, data, batch_size, train=True):
        n_data = len(data[0])
        predictions_salary, predictions_easy, loss = [], [], []
        for i in range(0, n_data, batch_size):
            # data_batch = [dt[i:min(i + batch_size, n_data)] for dt in data]
            data_batch = data
            if train:
                preds_easy, preds_salary, loss_step, _ = self.sess.run(
                    (self.v_difficulty, self.v_salary, self.loss, self.optimizer),
                    feed_dict=self.get_dict(data_batch, train=True))

            else:
                # retrieve the corresponding data run( , )
                preds_easy, preds_salary = self.sess.run((self.v_difficulty, self.v_salary),
                                                         feed_dict=self.get_dict(data_batch, train=False))
            predictions_salary.extend(preds_salary)
            predictions_easy.extend(preds_easy)
            loss.append(loss_step)

        return predictions_salary, predictions_easy, loss

    def get_dict(self, data, train=True):
        feed_dict = {
            self.input_state: data[0],
            self.input_action: data[1],
            self.salary_label: data[2],
            self.difficulty_label: data[3],
            self.frequency_set: data[4],
            self.deep_salary_dropout: self.dropout_salary_feed if train else [1] * len(self.dropout_salary_feed),
            self.deep_difficulty_dropout: self.dropout_difficulty_feed if train else [1] * len(self.dropout_difficulty_feed),
            self.deep_encoder_salary_action_dropout: self.dropout_salary_feed if train else [1] * len(self.dropout_salary_feed),
            self.deep_encoder_salary_action_dropout2: self.dropout_salary_feed if train else [1] * len(self.dropout_salary_feed),
            self.deep_encoder_salary_action_dropout3: self.dropout_salary_feed if train else [1] * len(self.dropout_salary_feed),
            self.deep_encoder_difficulty_action_dropout: self.dropout_difficulty_feed if train else [1] * len(self.dropout_difficulty_feed),
            self.deep_decoder_salary_dropout: self.dropout_salary_feed if train else [1] * len(self.dropout_salary_feed),
            self.deep_decoder_difficulty_dropout: self.dropout_difficulty_feed if train else [1] * len(self.dropout_difficulty_feed),
            self.deep_encoder_difficulty_action_dropout2: self.dropout_difficulty_feed if train else [1] * len(self.dropout_difficulty_feed),
            self.deep_encoder_difficulty_action_dropout3: self.dropout_difficulty_feed if train else [1] * len(self.dropout_difficulty_feed),
        }
        return feed_dict

    def _count_parameters(self, print_count=False):
        total_parameters = 0
        for name, variable in self.weights.items():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            if print_count:
                print(name, variable_parameters)
            total_parameters += variable_parameters
        return total_parameters
    
    def _build_network(self):
        # --------------------------input ------------------------
        self.input_state = tf.placeholder(tf.float32, shape=[None, self.n_s], name = "%s_input_state" % self.name) # -1, n_s
        self.salary_label = tf.placeholder(tf.float32, shape = [None], name = "%s_input_salary_label" % self.name)
        self.difficulty_label = tf.placeholder(tf.float32, shape = [None], name = "%s_input_difficulty_label" % self.name)
        self.input_action = tf.placeholder(tf.int32, shape=[None, self.pool_size], name="%s_input_action" % self.name) # num, pool_size
        self.frequency_set = tf.placeholder(tf.float32, shape=[None, self.n_s], name = "%s_frequency_set" % self.name)# num * 10, n_s

        # ----------------------- multi-hot processing ----------------------
        input_set = tf.reshape(tf.tile(self.input_state, [1, self.pool_size]), [-1,  self.n_s]) # num * pool_size, n_s
        action_set = tf.reshape(tf.one_hot(tf.reshape(self.input_action, [-1, 1]), self.n_s), [-1, self.n_s]) # num * pool_size, n_s

        # concatenate two input tensors
        len_input_set = tf.shape(input_set)[0]
        mix_input = tf.concat([input_set, self.frequency_set], axis=0)

        # consider using sparse
        mix_input = tf.sparse.from_dense(mix_input)
        action_set = tf.sparse.from_dense(action_set)
        # ----------------------------------------------------------------
        # ---------------------- Part of Salary --------------------------
        # ----------------------------------------------------------------
        # ---------------------- encoder network -------------------------
        self.deep_salary_dropout, y_deep_encoder, _ = self._mlp(mix_input, self.n_s, self.salary_encoder_layers,
                                                        name="%s_deep_encoder_salary" % self.name,
                                                        activation=self.activation, bias=True, sparse_input=True, batch_norm=False)
        
        salary_embedding_mix, _ = self._fc(y_deep_encoder, self.salary_encoder_layers[-1], self.embsize, name="%s_salary_encoder_emb" % self.name, l2_reg=0.0, activation=None, bias=True, sparse=False) 


        self.salary_embedding = tf.slice(salary_embedding_mix, [0, 0], [len_input_set, -1])
        self.salary_embedding_freq = tf.slice(salary_embedding_mix, [len_input_set, 0], [-1, -1])

        # ---------------------- decoder network ------------------
        self.deep_decoder_salary_dropout, y_deep_decoder, _ = self._mlp(self.salary_embedding_freq, self.embsize, self.salary_decoder_layers,
                                                        name="%s_deep_decoder_salary" % self.name,
                                                        activation=self.activation, bias=True, sparse_input=False, batch_norm=False)

        self.salary_decoder, _ = self._fc(y_deep_decoder, self.salary_decoder_layers[-1], self.n_s, name="%s_salary_decoder_output" % self.name, l2_reg=0.0, activation= tf.nn.sigmoid, bias=True, sparse=False)

        # ---------------------- decoder network loss function ------------------
        # salary_l1_norm = tf.norm(self.salary_decoder, ord=1)
        # bce = tf.keras.losses.BinaryCrossentropy()
        # self.encoder_loss = bce(self.frequency_set, self.salary_decoder) 
        self.encoder_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(self.frequency_set, self.salary_decoder))

        if "%s_salary_protoype" % self.name not in self.weights:
            self.weights["%s_salary_protoype" % self.name] =  tf.Variable(tf.random_uniform(shape=[self.salary_prototype, self.embsize], dtype=tf.float32), name = "%s_salary_protoype" % self.name) # prototype_num, emdsize

        # ---------------------- action network ------------------ 
        self.deep_encoder_salary_action_dropout, y_deep_action1, _ = self._mlp(action_set, self.n_s, self.salary_encoder_layers,
                                                        name="%s_deep_encoder_salary_action" % self.name,
                                                        activation=self.activation, bias=True, sparse_input=True, batch_norm=False)
        
        y_deep_action2 = tf.repeat(y_deep_action1, repeats = self.salary_prototype, axis = 0)
        y_deep_action3 = tf.tile(tf.stop_gradient(self.weights["%s_salary_protoype" % self.name]), [tf.shape(action_set)[0], 1])

        y_deep_action4 = tf.concat([y_deep_action2, y_deep_action3], 1)
        y_deep_action5 = tf.math.add(y_deep_action2, y_deep_action3)

        self.deep_encoder_salary_action_dropout2, y_deep_action6, _ = self._mlp(y_deep_action4, 512, [256, 256],
                                                        name="%s_deep_encoder_salary_action2" % self.name,
                                                        activation=self.activation, bias=True, sparse_input=False, batch_norm=False)

        # For last layer, we use SoftPlus function
        self.salary_action_value_concat, _ = self._fc(y_deep_action6, self.salary_encoder_layers[-1], 1, name="%s_salary_encoder_action" % self.name, l2_reg=0.0, activation = tf.nn.softplus, bias=True, sparse=False) 
        
        self.deep_encoder_salary_action_dropout3, y_deep_action7, _ = self._mlp(y_deep_action5, self.embsize, [256, 256],
                                                        name="%s_deep_encoder_salary_action3" % self.name,
                                                        activation=self.activation, bias=True, sparse_input=False, batch_norm=False)

        # For last layer, we use SoftPlus function
        self.salary_action_value_mixed, _ = self._fc(y_deep_action7, self.salary_encoder_layers[-1], 1, name="%s_salary_encoder_action2" % self.name, l2_reg=0.0, activation = tf.nn.softplus, bias=True, sparse=False) 

        self.salary_action_value = self.salary_action_value_mixed + self.salary_action_value_concat
        self.salary_action_value = tf.reshape(self.salary_action_value, [-1, self.salary_prototype])

        # range from [0, self.salary_q_value]
        # self.salary_action_value = tf.keras.activations.relu(self.salary_action_value, max_value=self.salary_q_value)

        # ---------------------- salary prototype ------------------
        self.q_salary, salary_diversity_loss, salary_freq_loss, salary_weights, self.skill_salary_attention = self._prototype_layer(self.salary_embedding, self.salary_embedding_freq, self.salary_prototype, self.embsize, self.salary_action_value, name="%s_salary" % self.name, diversity1=self.diversity1, diversity2=self.diversity2)
        self.q_salary = tf.reshape(self.q_salary, [-1, self.pool_size])
        self.frequency_loss = salary_freq_loss

        
        # ---------------------- mapping prototypes back ------------------
        _ , y_deep_prototype, _ = self._mlp(salary_weights, self.embsize, self.salary_decoder_layers,
                                                        name="%s_deep_decoder_salary" % self.name,
                                                        activation=self.activation, bias=True, sparse_input=False, batch_norm=False)

        self.skill_salary_prototypes, _ = self._fc(y_deep_prototype, self.salary_decoder_layers[-1], self.n_s, name="%s_salary_decoder_output" % self.name, l2_reg=0.0, activation= tf.nn.sigmoid, bias=True, sparse=False) # prototype_num * n_s
        

        # ---------------------------------------------------------------
        # ---------------------- Part of Difficulty ---------------------
        # ---------------------------------------------------------------
        # ---------------------- encoder network ------------------------
        self.deep_difficulty_dropout, y_deep_encoder, _ = self._mlp(mix_input, self.n_s, self.difficulty_encoder_layers,
                                                        name="%s_deep_encoder_difficulty" % self.name,
                                                        activation=self.activation, bias=True, sparse_input=True, batch_norm=False)
        
        difficulty_embedding_mix, _ = self._fc(y_deep_encoder, self.difficulty_encoder_layers[-1], self.embsize, name="%s_difficulty_encoder_emb" % self.name, l2_reg=0.0, activation=None, bias=True, sparse=False) 

        self.difficulty_embedding = tf.slice(difficulty_embedding_mix, [0, 0], [len_input_set, -1])
        self.difficulty_embedding_freq = tf.slice(difficulty_embedding_mix, [len_input_set, 0], [-1, -1])

        # debug unit
        # check0 = tf.debugging.assert_equal(self.salary_embedding_freq, self.difficulty_embedding_freq)

        # ---------------------- decoder network ------------------
        self.deep_decoder_difficulty_dropout , y_deep_decoder, _ = self._mlp(self.difficulty_embedding_freq, self.embsize, self.difficulty_decoder_layers,
                                                        name="%s_deep_decoder_difficulty" % self.name,
                                                        activation=self.activation, bias=True, sparse_input=False, batch_norm=False)

        self.difficulty_decoder, _ = self._fc(y_deep_decoder, self.difficulty_decoder_layers[-1], self.n_s, name="%s_difficulty_decoder_output" % self.name, l2_reg=0.0, activation= tf.nn.sigmoid, bias=True, sparse=False)

        # ---------------------- decoder network loss function ------------------
        # self.encoder_loss += self.lambda_d * bce(self.frequency_set, self.difficulty_decoder)
        # self.encoder_loss += self.lambda_d * tf.norm(self.difficulty_decoder, ord=1)
        self.encoder_loss += tf.reduce_mean(tf.keras.losses.binary_crossentropy(self.frequency_set, self.difficulty_decoder))

        if "%s_difficulty_protoype" % self.name not in self.weights:
            self.weights["%s_difficulty_protoype" % self.name] =  tf.Variable(tf.random_uniform(shape=[self.difficulty_prototype, self.embsize], dtype=tf.float32), name = "%s_difficulty_protoype" % self.name) # prototype_num, emdsize

        # ---------------------- action network ------------------ 
        self.deep_encoder_difficulty_action_dropout, y_deep_action1, _ = self._mlp(action_set, self.n_s, self.difficulty_encoder_layers,
                                                        name="%s_deep_encoder_difficulty_action" % self.name,
                                                        activation=self.activation, bias=True, sparse_input=True, batch_norm=False)

        y_deep_action2 = tf.repeat(y_deep_action1, repeats = self.difficulty_prototype, axis = 0)
        y_deep_action3 = tf.tile(tf.stop_gradient(self.weights["%s_difficulty_protoype" % self.name]), [tf.shape(action_set)[0], 1])

        y_deep_action4 = tf.concat([y_deep_action2, y_deep_action3], 1)
        y_deep_action5 = tf.math.add(y_deep_action2, y_deep_action3)

        self.deep_encoder_difficulty_action_dropout2, y_deep_action6, _ = self._mlp(y_deep_action4, 512, [256, 256],
                                                        name="%s_deep_encoder_difficulty_action2" % self.name,
                                                        activation=self.activation, bias=True, sparse_input=False, batch_norm=False)

        self.difficulty_action_value_concat, _ = self._fc(y_deep_action6, self.difficulty_encoder_layers[-1], 1, name="%s_difficulty_encoder_action" % self.name, l2_reg=0.0, activation = tf.nn.softplus, bias=True, sparse=False) 

        
        self.deep_encoder_difficulty_action_dropout3, y_deep_action7, _ = self._mlp(y_deep_action5, self.embsize, [256, 256],
                                                        name="%s_deep_encoder_difficulty_action3" % self.name,
                                                        activation=self.activation, bias=True, sparse_input=False, batch_norm=False)

        self.difficulty_action_value_mixed, _ = self._fc(y_deep_action7, self.difficulty_encoder_layers[-1], 1, name="%s_difficulty_encoder_action2" % self.name, l2_reg=0.0, activation = tf.nn.softplus, bias=True, sparse=False) 

        self.difficulty_action_value = self.difficulty_action_value_mixed + self.difficulty_action_value_concat
        self.difficulty_action_value = tf.reshape(self.difficulty_action_value, [-1, self.difficulty_prototype])
        # self.difficulty_action_value = tf.keras.activations.relu(self.difficulty_action_value, max_value=self.difficulty_q_value)

        # debug
        # self.difficulty_action_value = tf.multiply(self.difficulty_action_value, tf.ones([6400, self.difficulty_prototype], tf.float32))
        # assert_op0 = tf.assert_equal(self.difficulty_prototype, 20)
        # assert_op1 = tf.debugging.assert_shapes([6400, self.difficulty_prototype], data=self.difficulty_action_value)
        # assert_op2 = tf.assert_equal(self.embsize, 256)

        # debug
        # test = tf.ones([6400, 1, 1], tf.float32)
        # opx = tf.assert_equal(y_deep_action.shape, test.shape)

        # ---------------------- difficulty prototype ---------------------
        self.q_difficulty, difficulty_diversity_loss, difficulty_frequenct_loss, difficulty_weights, self.skill_difficulty_attention = self._prototype_layer(self.difficulty_embedding, self.difficulty_embedding_freq, self.difficulty_prototype, self.embsize, self.difficulty_action_value, name="%s_difficulty" % self.name, diversity1=self.diversity1, diversity2=self.diversity2)
        self.q_difficulty = tf.negative(tf.reshape(self.q_difficulty, [-1, self.pool_size]))
        self.frequency_loss += self.lambda_d * difficulty_frequenct_loss
        
        
        # ---------------------- mapping prototypes back ------------------
        _ , y_deep_prototype, _ = self._mlp(difficulty_weights, self.embsize, self.difficulty_decoder_layers,
                                                        name="%s_deep_decoder_difficulty" % self.name,
                                                        activation=self.activation, bias=True, sparse_input=False, batch_norm=False)

        self.skill_difficulty_prototypes, _ = self._fc(y_deep_prototype, self.difficulty_decoder_layers[-1], self.n_s, name="%s_difficulty_decoder_output" % self.name, l2_reg=0.0, activation= tf.nn.sigmoid, bias=True, sparse=False) # prototype_num * n_s
        

        # ---------------------------------------------------------------------
        # --------------------- Choose an action ------------------------------
        # ---------------------------------------------------------------------
        self.q = self.q_salary + self.lambda_d * self.q_difficulty # -1, pool_size    

        self.v_place = tf.argmax(self.q, 1)

        onehot_action_nxt = tf.one_hot(self.v_place, self.pool_size)  # should be: data batch, pool_size  

        # ----------- has a problem -----------
        self.v_skill = tf.cast(tf.reduce_sum(tf.multiply(onehot_action_nxt, tf.cast(self.input_action, np.float32)), 1), np.int32)        
        self.v = tf.reduce_sum(tf.multiply(onehot_action_nxt, self.q), 1)
        self.v_salary = tf.reduce_sum(tf.multiply(onehot_action_nxt, self.q_salary), 1)
        self.v_difficulty = tf.reduce_sum(tf.multiply(onehot_action_nxt, self.q_difficulty), 1)

        # ------ attention selections-----------------------
        self.skill_salary_attention = tf.reshape(self.skill_salary_attention, [-1, self.pool_size, self.salary_prototype])
        self.skill_difficulty_attention = tf.reshape(self.skill_difficulty_attention, [-1, self.pool_size, self.difficulty_prototype])

        att_onehot_salary = tf.tile(tf.reshape(onehot_action_nxt, [-1, self.pool_size, 1]), [1,1, self.salary_prototype])
        att_onehot_difficulty = tf.tile(tf.reshape(onehot_action_nxt, [-1, self.pool_size, 1]), [1,1, self.difficulty_prototype])

        self.skill_salary_attention_final =tf.reduce_sum(tf.multiply(att_onehot_salary, self.skill_salary_attention), axis=1)
        self.skill_difficulty_attention_final = tf.reduce_sum(tf.multiply(att_onehot_difficulty, self.skill_difficulty_attention), axis=1)

        # --------------------- loss ----------------------------------
        # --------------------- need to add lambda_d to restrict ------
        self.diversity_loss = tf.reduce_mean(salary_diversity_loss)
        self.diversity_loss += self.lambda_d * tf.reduce_mean(difficulty_diversity_loss)

        self.loss = tf.reduce_mean(tf.square(self.v_salary - self.salary_label))
        self.loss += self.lambda_d * tf.reduce_mean(tf.square(self.v_difficulty - self.difficulty_label))
        self.loss += self.lambda_div * self.diversity_loss
        self.loss += self.lambda_enc * self.encoder_loss
        self.loss += self.lambda_freq * self.frequency_loss
        return 

    # contain the prototype duplicates
    def _prototype_layer(self, tensor, frequency_set, prototype_size, embedding_size, q_value, name, diversity1=False, diversity2=True):
        if "%s_protoype" % name not in self.weights:
            self.weights["%s_protoype" % name] =  tf.Variable(tf.random_uniform(shape=[prototype_size, embedding_size], dtype=tf.float32), name = "%s_protoype" % name) # prototype_num, emdsize
        
        tensor_prot = tf.tile(tf.reshape(tensor, [-1, 1, embedding_size]), [1, prototype_size, 1]) # num * pool_size, prototype_num, emdsize

        # normalization
        tensor_prot = tf.math.l2_normalize(tensor_prot, axis=2)
        normalized_weights = tf.math.l2_normalize(self.weights["%s_protoype" % name], axis=1)

        # use euclidean distance to measure the distance
        x2_res = tf.cast(tensor_prot - normalized_weights, tf.float32) 
        # x2_res = tf.reduce_sum(tf.multiply(normalize_a,normalize_b))
        
        similarity = tf.math.negative(tf.norm(x2_res, ord='euclidean', axis=2))
        #self.attention["%s_prototype_weights" % name] = tf.nn.softmax(similarity, axis=1) # -1, prototype_num
        attention = tf.nn.softmax(similarity, axis=1) # -1, prototype_num
        q_result = tf.reduce_sum(tf.multiply(attention, q_value), axis=1, keepdims=True)

        # frequency loss
        frequency_loss = 0
        
        freq_prot = tf.tile(tf.reshape(frequency_set, [-1, 1, embedding_size]), [1, prototype_size, 1]) # num *10, prototype_num, emdsize
        # normalization 
        freq_prot = tf.math.l2_normalize(freq_prot, axis=2)

        freq_res = tf.cast(freq_prot - normalized_weights, tf.float32)
        freq_similarity = tf.norm(freq_res, ord='euclidean', axis=2) # -1, prot_num
       
        # for each frequency set 
        freq_min_loss = tf.reduce_mean(tf.reduce_min(freq_similarity, axis=1))
        frequency_loss += freq_min_loss

        # for each prototype
        freq_prototype_loss = tf.reduce_mean(tf.reduce_min(freq_similarity, axis=0))
        frequency_loss += freq_prototype_loss
        

        # diversity loss
        diversity_loss = 0
        if diversity1:
            temp = tf.matmul(attention, tf.transpose(attention))
            data_length = tf.shape(temp)[0]
            diversity_loss += tf.norm(temp - tf.eye(data_length) , ord='euclidean')
        
        if diversity2:
            prototype_embeddings_plus = tf.reshape(tf.tile(normalized_weights, [1, prototype_size]), [-1, embedding_size])
            prototype_embeddings_minus = tf.tile(normalized_weights, [prototype_size, 1])
            prototype_distance = tf.reduce_sum(tf.multiply(prototype_embeddings_plus, prototype_embeddings_minus), axis=1)
            # 归一化到 [0, 1]
            prototype_distance = 0.5 * prototype_distance + 0.5

            prototype_distance = tf.reshape(prototype_distance, [prototype_size, prototype_size])
            prototype_distance = tf.matrix_set_diag(prototype_distance, tf.zeros(prototype_distance.shape[0:-1]))

            diversity_loss += tf.reduce_sum(tf.math.maximum(tf.zeros_like(prototype_distance), prototype_distance - 0.6 * tf.ones_like(prototype_distance)))

        return q_result, diversity_loss, frequency_loss, self.weights["%s_protoype" % name], attention


    def _fc(self, tensor, dim_in, dim_out, name, l2_reg, activation=None, bias=True, sparse=False):
        glorot = np.sqrt(2.0 / (dim_in + dim_out))
        # add if condition, if not existing 
        if "%s_w" % name not in self.weights:
            self.weights["%s_w" % name] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(dim_in, dim_out)),
                                                  dtype=np.float32, name="%s_w" % name)
        if not sparse:
            y_deep = tf.matmul(tensor, self.weights["%s_w" % name])
        else:
            y_deep = tf.sparse_tensor_dense_matmul(tensor, self.weights["%s_w" % name])
        if bias:
            if "%s_b" % name not in self.weights:
                self.weights["%s_b" % name] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, dim_out)),
                                                          dtype=np.float32, name="%s_b" % name)
                y_deep += self.weights["%s_b" % name]
        if activation is not None:
            y_deep = activation(y_deep)
        return y_deep, tf.contrib.layers.l2_regularizer(l2_reg)(self.weights["%s_w" % name])

    def _mlp(self, tensor, dim_in, layers, name, activation=None, bias=True, sparse_input=False, batch_norm=False):
        if name not in self.dropouts:
            self.dropouts[name] = tf.placeholder(tf.float32, shape=[None], name="%s_dropout" % name)
        dropout = self.dropouts[name]
        #y_deep = tf.nn.dropout(tensor, rate = 1.0 - dropout[0])
        y_deep = tensor
        lst = []
        loss = 0
        for i, layer in enumerate(layers):
            if i == 0 and sparse_input:
                y_deep, _ = self._fc(y_deep, dim_in, layer, l2_reg=self.l2_reg, name="%s_%d" % (name, i),
                                            bias=bias, activation=activation, sparse=True)
            else:
                y_deep, _ = self._fc(y_deep, dim_in, layer, l2_reg=self.l2_reg, name="%s_%d" % (name, i),
                                            bias=bias, activation=activation, sparse=False)
            # add NB
            #if batch_norm:
            # y_deep = tf.nn.leaky_relu(tf.layers.batch_normalization(y_deep, training=self.train))
            y_deep = tf.nn.dropout(y_deep, rate = 1.0 - dropout[i + 1])
            lst.append(y_deep)
            dim_in = layer
        return dropout, y_deep, loss

    def save(self, save_path):
        self.saver.save(self.sess, save_path)

    def load(self, load_path):
        self.saver.restore(self.sess, load_path)

    def evaluate_metrics(self, y_pred_salary, y_true_salary, y_pred_easy, y_true_easy):
        salary_mse = mean_squared_error(y_true_salary, y_pred_salary)
        easy_mse = mean_squared_error(y_true_easy, y_pred_easy)
        return [("salary_mse", salary_mse), ("easy_mse", easy_mse)]

    def predict(self, data, batch_size=32):
        predictions_salary, predictions_easy = self.run(data, batch_size, train=False)
        return predictions_salary, predictions_easy

    def print_result(self, data_eval, endch="\n"):
        print_str = ""
        for i, name_val in enumerate(data_eval):
            if i != 0:
                print_str += ','
            print_str += "%s: %f" % name_val
        print(print_str, end=endch)

    def estimate_maxq_action(self, state_vis, act_pool):
        data = [[state_vis], [act_pool], [0], [0], [[0] * self.n_s]]
        act, v_salary, v_easy = self.sess.run((self.v_skill, self.v_salary, self.v_difficulty), self.get_dict(data, train=False))
        return (v_salary[0], v_easy[0]), act[0]

    def estimate_maxq_batch(self, data_state, data_pool):
        n_data = len(data_state)
        act_lst, v_salary, v_easy = self.sess.run((self.v_skill, self.v_salary, self.v_difficulty),
                                                  self.get_dict((data_state, data_pool, [0] * n_data, [0] * n_data, [[0] * self.n_s] * n_data * 10), train=False))
        return (v_salary, v_easy), act_lst    

    def estimate_maxq_batch_sample(self, data_state, data_pool):
        # n_data = len(data_state)
        act_lst, q_salary, q_easy, q = self.sess.run((self.input_action, self.q_salary, self.q_difficulty, self.q),
                                                  self.get_dict(([data_state], [data_pool], [0], [0], [[0] * self.n_s]), train=False))
        return (q_salary, q_easy), act_lst, q
    
    def estimate_maxq_action_with_prototypes(self, state_vis, act_pool):
        # here, we radomly select the frequent set to test the decoder accuracy
        data = [[state_vis], [act_pool], [0], [0], [sparse_to_dense([107, 356, 639, 953], self.n_s)]]
        act, v_salary, v_easy, attention_salary, salary_prototypes, attention_difficulty, difficulty_prototypes, salary_action_value, difficulty_action_value, salary_mapped, original_salary_mapped, v_place, difficulty_mapped = self.sess.run((
                                               self.v_skill, self.v_salary, self.v_difficulty, 
                                               self.skill_salary_attention_final, self.skill_salary_prototypes,
                                               self.skill_difficulty_attention_final, self.skill_difficulty_prototypes, self.salary_action_value, self.difficulty_action_value, self.salary_decoder, self.frequency_set, self.v_place, self.difficulty_decoder), self.get_dict(data, train=False))

        return (v_salary[0], v_easy[0]), act[0], attention_salary, salary_prototypes, attention_difficulty, difficulty_prototypes, salary_action_value, difficulty_action_value, salary_mapped, original_salary_mapped, v_place, difficulty_mapped

    def estimate_maxq_action_with_prototypes_with_skill(self, state_vis, s):
        # here, we radomly select the frequent set to test the decoder accuracy
        data = [[state_vis], [[s] * self.pool_size], [0], [0], [sparse_to_dense([107, 356, 639, 953], self.n_s)]]
        act, v_salary, v_easy, attention_salary, salary_prototypes, attention_difficulty, difficulty_prototypes, salary_action_value, difficulty_action_value, salary_mapped, original_salary_mapped, v_place, difficulty_mapped = self.sess.run((
                                               self.v_skill, self.v_salary, self.v_difficulty, 
                                               self.skill_salary_attention_final, self.skill_salary_prototypes,
                                               self.skill_difficulty_attention_final, self.skill_difficulty_prototypes, self.salary_action_value, self.difficulty_action_value, self.salary_decoder, self.frequency_set, self.v_place, self.difficulty_decoder), self.get_dict(data, train=False))

        return (v_salary[0], v_easy[0]), act[0], attention_salary, salary_prototypes, attention_difficulty, difficulty_prototypes, salary_action_value, difficulty_action_value, salary_mapped, original_salary_mapped, v_place, difficulty_mapped

    def evaluate_internal_value_functions(self, state_vis, act_pool):
        data = [[state_vis], [act_pool], [0], [0], [sparse_to_dense([416, 222, 183, 743, 746], self.n_s)]]
        loss, encoder_loss, div_loss, freq_loss = self.sess.run((self.loss, self.encoder_loss, self.diversity_loss, self.frequency_loss), self.get_dict(data, train=False))
        return loss, encoder_loss, div_loss, freq_loss
    
    def get_q_list(self, skill_vis, act_pool):
        data = [[skill_vis], [act_pool], [0], [0], [[0] * self.n_s]]
        q_salary_lst, q_easy_lst = self.sess.run((self.q_salary, self.q_difficulty), self.get_dict(data, train=False))
        return q_salary_lst[0], q_easy_lst[0]

    # 
    def get_embeddings(self, state_vis, flag):
        input_state = [sparse_to_dense(state_vis, self.n_s)]
        data = [input_state, [[0] * self.pool_size], [0], [0], [[0] * self.n_s]]
        if flag: 
           embeddings = self.sess.run((self.salary_embedding), self.get_dict(data, train=False))
        else:
           embeddings = self.sess.run((self.difficulty_embedding), self.get_dict(data, train=False))
        return embeddings[0]

    def get_salary_prototypes(self):
        data = [[[0] * self.n_s], [[0] * self.pool_size], [0], [0], [[0] * self.n_s]]
        decoder_result, embeddings = self.sess.run((self.skill_salary_prototypes, self.weights["%s_salary_protoype" % self.name]), self.get_dict(data, train=False))
        return decoder_result, embeddings

    def get_difficulty_prototypes(self):
        data = [[[0] * self.n_s], [[0] * self.pool_size], [0], [0], [[0] * self.n_s]]
        decoder_result, embeddings = self.sess.run((self.skill_difficulty_prototypes, self.weights["%s_difficulty_protoype" % self.name]), self.get_dict(data, train=False))
        return decoder_result, embeddings