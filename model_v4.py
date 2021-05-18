#coding=utf-8
import tensorflow as tf
import numpy as np
import pdb
from tensorflow.contrib.rnn import GRUCell

class model_ig():
    def __init__(self, batch_size, embeddings, embedding_size, update_embedding, hidden_dim, num_tags, l2_lambda=None):
        self.batch_size = batch_size
        self.embeddings = embeddings
        self.embedding_size = embedding_size
        self.update_embedding = update_embedding
        self.hidden_dim = hidden_dim
        self.num_tags = num_tags
        self.l2_lambda = l2_lambda
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.5)
        #****Encode Sentence_x****
        x_state, seq_len_x, seq_len_m = self.encode_x_sentence()
        # x_state [batch_size, seq_len_x, hidden_dim*2]
        x_state_out = self.GCNLayer( gcn_in = x_state, in_dim = self.hidden_dim*2, gcn_dim = self.hidden_dim*2, 
                                batch_size = self.batch_size, max_nodes = seq_len_x,  
                                adj = self.adj_x, num_layers = 1, name = "GCN_XS")
        x_state = x_state_out[-1] 
        # x_state [batch_size, seq_len_x, hidden_dim*2]

        # id_logits = self.mlp_identification(x_state)
        # # id_logits [batch_size, seq_len_x, 2]

        #****Encode Memory****
        memory_seq_state = self.encode_memory()
        # memory_seq_state [batch_size, 7, seq_len_m, hid_dim*2]  sentence_state [batch_size, 7, hidden_dim*2]
        memory_seq_gcn_in = tf.reshape(memory_seq_state, [self.batch_size*7, -1, self.hidden_dim*2])
        # # memory_seq_state [batch_size*7, seq_len_m, hid_dim*2]
        memory_seq_state_out = self.GCNLayer( gcn_in = memory_seq_gcn_in, in_dim = self.hidden_dim*2, gcn_dim = self.hidden_dim*2, 
                                batch_size = self.batch_size*7, max_nodes = seq_len_m, 
                                adj = self.adj_m, num_layers = 1, name = "GCN_MS")
        memory_seq_state = tf.reshape(memory_seq_state_out[-1], [self.batch_size, 7, -1, self.hidden_dim*2])
        # memory_seq_state [batch_size, 7, seq_len_m, hid_dim*2]

        #****Sentence And Memory Mask****
        x_mask = tf.tile(tf.expand_dims(tf.sequence_mask(self.sequence_lengths, seq_len_x), axis=-1), [1,1,self.hidden_dim*2])
        x_sentence_state = tf.expand_dims(tf.reduce_mean(x_state * tf.cast(x_mask, tf.float32), axis=1), axis=1) 
        # x_sentence_state [batch_size, 1, hidden_dim*2]
        memory_mask = tf.tile(tf.expand_dims(tf.sequence_mask(self.m_seq_len, seq_len_m), axis=-1), [1,1,1,self.hidden_dim*2])
        memory_sentence_state = tf.reduce_mean(memory_seq_state * tf.cast(memory_mask, tf.float32), axis=2)
        # memory_sentence_state [batch_size, 7, hid_dim*2]
        gcn_rel_in = tf.concat([memory_sentence_state[:,:5,:], x_sentence_state, memory_sentence_state[:,5:,:]], axis=1)
        # gcn_rel_in [batch_size, 8, hid_dim*2]

        #****Relation GCN Layer****
        sen_state_gcn_out = self.GCN_Rel_Layer(gcn_in = gcn_rel_in, rel_emb = self.relation_embedding, in_dim = self.hidden_dim*2,
                                gcn_dim = self.hidden_dim*2, batch_size = self.batch_size, max_nodes = 8, 
                                max_labels = 18, adj = self.adj_rel, num_layers=3, name="rel_GCN")
        sen_state_gcn = sen_state_gcn_out[-1]
        # sen_state_gcn_out [batch_size, 8, hid_dim*2]

        x_sentence_state = sen_state_gcn[:,5:6,:] # x_sentence_state [batch_size, 1, hid_dim*2]
        print('x_sentence_state: ', x_sentence_state)
        memory_sentence_state = tf.concat([sen_state_gcn[:,:5,:], sen_state_gcn[:,6:,:]], axis=1) 
        # memory_sentence_state [batch_size, 7, hid_dim*2]

        #****update word state according to rel_GCN***
        #x_state, memory_seq_state = self.updata_word_state(x_state, memory_seq_state, x_sentence_state, memory_sentence_state, seq_len_x, seq_len_m)


        self.sentence_attention = self.sentence_weight(x_state, memory_sentence_state)
        # sentence_attention [batch_size, seq_len_x, 7]
        sentence_attention_state,sentence_x_state = self.x_update_sentence(self.sentence_attention, x_state, memory_sentence_state)
        # sentence_x_state [batch_size, seq_len_x, hidden_dim*2]
        self.word_attention, l_g = self.word_weight(sentence_x_state, memory_seq_state)
        # word_attention [batch_size,seq_len_x,7,seq_len_m]
        word_x_state = self.x_update_word(memory_seq_state, self.word_attention, self.sentence_attention)
        # word_x_state [batch_size, seq_len_x, hidden_dim*2]
        self.logits = self.mlp_predict(x_state, sentence_attention_state, word_x_state)
        # logits [sen_num, seq_len, num_tags]

        with tf.name_scope("loss"):
            # cross_entropy_i = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=id_logits, labels=self.input_yi, name='loss_i')
            cross_entropy_g = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_yg, name='loss_g')
            mask0_g = tf.to_float(tf.equal(self.input_yg, 0))
            mask1_g = tf.to_float(tf.equal(self.input_yg, 1))
            mask2_g = tf.to_float(tf.equal(self.input_yg, 2))
            mask3_g = tf.to_float(tf.equal(self.input_yg, 3))
            mask4_g = tf.to_float(tf.equal(self.input_yg, 4))
            mask5_g = tf.to_float(tf.equal(self.input_yg, 5))
            mask6_g = tf.to_float(tf.equal(self.input_yg, 6))
            mask7_g = tf.to_float(tf.equal(self.input_yg, 7))
            mask8_g = tf.to_float(tf.equal(self.input_yg, 8))
            mask9_g = tf.to_float(tf.equal(self.input_yg, 9))
            mask10_g = tf.to_float(tf.equal(self.input_yg, 10))
            mask11_g = tf.to_float(tf.equal(self.input_yg, 11))
            mask12_g = tf.to_float(tf.equal(self.input_yg, 12))
            mask13_g = tf.to_float(tf.equal(self.input_yg, 13))
            mask14_g = tf.to_float(tf.equal(self.input_yg, 14))
            mask15_g = tf.to_float(tf.equal(self.input_yg, 15))
            mask16_g = tf.to_float(tf.equal(self.input_yg, 16))
            mask_g = tf.multiply(mask0_g,0.5)+tf.multiply(mask1_g,1.2)+tf.multiply(mask2_g,1.9)+tf.multiply(mask3_g,1)+tf.multiply(mask4_g,1.50)+\
                     tf.multiply(mask5_g,2)+tf.multiply(mask6_g,4)+tf.multiply(mask7_g,2.5)+tf.multiply(mask8_g,2.500)+\
                     tf.multiply(mask9_g,5.5)+tf.multiply(mask10_g,2.50)+tf.multiply(mask11_g,4)+tf.multiply(mask12_g,6.5)+tf.multiply(mask13_g,5)+\
                     tf.multiply(mask14_g,5.5)+tf.multiply(mask15_g,5)+tf.multiply(mask16_g,7)
            mask0_i = tf.to_float(tf.equal(self.input_yi, 0))
            mask1_i = tf.to_float(tf.equal(self.input_yi, 1))
            mask_i = tf.multiply(mask0_i,0.134)+tf.multiply(mask1_i,0.866)
            # self.cross_entropy_i = tf.multiply(cross_entropy_i,mask_i)
            self.cross_entropy_g = tf.multiply(cross_entropy_g,mask_g)

            # cross_entropy [sen_num, seq_len]
            mask = tf.sequence_mask(self.sequence_lengths)
            # self.loss_i = tf.boolean_mask(self.cross_entropy_i, mask)
            self.loss_g = tf.boolean_mask(self.cross_entropy_g, mask)
            if self.l2_lambda:
                vars = tf.trainable_variables()
                lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars])
                self.loss = tf.reduce_mean(self.loss_g) + self.l2_lambda * lossL2+l_g*0
            else:
                self.loss = tf.reduce_mean(self.loss_g) + l_g*0

    def encode_x_sentence(self):
        # '''
        # encode 输入的x
        # :return: x_state [batch_size, seq_len_x, hidden_dim*2]
        # '''
        with tf.name_scope('placeholders'):
            self.input_x = tf.placeholder(tf.int32, [self.batch_size, None], name="input_x")
            self.shape = tf.shape(self.input_x)
            # input_x [batch_size, seq_len_x]
            self.input_yg = tf.placeholder(tf.int32, [self.batch_size, None], name="input_yg")
            # input_y [batch_size, seq_len_x]
            self.input_yi = tf.placeholder(tf.int32, [self.batch_size, None], name="input_yi")
            # input_yi [batch_size, seq_len_x]
            self.memory = tf.placeholder(tf.int32, [self.batch_size, 7, None], name="memory")
            self.m_shape = tf.shape(self.memory)
            # memory [batch_size, 7, seq_len_m]
            self.sequence_lengths = tf.placeholder(tf.int32, [self.batch_size], name="sequence_lengths")
            # sequence_length [batch_size]
            self.m_seq_len = tf.placeholder(tf.int32, [self.batch_size, 7], name="memory_sequence_lengths")
            # m_seq_len [batch_size, 7]
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

            self.adj_x = tf.placeholder(tf.float32,  shape=[self.batch_size, None, None]) 
            # self.adj_x [batch_size, seq_len, seq_len]  dynamic
            self.adj_m = tf.placeholder(tf.float32,  shape=[self.batch_size*7, None, None]) 
            # self.adj_m [batch_size*7, seq_len, seq_len] not add memory now
            self.adj_rel = tf.placeholder(tf.float32,  shape=[self.batch_size, 18, None, None]) 
            # self.adj_x [batch_size, 18, 8, 8]


        with tf.device("/cpu:0"), tf.name_scope("embedding_layer"):
            self.embedding_w1 = tf.Variable(self.embeddings, dtype=tf.float32, trainable=self.update_embedding,
                                     name="embedding_w1")
            self.embedding_w2 = tf.get_variable(name="embedding_w2", shape=[self.embedding_w1.shape[0], 50],
                                 initializer=tf.truncated_normal_initializer(stddev=0.01),
                                 trainable= True, dtype=tf.float32)
            self.embedding_w = tf.concat([self.embedding_w1, self.embedding_w2], axis=-1)

            self.relation_embedding = tf.get_variable(name="relation_embedding", shape=[18, self.hidden_dim*2],
                                 initializer=tf.truncated_normal_initializer(stddev=0.01),
                                 trainable= True, dtype=tf.float32)

            self.word_embeddings = tf.nn.embedding_lookup(self.embedding_w, self.input_x)
            # #self.word_embeddings = tf.nn.dropout(self.word_embeddings, self.dropout_keep_prob)

        with tf.name_scope('bilstm'):
            self.cell_fw = GRUCell(self.hidden_dim)
            self.cell_bw = GRUCell(self.hidden_dim)
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=self.cell_fw,
                cell_bw=self.cell_bw,
                inputs=self.word_embeddings,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)

            x_lstm_out = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            x_lstm_out = tf.nn.dropout(x_lstm_out, self.dropout_keep_prob)
            # x_lstm_out [batch_size, seq_len_x, hidden_dim*2](300)
        return x_lstm_out, self.shape[1], self.m_shape[-1]

    def encode_memory(self):
        # '''
        # encode memory
        # :return: memory_state [batch_size, 7, hid_dim*2]
        # '''
        memory = tf.reshape(self.memory, [self.batch_size*7, -1])
        # memory [batch_size*7, seq_len_m]
        m_seq_len = tf.reshape(self.m_seq_len, [-1])
        # m_seq_len [batch_size*7]
        self.memory_embeddings = tf.nn.embedding_lookup(self.embedding_w, memory)
        # self.memory_embeddings [batch_size*7, seq_len_m, emb_size]
        (memory_fw_seq, memory_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=self.cell_fw,
            cell_bw=self.cell_bw,
            inputs=self.memory_embeddings,
            sequence_length=m_seq_len,
            dtype=tf.float32)
        # memory_state = tf.reshape(tf.concat(memory_state, axis=1), [self.batch_size, 7, self.hidden_dim*2])
        # # memory_state [batch_size, 7, hid_dim*2]
        memory_seq_state = tf.reshape(tf.concat([memory_fw_seq, memory_bw_seq], axis=-1), [self.batch_size, 7, -1, self.hidden_dim*2])
        # memory_seq_state [batch_size, 7, seq_len_m, hid_dim*2]
        return memory_seq_state

    def sentence_weight(self, x_state, sentence_state):
        # '''
        # 计算每个词的state对文章中每个句子的attention
        # ：input：x_state [batch_size, seq_len_x, hidden_dim*2]
        #          sentence_state [batch_size, 7, hidden_dim*2]
        # :return: sentence_att [batch_size, seq_len_x, 7]
        # '''
        sentence_state = tf.transpose(sentence_state, [0,2,1])
        # sentence_state [batch_size, hidden_dim*2, 7]
        sentence_att = tf.nn.softmax(tf.matmul(x_state, sentence_state))
        # sentence_att [batch_size, seq_len_x, 7]
        #with tf.name_scope('normalization'):
            #mean = tf.tile(tf.expand_dims(tf.reduce_mean(sentence_att, axis=2), -1), [1,1,7])
            # mean [batch_size, seq_len_x, 7]
            #sentence_att = tf.nn.relu(sentence_att - mean)
            # sentence_att [batch_size, seq_len_x, 7]
            #mask = tf.cast(sentence_att, tf.bool)
            # mask [batch_size, seq_len_x, 7]
            #sentence_att = tf.nn.softmax(tf.where(mask, sentence_att, tf.ones_like(sentence_att) * (-2 ** 32 + 1)), 2)
        return  sentence_att

    def x_update_sentence(self, sentence_attention, x_state, sentence_state):
        # '''
        # 用算出来的sentence attention权重更新初始x_state
        # :param sentence_attention: [batch_size, seq_len_x, 7]
        # :param x_state: [batch_size, seq_len_x, hidden_dim*2]
        # :param sentence_state: [batch_size, 7, hid_dim*2]
        # :return: update_x_state [batch_size, seq_len_x, hidden_dim]
        # '''
        sentence_attention_state = tf.matmul(sentence_attention, sentence_state)
        # sentence_attention_state [batch_size, seq_len_x, hidden_dim*2]
        with tf.name_scope('update_xstate'):
            x_state = tf.reshape(tf.concat([x_state, sentence_attention_state], axis=-1), [-1, self.hidden_dim*4])
            # x_state [batch_size*seq_len_x, hidden_dim*4]
            W1 = tf.get_variable(name="W1",
                                shape=[4 * self.hidden_dim, self.hidden_dim*2],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)
            b1 = tf.get_variable(name="b1",
                                shape=[self.hidden_dim*2],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)
            update_x_state = tf.reshape(tf.matmul(x_state, W1) + b1, [-1, self.shape[1], self.hidden_dim*2])
            # update_x_state [batch_size, seq_len_x, hidden_dim*2]
            # print('update_x_state: ', update_x_state)
        return sentence_attention_state,update_x_state


    def word_weight(self, sentence_x_state, memory_seq_state):
        #'''
        #用更新后的x_state对文章中每个词做attention---element wise
        #:param sentence_x_state: [batch_size, seq_len_x, hidden_dim*2]
        #:param memory_seq_state: [batch_size, 7, seq_len_m, hidden_dim*2]
        #:return: word_attention: [batch_size, seq_len_x, 7, seq_len_m]
        #'''
        #self.memory_embedding_reshape = tf.reshape(self.memory_embeddings, [self.batch_size, 7, -1, self.embedding_size])
        # memory_embedding_reshape [batch_size, 7, seq_len_m, emb_size]

        #print('sentence_x_state: ', sentence_x_state)
        W5 = tf.get_variable(name="W5",
                             shape=[self.hidden_dim*2, 1],
                             initializer=tf.contrib.layers.xavier_initializer(),
                             dtype=tf.float32)
        b5 = tf.get_variable(name="b5",
                             shape=[1],
                             initializer=tf.zeros_initializer(),
                             dtype=tf.float32)
        temp_count = tf.constant(0)
        word_attention = tf.zeros([self.batch_size, 7, self.m_shape[2]], tf.float32)
        # word_attention [batch_size, 7, seq_len_m]
        def cond(temp_count, word_attention):
            return temp_count < self.shape[1]

        def body(temp_count, word_attention):
            temp = tf.transpose(tf.expand_dims(tf.expand_dims(sentence_x_state[:,temp_count,:], -1), -1), [0,3,2,1])
            #print('temp: ', temp)
            # temp [batch_size, 1, 1, hid_dim*2]
            temp_1 = tf.reshape(tf.multiply(memory_seq_state, temp), [-1, self.hidden_dim*2])
            #print('temp_1: ', temp_1)
            # temp_1 [batch_size*7*seq_len_m, hid_dim*2]
            temp_1 = tf.reshape(tf.matmul(temp_1, W5)+b5, [self.batch_size, 7, -1])
            word_attention = tf.concat([word_attention, temp_1], 1)
            return temp_count + 1, word_attention

        temp_count, word_attention = tf.while_loop(cond, body, [temp_count, word_attention],
                                                   shape_invariants=[temp_count.get_shape(), tf.TensorShape([self.batch_size, None, None])])
        word_attention = word_attention[:, 7:, :]
        word_attention = tf.reshape(word_attention,[self.batch_size,self.shape[1],7,tf.shape(self.memory)[2]])
        #print("word_attention",word_attention.shape)
        #word_attention = tf.Print(word_attention,[word_attention],"word_attention_before:",summarize=1000)
        #print("word_att_pad:",word_attention)
        # word_attention [batch_size,seq_len_x,7,seq_len_m]

        mask_tf = tf.reshape(tf.sequence_mask(tf.reshape(tf.tile(self.m_seq_len, [1, self.shape[1]]), [-1])), [self.batch_size, self.shape[1], 7, -1])
        # mask_tf [batch_size,seq_len_x,7,seq_len_m]
        word_attention = tf.where(mask_tf, word_attention, tf.ones_like(word_attention)*(-2**32+1))  # mask后的word attention
        word_attention = tf.nn.softmax(word_attention, dim=-1)
        #print("word_stt_mask",word_attention)
        #word_attention = tf.Print(word_attention,[word_attention],"word_attention",summarize = 100)
        l_g = tf.reduce_sum(tf.abs(word_attention[:,:,:,1:]-word_attention[:,:,:,:-1]))
        return word_attention,l_g

    def x_update_word(self, memory_seq_state, word_attention, sentence_attention):
        # '''
        # 用算出来的word attention更新x_state
        # :param word_attention:[batch_size,seq_len_x,7,seq_len_m]
        # :param sentence_attention:[batch_size, seq_len_x, 7]
        # :param memory_seq_state: [batch_size, 7, seq_len_m, hid_dim*2]
        # :return: [sen_num, seq_len, hidden_dim]
        # '''
        memory_reshape = tf.reshape(memory_seq_state, [self.batch_size, -1, self.hidden_dim*2])
        # memory_reshape [batch_size, 7*seq_len_m, hid_dim*2]
        sentence_attention = tf.transpose(tf.expand_dims(sentence_attention, -1), [0,1,3,2])
        # sentence_attention [batch_size, seq_len_x, 1, 7]
        word_attention = tf.transpose(word_attention, [0,1,3,2])
        # word_attention: [batch_size, seq_len_x, seq_len_m, 7]
        word_attention = tf.transpose(tf.multiply(word_attention, sentence_attention), [0,1,3,2])
        # word_attention [batch_size, seq_len_x, 7, seq_len_m]
        word_attention = tf.reshape(word_attention, [self.batch_size, self.shape[1], -1])
        # word_attention [batch_size, seq_len_x, 7*seq_len_m]
        word_x_state = tf.matmul(word_attention, memory_reshape)
        # word_x_state [batch_size, seq_len_x, hidden_dim*2]
        return word_x_state

    def mlp_predict(self, x_state, sentence_x_state, word_x_state):
        # '''
        # 输出序列预测结果
        # :param x_state: [batch_size, seq_len_x, hidden_dim*2]
        # :param sentence_x_state: [batch_size, seq_len_x, hidden_dim]
        # :param word_x_state: [batch_size, seq_len_x, hidden_dim]
        # :return:[sen_num, seq_len, num_tags]
        # '''
        x_state = tf.reshape(tf.concat([x_state, sentence_x_state, word_x_state], axis=-1), [-1, self.hidden_dim*6])
        W3 = tf.get_variable(name="W3",
                             shape=[6*self.hidden_dim, self.hidden_dim],
                             initializer=tf.contrib.layers.xavier_initializer(),
                             dtype=tf.float32)
        b3 = tf.get_variable(name="b3",
                             shape=[self.hidden_dim],
                             initializer=tf.zeros_initializer(),
                             dtype=tf.float32)
        temp_state = tf.nn.tanh(tf.matmul(x_state, W3) + b3)
        #temp_state = tf.nn.dropout(temp_state, self.dropout_keep_prob)
        W4 = tf.get_variable(name="W4",
                             shape=[self.hidden_dim, self.num_tags],
                             initializer=tf.contrib.layers.xavier_initializer(),
                             dtype=tf.float32)
        b4 = tf.get_variable(name="b4",
                             shape=[self.num_tags],
                             initializer=tf.zeros_initializer(),
                             dtype=tf.float32)
        logits = tf.reshape(tf.matmul(temp_state, W4) + b4, [self.batch_size, self.shape[1], self.num_tags])
        # logits = tf.nn.dropout(logits_temp, self.dropout_keep_prob)
        # logits [batch_size, seq_len_x, num_tags]
        self.prediction = tf.argmax(logits, axis=2)
        # prediction [batch_size, seq_len_x]
        return logits

    def mlp_identification(self, x_state):
        '''
        输出identification预测结果
        :param x_state: [batch_size, seq_len_x, hidden_dim*2]
        :return:[sen_num, seq_len, num_tags]
        '''
        x_state = tf.reshape(x_state, [-1, self.hidden_dim*2])
        W2 = tf.get_variable(name="W2",
                             shape=[self.hidden_dim*2,  2],
                             initializer=tf.contrib.layers.xavier_initializer(),
                             dtype=tf.float32)
        b2 = tf.get_variable(name="b2",
                             shape=[2],
                             initializer=tf.zeros_initializer(),
                             dtype=tf.float32)
        id_logits = tf.reshape(tf.matmul(x_state, W2) + b2, [self.batch_size, self.shape[1], 2])
        # id_logits [batch_size, seq_len_x, 2]
        return id_logits

    def GCNLayer(self, gcn_in, in_dim, gcn_dim, batch_size, max_nodes, adj, num_layers=1, name="GCN"):
        out = []
        out.append(gcn_in)

        for layer in range(num_layers):
            gcn_in = out[-1]                     

            if len(out) > 1: in_dim = gcn_dim               

            with tf.name_scope('%s-%d' % (name,layer)): 
                act_sum = tf.zeros([batch_size, max_nodes, gcn_dim]) #[batch_size, seq_len, hid_dim*2]
                with tf.variable_scope('label_name-%s_layer-%d' % (name, layer)) as scope:
                    w_in   = tf.get_variable('w_in',   [in_dim, gcn_dim],   initializer=tf.contrib.layers.xavier_initializer(),     regularizer=self.regularizer)
                    b_in   = tf.get_variable('b_in',   [1, gcn_dim],    initializer=tf.constant_initializer(0.0),       regularizer=self.regularizer)
                    w_out  = tf.get_variable('w_out',  [in_dim, gcn_dim],   initializer=tf.contrib.layers.xavier_initializer(),     regularizer=self.regularizer)
                    b_out  = tf.get_variable('b_out',  [1, gcn_dim],    initializer=tf.constant_initializer(0.0),       regularizer=self.regularizer)
                    w_loop = tf.get_variable('w_loop', [in_dim, gcn_dim],   initializer=tf.contrib.layers.xavier_initializer(),     regularizer=self.regularizer)
                    w_gin  = tf.get_variable('w_gin',  [in_dim, 1],     initializer=tf.contrib.layers.xavier_initializer(),     regularizer=self.regularizer)
                    b_gin  = tf.get_variable('b_gin',  [1],         initializer=tf.constant_initializer(0.0),       regularizer=self.regularizer)
                    w_gout = tf.get_variable('w_gout', [in_dim, 1],     initializer=tf.contrib.layers.xavier_initializer(),     regularizer=self.regularizer)
                    b_gout = tf.get_variable('b_gout', [1],         initializer=tf.constant_initializer(0.0),       regularizer=self.regularizer)
                    #w_gloop = tf.get_variable('w_gloop',[in_dim, 1],    initializer=tf.contrib.layers.xavier_initializer(),     regularizer=self.regularizer)

                with tf.name_scope('in_arcs_name-%s_layer-%d' % (name, layer)):
                    inp_in  = tf.tensordot(gcn_in, w_in, axes=[2,0]) + tf.expand_dims(b_in, axis=0) 
                    # inp_in: [batch_size, seq_len, hid_dim*2(gcn_dim)]
                    in_t    = tf.stack([tf.matmul(adj[i], inp_in[i]) for i in range(batch_size)])
                    # adj[i]: [seq_len, seq_len]   inp_in[i]:[seq_len, hid_dim*2(gcn_dim)]   in_t: [batch_size, seq_len, hid_dim*2]
                    in_t = tf.nn.dropout(in_t, keep_prob=self.dropout_keep_prob)
                    inp_gin = tf.tensordot(gcn_in, w_gin, axes=[2,0]) + tf.expand_dims(b_gin, axis=0)
                    # inp_gin: [batch_size, seq_len]
                    in_gate = tf.stack([tf.matmul(adj[i], inp_gin[i]) for i in range(batch_size)])
                    # in_gate: [batch_size, seq_len]
                    in_gsig = tf.sigmoid(in_gate)
                    in_act   = in_t * in_gsig  # in_t: [batch_size, seq_len, hid_dim*2]
                    #in_act = in_t

                with tf.name_scope('out_arcs_name-%s_layer-%d' % (name, layer)):
                    inp_out  = tf.tensordot(gcn_in, w_out, axes=[2,0]) + tf.expand_dims(b_out, axis=0)
                    out_t    = tf.stack([tf.matmul(tf.transpose(adj[i]), inp_out[i]) for i in range(batch_size)])
                    # out_t: [batch_size, seq_len, hid_dim*2]
                    out_t    = tf.nn.dropout(out_t, keep_prob=self.dropout_keep_prob)
                    inp_gout = tf.tensordot(gcn_in, w_gout, axes=[2,0]) + tf.expand_dims(b_gout, axis=0)
                    out_gate = tf.stack([tf.matmul(tf.transpose(adj[i]), inp_gout[i]) for i in range(batch_size)])
                    out_gsig = tf.sigmoid(out_gate)
                    out_act  = out_t * out_gsig  # out_t: [batch_size, seq_len, hid_dim*2]

                with tf.name_scope('self_loop'):
                    # inp_loop  = tf.tensordot(gcn_in, w_loop,  axes=[2,0])
                    # inp_loop  = tf.nn.dropout(inp_loop, keep_prob=self.dropout_keep_prob)
                    # inp_gloop = tf.tensordot(gcn_in, w_gloop, axes=[2,0])
                    # loop_gsig = tf.sigmoid(inp_gloop)
                    #loop_act  = inp_loop * loop_gsig
                    loop_act  = gcn_in

                act_sum = act_sum + in_act + loop_act + out_act
                #act_sum += in_act
                gcn_out = tf.nn.relu(act_sum)
                out.append(gcn_out)

        return out


    def GCN_Rel_Layer(self, gcn_in, rel_emb, in_dim, gcn_dim, batch_size, max_nodes, max_labels, adj, num_layers=1, name="GCN"):

        out = []
        out.append(gcn_in)

        for layer in range(num_layers):
            gcn_in    = out[-1]     # [batch_size, seq_len, hid_dim*2]
            
            if len(out) > 1: in_dim = gcn_dim               # After first iteration the in_dim = gcn_dim

            with tf.name_scope('%s-%d' % (name,layer)): 

                act_sum = tf.zeros([batch_size, max_nodes, gcn_dim])
                
                for lbl in range(max_labels):  # relation number
                    rel_emb_lbl = tf.tile(tf.expand_dims(tf.expand_dims(rel_emb[lbl], axis=0), axis=0), [batch_size, max_nodes, 1])  # [batch_size, seq_len, hid_dim*2]

                    with tf.variable_scope('label-%d_name-%s_layer-%d' % (lbl, name, layer)) as scope:
                        w_rel   = tf.get_variable('w_in',   [in_dim, gcn_dim],   initializer=tf.contrib.layers.xavier_initializer(),     regularizer=self.regularizer)
                        b_rel   = tf.get_variable('b_in',   [1, gcn_dim],    initializer=tf.constant_initializer(0.0),       regularizer=self.regularizer)

                    with tf.name_scope('arcs-%s_name-%s_layer-%d' % (lbl, name, layer)):
                        inp_in  = gcn_in * rel_emb_lbl   # [batch_size, seq_len, hid_dim*2]
                        inp_in  = tf.tensordot(gcn_in, w_rel, axes=[2,0]) + tf.expand_dims(b_rel, axis=0)  # [batch_size, seq_len, hid_dim*2]
                        in_t    = tf.stack([tf.matmul(adj[i][lbl], inp_in[i]) for i in range(batch_size)])
                        in_t    = tf.nn.dropout(in_t, keep_prob=self.dropout_keep_prob)

                    act_sum += in_t 
                gcn_out = tf.nn.relu(act_sum)
                out.append(gcn_out)
        return out

    def updata_word_state(self, x_state, memory_seq_state, x_sentence_state, memory_sentence_state, seq_len_x, seq_len_m):
        # update the state of x_state and memory_seq_state
        # x_state [batch_size, seq_len_x, hidden_dim*2]
        # memory_seq_state [batch_size, 7, seq_len_m, hid_dim*2]
        # x_sentence_state [batch_size, 1, hid_dim*2]
        # memory_sentence_state [batch_size, 7, hid_dim*2]
        tmp_x_state = tf.concat([x_state, tf.tile(x_sentence_state, [1, seq_len_x, 1])], axis=-1)
        tmp_memory_seq_state = tf.concat([memory_seq_state, tf.tile(tf.expand_dims(memory_sentence_state, axis=2), [1, 1, seq_len_m, 1])], axis=-1)
        W_up = tf.get_variable(name="W_up",
                             shape=[self.hidden_dim*4,  self.hidden_dim*2],
                             initializer=tf.contrib.layers.xavier_initializer(),
                             dtype=tf.float32)
        b_up = tf.get_variable(name="b_up",
                             shape=[self.hidden_dim*2],
                             initializer=tf.zeros_initializer(),
                             dtype=tf.float32)
        x_state = tf.matmul(tmp_x_state, W_up) + b_up
        memory_seq_state = tf.matmul(tmp_memory_seq_state, W_up) + b_up
        return x_state, memory_seq_state

