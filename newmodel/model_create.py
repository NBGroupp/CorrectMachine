# -- coding: utf-8 --
#=====================================================================
import tensorflow as tf
import numpy as np
import math

#定义相关参数

HIDDEN_SIZE = 100
NUM_LAYERS = 2
VOCAB_SIZE = 100000

TRAIN_BATCH_SIZE = 10
TEST_BATCH_SIZE = 10

DATA_SIZE=500
TRAIN_DATA_SIZE = int(DATA_SIZE*1.0)
# TEST_DATA_SIZE = int(DATA_SIZE-TRAIN_BATCH_SIZE)
TEST_DATA_SIZE = TRAIN_DATA_SIZE

TRAIN_EPOCH_SIZE = math.ceil(TRAIN_DATA_SIZE / TRAIN_BATCH_SIZE)
TEST_EPOCH_SIZE = math.ceil(TEST_DATA_SIZE / TEST_BATCH_SIZE)

NUM_EPOCH = 30   #训练轮数

TRAIN_NUM_STEP =3  #前后多少用来预测
TEST_NUM_STEP = 3  #前后多少用来预测
LEARNING_RATE = 0.1

KEEP_PROB = 0.95


sentence = "令天天气很好，今我很高兴。"
target= "今天天气很好，令我很高兴。"
char_set=[]

#定义一个类来描述模型结构
class Proofreading_Model(object):
    def __init__(self, is_training, batch_size, num_steps):

        self.batch_size = batch_size
        self.num_steps = num_steps
        self.targets = tf.placeholder(tf.int32, [None, 1])
        embedding = tf.get_variable("embedding", [VOCAB_SIZE, HIDDEN_SIZE]) #embedding矩阵

        with tf.variable_scope('First') as scope:
            self.input_data1 = tf.placeholder(tf.int32, [None, num_steps])
            # 定义使用LSTM结构及训练时使用dropout。
            lstm_cell1 = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE)
            if is_training:
                lstm_cell1 = tf.contrib.rnn.DropoutWrapper(lstm_cell1, output_keep_prob=KEEP_PROB)
            cell1 = tf.contrib.rnn.MultiRNNCell([lstm_cell1] * NUM_LAYERS)
            self.initial_state1 = cell1.zero_state(batch_size, tf.float32) # 初始化最初的状态。
            # inputs1=[batch_size,num_steps,hidden_size]
            inputs1 = tf.nn.embedding_lookup(embedding, self.input_data1) #将原本单词ID转为单词向量。
            if is_training:
                inputs1 = tf.nn.dropout(inputs1, KEEP_PROB)
            ''' ouputs1=[batch_size,num_steps,hidden_size]
                state1=[batch_size,hidden_size]
            '''
            outputs1, state1 = tf.nn.dynamic_rnn(cell1, inputs1, initial_state=self.initial_state1, dtype=tf.float32)
            cell_output1 = outputs1[:, -1, :]
            # output1, state1 = cell1(cell_output1, state1)
            output1=cell_output1
            self.final_state1 = state1

        with tf.variable_scope('Second') as scope:
            self.input_data2 = tf.placeholder(tf.int32, [None, num_steps])
            lstm_cell2 = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE)
            if is_training:
                lstm_cell2 = tf.contrib.rnn.DropoutWrapper(lstm_cell2, output_keep_prob=KEEP_PROB)
            cell2 = tf.contrib.rnn.MultiRNNCell([lstm_cell2] * NUM_LAYERS)
            self.initial_state2 = cell2.zero_state(batch_size, tf.float32)
            inputs2 = tf.nn.embedding_lookup(embedding, self.input_data2)
            if is_training:
                inputs2 = tf.nn.dropout(inputs2, KEEP_PROB)
            outputs2, state2 = tf.nn.dynamic_rnn(cell2, inputs2, initial_state=self.initial_state2, dtype=tf.float32)
            cell_output2 = outputs2[:, -1, :]
            # output2, state2 = cell2(cell_output2, state2)
            output2=cell_output2
            self.final_state2 = state2

        #全连接层
        self.output=(output1+output2)/2
        weight = tf.get_variable("weight", [HIDDEN_SIZE, VOCAB_SIZE])
        bias = tf.get_variable("bias", [VOCAB_SIZE])
        self.logits = tf.matmul(self.output, weight) + bias  # logits=[batch_size,vocab_size]
        # print(self.logits.shape)

        ''' 定义交叉熵损失函数和平均损失。
        logits中在vocab_size个结果中选择概率最大的结果与相应的targets结果比较计算loss值
        返回一个 [batch_size] 的1维张量 
        w = tf.ones([5], dtype=tf.float32)表示每个batch loss值的权重
        '''
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [self.logits],
            [self.targets],
            [tf.ones([batch_size], dtype=tf.float32)])

        #self.cost = tf.reduce_sum(loss) / batch_size
        self.cost = tf.reduce_mean(loss)
        # 只在训练模型时定义反向传播操作。
        if not is_training: return

        self.train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(self.cost)

# 使用给定的模型model在数据data上运行train_op并返回在全部数据上的cost值
def run_epoch(session, model, data, train_op, is_training, epoch_size):

    total_costs = 0.0
    state1 = session.run(model.initial_state1)
    state2 = session.run(model.initial_state2)

    dataX1,dataX2,dataY=data
    max_cnt=len(dataY)
    cnt=0  #现在取第cnt个输入
    correct_num=0

    # 训练一个epoch。
    for step in range(epoch_size):
        if(cnt+TRAIN_BATCH_SIZE>max_cnt):
            cnt=max_cnt-TRAIN_BATCH_SIZE
        x1=dataX1[cnt:cnt+TRAIN_BATCH_SIZE]
        # print(x1)
        x2=dataX2[cnt:cnt+TRAIN_BATCH_SIZE]
        y=dataY[cnt:cnt+TRAIN_BATCH_SIZE]
        # print(y)
        cost,state1,state2,outputs, _ = session.run([model.cost, model.final_state1, model.final_state2,
                                                     model.logits, train_op],
                                     feed_dict={model.input_data1: x1, model.input_data2: x2,
                                                model.targets: y,
                                                model.initial_state1: state1,
                                                model.initial_state2: state2
                                                })
        total_costs += cost
        # if output_log and step % 100 == 0:
        if (step+1) % 1 == 0:
            print("After %d steps, cost : %.3f" % (step, total_costs / (step+1)))
            if(is_training):
                with open('../results.txt', 'a') as f:
                    f.write("After %d steps, cost : %.3f" % (step, total_costs / (step+1))+'\n')
            else:
                with open('../test_results.txt', 'a') as f:
                    f.write("After %d steps, cost : %.3f" % (step, total_costs / (step + 1)) + '\n')

            # print(outputs.shape)
            index = np.argmax(outputs, axis=1)
            # print(index.shape)
            # print(np.shape(char_set))
            target_index=np.array(y).ravel()
            print("outputs: "+' '.join([char_set[t] for t in index]))
            print("targets: "+' '.join([char_set[t] for t in target_index]))
            correct_num=correct_num+sum(index==target_index)

            if (is_training):
                with open('../results.txt','a') as f:
                    f.write(' '.join([char_set[t] for t in index])+'\n')
                    f.write(' '.join([char_set[t] for t in target_index])+'\n'+'\n')
            else:
                with open('../test_results.txt', 'a') as f:
                    f.write(' '.join([char_set[t] for t in index]) + '\n')
                    f.write(' '.join([char_set[t] for t in target_index]) + '\n'+'\n')

        cnt += TRAIN_BATCH_SIZE
        if (cnt >= max_cnt):
            cnt = 0

    if not is_training:
        acc=correct_num/TEST_DATA_SIZE
        print("acc:",acc)
        with open('../test_results.txt', 'a') as f:
            f.write("acc:"+str(acc))

#定义主函数并执行
def main():

    with open('../model_data/data1.18876', 'r') as f:
        rows = f.read().split('\n')
        data1 = [one.split() for one in rows]
        for one in data1:
            for index, ele in enumerate(one):
                one[index]=int(ele)
    with open('../model_data/data2.18876', 'r') as f:
        rows = f.read().split('\n')
        data2 = [one.split() for one in rows]
        for one in data2:
            for index, ele in enumerate(one):
                one[index]=int(ele)
    with open('../model_data/target.18876', 'r') as f:
        rows = f.read().split('\n')
        target = [one.split() for one in rows]
        for one in target:
            for index, ele in enumerate(one):
                one[index]=int(ele)
    with open('../model_data/vocab.100000', 'r') as f:
        global char_set
        char_set = f.read().split('\n')

    # global DATA_SIZE=len(target)
    train_data=(data1[0:TRAIN_DATA_SIZE],data2[0:TRAIN_DATA_SIZE],target[0:TRAIN_DATA_SIZE])
    test_data=train_data
    # test_data=(data1[TRAIN_DATA_SIZE:DATA_SIZE],data2[TRAIN_DATA_SIZE:DATA_SIZE],target[TRAIN_DATA_SIZE:DATA_SIZE])

    # train_data = data_init()
    # test_data=train_data

    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    with tf.variable_scope("Proofreading_model", reuse=None, initializer=initializer):
        train_model = Proofreading_Model(True, TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)

    with tf.variable_scope("Proofreading_model", reuse=True, initializer=initializer):
        eval_model = Proofreading_Model(False, TEST_BATCH_SIZE, TEST_NUM_STEP)

    saver=tf.train.Saver()
    # 训练模型。
    with tf.Session() as session:
        tf.global_variables_initializer().run()

        print("In training:")
        with open('../results.txt', 'a') as f:
            f.write("In training:"+'\n'+'\n')
        for i in range(NUM_EPOCH):
            print("In iteration: %d " % (i + 1))
            run_epoch(session, train_model, train_data, train_model.train_op, True, TRAIN_EPOCH_SIZE)

            #valid_perplexity = run_epoch(session, eval_model, eval_queue, tf.no_op(), False, valid_epoch_size)
            #print("Epoch: %d Validation Perplexity: %.3f" % (i + 1, valid_perplexity))
        print("In testing:")
        with open('../test_results.txt', 'a') as f:
            f.write("In training:" + '\n'+'\n')
        run_epoch(session, eval_model, test_data, tf.no_op(), False, TEST_EPOCH_SIZE)
        saver.save(session,"../ckpt/model.ckpt")


def data_init():
    global VOCAB_SIZE
    global TRAIN_BATCH_SIZE
    tmp = list(set(sentence))
    char_set.append('_')
    char_set.extend(tmp)

    VOCAB_SIZE=len(char_set)
    # print (VOCAB_SIZE)

    # char_dic={w:i+1 for i,w in enumerate(char_set)}
    char_dic = {}
    for i, w in enumerate(char_set):
        char_dic[w] = i
    # print(char_set)
    # print(char_dic)

    #dataX1为待遇测词前TRAIN_NUM_STEP个词,dataX2为待预测词后TRAIN_NUM_STEP个词
    #dataY为目标输出
    dataX1 = []
    dataX2 = []
    dataY = []

    length = len(sentence)
    for i in range(length):
        if i < TRAIN_NUM_STEP:
            pre_fill = TRAIN_NUM_STEP - i
            x_str1 = ['_'] * pre_fill
            for j in range(i):
                x_str1.append(sentence[j])
        else:
            x_str1 = list(sentence[i - TRAIN_NUM_STEP:i])
        # print("x_str1: ",x_str1)

        if length - (i + 1) < TRAIN_NUM_STEP:
            suf_fill = TRAIN_NUM_STEP - (length - (i + 1))
            x_str2 = []
            for j in range(i + 1, length):
                x_str2.append(sentence[j])
            for j in range(suf_fill):
                x_str2.append('_')
            x_str2.reverse()
        else:
            x_str2 = list(sentence[i + 1:i + 1 + TRAIN_NUM_STEP])
            x_str2.reverse()
        # print("x_str2: ",x_str2)

        y_str = list(target[i])
        # print("y_str: ",y_str)

        x1 = [char_dic[c] for c in x_str1]
        x2 = [char_dic[c] for c in x_str2]
        y = [char_dic[c] for c in y_str]
        dataX1.append(x1)
        dataX2.append(x2)
        dataY.append(y)

    # print ("dataX1: ", dataX1)
    # print ("dataX2: ", dataX2)
    # print ("datay: ", dataY)

    #TRAIN_BATCH_SIZE=len(dataY)
    # print (TRAIN_BATCH_SIZE)
    return dataX1, dataX2, dataY

if __name__ == "__main__":
    main()


