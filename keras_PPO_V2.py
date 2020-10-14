import tensorflow as tf
from tensorflow import keras
import numpy as np

"""
keras_PPO_V2 版本参照了V1版本的代码风格，同时根据github改善了部分结构，
GitHub网址为：https://github.com/nric/ProximalPolicyOptimizationKeras/blob/master/ppo_deterministic_v2.py
后期的代码会采用V2代码的自定义loss函数，替代tf.Gradient()结构，同时V1版本修改其参数
"""
"""
参数说明：
a_dims:动作的维度，整数
s_dims:状态的维度，整数
upper_bound:动作的上限
epsilon:clip的参数，float
GAMA:reward衰减值
GAE_LAMBDA:GAE参数
update_alpha:软更新参数
entropy_loss_ratio:可选择参数，用于ppo_loss，不用时参数设置为0即可
batch_size:时间T，以整数代替步长，多长时间进行一次训练
"""


class PPO:
    def __init__(self, a_dims, s_dims,
                 upper_bound,
                 epsilon=0.2,
                 GAMA=0.99,
                 GAE_LAMBDA=0.95,
                 update_alpha=0.95,
                 entropy_loss_ratio=0.001,
                 batch_size=32
                 ):
        # 网络参数的定义
        self.a_dims = a_dims
        self.s_dims = s_dims
        self.epsilon = epsilon
        self.entropy_loss_ratio = entropy_loss_ratio

        # 构建网络实体
        self.new_actor = self.actor_net()
        self.old_actor = self.actor_net()
        self.critic = self.critic_net()
        self.old_actor.set_weights(self.new_actor.get_weights())

        # 数据中心的定义,该创建遵循V1统一原则
        self.batch_size = batch_size
        self.state_batch = np.zeros((batch_size, s_dims))
        self.next_state_batch = np.zeros((batch_size, s_dims))
        self.action_batch = np.zeros((batch_size, a_dims))
        self.reward_batch = np.zeros((batch_size, 1))
        self.step_counter = 0

        # 充数给action_policy使用
        self.cheat_advantage = np.zeros((1, 1))
        self.cheat_old_prediction = np.zeros((1, self.a_dims))

        # 环境参数
        self.upper_bound = upper_bound
        self.GAMA = GAMA
        self.GAE_LAMBDA = GAE_LAMBDA
        self.update_alpha = update_alpha

    # 这里的actor_net为了方便使用自定义loss函数，与V1版本有点区别
    # 为了方便loss函数的参数的引入，额外引入advantage，old_predict层
    # 针对pendulum创建的网络
    def actor_net(self):
        # 输入层的定义
        # 由于使用的是自定义环境，state输入shape采用(self.s_dims,)，源代码中使用的是shape=self.s_dims
        state = keras.Input(shape=(self.s_dims), name='state_input')
        advantage = keras.Input(shape=(1), name="advantage_input")
        old_prediction = keras.Input(shape=(self.a_dims), name="old_prediction_input")

        # 隐藏层的定义
        # 这里可以结合V1版本中的创建风格，这里只采用源代码中风格
        # 由该层可以看出，虽然输入层有三种，但是后两种只是用于loss函数的传参
        dense = keras.layers.Dense(64, activation='relu', name='dense1')(state)
        dense = keras.layers.BatchNormalization()(dense)
        dense = keras.layers.Dense(64, activation='relu', name='dense')(dense)
        dense = keras.layers.BatchNormalization()(dense)

        # 输出层定义
        outputs = keras.layers.Dense(self.a_dims, activation='tanh', name='actor_output_layer')(dense)
        # 这里采用V1风格
        actor_network = keras.Model(inputs=[state, advantage, old_prediction], outputs=outputs)
        actor_network.compile(
            optimizer='Adam',
            loss=self.actor_loss(advantage=advantage, old_prediction=old_prediction)
        )
        return actor_network

    def critic_net(self):
        # 输入层的定义
        state = keras.layers.Input(shape=(self.s_dims,), name='state_input')

        # 定义隐藏层
        dense = keras.layers.Dense(64, activation='relu', name='dense1')(state)
        dense = keras.layers.Dense(64, activation='relu', name='dense2')(dense)

        # 输出层
        V = keras.layers.Dense(1, name='actor_output_layer')(dense)
        critic_network = keras.Model(state, V)
        critic_network.compile(optimizer='Adam', loss='mean_squared_error')
        return critic_network

    def actor_loss(self, advantage, old_prediction):
        """
        这里采用github中的loss计算方法，使用了交叉熵，后期测试去除交叉熵的效果
        :param advantage: 优势值
        :param old_prediction: 老的神经网络得出的值
        :return: 返回loss
        """

        def loss(y_true, y_pred):
            """

            :param y_true: one_hot 标签
            :param y_pred:每个动作的概率
            :return:
            """
            prob = keras.backend.sum(y_true * y_pred)
            old_prob = keras.backend.sum(y_true * old_prediction)
            ratio = prob / (old_prob + 1e-10)
            # print(ratio)
            # np_ratio = ratio.
            clip_ratio = keras.backend.clip(ratio, min_value=1. - self.epsilon,
                                            max_value=1. + self.epsilon)
            surrogate1 = ratio * advantage
            surrogate2 = clip_ratio * advantage
            # 没有弄清楚这地方是干啥
            # 貌似这个可以选择不加入
            # entropy_loss = (prob * keras.backend.log(prob + 1e-10))
            actor_loss = -keras.backend.mean(keras.backend.minimum(surrogate1, surrogate2))
            # + self.entropy_loss_ratio * entropy_loss)
            return actor_loss

        return loss

    # V1风格
    def store(self, state, action, reward, next_state):
        state = np.array(state).reshape((1, self.s_dims))
        action = np.array(action).reshape((1, self.a_dims))
        next_state = np.array(next_state).reshape((1, self.s_dims))

        index = self.step_counter % self.batch_size
        self.state_batch[index] = state
        self.next_state_batch[index] = next_state
        self.action_batch[index] = action
        self.reward_batch[index] = reward
        self.step_counter += 1

        # 返回当前是否存满
        if index + 1 == self.batch_size:
            return True
        else:
            return False

    def clean_date(self):
        self.state_batch = np.zeros((self.batch_size, self.s_dims))
        self.next_state_batch = np.zeros((self.batch_size, self.s_dims))
        self.action_batch = np.zeros((self.batch_size, self.a_dims))
        self.reward_batch = np.zeros(self.batch_size, 1)

    # 采用V1风格
    # 争对不同的环境需要修改网络层，尤其是输出部分
    def action_policy(self, state):
        """

        :param state: 形式为np.ones((1,s_dims))
        :return: action
        """
        # print(state)
        # state = tf.convert_to_tensor(state)
        # print(state.reshape((2,1)))
        state = np.reshape(state, (-1, self.s_dims))
        # print(state)
        action = self.new_actor.predict_on_batch([state, self.cheat_advantage, self.cheat_old_prediction]).flatten()
        action = action * self.upper_bound

        return action

    def discount_reward(self):
        """Generates GAE type rewards and pushes them into memory object
        #GAE algorithm:
            #delta = r + gamma * V(s') * mask - V(s)  |aka advantage
            #gae = delta + gamma * lambda * mask * gae |moving average smoothing
            #return(s,a) = gae + V(s)  |add value of state back to it.
        """
        gae = 1
        mask = 1
        discount_reward = np.zeros((self.batch_size, 1))
        for i in reversed(range(self.batch_size)):
            # print(self.state_batch[i])
            state = self.state_batch[i]
            next_state = self.next_state_batch[i]
            state = np.reshape(state, (-1, self.s_dims))
            next_state = np.reshape(next_state, (-1, self.s_dims))
            v = self.get_v(state)
            v = np.squeeze(v)
            # print(v)
            next_v = self.get_v(next_state)
            next_v = np.squeeze(next_v)
            delta = self.reward_batch[i] + self.GAMA * next_v * mask - v
            gae = delta + self.GAMA * self.GAE_LAMBDA * mask * gae
            discount_reward[i] = gae + v
        # print(discount_reward.shape)
        return discount_reward

    def get_v(self, state):
        v = self.critic.predict_on_batch(state).flatten()
        v = np.squeeze(v)
        return v

    def get_old_predict(self, state):
        # print(state.shape)
        state = np.reshape(state, (-1, self.s_dims))
        old_predict = self.old_actor.predict_on_batch(
            [state, self.cheat_advantage, self.cheat_old_prediction]).flatten()
        result = np.squeeze(old_predict)
        # print(result)
        return result

    # test函数
    def get_new_predict(self, state):
        # print(state.shape)
        state = np.reshape(state, (-1, self.s_dims))
        old_predict = self.new_actor.predict_on_batch(
            [state, self.cheat_advantage, self.cheat_old_prediction]).flatten()
        result = np.squeeze(old_predict)
        # print(result)
        return result

    def learn(self):
        """Train the actor and critic networks using GAE Algorithm.
        1. Get GAE rewards
        2. reshape batches s,a,gae_r baches
        3. get value of state
        4. calc advantage
        5. get "old" precition (of target network)
        6. fit actor and critic network
        7. soft update target "old" network
        """
        # batch表示数量
        discount_reward = self.discount_reward()
        # print(discount_reward.shape)
        batch_v = self.get_v(self.state_batch)
        batch_v = np.reshape(batch_v, (-1, 1))
        # print(batch_v.shape)
        batch_advantage = discount_reward
        # print(batch_advantage.shape)
        batch_old_prediction = self.get_old_predict(self.state_batch)
        # batch_new_prediction = self.get_new_predict(self.state_batch)
        # one_hot标签的制作
        batch_action_hot = np.ones((self.batch_size, self.a_dims))
        batch_action_hot = np.reshape(batch_action_hot, (self.batch_size, self.a_dims))
        state = np.reshape(self.state_batch, (self.batch_size, self.s_dims))
        # batch_advantage = np.reshape(batch_advantage, (self.batch_size, 1))
        # print(batch_advantage)
        # batch_old_prediction = np.reshape(batch_old_prediction, (self.batch_size, self.a_dims))

        # ratio = batch_new_prediction / (batch_old_prediction + 1e-10)
        # print(ratio)

        # 训练
        self.new_actor.fit(x=[state, batch_advantage, batch_old_prediction], y=batch_action_hot, verbose=0)
        self.critic.fit(x=self.state_batch, y=discount_reward, verbose=0)
        self.soft_update()

    def soft_update(self):
        # new_weights = []
        # target_variables = self.old_actor.weights
        # for i, variable in enumerate(self.new_actor.weights):
        #     new_weights.append(variable * self.update_alpha + target_variables[i] * (1 - self.update_alpha))
        #
        # self.old_actor.set_weights(new_weights)
        self.old_actor.set_weights(self.new_actor.get_weights())

    def save_model(self):
        self.new_actor.save_weights("pendulum_new_actor.h5")
        self.old_actor.save_weights("pendulum_old_actor.h5")

        self.critic.save_weights("pendulum_critic.h5")
