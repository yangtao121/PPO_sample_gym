import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np


class PPO:
    """
    NOTICE: the batch_size is much less than the episodes!
    I thought the episodes can be set like 10*batch_size
    """

    def __init__(self, s_dims, a_dims, action_upper_bound,
                 action_lower_bound,
                 actor_lr=0.001,
                 critic_lr=0.002,
                 batch_size=32,
                 GAMA=0.9,
                 lambada=0.95,
                 epsilon=0.2,
                 tao=0.9
                 ):
        # 网络参数的设置
        self.s_dims = s_dims  # 状态的维度
        self.a_dims = a_dims  # 动作的维度
        self.epsilon = epsilon
        self.soft_update = tao

        # 优化器的定义
        self.actor_optimizer = keras.optimizers.Adam(actor_lr)
        self.critic_optimizer = keras.optimizers.Adam(critic_lr)

        # reward衰减||GAE参数
        self.GAMA = GAMA
        self.lambada = lambada

        # 动作的上下限
        self.action_lower_bound = action_lower_bound
        self.action_upper_bound = action_upper_bound

        # 容器的参数
        self.batch_size = batch_size
        self.store_counter = 0

        self.state_batch = np.zeros((self.batch_size, self.s_dims))
        self.next_state_batch = np.zeros((self.batch_size, self.s_dims))
        self.action_batch = np.zeros((self.batch_size, self.a_dims))
        self.reward_batch = np.zeros((self.batch_size, 1))
        self.maks_flag_batch = np.zeros((self.batch_size, 1))

        # actor net
        self.new_actor = self.actor_net()
        self.old_actor = self.actor_net()

        # critic net
        self.critic = self.critic_net()

    def actor_net(self):
        # 初始化权值
        last_init = tf.random_uniform_initializer(minval=-1, maxval=1)

        inputs = layers.Input(shape=(self.s_dims,))
        # PPO算法速度优于DDPG可以设置大一点的网络
        out = layers.Dense(64, activation="relu")(inputs)
        out = layers.BatchNormalization()(out)
        out = layers.Dense(64, activation="relu")(out)
        out = layers.BatchNormalization()(out)
        outputs = layers.Dense(self.a_dims, activation="tanh", kernel_initializer=last_init)(out)

        # outputs = outputs * self.action_upper_bound
        model = tf.keras.Model(inputs, outputs)

        return model

    def critic_net(self):
        # 输入为state，输出为v
        # 网络的创建需要优化
        inputs = layers.Input(shape=(self.s_dims,))
        out = layers.Dense(16, activation='relu')(inputs)
        out = layers.Dense(32, activation='relu')(out)
        out_v = layers.Dense(1)(out)

        model = tf.keras.Model(inputs, out_v)
        return model

    def action_policy(self, state):
        tf_state = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
        sample_actions = tf.squeeze(self.new_actor(tf_state) * self.action_upper_bound)
        sample_actions = sample_actions.numpy()
        legal_action = np.clip(sample_actions, self.action_lower_bound, self.action_upper_bound)
        # self.store_counter += 1
        return [np.squeeze(legal_action) * self.action_upper_bound]

    '''
    store函数返回当前batch是否已满，能不能进行网络的更新
    '''

    def store(self, state, action, reward, next_state, mask_flag):
        """

        :param state: pre_state
        :param action: action
        :param reward: reward given by env
        :param next_state: next_state(state)
        :param mask_flag: control GAE mask, when the end state is controlled by total steps, it can be always set 1
        :return:
        """
        index = self.store_counter % self.batch_size

        self.state_batch[index] = state
        self.next_state_batch[index] = next_state
        self.action_batch[index] = action
        self.reward_batch[index] = reward
        self.maks_flag_batch[index] = mask_flag

        # print(index)
        if index + 1 == self.batch_size:
            done = True
        else:
            done = False

        self.store_counter += 1
        return done

    '''
    由critic生成的v
    可能存在一些问题
    '''

    def get_v(self, state):
        """
        get the V
        :param state: input state
        :return: a numpy matrix
        """
        tf_state = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float64), 0)
        v = tf.squeeze(self.critic(tf_state))
        v = v.numpy()
        return v

    # 这里有几种算法实现，先用简单算法
    # 还有PPO2中的算法
    def discount_reward(self):
        discount_reward = np.zeros((self.batch_size, 1))
        for i in range(self.batch_size - 1):
            discount_reward[i + 1] = self.GAMA * discount_reward[i] + self.reward_batch[i + 1]

        # 归一化
        discount_reward = (discount_reward - np.mean(discount_reward)) / np.std(discount_reward)

        return discount_reward

    def Advantage(self):
        """
        参考网站为：https://www.youtube.com/watch?v=WxQfQW48A4A&feature=youtu.be
        使用GAE算法
        具体算法如下所示：
        1. mask is 0, if the state is terminal, otherwise is 1
        2. init gae=0, look backward from last step
        3. delta = r + gamma * V(s') * mask - V(s)  |aka advantage
        4. gae = delta + gamma * lambda * mask * gae |moving average smoothing
        5. return(s,a) = gae |add value of state back to it.
        :return:
        """
        gae = 0
        advantage = np.zeros((self.batch_size, 1))
        for i in reversed(range(self.batch_size)):
            V_next = self.get_v(self.next_state_batch[i])
            V = self.get_v(self.state_batch[i])
            delta = self.reward_batch[i] + self.GAMA * V_next * self.maks_flag_batch[i] - V
            gae = delta + self.GAMA * self.lambada * self.maks_flag_batch[i] * gae
            advantage[i] = gae
            # print(GAE_reward[i])

        return advantage

    def learn(self):
        """
        参考网址为：https://www.youtube.com/watch?v=WxQfQW48A4A&feature=youtu.be
        method: using GAE reward
        1. Get GAE rewards
        2. reshape batches s,a,gae_r batches
        3. get value of state
        4. calc advantage
        5. get "old" precition (of target network)
        6. fit actor and critic network
        7. soft update target "old" network
        :return:
        """
        # 将数据转换为张量
        state_batch = tf.convert_to_tensor(self.state_batch)
        # state_batch = tf.expand_dims(tf.convert_to_tensor(self.state_batch, dtype=tf.double), 0)
        # state_batch = tf.convert_to_tensor(state_batch, dtype=tf.double)
        # print(state_batch)
        # action_batch = tf.convert_to_tensor(self.action_batch)
        # reward_batch = tf.convert_to_tensor(self.reward_batch)
        # reward_batch = tf.cast(reward_batch, dtype=tf.float32)  # 数据类型转换

        # 求discount_reward
        # discount_reward = self.discount_reward()
        # discount_reward = tf.convert_to_tensor(discount_reward, dtype=tf.float32)

        # get GAE reward
        advantage = self.Advantage()
        # print(GAE_reward)
        tf_advantage = tf.convert_to_tensor(advantage, dtype=tf.float32)
        # tf_GAE_reward = tf.squeeze(tf_GAE_reward)
        # print(tf_GAE_reward)

        # update critic
        with tf.GradientTape() as tape:
            tf_v = self.critic(state_batch)
            # tf_v = tf.squeeze(tf_v)
            # print(tf_GAE_reward)
            # print(tf_v)
            y = tf_v - tf_advantage
            # tf_v = tf.convert_to_tensor(tf_v, dtype=tf.double)
            # print(tf_v)
            # tf_v = tf.convert_to_tensor(v, dtype=tf.float64)
            # v = self.get_v(self.state_batch)
            # y = GAE_reward - v
            # tf_y = tf.convert_to_tensor(y)
            critic_loss = tf.reduce_mean(tf.math.square(y))
            # print(critic_loss)
            # print(critic_loss.numpy())
        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic.trainable_variables)
        )

        # update actor
        with tf.GradientTape() as tape:
            ratio = self.new_actor(state_batch) / (self.old_actor(state_batch) + 1e-10)
            # print(ratio)
            clip = tf.clip_by_value(ratio, clip_value_min=1. - self.epsilon, clip_value_max=1. + self.epsilon)
            surrogate1 = ratio * tf_advantage
            surrogate2 = clip * tf_advantage
            actor_loss = -tf.reduce_mean(tf.math.minimum(surrogate1, surrogate2))
            # print(actor_loss)
        actor_grad = tape.gradient(actor_loss, self.new_actor.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.new_actor.trainable_variables)
        )

        # if self.store_counter % 20 == 0:
        #     self.update_old_act()

        # 计算优势函数（PPO1）
        # 存在问题
        # v = self.get_v(self.state_batch)
        # advantage = discount_reward - v
        # adv_tensor = tf.convert_to_tensor(advantage)

        # 神经网络的更新
        # critic的更新
        # with tf.GradientTape() as tape:
        #     v = self.get_v(state_batch)
        #     y = discount_reward - v
        #     critic_loss = tf.math.reduce_mean(tf.square(y))
        #     # print(critic_loss)
        #     # print("***********************")
        # critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        # # print(critic_grad)
        # self.critic_optimizer.apply_gradients(
        #     zip(critic_grad, self.critic.trainable_variables)
        # )
        #
        # # new_actor的更新
        # with tf.GradientTape() as tape:
        #     ratio = self.new_actor(state_batch) / (self.old_actor(
        #         state_batch) + 1e-10)  # 防止0的出现
        #     print(ratio)
        #     adv = discount_reward - self.get_v(state_batch)
        #     adv_rat = ratio * adv
        #     clip = tf.clip_by_value(ratio, 1. - self.epsilon, 1. + self.epsilon)
        #     actor_loss = -tf.math.reduce_mean(tf.math.minimum(
        #         adv_rat,
        #         tf.clip_by_value(ratio, 1. - self.epsilon, 1. + self.epsilon) * adv
        #     ))  # print(actor_loss)
        #     # print("***********************************")
        #
        # actor_grad = tape.gradient(actor_loss, self.new_actor.trainable_variables)
        # # print(actor_grad)
        # self.actor_optimizer.apply_gradients(
        #     zip(actor_grad, self.new_actor.trainable_variables)
        # )

    def update_old_act(self):
        """
        该函数支持soft模式,soft_update为1时为完全更新，0时为不更新
        :return:
        """
        new_weights = []
        target_variables = self.old_actor.weights
        for i, variable in enumerate(self.new_actor.weights):
            new_weights.append(variable * self.soft_update + target_variables[i] * (1 - self.soft_update))

        self.old_actor.set_weights(new_weights)

    def save_model(self):
        self.new_actor.save_weights("pendulum_online_actor.h5")
        self.old_actor.save_weights("pendulum_online_critic.h5")

        self.critic.save_weights("pendulum_target_actor.h5")

    def clean_data(self):
        self.state_batch = np.zeros((self.batch_size, self.s_dims))
        self.next_state_batch = np.zeros((self.batch_size, self.s_dims))
        self.action_batch = np.zeros((self.batch_size, self.a_dims))
        self.reward_batch = np.zeros((self.batch_size, 1))
        self.maks_flag_batch = np.zeros((self.batch_size, 1))
