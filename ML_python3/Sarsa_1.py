import numpy as np
import pandas as pd
import time
import tkinter as tk

'''
 4*4 的迷宫：
---------------------------
| 入口 |      |      |      |
---------------------------
|      |     | 陷阱  |      |
---------------------------
|      | 陷阱 |  终点 |      |  
---------------------------
|      |     |      |      |  
---------------------------
'''

UNIT = 40  # pixels
MAZE_H = 4  # grid height
MAZE_W = 4  # grid width


class Maze(tk.Tk, object):
    """
    环境类GUI可视化
    """

    def __init__(self):
        # super() 函数是用于调用父类(超类)的一个方法
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('迷宫')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))  # 窗口大小
        self._build_maze()

    def _build_maze(self):
        # 设置画布
        self.canvas = tk.Canvas(self, bg='white',
                                height=MAZE_H * UNIT,
                                width=MAZE_W * UNIT)

        # 画网格
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # 创造起点
        origin = np.array([20, 20])

        # 陷阱
        hell1_center = origin + np.array([UNIT * 2, UNIT])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black')
        # 陷阱
        hell2_center = origin + np.array([UNIT, UNIT * 2])
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0] - 15, hell2_center[1] - 15,
            hell2_center[0] + 15, hell2_center[1] + 15,
            fill='black')

        # 画终点
        oval_center = origin + UNIT * 2
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        # 画玩家
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        # 显示画作
        self.canvas.pack()

    def reset(self):
        self.update()  # tkinter内置的update
        time.sleep(0.5)
        self.canvas.delete(self.rect)  # 删除玩家原位置
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        # 返回初始位置
        return self.canvas.coords(self.rect)

    def step(self, action):
        s = self.canvas.coords(self.rect)  # 得到现在位置
        base_action = np.array([0, 0])
        # 根据不同行为改变位置
        if action == 0:  # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:  # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.rect, base_action[0], base_action[1])  # 移动

        s_ = self.canvas.coords(self.rect)  # 更新下一位置

        # 是否达到终点
        if s_ == self.canvas.coords(self.oval):
            # 是则给与奖励 完成该回合
            reward = 5
            done = True
            s_ = 'terminal'
            # 若是陷阱  也停止回合  给予惩罚
        elif s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]:
            reward = -10
            done = True
            s_ = 'terminal'
        else:
            # 否则继续
            reward = 0
            done = False

        return s_, reward, done

    def render(self):
        # 更新
        time.sleep(0.1)
        self.update()


# 我们甚至可以定义一个 主class RL, 然后将 QLearningTable 和 SarsaTable 作为 主class RL 的衍生, 这个主 RL 可以这样定义.
# 所以我们将之前的 __init__, check_state_exist, choose_action, learn 全部都放在这个主结构中, 之后根据不同的算法更改对应的内容就好了.
class RL(object):
    # 初始化
    def __init__(self, action_space, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = action_space  # 行为表
        self.lr = learning_rate  # 学习率
        self.gamma = reward_decay  # 奖励衰减
        self.epsilon = e_greedy  # 贪婪度

        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    # 检测 state 是否存在
    # 这个功能就是检测 q_table 中有没有当前 state 的步骤了, 如果还没有当前 state,
    # 那我我们就插入一组全 0 数据, 当做这个 state 的所有 action 初始 values.
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    # 选择 action
    # 这里是定义如何根据所在的 state, 或者是在这个 state 上的 观测值 (observation) 来决策.
    def choose_action(self, observation):
        self.check_state_exist(observation)  # 检测本 state 是否在 q_table 中存在
        if np.random.rand() < self.epsilon:
            # 选择 Q value 最高的 action
            state_action = self.q_table.loc[observation, :]
            # 同一个 state, 可能会有多个相同的 Q action value, 所以我们乱序一下
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # 随机选择 action
            action = np.random.choice(self.actions)
        return action

    # 学习更新参数
    def learn(self, *args):
        pass  # 每种的都有点不同, 所以用 pass


# off-policy
class QLearningTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(QLearningTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)  # 表示继承关系

    # learn 的方法在每种类型中有不一样, 需重新定义
    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update


# on-policy
class SarsaTable(RL):
    # 有了父类的 RL, 我们这次的编写就很简单, 只需要编写 SarsaTable 中 learn 这个功能就完成了. 因为其他功能都和父类是一样的
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(SarsaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)   # 表示继承关系

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, a_]   # q_target 基于选好的 a_ 而不是 Q(s_) 的最大值
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # 更新 q_table


def update():
    for episode in range(100):
        # 初始化环境
        observation = env.reset()

        # Sarsa 根据 state 观测选择行为
        action = RL.choose_action(str(observation))

        while True:
            # 刷新环境
            env.render()

            # 在环境中采取行为, 获得下一个 state_ (obervation_), reward, 和是否终止
            observation_, reward, done = env.step(action)

            # 根据下一个 state (obervation_) 选取下一个 action_
            action_ = RL.choose_action(str(observation_))

            # 从 (s, a, r, s, a) 中学习, 更新 Q_tabel 的参数 ==> Sarsa
            RL.learn(str(observation), action, reward, str(observation_), action_)

            # 将下一个当成下一步的 state (observation) and action
            observation = observation_
            action = action_

            # 终止时跳出循环
            if done:
                break

    # 大循环完毕
    print('game over')
    env.destroy()


if __name__ == "__main__":
    env = Maze()
    RL = SarsaTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()
