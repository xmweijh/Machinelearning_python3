import numpy as np
import random

# 初始化奖赏函数  事先设计好的
r = np.array([[-1, -1, -1, -1, 0, -1], [-1, -1, -1, 0, -1, 100], [-1, -1, -1, 0, -1, -1], [-1, 0, 0, -1, 0, -1],
              [0, -1, -1, 0, -1, 100], [-1, 0, -1, -1, 0, 100]])

# 初始化q值 暂时设置为0
q = np.zeros([6, 6], np.float)

# 贪婪率
greed = 0.8

episode = 0

while episode < 1000:
    state = np.random.randint(0, 6)
    if state != 5:
        next_state_list = []
        # 遍历找到下一步能够到达的点
        for i in range(6):
            if r[state, i] != -1:
                next_state_list.append(i)
        if len(next_state_list) > 0:
            # 从能够到达的点随机选择一个状态
            next_state = next_state_list[random.randint(0, len(next_state_list) - 1)]
            # 计算q值
            q[state, next_state] = r[state, next_state] + greed * max(q[next_state])
    episode = episode + 1

# 设定初始状态
i = 2
state = i
count = 0
list1 = []
while state != 5:
    if count > 11:
        print("failed ! \n")
        break
    list1.append(state)
    # #取出元素最大值所对应的索引
    next_state = q[state].argmax()
    count = count + 1
    state = next_state
list1.append(5)
print('path is :')
print(list1)
