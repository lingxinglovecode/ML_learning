import numpy as np
import torch.nn as nn



##位置编码
flag = True
if flag:
    d_hid = 512
    def get_position_angle_vec(position):
        # hid_j是0-511,d_hid是512，position表示单词位置0～N-1
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    n_position = 10
    # 每个单词位置0～N-1都可以编码得到512长度的向量
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    # 偶数列进行sin
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    # 奇数列进行cos
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

class  ScaledDotProductAttition()