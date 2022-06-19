# -*- coding: utf-8 -*-
"""
Created on Fri May 27 21:44:22 2022

@author: Administrator
"""

import numpy as np
import copy


class kd_node:
    '''
    KD树的节点
    '''
    def __init__(self, point, region, dim):
        self.point = point   # point stored in this kd-tree node
        self.region = region # [[lower_left_x, lower_left_y], [upper_right_x, upper_right_y]]
        self.split_dim = dim # on which dimension this point splits
        self.left = None
        self.right = None


def median(data_lst, split_dim):
    '''找出data_lst的中位数'''
    d = len(data_lst) // 2
    
    # make sure that on split_dim dimension all the points in right subtree 
    # are equal or greater than the point stored in root node
    l = 0
    h = d
    while l < h:
        m = (l + h) // 2
        
        if data_lst[m][split_dim] < data_lst[h][split_dim]:
            l = m + 1
        else:
            h = m
    return data_lst[h], h


def get_split_dim(data_lst):
    """
    计算points在每个维度上的方差, 选择在方差最大的维度上进行切割
    """
    var_lst = np.var(data_lst, axis=0)
    split_dim = 0
    for v in range(1, len(var_lst)):
        if var_lst[v] > var_lst[split_dim]:
            split_dim = v
    return split_dim


def build_kdtree(data_lst, region, square_list):
    '''构建kd树'''
    split_dim = get_split_dim(data_lst)
    
    data_lst = sorted(data_lst, key=lambda x: x[split_dim])
   
    point, m = median(data_lst, split_dim)
   
    tree_node = kd_node(point, region, split_dim)
    square_list.append(region)

    if m > 0:
        sub_region = copy.deepcopy(region)
        sub_region[1][split_dim] = point[split_dim]
        tree_node.left = build_kdtree(data_lst[:m], sub_region, square_list)

    if len(data_lst) > m + 1:
        sub_region = copy.deepcopy(region)
        sub_region[0][split_dim] = point[split_dim]
        tree_node.right = build_kdtree(data_lst[m + 1:], sub_region, square_list)

    return tree_node


def euclid_distance(d1, d2):
    dist = np.linalg.norm(np.array(d1) - np.array(d2))
    return dist


class NeiNode:
    '''neighbor node'''
    def __init__(self, p, d):
        self.__point = p
        self.__dist = d

    def get_point(self):
        return self.__point

    def get_dist(self):
        return self.__dist

class BPQ:
    '''bounded priority queue'''  #有界的优先队列
    def __init__(self, k):
        self.__K = k
        self.__pos = 0
        self.__bpq = [0] * (k + 2)

    def add_neighbor(self, neighbor):
        self.__pos += 1
        self.__bpq[self.__pos] = neighbor
        self.__swim_up(self.__pos)
        if self.__pos > self.__K:
            self.__exchange(1, self.__pos)
            self.__pos -= 1
            self.__sink_down(1)

    def get_knn_points(self):
        return [neighbor.get_point() for neighbor in self.__bpq[1:self.__pos + 1]]

    def get_max_distance(self):
        if self.__pos > 0:
            return self.__bpq[1].get_dist()
        return 0

    def is_full(self):
        return self.__pos >= self.__K

    def print_bpq(self):
        neighbor_list = []
        dis_list = []
        if self.__pos < 1:
            print ('no neighbor')
        for p in self.__bpq[1: self.__pos + 1]:
            neighbor_list.append(p.get_point())
            dis_list.append(p.get_dist())
        return neighbor_list, dis_list


    def __swim_up(self, n):
        while n > 1 and self.__less(n//2, n):
            self.__exchange(n//2, n)
            n = n//2

    def __sink_down(self, n):
        while 2*n <= self.__pos:
            j = 2*n
            if j < self.__pos and self.__less(j, j+1):
                j += 1
            if not self.__less(n, j):
                break
            self.__exchange(n, j)
            n = j

    def __less(self, m, n):
        return self.__bpq[m].get_dist() < self.__bpq[n].get_dist()

    def __exchange(self, m, n):
        tmp = self.__bpq[m]
        self.__bpq[m] = self.__bpq[n]
        self.__bpq[n] = tmp

def knn_search_kd_tree(knn_bpq, tree, target, search_track):
    '''kd树的近邻搜索'''
    track_node = []
    node_ptr = tree
    while node_ptr:
        while node_ptr:
            track_node.append(node_ptr)
            search_track.append([node_ptr.point, knn_bpq.get_knn_points(), knn_bpq.get_max_distance()])
            dist = euclid_distance(node_ptr.point, target)
            knn_bpq.add_neighbor(NeiNode(node_ptr.point, dist))
            
            search_track.append([None, knn_bpq.get_knn_points(), knn_bpq.get_max_distance()])

            split_dim = node_ptr.split_dim
            if target[split_dim] < node_ptr.point[split_dim]:
                node_ptr = node_ptr.left
            else:
                node_ptr = node_ptr.right

        while track_node:
            iter_node = track_node[-1]
            del track_node[-1]

            split_dim = iter_node.split_dim
            if not knn_bpq.is_full() or \
                    abs(iter_node.point[split_dim] - target[split_dim]) < knn_bpq.get_max_distance():
                if target[split_dim] < iter_node.point[split_dim]:
                    node_ptr = iter_node.right
                else:
                    node_ptr = iter_node.left

            if node_ptr:
                break

def print_kd_tree(tree):
    node_list = [tree]
    while node_list:
        next_node_list = []
        for node in node_list:
            if node.left:
                next_node_list.append(node.left)
            if node.right:
                next_node_list.append(node.right)
        node_list = next_node_list
        
##############获取delta########################
def getGammaOrderIndex(X_data, p_ratio, n_data,d_data):
    global index, orignOrder,k_value
    #初始化delta 和 leader
    delta = np.zeros(len(X_data))
    leader = np.ones(len(X_data),int) * (-1)
    dist = []
    for i in range(1,len(X_data)):
        distance = np.linalg.norm(np.array(X_data[0] - np.array(X_data[i])))
        dist.append(distance)
    delta[0] = max(dist)
    leader[0]= -1
    a_list = []
    for k in range(1,len(X_data)):
        b_list = []
        a = orignOrder[str(X_data[k-1])]
        a_list.append(a)
        for j in range(k_value-1):
            b = orignOrder[str(n_data[k][j])]
            b_list.append(b)
        c = list((set(a_list)&set(b_list)))
        if c:                            
            m = np.zeros(len(c),int)
            dist = np.zeros(len(m))
            for i in range(len(c)):
                m[i] = np.argwhere(c[i] == np.array(b_list))[0][0]
            for j in range(len(m)):
                dist[j] = d_data[k][m[j]]
            p = np.argwhere(dist.min() == dist)[0][0]   
            up_index = np.argwhere(c[p] ==index)[0][0]
            delta[k] = dist.min()
            leader[k] = up_index        
        else:                            
            min_dist = np.zeros(k)
            min_index = np.zeros(k)
            for i in range(len(X_data[0:k])):
                min_dist[i] = np.linalg.norm(np.array(X_data[k]) - np.array(X_data[i]))
                min_index[i] = i
            delta[k] = min_dist.min()
            minindex = np.argwhere(min_dist.min() == min_dist)[0][0]
            orignindex = min_index[minindex]
            leader[k] = orignindex        
    gamma = delta * p_ratio
    gammaOrderIndex = gamma.argsort()[::-1]
    return delta, gamma, gammaOrderIndex, leader  
##############筛选聚类中心########################
def getCluster(X_data, p_ratio, n_data,d_data, n):
    delta, gamma, r, leader = getGammaOrderIndex(X_data, p_ratio, n_data,d_data)
    DN = int(np.sqrt(n))
    dis = gamma[r[0]]-gamma[r[n-1]]
    sp_max = 0.0
    blckNum = 0
    for i in range(1,DN):
       ui = gamma[r[i]]-gamma[r[i+1]]
       ui1 = gamma[r[i+1]]-gamma[r[i+2]]
       sp = abs(((i+1)**2/i**2)*(ui-ui1)/dis)
       if sp > sp_max:
          sp_max = sp
          blckNum = i+1        
    blockNum = []
    avg_rho = 0
    avg_delta = 0
    for i in range(blckNum):
        avg_rho += p_ratio[i]
    avg_rho = avg_rho / len(p_ratio)
    for i in range(blckNum):
        avg_rho += delta[i]
    avg_delta = avg_delta / len(delta)
    for i in range(blckNum):
        if (p_ratio[i] > avg_rho) & (delta[i] > avg_delta):
            blockNum.append(i) 
    return blockNum