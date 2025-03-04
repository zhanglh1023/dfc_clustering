import math
import queue

import numpy as np

from calculate_result import CalDistance
tot = 1
print(float("-inf") == float("-inf"))
def MCMF(points, centers, fair_num, kl, kr, dataToPoint, point_num) :

    a = []
    for i in range(len(centers)) :
        a.append([])
    for i in range(len(points)) :
        centers_id = points[i][1]
        a[centers_id].append(points[i])
    _k = 0
    for i in range(len(point_num)):
        if(point_num[i]): _k += 1
    C_NUM = len(points)
    center_num = len(centers)
    n = 1 + center_num + C_NUM + fair_num + 1 + 1 + 1
    m = center_num + C_NUM + C_NUM + fair_num + 1 + 1 + center_num + fair_num + 1
    N = n + 10
    M = m * 2 + 10
    ver = []
    edge = []
    cost = []
    Next = []
    Edge = []
    id = []
    for i in range(M) :
        ver.append(0)
        edge.append(0)
        cost.append(0.0)
        Next.append(0)
        Edge.append((-1, -1))
        id.append(0)
    head = []
    d = []
    incf = []
    pre = []
    v = []
    for i in range(N):
        head.append(0)
        d.append(float('inf'))
        incf.append(0)
        pre.append(0)
        v.append(0)
    global tot
    tot = 1
    def add(x, y, z, c, _id) :
        #print("Edge: ", "x: ", x, "y: ", y, "flow: ", z, "cost: ", c)
        global tot
        tot = tot + 1
        id[tot] = _id
        ver[tot] = y
        edge[tot] = z
        cost[tot] = c
        Next[tot] = head[x]
        head[x] = tot
        Edge[tot] = (x, y)
        tot = tot + 1
        id[tot] = _id
        ver[tot] = x
        edge[tot] = 0
        cost[tot] = -c
        Next[tot] = head[y]
        head[y] = tot
        Edge[tot] = (y, x)
    for i in range(center_num) :
        if(point_num[i]) :
            add(1, 2 + i, point_num[i] - 1, 0, -1)
    delt = 0
    for i in range(center_num) :
        for j in range(len(a[i])) :
            add(2 + i, 1 + center_num + delt + j + 1, 1, a[i][j][0], a[i][j][3])
            add(1 + center_num + delt + j + 1, 1 + center_num + C_NUM + 1 + a[i][j][2], 1, 0, -1)
        delt += point_num[i]
    for fair_id in range(fair_num) :
        add(1 + center_num + C_NUM + 1 + fair_id, 1 + center_num + C_NUM + fair_num + 1, kr[fair_id] - kl[fair_id], 0, -1)
    add(1 + center_num + C_NUM + fair_num + 1, 1, center_num - _k, 0, -1)
    S = 1 + center_num + C_NUM + fair_num + 2
    T = 1 + center_num + C_NUM + fair_num + 3
    kl_sum = 0
    for i in range(len(kl)):
        kl_sum += kl[i]
    print(kl)
    print("kl_sum:###################################")
    print(kl_sum)
    if(kl_sum - _k > 0) :
        add(S, 1 + center_num + C_NUM + fair_num + 1, kl_sum - _k, 0, -1)
    if(kl_sum - _k < 0) :
        add(1 + center_num + C_NUM + fair_num + 1, T, _k - kl_sum, 0, -1)
    for i in range(center_num) :
        if(point_num[i]) :add(S, 2 + i, 1, 0, -1)
    print(_k)
    for i in range(len(kl)) :
        add(1 + center_num + C_NUM + 1 + i, T, kl[i], 0, -1)
    '''for centers_id in range(center_num):
        for fair_id in range(fair_num):
            if(a[centers_id][fair_id][0] >= 0):
                add(centers_id + 2, fair_id + 2 + center_num, 1, a[centers_id][fair_id][0], a[centers_id][fair_id][3])
    for fair_id in range(fair_num):
        add(fair_id + 2 + center_num, 2 + center_num + fair_num, math.ceil(kr[fair_id]), 0, -1)'''
    maxflow = 0
    ans = 0
    print("kr")
    print(kr)
    def spfa() :
        q = queue.Queue()
        for i in range(N) :
            d[i] = float("inf")
        for i in range(N) :
            v[i] = 0
        q.put(S)
        d[S] = 0
        v[S] = 1
        incf[S] = float("inf")
        while q.qsize() :
            #print(q.qsize())
            x = q.get()
            #print(q.qsize())
            v[x] = 0
            i = head[x]
            while i :
                #print(i)
                if edge[i] == 0:
                    i = Next[i]
                    continue
                y = ver[i]
                if d[x] < float("inf") and d[y] > d[x] + cost[i] :
                    d[y] = d[x] + cost[i]
                    incf[y] = min(incf[x], edge[i])
                    pre[y] = i
                    if v[y] == 0:
                        v[y] = 1
                        q.put(y)
                i = Next[i]
        if d[T] == float("inf") : return False
        return True

    def update() :
        x = T
        while x != S:
            i = pre[x]
            edge[i] -= incf[T]
            #print(Edge[i][0], Edge[i][1], "-", incf[2 + center_num + fair_num])
            edge[i ^ 1] += incf[T]
            #print(Edge[i^1][0], Edge[i^1][1], "+", incf[2 + center_num + fair_num])
            x = ver[i ^ 1]
        return incf[T], d[T]
    while spfa() :
        x, y = update()
        maxflow += x
        ans += y
    print("最大流量", maxflow)
    print("最小费用", ans)
    print(tot)
    selected_point = []
    for i in range(2, tot + 1) :
        #print(i)
        if edge[i] == 0 :
            #print("#", Edge[i][0], Edge[i][1])
            if(Edge[i][0] < Edge[i][1] and Edge[i][0] > 1 and Edge[i][1] < 1 + center_num + C_NUM + 1) :
                print("#", Edge[i][0], Edge[i][1], id[i])
                selected_point.append(dataToPoint[id[i]])
    return selected_point



#dis:distance*point_num[lable[i]]、lable[i]:i's center_id、fair[i]:i's color、i:data_id(no point_id)
#points[dis, lable, fair, i]
def SNC(data, centers, lable, fair, kl, kr, point_num):
    centers_cnt = 0
    dis = []
    for i in range(len(data)) :
        dis.append(CalDistance(data[i], centers[lable[i]]) * point_num[lable[i]])
    points = []
    for i in range(len(dis)) :
        points.append((dis[i], lable[i], fair[i], i))
    sorted(points)
    #print(points)
    center_num = len(centers)
    dataToPoint = []
    for i in range(len(points)) :
        dataToPoint.append(0)
    for i in range(len(points)) :
        dataToPoint[points[i][3]] = i
    selected_points = MCMF(points, centers, len(kr), kl, kr, dataToPoint, point_num) #Return: Selected points
    Q = []
    for i in range(center_num) :
        Q.append([])
    Q_P = []
    for i in range(center_num):
        Q_P.append([])
    point_flag = []
    for i in range(len(points)) :
        point_flag.append(0)
    color_cnt = []
    for i in range(len(kl)) :
        color_cnt.append(0)
    _ = []
    for i in range(len(selected_points)) :
        p_id = selected_points[i]
        Q[points[p_id][1]].append(data[points[p_id][3]])
        _.append(points[p_id][3])
        #point_flag[p_id] = 1
        #color_cnt[points[p_id][2]] += 1

    '''for i in range(len(points)) :
        #if(center_flag[points[i][1]]) : continue
        if(color_cnt[points[i][2]] + 1 > kr[points[i][2]]) : continue
        #center_flag[points[i][1]] = 1
        color_cnt[points[i][2]] += 1
        Q[points[i][1]].append(data[points[i][3]])
        point_flag[i] = 1
        _.append(points[i][3])
        centers_cnt += 1'''
    '''for i in range(len(points)) :
        if(point_flag[i]) : continue
        if(color_cnt[points[i][2]] >= math.floor(kl[points[i][2]]) or color_cnt[points[i][2]] + 1 > math.ceil(kr[points[i][2]])) : continue
        color_cnt[points[i][2]] += 1
        Q_P[points[i][1]].append(i)
        selected_points.append(i)
        point_flag[i] = 1
    sorted(selected_points)
    for i in range(len(selected_points) - 1, -1, -1) :
        p_id = selected_points[i]
        if color_cnt[points[p_id][2]] - 1 < math.floor(kl[points[p_id][2]]) : continue
        for j in range(len(Q_P)):
            if len(Q_P[j]) > 1:
                for k in range(len(Q_P[j])):
                    if Q_P[j][k] == p_id:
                        del Q_P[j][k]
                        break'''
    '''for i in range(len(Q_P)):
        for j in range(len(Q_P[i])):
            p_id = Q_P[i][j]
            Q[i].append(data[points[p_id][3]])
            _.append(points[p_id][3])
            centers_cnt += 1'''
    centers_cnt = len(_)
    if centers_cnt > center_num :
        print("选择点的数量超过中心数")
    error = False
    for i in range(len(centers)) :
        if(point_num[i] and len(Q[i]) == 0) : error = True
    return Q, _, error

def Divide(data, gf_centers, Q, fair_num, data_distribution) :
    double_fair_centers = []
    double_fair_centers_num = 0
    double_fair_assignment = []
    for i in range(len(data)):
        double_fair_assignment.append(0)
    for i in range(len(gf_centers)):
        # print(i)
        id_array = []
        for j in range(len(Q[i])):
            # print(Q[i][j])
            double_fair_centers.append(Q[i][j])
            id_array.append(double_fair_centers_num)
            double_fair_centers_num += 1
        id = 0
        for j in range(fair_num):
            if j % 2 == 0:
                for k in range(len(data_distribution[i][j])):
                    double_fair_assignment[data_distribution[i][j][k]] = id_array[id]
                    id += 1
                    id %= len(id_array)
            else:
                for k in range(len(data_distribution[i][j]) - 1, -1, -1):
                    double_fair_assignment[data_distribution[i][j][k]] = id_array[id]
                    id += 1
                    id %= len(id_array)
    return double_fair_centers, double_fair_assignment
def Oral_Selected(data, centers, lable, fair, kl, kr, point_num):
    data_flag = []
    for i in range(len(data)) :
        data_flag.append(0)
    center_flag = []
    for i in range(len(centers)) :
        center_flag.append(0)
    Q = []
    for i in range(len(centers)) :
        Q.append([])
    color_cnt = []
    for i in range(len(kr)) :
        color_cnt.append(0)
    _ = []
    center_fair = []
    for i in range(len(centers)) :
        center_fair.append([])
        for j in range(len(kr)):
            center_fair[i].append([])
    for i in range(len(data)):
        center_id = lable[i]
        fair_id = fair[i]
        center_fair[center_id][fair_id].append(i)
    for i in range(len(centers)) :
        id = -1
        for j in range(len(kr)) :
            if(len(center_fair[i][j]) and color_cnt[j] < kl[j]) :
                id = j
                break
        if(id < 0) :
            for j in range(len(kr)) :
                if(len(center_fair[i][j]) and color_cnt[j] + 1 <= kr[j]) :
                #if (len(center_fair[i][j])):
                    id = j
                    break
        if(id < 0) :
            print("存在没选出中心点的簇")
            break
        color_cnt[id] += 1
        Q[i].append(data[center_fair[i][id][0]])
        _.append(center_fair[i][id][0])
        data_flag[center_fair[i][id][0]] = 1
    for i in range(len(kr)) :
        if color_cnt[i] >= kl[i] : continue
        while color_cnt[i] < kl[i] :
            for j in range(len(centers)) :
                if(len(center_fair[j][i]) == 0) : continue
                flag = 0
                for k in range(len(center_fair[j][i])) :
                    if data_flag[center_fair[j][i][k]] : continue
                    Q[j].append(data[center_fair[j][i][k]])
                    _.append(center_fair[j][i][k])
                    data_flag[center_fair[j][i][k]] = 1
                    color_cnt[i] += 1
                    flag = 1
                    break
                if flag :
                    break
    error = False
    for i in range(len(centers)) :
        if(point_num[i] and len(Q[i]) == 0) : error = True
    return Q, _, error


