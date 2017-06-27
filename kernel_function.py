import random
import math
import matplotlib.pyplot as plt
import numpy as np

def kernel_mean(data):
    center = []
    for i in range(2):
        rand = []
        rand.append(random.uniform(0.0, 1.0))
        '''sum'''
        rand.append(0)
        '''cnt'''
        rand.append(0)
        center.append(rand)

    changed = True
    while changed:
        changed = False
        for i in data:
            dist = 999999
            belong = -1
            for idxj, j in enumerate(center):
                tmp = abs(i[3] - j[0])
                if tmp < dist:
                    dist = tmp
                    belong = idxj
            center[belong][1] += dist
            center[belong][2] += 1
            if i[2] != belong:
                changed = True
                i[2] = belong
        '''calc new center'''
        for i in center:
            if i[2] != 0: i[0] = i[1]/i[2]
            #else: i[0] = i[0]/2
            i[1] = 0
            i[2] = 0
    
    for i in center:
        print i
    for i in data:
        plt.plot(i[0], i[1], color[i[2]])
    plt.show()

def k_mean(data):
    for k in K:
        center = []
        for i in range(k):
            rand = []
            rand.append(random.uniform(-2.0, 2.0))
            rand.append(random.uniform(0.0, 2.0))
            rand.append(0)
            rand.append(0)
            rand.append(0)
            center.append(rand)
            
        changed = True
        while changed:
            changed = False
            for i in data:
                dist = 999999
                belong = -1
                for idxj, j in enumerate(center):
                    tmp = math.pow(j[0] - i[0], 2) + math.pow(j[1] - i[1], 2)
                    if tmp < dist:
                        dist = tmp
                        belong = idxj
                center[belong][2] += i[0]
                center[belong][3] += i[1]
                center[belong][4] += 1
                if i[2] != belong:
                    changed = True
                    i[2] = belong
            '''calc new center'''
            for i in center:
                if i[4] != 0: i[0] = i[2]/i[4]
                else: i[0] = i[0]/2
                if i[4] != 0: i[1] = i[3]/i[4]
                else: i[1] = i[1]/2
                i[2] = 0
                i[3] = 0
                i[4] = 0
        
    
        for i in data:
            plt.plot(i[0], i[1], color[i[2]])
        plt.show()
        

def show(data, K, probabilities, color):
    for idx, i in enumerate(probabilities):
        max = -1
        tar = -1
        for k in range(K):
            if i[k] > max:
                max = i[k]
                tar = k
        tmp = data[idx].tolist()
        plt.plot(tmp[0], tmp[1], color[tar])

def maximize(mean, cnt_k, norm_data, cov, w, i):
    mean[i] = (1.0 / cnt_k[i]) * np.sum(norm_data[:, i] * data.T, axis = 1).T
    x_mean = np.matrix(data - mean[i])
    multiple = np.dot(np.multiply(x_mean.T,  norm_data[:, i]), x_mean)
    cov[i] = np.array((1 / cnt_k[i]) * multiple)
    w[i] = (1.0 / data.shape[0]) * cnt_k[i]


def expect(mean, norm_data, cov, w, i):
    p = pow(np.linalg.det(cov[i]), -0.5 ** (2 * np.pi) ** (-data.shape[1]/2.0))
    exp = np.exp(-0.5 * np.einsum('ij, ij -> i', data - mean[i], np.dot(np.linalg.inv(cov[i]) , (data - mean[i]).T).T ))
    norm_data[:, i] =  exp * p * w[i]


def EM_Algorithm(data, k, cnt):

    log_list = []
    res = []
    mean = []
    cov = []
    w = []
    norm_data = np.zeros((data.shape[0], k))
    
    for i in range(k):
        mean.append(random.choice(data))
        w.append(1.0 / k)
        cov.append(np.eye(2))

    
    
    while cnt >= 0:
        cnt -= 1
        # Expect
        for i in range(k):
            expect(mean, norm_data, cov, w, i)

        log_list.append(np.sum(np.log(np.sum(norm_data, axis = 1))))
        
        data_sum = np.sum(norm_data, axis = 1)
        norm_data = norm_data.T / data_sum
        norm_data = norm_data.T   
        cnt_k = np.sum(norm_data, axis = 0)
        
        # Maximumize
        for i in range(k):
            maximize(mean, cnt_k, norm_data, cov, w, i)

    res.append(mean)
    res.append(cov)
    res.append(log_list)
    return res, norm_data

   
def readData(filename):
    data = []
    file=open(filename,'r')
    for line in file:
        line = line.strip('\n')
        line = line.split(',')
        line[0] = float(line[0])
        line[1] = float(line[1])
        '''label for belong center'''
        line.append(0)
        '''for transformation'''
        line.append(math.pow(float(line[0]), 4) + math.pow(float(line[1]), 4))
        data.append(line)
    return data

if __name__ == '__main__':
    
    
    color = ['ro', 'gs', 'b<', 'yd', 'mp']
    K = [2, 3, 5]
    
    data = readData("blob.csv")
    k_mean(data)
    
    data = readData("circle.csv")
    k_mean(data)
    kernel_mean(data)
    
    data = np.genfromtxt('blob.csv', delimiter=',')
    best_param = []
    best_distribute = []
    best_probability = 0
    cluster = 3
    for cnt in range(5):
        params, probabilities = EM_Algorithm(data, cluster, 40)
        
        if best_probability < params[2][len(params[2]) - 1]:
            best_distribute = probabilities
            best_param = params
            best_probability = params[2][len(params[2]) - 1]
        
        plt.xlabel('iteration')
        plt.ylabel('log likelihood')
        plt.plot(params[2])
    plt.show()
    show(data, cluster, best_distribute, color)
    plt.show()
    
    
    print best_param[0]
    print best_param[1]
    pass