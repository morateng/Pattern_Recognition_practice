import numpy as np


def calc_euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))


def calc_weight(x, kernel='flat'):
    if x <= 1:
        if kernel.lower() == 'flat':
            return 1
        elif kernel.lower() == 'gaussian':
            return np.exp(-1 * (x ** 2))
        else:
            raise Exception("'%s' is invalid kernel" % kernel)
    else:
        return 0
    
def mean_shift(X, bandwidth, n_iteration=20, epsilon=0.001):
    history = {}
    for i in range(len(X)):
        history[i] = []
    centroids = np.zeros_like(X)   

    for i in range(len(X)):
        centroid = X[i].copy()  # 초기 중심점(t_0) 설정 -> 각 datapoint를 초기 중심점으로 할당
        prev = centroid.copy()
        history[i].append(centroid.copy())
        
        t = 0
        while True:
            if t > n_iteration:
                break

            # 종료 조건 및 중심점 이동 수식 구현
            numerator = 0
            denominator = 0
            for sample in X:
                distance = calc_euclidean_distance(centroid, sample)
                weight = calc_weight(distance/bandwidth, 'flat')
                numerator += ((sample - centroid) * weight)
                denominator += weight
                
            if denominator == 0:
                shift = 0
            else:
                shift = numerator / denominator
            centroid += shift
            
            if calc_euclidean_distance(centroid, prev) < epsilon:
                break
            
            prev = centroid.copy()
            t += 1

            history[i].append(centroid.copy())
        
        centroids[i] = centroid.copy()

    return centroids, history
