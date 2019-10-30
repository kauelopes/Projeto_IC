import numba
from numba import cuda
import numpy as np
import math
import timeit



from scipy.spatial import distance_matrix as dm

pontos = np.random.random([1000,2])

cuda.detect()
d_pontos = cuda.to_device(pontos)
d_distance_matrix = numba.cuda.device_array([d_pontos.shape[0],d_pontos.shape[0]])


@cuda.jit
def matrix_x_vezes_y(out):
    x,y = cuda.grid(2)
    for i in range(x,out.shape[0], cuda.gridDim.x):
        for j in range(y,out.shape[1],cuda.gridDim.y):
            out[i][j] = i*j




@cuda.jit
def adiciona_um(pontos, out):
    x,y = cuda.grid(2)
    for i in range(x,out.shape[0], cuda.gridDim.x):
        for j in range(y,out.shape[1],cuda.gridDim.y):
            out[i][j] = ((pontos[i][0]-pontos[j][0])**2 + (pontos[i][1] - pontos[j][1])**2)



matrix_x_vezes_y[32,32](d_distance_matrix)

print(d_distance_matrix.copy_to_host())



def dm_simples(pontos):
    start = timeit.timeit()
    resp = dm(pontos, pontos)
    end = timeit.timeit()
    print(end-start)

    return resp



def main():
    dm_simples(pontos)


