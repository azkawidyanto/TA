import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import math

#data irish
iris = load_iris()
data_input = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
targets = data_input.iloc[:,-1]
biases = pd.DataFrame(index=np.arange(len(data_input)), columns=np.arange(1))
biases = biases.fillna(1)

#data is training data, udh dipisah sama target, isinya cuma bias dan feature aja
data = pd.concat([biases.reset_index(drop=True), data_input.iloc[:, :-1].reset_index(drop=True)], axis=1, sort=False)

global delta_w, w, error, sigma, learning_rate


#ini buat testing yg datanya sama kayak PR, ntar apus ae
datatest1 = data_input.head(3)
datatest2 = data_input[data_input['target'] == 1].head(3)
datates = datatest1.append(datatest2, ignore_index=True)
biasesd = pd.DataFrame(index=np.arange(len(datates)), columns=np.arange(1))
biasesd = biasesd.fillna(1)
datatesbeneran = pd.concat([biasesd.reset_index(drop=True), datates.iloc[:, :-1].reset_index(drop=True)], axis=1, sort=False)
testarget = datates.iloc[:,-1]

############################################

# initialize yg dipake
delta_w  = {}

error = 0
sigma = 0
learning_rate = 0.1
out = []
delta = []
weight = {}
print(data)

# struktur out dan delta :
#   cth : out = [[0,0],[0,0],[0]] 
#           out[i,j] berarti output di node ke-j(+1) di hidden layer ke-i(+1)
#           out[<indeks terakhir>,0] adalah output dari target

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(out):
    return out * (1 - out)

def error_out(target, out):
    return dsigmoid(out) * (target - out)

def error_hidden(weight, error, out, i, j):
    sigma = 0

    for k in range(len(delta[i+1])):
        temp = (i+1,j,k)
        sigma = sigma + weight[temp] * delta[i+1][k]
    return dsigmoid(out[i][j]) * sigma

def update_deltaw(delta, out, i, j, k):
    # print("buat dicek ", learning_rate, " ", delta[i+1][k]," ", out)
    return learning_rate * delta[i+1][k] * out

def create_output_delta(n_hidden) :
    global out, delta, delta_w

    out = [0 for i in range (n_hidden+1)]
    delta = [0 for i in range (n_hidden+1)]
    for i in range (n_hidden) :
        print("jumlah node di hidden layer ",i+1,": ", end="")
        node = int(input())
        out[i] = [0 for j in range (node+1)]
        out[i][node] = 1
        delta[i] = [0 for j in range (node)]
        
    out[n_hidden] = [0]
    delta[n_hidden] = [0]
    
# struktur weight :
#   cth : temp = (a, b, c)
#         weight[temp] berarti weight di layer ke a 
#         dari node ke-b di layer a ke node ke-c di layer a+1

def create_weight_array(data) :
    global out, weight
    
    columns = len(data.columns)
    for i in range (len(out)):
        if (i==0) :
            for j in range (columns) :
                for k in range (len(out[i])) :
                    temp = (i,j,k)
                    weight[temp] = 1
        else :
            for j in range (len(out[i-1])) :
                for k in range(len(out[i])) :
                    temp = (i,j,k)
                    weight[temp] = 1

def create_delta_weight(data):
    global out, delta_w
    
    columns = len(data.columns)
    for i in range (len(out)):
        if (i==0) :
            for j in range (columns) :
                for k in range (len(out[i])) :
                    temp = (i,j,k)
                    delta_w[temp] = 0
        else :
            for j in range (len(out[i-1])) :
                for k in range(len(out[i])) :
                    temp = (i,j,k)
                    delta_w[temp] = 0

def mini_batch(data, targets, batch_size, max_epoch):
    global w, delta_w, sigma, learning_rate, error, out, weight
    
    n_hidden = int(input("Jumlah Hidden Layer: "))
    create_output_delta(n_hidden)
    create_weight_array(data)
    create_delta_weight(data)
    columns = len(data.columns)
    epoch = 0
    done = False
    # done tu kalo epoch kelar
    while not done:
        
        error = 0

        #row num itu buat nomor row nya
        row_num = 0

        #it buat ngecek batches
        it = 1
        
        #ini nge iterate data set
        for row in data.iterrows():
            # error = 0
            # ========= MULAI FEED FORWARD ==============
            for i in range (len(out)-1) :
                for k in range (len(out[i])-1) :
                    net = 0
                    if(i == 0) :
                        j = 0
                        for attr in row[1] :
                            temp = (i,j,k)
                            net += attr * weight[temp]
                            j+=1
                    else :
                        for j in range(len(out[i-1])) :
                            temp = (i,j,k)
                            net += out[i-1][j] * weight[temp]
                    y = 1 / ( 1 + math.exp(-1*net))
                    out[i][k] = y

            i =  len(out)-1
            k = 0
            net = 0
            for j in range(len(out[i-1])):
                temp = (i, j, k)
                net += out[i-1][j] * weight[temp]
            y = 1 / (1 + math.exp(-1*net))
            out[i][k] = y
            # print("ini out")
            # print(out)

            # ========= SELESAI FEED FORWARD ==============
            error += ((targets[row_num] - out[len(out)-1][0])**2)/2
            # x = input("ctr+c aja habis ini belum dibenerin")

            # ========= MULAI BACKWARD PHASE ==============
            
            delta[len(delta)-1][0] = error_out(targets[row_num],out[len(out)-1][0])
            for layer in reversed(range(len(delta)-1)):
                for node in range(len(delta[layer])):
                    delta[layer][node] = error_hidden(weight,delta,out,layer,node)

            # print("ini delta")
            # print(delta)

            for layer in reversed(range(len(delta)-1)):
                for node in range(len(delta[layer])+1):
                    for sebelah in range(len(delta[layer+1])):
                        temp = (layer+1, node, sebelah)
                        # print("ini layer", layer+1, node, sebelah)
                        delta_w[temp] += update_deltaw(delta,out[layer][node],layer,node, sebelah)
            
            layer = 0
            node = 0
            for attr in row[1]:
                for sebelah in range(len(delta[layer])):
                    temp = (layer, node, sebelah)
                    # print("ini layer", layer, node, sebelah)
                    delta_w[temp] += update_deltaw(delta,attr,layer,node, sebelah)            
                node += 1
            
            # x = input("ctr+c aja habis ini belum dibenerin")

            if it == batch_size:
                for i in range (len(out)):
                    if (i==0) :
                        for j in range (columns) :
                            for k in range (len(out[i])) :
                                temp = (i,j,k)
                                weight[temp] += delta_w[temp]
                    else :
                        for j in range (len(out[i-1])) :
                            for k in range(len(out[i])) :
                                temp = (i,j,k)
                                weight[temp] += delta_w[temp]
                print("ini weight")
                print(weight)
                print("ini error")
                print(error)                 
                it = 0
                create_delta_weight(data)
                #tiap batch delta sama error di 0 in
                
            # print("deltaw")
            # print(delta_w)
            # print("w")
            # print(w)

            row_num += 1
            it += 1

        epoch += 1
        print("ini epochh yaa", epoch)
        if (epoch >= max_epoch or error < 0.1):
            print("mask done")
            done = True
    return       

mini_batch(data, targets, 3, 100)
# print(error)
# print(w)
# print(delta_w)

#n_hidden = int(input("Jumlah Hidden Layer: "))
#out = []
#create_output_array(n_hidden)
#print(out)
#
#create_weight_array(datatesbeneran)

