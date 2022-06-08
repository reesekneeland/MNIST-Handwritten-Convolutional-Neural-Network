import cv2
import numpy as np
import os
import time
import scipy.io as sio
import matplotlib.pyplot as plt
import math
import main_functions as main

#DONE
def get_mini_batch(im_train, label_train, batch_size):
    num_samples = len(im_train[0])
    num_batches = np.floor(num_samples/batch_size)
    random_perm = np.random.permutation(num_samples)
    
    oh_labels = np.zeros((10, num_samples))
    perm_im = np.zeros((196, num_samples))
    for i, n in enumerate(random_perm):
        oh_labels[label_train[0, n], i] = 1
        perm_im[:,i] = im_train[:,n]
    mini_batch_x = np.array(np.array_split(perm_im, num_samples/batch_size, axis = 1))
    mini_batch_y = np.array(np.array_split(oh_labels, num_samples/batch_size, axis = 1))
    return mini_batch_x, mini_batch_y

#DONE
def fc(x, w, b):
    y = (w @ x) + b
    return y

# DONE
def fc_backward(dl_dy, x, w, b, y):
    dl_dy = dl_dy.reshape((dl_dy.shape[0], 1))
    dl_dx = w.transpose() @ dl_dy
    dl_dw = dl_dy @ x.transpose()
    dl_db = dl_dy 
    return dl_dx, dl_dw, dl_db

#DONE
def loss_euclidean(y_tilde, y):
    dif = y_tilde - y
    l = math.pow(np.linalg.norm(dif, 2), 2)
    dl_dy = 2*dif
    return l, dl_dy

#DONE
def train_slp_linear(mini_batch_x, mini_batch_y):
    #set the learning and decay rates
    gamma = 0.0001
    lamda = 0.8
    L = []
    w = np.random.normal(0, 0.5, (10, 196))
    b = np.random.normal(0, 0.5, (10, 1))
    k=1
    for i in range(1, 5000):
        if(i % 500 == 0):
            gamma *=  lamda
        l=0
        dL_dw = np.zeros(w.shape)
        dL_db = np.zeros(b.shape)
        
        for j in range(mini_batch_x[k].shape[1]):
            cur_mini_x = mini_batch_x[k,:,j].reshape(196, 1)
            cur_mini_y = mini_batch_y[k,:,j].reshape(10, 1)
            #forward pass
            y_t = fc(cur_mini_x, w, b)

            loss, dl_dy = loss_euclidean(y_t, cur_mini_y)
            l += np.linalg.norm(loss)

            #backward pass
            dl_dx, new_dL_dw, new_dL_db = fc_backward(dl_dy, cur_mini_x, w, b, y_t)

            dL_dw = dL_dw + new_dL_dw
            dL_db = dL_db + new_dL_db
        w -= gamma * dL_dw
        b -= gamma * dL_db
        L.append(l/mini_batch_x[k].shape[1])
        k+=1
        if(k>=mini_batch_x.shape[0]):
            k=1
    plt.plot(list(range(4999)), L)
    plt.show()
    return w, b

#DONE
def loss_cross_entropy_softmax(x, y):
    y_softmax = np.exp(x)/np.sum(np.exp(x))
    l = -np.sum(y * np.log(y_softmax))
    dl_dy = y_softmax - y
    return l, dl_dy

def train_slp(mini_batch_x, mini_batch_y):
    #set the learning and decay rates
    gamma = 0.005
    lamda = 0.9
    L = []
    w = np.random.normal(0, 1, (10, 196))
    b = np.random.normal(0, 1, (10, 1))
    k=1
    for i in range(1, 5000):
        if(i % 1000 == 0):
            gamma *=  lamda
        l=0
        dL_dw = np.zeros(w.shape)
        dL_db = np.zeros(b.shape)
        
        for j in range(mini_batch_x[k].shape[1]):
            cur_mini_x = mini_batch_x[k,:,j].reshape(196, 1)
            cur_mini_y = mini_batch_y[k,:,j].reshape(10, 1)
            
            #forward pass
            y_t = fc(cur_mini_x, w, b)

            loss, dl_dy = loss_cross_entropy_softmax(y_t, cur_mini_y)
            l += np.linalg.norm(loss)

            #backward pass
            dl_dx, new_dL_dw, new_dL_db = fc_backward(dl_dy, cur_mini_x, w, b, y_t)

            dL_dw = dL_dw + new_dL_dw
            dL_db = dL_db + new_dL_db
        w -= gamma * dL_dw
        b -= gamma * dL_db
        L.append(l/mini_batch_x[k].shape[1])
        k+=1
        if(k>=mini_batch_x.shape[0]):
            k=1
    plt.plot(list(range(4999)), L)
    plt.show()
    return w, b

#DONE
def relu(x):
    x_flat = x.flatten()
    y = np.zeros(x_flat.shape)
    for i, n in enumerate(y):
        y[i] = max(0, x_flat[i])
    return y.reshape(x.shape)

#DONE
def relu_backward(dl_dy, x, y):
    dl_dy_flattened = dl_dy.flatten()
    x_flattened = x.flatten()
    dl_dx = np.zeros(dl_dy_flattened.shape)
    for i, n in enumerate(dl_dx):
        if x_flattened[i] > 0:
            dl_dx[i] = dl_dy_flattened[i]
        else:
            dl_dx[i] = 0
    dl_dx = dl_dx.reshape(dl_dy.shape)
    return dl_dx

#DONE, needs some cleanup
def train_mlp(mini_batch_x, mini_batch_y):
    #set the learning and decay rates
    gamma = 0.012
    lamda = 0.8
    L = []
    w1 = np.random.normal(0, 0.5, (30, 196))
    b1 = np.random.normal(0, 0.5, (30, 1))
    w2 = np.random.normal(0, 0.5, (10, 30))
    b2 = np.random.normal(0, 0.5, (10, 1))
    k=1
    for i in range(1, 5000):
        if(i % 1000 == 0):
            gamma *=  lamda
        l=0
        dL_dw1 = np.zeros(w1.shape)
        dL_db1 = np.zeros(b1.shape)
        dL_dw2 = np.zeros(w2.shape)
        dL_db2 = np.zeros(b2.shape)
        
        for j in range(mini_batch_x[k].shape[1]):
            cur_mini_x = mini_batch_x[k,:,j].reshape(196, 1)
            cur_mini_y = mini_batch_y[k,:,j].reshape(10, 1)
            #forward pass
            step1 = fc(cur_mini_x, w1, b1)
            step2 = relu(step1)
            y_t = fc(step2, w2, b2)
            
            loss, dl_dy = loss_cross_entropy_softmax(y_t, cur_mini_y)
            l += np.linalg.norm(loss)
            #backward pass
            dl_dx_step1, dl_dw2, dl_db2 = fc_backward(dl_dy, step2, w2, b2, y_t)
            dl_dx_step2 = relu_backward(dl_dx_step1, step1, step2)
            dl_dx_step3, dl_dw1, dl_db1 = fc_backward(dl_dx_step2, cur_mini_x, w1, b1, step1)
            
            dL_dw1 += dl_dw1
            dL_db1 += dl_db1
            dL_dw2 += dl_dw2
            dL_db2 += dl_db2
        w1 -= gamma * dL_dw1
        b1 -= gamma * dL_db1
        w2 -= gamma * dL_dw2
        b2 -= gamma * dL_db2
        L.append(l/mini_batch_x[k].shape[1])
        k+=1
        if(k>=mini_batch_x.shape[0]):
            k=1
    plt.plot(list(range(4999)), L)
    plt.show()
    return w1, b1, w2, b2

#DONE
def conv(x, w_conv, b_conv):
    y = np.zeros((x.shape[0], x.shape[1], w_conv.shape[3]))

    padded_x = np.pad(x,((1, 1),(1, 1), (0, 0)), mode='constant')

    for channel in range(w_conv.shape[3]):
        cfilt = w_conv[ :, :, :, channel]

        for x_pix in range(x.shape[0]):

            for y_pix in range(x.shape[1]):

                padded_im = padded_x[x_pix:x_pix+w_conv.shape[0], y_pix:y_pix+w_conv.shape[1], :]
                padded_im = padded_im.reshape(w_conv.shape[0], w_conv.shape[1])
                filtered_im = sum(cfilt.reshape(w_conv.shape[0], w_conv.shape[1]) * padded_im)
                convolved_im = sum(filtered_im) + b_conv[channel]
                y[x_pix, y_pix, channel] = convolved_im

    return y

#DONE
def conv_backward(dl_dy, x, w_conv, b_conv, y):
    padded_x = np.pad(x,((1, 1),(1, 1), (0, 0)), mode='constant')
    dl_dw = np.zeros(w_conv.shape)
    dl_db = np.zeros(b_conv.shape)

    for channel2 in range(w_conv.shape[3]):
        chanslice = dl_dy[:,:,channel2]
        dl_db[channel2] = np.sum(chanslice)
        for channel1 in range(w_conv.shape[2]):
            for x_pix in range(w_conv.shape[0]):
                for y_pix in range(w_conv.shape[1]):
                    padded_im = padded_x[x_pix:x_pix+x.shape[0], y_pix:y_pix+x.shape[1], channel1]
                    padded_im = padded_im.reshape(y.shape[0], y.shape[1])
                    dl_dw[x_pix, y_pix, channel1, channel2] = np.sum(chanslice * padded_im)

    return dl_dw, dl_db

#DONE
def pool2x2(x):
    height = int(x.shape[0]/2)
    width = int(x.shape[1]/2)
    y = np.zeros((height, width, x.shape[2]))
    for channel in range(x.shape[2]):
        for h in range(0, x.shape[0], 2):
            for w in range(0, x.shape[1], 2):
                y[int(h/2), int(w/2), channel] = np.max(x[h:h+2,w:w+2,channel])  
    return y

#DONE
def pool2x2_backward(dl_dy, x, y):
    dl_dx = np.zeros(x.shape)
    for channel in range(x.shape[2]):
        for h in range(0, x.shape[0], 2):
            for w in range(0, y.shape[1], 2):
                pos = np.argmax(x[h:h+2,w:w+2,channel])
                dl_dx[h + int(pos/2), h + pos%2, channel] = dl_dy[int(h/2), int(w/2), channel]

    return dl_dx

#DONE
def flattening(x):
    x_flat = x.reshape(-1, order='F')
    y = x_flat.reshape((x_flat.shape[0], 1))
    return y

#DONE
def flattening_backward(dl_dy, x, y):
    return dl_dy.reshape(x.shape, order = 'F')

#DONE
def train_cnn(mini_batch_x, mini_batch_y):
    #set the learning and decay rates
    gamma = 0.005
    lamda = 0.95
    
    k=0
    L = []
    w_conv = np.random.normal(0, 0.25, size=(3,3,1,3))
    b_conv = np.random.normal(0, 0.25, size=(3))
    w_fc = np.random.normal(0, 0.25, size=(10,147))
    b_fc = np.random.normal(0, 0.25, size=(10, 1))
    
    for i in range(9000):
        print("Iter #"+str(i))
        if i % 250 == 0:
            gamma *= lamda
        
        dL_dw_conv = np.zeros(w_conv.shape)
        dL_db_conv = np.zeros(b_conv.shape)
        dL_dw_fc = np.zeros(w_fc.shape)
        dL_db_fc = np.zeros(b_fc.shape)
        l=0
        
        for j in range(mini_batch_x[k].shape[1]):
            cur_mini_x = mini_batch_x[k,:,j].reshape((14, 14, 1), order='F')
            cur_mini_y = mini_batch_y[k,:,j]

            #forward pass
            conv_x = conv(cur_mini_x, w_conv, b_conv)
            relu_x = relu(conv_x)
            pool = pool2x2(relu_x)
            flat = flattening(pool)
            y_t = fc(flat, w_fc, b_fc)
            
            loss, dl_dy = loss_cross_entropy_softmax(y_t.reshape(-1), cur_mini_y)
            l += np.linalg.norm(loss)
            #backward pass
            dl_dy, dl_dw2, dl_db2 = fc_backward(dl_dy, flat, w_fc, b_fc, y_t)
            dl_dy = flattening_backward(dl_dy, pool, flat)    
            dl_dy = pool2x2_backward(dl_dy, relu_x, pool)    
            dl_dy = relu_backward(dl_dy, conv_x, relu_x) 
            dl_dw_conv, dl_db_conv = conv_backward(dl_dy, cur_mini_x, w_conv, b_conv, conv_x) 

            dL_dw_conv += dl_dw_conv
            dL_db_conv += dl_db_conv
            dL_dw_fc += dl_dw2.reshape(dL_dw_fc.shape)
            dL_db_fc  += dl_db2.reshape(dL_db_fc.shape)
        print("Loss " + str(L/mini_batch_x[k].shape[1]))
        L.append(l/mini_batch_x[k].shape[1])
        k += 1
        if(k>=mini_batch_x.shape[0]):
            k=0
        w_conv -= gamma * dL_dw_conv
        b_conv -= gamma * dL_db_conv         
        w_fc -= gamma * dL_dw_fc.reshape(w_fc.shape)
        b_fc -= gamma * dL_db_fc.reshape(b_fc.shape) 
    plt.plot(list(range(9000)), L)
    plt.show()
    return w_conv, b_conv, w_fc, b_fc

if __name__ == '__main__':
    main.main_slp_linear()
    main.main_slp()
    main.main_mlp()
    main.main_cnn()