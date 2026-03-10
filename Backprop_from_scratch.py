# Simple Neural Network From Scratch
# Architecture: 3 → 6 → 4 → 1
# Activation: Sigmoid
# Loss: Mean Squared Error
# Training: Gradient Descent + Backpropagation

import math

def sigmoid(paraa):  # the sigmoid function
    sig=1/(1+math.exp(-paraa))
    return sig

X = [                       # some random input of x
    [0.1, 0.2, 0.1],
    [0.5, 0.4, 0.2],
    [0.9, 0.1, 0.3],
    [0.3, 0.8, 0.2],
    [0.6, 0.6, 0.6],
    [0.2, 0.1, 0.2],
    [0.7, 0.5, 0.4],
    [0.05, 0.05, 0.1]
]

Y = [                    # random input for y
    0,
    1,
    1,
    1,
    1,
    0,
    1,
    0
]

w1 = [                              # initial weights for every neuron
    [1,1,1],
    [1,1,1],
    [1,1,1],
    [1,1,1],
    [1,1,1],
    [1,1,1]
]
w2 = [                                              
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1]
]

w3 = [
    [1, 1, 1, 1]
]

b1 = [1, 1, 1, 1, 1, 1]
b2 = [1, 1, 1, 1]
b3 = [1]


epoch =15000                 #initializing the epochs
for epochh in range(epoch):
    totalloss=0
    for sample in range(len(X)):
        empty_z1=[]       #output of first layer
        empty_z2=[]       #output of second layer
        empty_z3=[]       #output of third layer
        empty_a1=[]       #sigmoid to z1
        empty_a2=[]       #sigmoid to z2
        empty_a3=[]       #sigmoid to z3
        L=X[sample]       # set of inputs of x
        O=Y[sample]       # set of inputs of y
        for i in range(len(w1)):       #stores z1 and a1 by doing operations on set of inputs
            summ=0
            for j in range(len(w1[i])):
                summ+=w1[i][j]*L[j]
            summ+=b1[i]

            empty_z1.append(summ)
            empty_a1.append(sigmoid(summ))


        for i in range(len(w2)):      #stores z2 and a2 by doing operations on values of a1
            summ=0
            for j in range(len(empty_a1)):
                summ+=w2[i][j]*empty_a1[j]
            summ+=b2[i]
            empty_z2.append(summ)
            empty_a2.append(sigmoid(summ))

        for i in range(len(w3)):       #stores z3 and a3 by doing operations on values of a2
            summ=0
            for j in range(len(empty_a2)):
                summ+=w3[i][j]*empty_a2[j]
            summ+=b3[i]
            empty_z3.append(summ)
            empty_a3.append(sigmoid(summ))

        y_cap=empty_a3[0]              #calculates the loss function
        loss=0.5*(y_cap-O)**2
        totalloss+=loss



        a3=empty_a3[0]                 
        del3=a3*(1-a3)*(a3-Y[sample])        #calculating the the set of derivatives for 3rd layer
        del2=[]
        for j in range(len(empty_a2)):       #calculating the the set of derivatives for 2nd layer
            val = del3 * w3[0][j] * empty_a2[j] * (1 - empty_a2[j])
            del2.append(val)
        
        del1=[]
        for j in range(len(empty_a1)):  # 6 neurons     #calculating the the set of derivatives for 1st layer
            summ = 0
            for k in range(len(del2)):  # 4 neurons
                summ += del2[k] * w2[k][j]

            val = summ * empty_a1[j] * (1 - empty_a1[j])
            del1.append(val)

        for j in range(len(w3[0])):                  #updating the values of weights in 3rd layer
            w3[0][j] -= 0.1 * del3 * empty_a2[j]

        b3[0] -= 0.1 * del3          #updating the values of biases in 3rd layer

        for i in range(len(w2)):
            for j in range(len(w2[i])):
                w2[i][j] -= 0.1 * del2[i] * empty_a1[j]       #updating the values of weights in 2nd layer

        for i in range(len(b2)):
            b2[i] -= 0.1 * del2[i]            #updating the values of biases in 2nd layer

        for i in range(len(w1)):
            for j in range(len(w1[i])):
                w1[i][j] -= 0.1 * del1[i] * L[j]          ##updating the values of weights in 1st layer

        for i in range(len(b1)):
            b1[i] -= 0.1 * del1[i]            #updating the values of biases in 1st layer

    print("Epoch", epochh, "Loss:", totalloss)

    