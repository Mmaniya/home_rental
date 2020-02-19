def mean_std(my_list):
    """Takes a list as an input 
    and returns its mean and standard deviation"""
    
    my_list = np.array(my_list)
    mean = round(np.mean(my_list),2)
    std = round(np.std(my_list),2)
    return mean,std

def normalize(data):
    """Takes a file as an input. Creates a file with elements from
    input file normalized. Returns the mean and std of the features
    from the input file"""
    
    with open(data,'r') as file:
        l = []
        for i in range(47):
            content = file.readline()
            y = content.split(',')
            l.append(y)
        print(l)
        for i in range(len(l)):
            x = l[i][0]
            y = l[i][1]
            z = l[i][2]
            SIZE.append(int(x))
            ROOMS.append(int(y))
            PRICE.append(int(z))
    mean1, std1 = mean_std(SIZE)
    mean2, std2 = mean_std(ROOMS)
    with open('normalize.txt','w') as file:
        for i in range(47):
            s = (SIZE[i] - mean1)/std1
            r  = (ROOMS[i] - mean2)/std2
            p = PRICE[i]
            file.write(str(s))
            file.write(',')
            file.write(str(r))
            file.write(',')
            file.write(str(p))
            file.write('\n')
    return(mean1,std1,mean2,std2)
    
def gradient_descent(iterations,alpha,x,y):
    """Implements Vanilla Gradient Descent Algorithm. Takes as inputs no. of iterations
    allowed, learning rate as alpha, lists x and y from data file. Returns the vector w"""

    with open('normalize.txt','r') as file:
        a = []
        c = file.readlines()
        for i in c:
            a.append((i.split(',')))
        for i in a:
            for j in i:
                i[i.index(j)] = float(j)
        for i in a:
            i.insert(0,1)
            x.append(i[:3])
            y.append(i[-1])
        w = [0 for i in range(3)]
        w2 = [0 for i in range(3)] #to simultaneously update w0, w1, and w2
        stop = 0
        cost_func = 0
        while stop < iterations:
            for j in range(len(x[0])):
                der = 0
                for i in range(len(x)):
                    der = der + (np.dot(w,x[i])-y[i])*x[i][j]
                    w2[j]= (w[j] - ((alpha/len(x)*der)))
            w = list(w2)
            stop += 1
        for i in range(len(x)):
            cost_func += ((np.dot(w,x[i]) - y[i])**2)
        cost_func = cost_func/(2*47)
        print()
        print("(Vanilla GD) - Cost Function after respective iterations: ", cost_func)
        return w

def plotting(alpha):
    """Takes the learing rate as alpha and plots a graph showing no. of iterations
    on x-axis and cost function of y-axis (for w from gradient descent algorithm) for 
    iterations = 10,20,30,40,50,60,70,80"""
    
    cost = []
    iterations = [10,20,30,40,50,60,70,80]
    for i in iterations:
        start_time = time.time()
        x=[]
        y=[]
        w = gradient_descent(i,alpha,x,y)
        end_time = time.time()
        time1 = round((end_time - start_time),4)
        print("vector w for respective iteration:",w)
        print('Time taken when',i,'iterations: ', time1,'seconds')
        summation = 0
        for j in range(len(x)):
            summation = summation + (np.dot(w,x[j])- y[j])**2
        cost.append((1/(2*len(x)))*summation)
    plt.plot(iterations,cost,'-g',label = 'Cost Function against Iterations')
    plt.legend(loc = 'upper right')
    
def sto_gradient(iterations, alpha,x,y):
    """Implements Stochastic Gradient Descent. Takes as input no. of iterations, 
    learning rate as alpha, and two list to manage dat from the input file.
    Returns the vector w developed"""
    
    data = []
    for i in range(len(x)):
        temp=[]
        for j in range(3):
            temp.append(x[i][j])
        temp.append(y[i])
        data.append(temp)
    w = [0 for i in range(3)]
    w2 = [0 for i in range(3)]
    for r in range(iterations):
        start_time = time.time()
        for i in range(len(data)):
            der = 0
            for j in range((len(data[0])-1)):
                der = der + (np.dot(w,data[i][:3])-data[i][-1])*data[i][j]
                w2[j]= w[j] - (alpha)*der
                w = list(w2)
        end_time = time.time()
        time2 = round((end_time - start_time),6)
        np.random.shuffle(data) #shuffling the data
        cost = 0
        for j in range(len(data)):
            cost = cost + (np.dot(w,data[i][:3]) - data[i][-1])**2
        cost_func = ((cost)*(1/(2*len(data))))
        print("(Schotastic GD) - Cost after",r+1,"iterations:",cost_func)
        print("Vector w after respective iterations is:", w)
        print("(Schotastic GD) - Time taken after",r+1,"iterations:",time2,'seconds')
        print()
    return w

def predict(w,size,rooms):
    """Function to predict the price (hypothesis) using the inputs: vector w,
    size of the house (x1) and no. of rooms (x2). Returns that price"""
    
    size = (size-m1)/s1
    rooms = (rooms-m2)/s2
    y = w[0] + w[1]*size + w[2]*rooms
    return y

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    
    SIZE = []
    ROOMS = []
    PRICE = []
    m1,s1,m2,s2 = normalize('housing.txt') # m1,s1= mean of x1 data points and m2,s2 for x2
    x = []
    y = []
    w = gradient_descent(80,0.3,x,y)
    print()
#    print(w)
    print("Prediction of the house with size 1650 sq ft and 3 rooms using Gradient Descent is: ",predict(w,1650,3))
    print()
    w2 = sto_gradient(3,0.1,x,y)
    print()
#    print(w2)
    print("Prediction of the house with size 1650 sq ft and 3 rooms using Schotastic Gradient Descent is:",predict(w2,1650,3))
    plotting(0.3)
