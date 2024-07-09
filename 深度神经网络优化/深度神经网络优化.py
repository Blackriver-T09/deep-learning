from PIL import Image
import numpy as np
import os



# 数据加载模块
def img2array(file_path,size):
    try: 
        image = Image.open(file_path)  
        image = image.resize(size)  
        image_array = np.array(image) / 255.0  # 转化成array，并且除以 255 进行数据标准化方法。
        # 神经网络训练时，较小的数值范围（如 0 到 1）可以帮助模型更快地收敛，因为大的数值范围或极端值可能会导致训练过程中的数值不稳定。
        # 同时因为sigmoid函数在z很大的时候会变得很平缓，所以小一点会训练快一点
        if image_array.shape[2] != 3:  # 确认是RGB格式
            print("图片不是RGB格式")
            return None
        else:
            image_array = image_array.reshape(-1, 1)  # -1指自动计算这个维度应该有多少元素。1制定了只有一列
            print(f"{file_path}处理完成，准备好用于模型输入")
            return image_array
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def load_pictures(folder_path,size):
    X = np.empty((size[0]*size[1]*3, 0))  # 行数不变（每一张照片都是64**2*3个元素），列数可以变（逐个加载照片）
    Y = np.empty((1, 0))  # 行数不变（每一张照片只有一个结果，是不是猫），列数可以变（逐个加载照片）
    file_list = [file for file in os.scandir(folder_path) if file.is_file()]
    m=len(file_list)

    for file in file_list:
        file_path = os.path.join(folder_path, file.name)
        img_array = img2array(file_path,size)

        if img_array is not None:
            X = np.hstack((X, img_array)) 
            if 'cat' in file.name:
                Y = np.hstack((Y, np.array([[1]])))  
            else:
                Y = np.hstack((Y, np.array([[0]])))  

    print('所有图片处理完成')
    print('————————————————————————————————————————————————————————————————————————————————————————')
    return X, Y,m


# 数学函数模块
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_d(z):
    return sigmoid(z)*(1-sigmoid(z))

def tanh(z):
    return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))

def tanh_d(z):
    return 1-tanh(z)**2

def ReLU(z):
    return np.maximum(0,z)  #注意不要使用内置的max！

def ReLU_d(Z):  #注意不要用if else判断语句，不然是不能进行批量操作的
    return np.where(Z > 0, 1, 0)





# 随机初始化模块

def initialize_parameters_deep(layer_dims):
    parameters = {}
    number_of_layers = len(layer_dims)
    for l in range(1, number_of_layers):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2. / layer_dims[l-1])  #HE初始化
#       parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01   

        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


# 神经网络核心代码
def compute_cost(A, Y):
    m = Y.shape[1]  # m是样本数
    A = np.clip(A, 1e-8, 1 - 1e-8)  # 防止log(0)
    cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / m
    cost = np.squeeze(cost)  # 确保成本是标量
    return cost





def forward_propagation(A_prev, W, b, activation_func, keep_prob):
    Z = np.dot(W, A_prev) + b
    A = activation_func(Z)

    D = np.random.rand(A.shape[0], A.shape[1]) < keep_prob  
    A *= D 
    A /= keep_prob 
    return A, Z, D


    if keep_prob < 1:
        D = np.random.rand(A.shape[0], A.shape[1]) < keep_prob  # 每次调用时都生成新的Dropout掩码
        A *= D 
        A /= keep_prob 
        return A, Z, D
    return A, Z, None



def forward_propagation_complete(X,Y, parameters,chosen_func, Z_cache,A_cache,D_cache, keep_probs): #管理所有向前传播细节的函数

    A = X
    L = len(parameters) // 2   #总层数 是 W和b的总数的一般
    A_cache['A0']=X

    for l in range(1, L+1):
        A_prev = A
        func=chosen_func[l]
        W=parameters['W' + str(l)]
        b=parameters['b' + str(l)]


        keep_prob = keep_probs[l] 
        A, Z, D = forward_propagation(A_prev, W, b, func, keep_prob)

        A_cache['A'+str(l)]=A
        Z_cache['Z'+str(l)]=Z
        D_cache['D'+str(l)]=D
    
    cost=compute_cost(A, Y)
    
    return A_cache,Z_cache,D_cache,cost






def backward_propagation(dA, Z, A_prev, W, activation_func_d,m, D, keep_prob, lambda_):
    if D is not None:  #对于反向传播中的对应的节点也需要进行对应dropout处理
        dA *= D  
        dA /= keep_prob 
    dZ = dA * activation_func_d(Z)
    # dW = np.dot(dZ, A_prev.T) / m 
    dW = (np.dot(dZ, A_prev.T) + (lambda_ / m) * W) / m   #启用L2正则化

    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)
    return dW, db, dA_prev


def backward_propagation_complete(X, Y, parameters, chosen_func_d, Z_cache, A_cache,D_cache,   keep_probs, dW_cache, db_cache, lambda_):
    m = X.shape[1]

    L = len(parameters) // 2  # 总层数是W和b的总数的一半
    AL = A_cache.get('A' + str(L))
    dA = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    for l in range(L, 0, -1):
        Z = Z_cache.get('Z' + str(l))
        D = D_cache.get('D' + str(l))
        prev_A = A_cache.get('A' + str(l-1))
        W = parameters['W' + str(l)]
        activation_func_d = chosen_func_d[l]
        keep_prob = keep_probs[l] 
        
        dW, db, dA = backward_propagation(dA, Z, prev_A, W, activation_func_d, m, D, keep_prob, lambda_)
                                          
        dW_cache['dW' + str(l)] = dW    
        db_cache['db' + str(l)] = db

    return dW_cache, db_cache



def update_parameters(parameters, dW_cache, db_cache, learning_rate):
    L = len(parameters) // 2
    for l in range(1, L+1):
        parameters['W' + str(l)] -= learning_rate * dW_cache['dW' + str(l)]
        parameters['b' + str(l)] -= learning_rate * db_cache['db' + str(l)]
    return parameters



#核心控制模块
def train(X, Y, num_iterations, learning_rate ,parameters,chosen_func,chosen_func_d,lambda_, dropout_keep_prob):
    A_cache={}
    Z_cache={}
    D_cache={}
    dW_cache={}
    db_cache={}
    

    costs=[]

    L = len(parameters) // 2   #总层数 是 W和b的总数的一般
    
    for i in range(num_iterations):
        A_cache,Z_cache,D_cache,cost=forward_propagation_complete(X,Y, parameters,chosen_func,   Z_cache,A_cache,D_cache, dropout_keep_prob)
        dW_cache,db_cache=backward_propagation_complete(          X,Y, parameters,chosen_func_d, Z_cache,A_cache,D_cache, dropout_keep_prob,dW_cache,db_cache,lambda_)
        parameters=update_parameters(parameters,dW_cache,db_cache,learning_rate)
        
        if i % 100 == 0:  #每训练100次输出一次
            costs.append(cost)
            print(f"Cost after iteration {i}: {cost}")


    return parameters, costs



# 测试模块

def test(X_test, parameters,chosen_func,m_test): #管理所有向前传播细节的函数
    Y_prediction = np.zeros((1, m_test))

    A = X_test
    L = len(parameters) // 2   #总层数 是 W和b的总数的一般

    for l in range(1, L+1):
        A_prev = A
        activation_func=chosen_func[l]
        W=parameters['W' + str(l)]
        b=parameters['b' + str(l)]


        Z = np.dot(W, A_prev) + b
        A = activation_func(Z)

    for i in range(A.shape[1]):
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0  
        # 把Y_predict中大于0.5的改成1，否则改成0，用于表示每一个最终预测结果
    
    # 这里的for循环可以使用下面这一行代替
    # Y_prediction = (A2 > 0.5).astype(int)

    return Y_prediction





if __name__ == "__main__":
    # train_path = r"C:\Users\28121\Desktop\deep_learning\deep-learning\浅层神经网络\train"
    # test_path = r"C:\Users\28121\Desktop\deep_learning\deep-learning\浅层神经网络\test"
    train_path = r".\train_heavy"
    test_path = r".\test"


    func_list={
        1:sigmoid,
        2:tanh,
        3:ReLU,

        0:None
    }
    func_d_list={
        1:sigmoid_d,
        2:tanh_d,
        3:ReLU_d,

        0:None
    }


    # 超参数
    size=(64,64)     #图片压缩后的大小
    dim=size[0]*size[1]*3  #图片的输入形状 64*64*3
    layer_dims = [dim, 25, 15, 10,  5,  1]  # 本参数可以任意修改隐藏层数和每个层的节点数
    func_choose= [0,   3,  3,  3,   3,  1]       # 本层可以设置每层的激活函数，除了第一层没有激活函数是None
    learning_rate= 0.01
    num_iterations=2000
    lambda_=0.0001      #是否启用L2正则化,设置为0关闭L2正则化
    dropout_keep_prob=[1, 0.7, 0.8, 0.8, 0.9, 1 ]   #dropout机制：表示各层保留概率，全部为1则关闭Dropout机制，首层和末层必须是1




    X, Y, m = load_pictures(train_path,size)  #m是训练样本数，Y是标注集，X是训练集
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    # 获得初始化参数 
    parameters=initialize_parameters_deep(layer_dims)  
    chosen_func=[func_list.get(i) for i in func_choose] 
    chosen_func_d=[func_d_list.get(i) for i in func_choose] 
    print(chosen_func)
    print(chosen_func_d)


    parameters, costs = train(X, Y, num_iterations, learning_rate ,parameters,chosen_func,chosen_func_d, lambda_, dropout_keep_prob)  

    print('————————————————————————————————————————————————————————————————————————————————————————————————————')

    #测试准确度
    # 册数数据集中，全是猫的图片，用来检测二分类准确性 
    X_test, Y_test, m_test = load_pictures(test_path,size)
    Y_prediction = test(X_test, parameters,chosen_func,m_test)
    print(f"Accuracy: {np.mean(Y_prediction == Y_test)*100}%")
    # 逐个元素进行比较是否相同，结果是一个布尔值数组。布尔值 True 可以被当作 1 处理，False 被当作 0。
    print("Test finished.")