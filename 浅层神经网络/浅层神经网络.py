from PIL import Image
import numpy as np
import os



# 数据加载模块
def img2array(file_path):
    try: 
        image = Image.open(file_path)  
        image = image.resize((64, 64))  
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

def load_pictures(folder_path):
    X = np.empty((12288, 0))  # 行数不变（每一张照片都是64**2*3个元素），列数可以变（逐个加载照片）
    Y = np.empty((1, 0))  # 行数不变（每一张照片只有一个结果，是不是猫），列数可以变（逐个加载照片）
    file_list = [file for file in os.scandir(folder_path) if file.is_file()]
    m=len(file_list)

    for file in file_list:
        file_path = os.path.join(folder_path, file.name)
        img_array = img2array(file_path)

        if img_array is not None:
            X = np.hstack((X, img_array)) 
            if 'cat' in file.name:
                Y = np.hstack((Y, np.array([[1]])))  
            else:
                Y = np.hstack((Y, np.array([[0]])))  

    return X, Y,m




# 随机初始化模块
def initialize_parameters(n_x, n_h, n_y):
    # np.random.seed(2)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    

    return W1,b1,W2,b2



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




# 神经网络核心代码

def compute_cost(A2, Y):
    m = Y.shape[1]   #m是样本数
    cost = -np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2)) / m
    cost = np.squeeze(cost)
    return cost

def forward_propagation(X, W1,b1,W2,b2):
    Z1 = np.dot(W1, X) + b1
    A1 = ReLU(Z1)  # ReLU activation
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)  # Sigmoid activation
    
    return A2, Z2, A1,Z1


def backward_propagation(W1, W2, A2, Z2, A1, Z1, X, Y):
    m = X.shape[1]


    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = np.dot(W2.T, dZ2) * (A1 > 0)  # Derivative of ReLU
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    
    return dW1,db1,dW2,db2

def update_parameters(W1,b1,W2,b2,    dW1,db1,dW2,db2,    learning_rate=0.01):
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    
    return W1,b1,W2,b2




#核心控制模块
def train(W1,b1,W2,b2, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []

    for i in range(num_iterations):
        A2, Z2, A1,Z1=forward_propagation(X, W1,b1,W2,b2)
        cost=compute_cost(A2, Y)

        dW1,db1,dW2,db2=backward_propagation(W1, W2, A2, Z2, A1, Z1, X, Y)
        W1,b1,W2,b2=update_parameters(W1,b1,W2,b2,    dW1,db1,dW2,db2,    learning_rate)


        if i % 100 == 0:  #每训练100次输出一次
            costs.append(cost)

            print(f"Cost after iteration {i}: {cost}")

    return W1,b1,W2,b2, costs




# 测试模块
def test(final_W1,final_b1,final_W2,final_b2,X_test,m_test):
    Y_prediction = np.zeros((1, m_test))

    Z1 = np.dot(final_W1, X_test) + final_b1
    A1 = ReLU(Z1)  # ReLU activation
    Z2 = np.dot(final_W2, A1) + final_b2
    A2 = sigmoid(Z2)  # Sigmoid activation


    
    for i in range(A2.shape[1]):
        Y_prediction[0, i] = 1 if A2[0, i] > 0.5 else 0  
        # 把Y_predict中大于0.5的改成1，否则改成0，用于表示每一个最终预测结果
    
    # 这里的for循环可以使用下面这一行代替
    # Y_prediction = (A2 > 0.5).astype(int)

    return Y_prediction






if __name__ == "__main__":
    train_path = r"C:\Users\28121\Desktop\deep_learning\deep-learning\浅层神经网络\train"
    test_path = r"C:\Users\28121\Desktop\deep_learning\deep-learning\浅层神经网络\test"

    X, Y, m = load_pictures(train_path)  #m是训练样本数，Y是标注集，X是训练集
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    # 超参数
    dim=12288  #单张图片的大小64*64*3
    n_x=dim    #单张图片的样本数
    n_h=9      #隐藏层单元个数
    n_y=1      #输出层单元个数
    num_iterations=2000
    learning_rate=0.005

    W1,b1,W2,b2=initialize_parameters(n_x, n_h, n_y)
    print(f'orinial paramiters:')
    print(f'W1:{W1}')
    print(f'b1:{b1}')
    print(f'W2:{W2}')
    print(f'b2:{b2}')


    final_W1,final_b1,final_W2,final_b2,costs = train(W1,b1,W2,b2, X, Y, num_iterations, learning_rate)  #训练2k次，每次的步长为0.005

    print("Optimization finished.")
    print(f'final paramiters:')
    print(f'W1:{final_W1}')
    print(f'b1:{final_b1}')
    print(f'W2:{final_W2}')
    print(f'b2:{final_b2}')

    print('————————————————————————————————————————————————————————————————————————————————————————————————————')

    #测试准确度
    # 册数数据集中，全是猫的图片，用来检测二分类准确性 
    X_test, Y_test, m_test = load_pictures(test_path)
    Y_prediction = test(final_W1,final_b1,final_W2,final_b2,X_test,m_test)
    print(f"Accuracy: {np.mean(Y_prediction == Y_test)*100}%")
    # 逐个元素进行比较是否相同，结果是一个布尔值数组。布尔值 True 可以被当作 1 处理，False 被当作 0。
    print("Test finished.")