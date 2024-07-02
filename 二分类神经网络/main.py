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
    Y = np.empty((1, 0))  # 行数不变（每一张照片只有一个特征，是不是猫），列数可以变（逐个加载照片）
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



# 数学计算模块
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def optimize(w, b, X, Y,learning_rate):   #单次传播（也就是一次优化）
    m = X.shape[1]   #训练样本数

    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)
    cost = (-1/m) * np.sum(Y * np.log(A) + (1-Y) * np.log(1-A))  #计算本次所有的损失

    dZ = A - Y
    dw = (1/m) * np.dot(X, dZ.T)
    db = (1/m) * np.sum(dZ)

    w = w - learning_rate * dw
    b = b - learning_rate * db

    return w, b, dw, db, cost



# 核心控制模块
def main(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []

    for i in range(num_iterations):
        w, b ,dw, db, cost = optimize(w, b, X, Y,learning_rate)

        if i % 100 == 0:  #每训练100次输出一次
            costs.append(cost)

            if print_cost:
                print(f"Cost after iteration {i}: {cost}")

    return w, b, dw, db, costs




# 测试模块
def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)
    
    for i in range(A.shape[1]):
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0
    
    return Y_prediction








if __name__ == "__main__":
    train_path = r"C:\Users\28121\Desktop\deep_learning\deep-learning\二分类神经网络\train"
    test_path = r"C:\Users\28121\Desktop\deep_learning\deep-learning\二分类神经网络\test"

    X, Y, m = load_pictures(train_path)  #m是训练样本数，Y是标注集，X是训练集
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)


    # 初始化参数
    dim=12288  #单张图片的大小64*64*3
    w = np.zeros((dim, 1))
    b = 0

    final_w, final_b, final_dw, final_db, costs = main(w, b, X, Y, num_iterations=2000, learning_rate=0.005, print_cost=True)  #训练2k次，每次的步长为0.005

    print("Optimization finished.")




    X_test, Y_test, m_test = load_pictures(test_path)
    Y_prediction = predict(w, b, X_test)
    print("Accuracy:", np.mean(Y_prediction == Y_test))
    print("Optimization finished.")





