from PIL import Image
import numpy as np



def load_img(path):
    image = Image.open(path) # 加载图片
    image = image.resize((64, 64))      # 调整图片大小
    image_array = np.array(image) / 255.0      # 将图片转换为数组，并标准化像素值
    if image_array.shape[2] != 3:      # 确保图片是三通道的 RGB 格式

        print("图片不是RGB格式")
    else:
        # print(image_array.shape)
        a=image_array.shape[0]
        b=image_array.shape[1]
        c=image_array.shape[2]
        v=image_array.reshape(a*b*c,1)
        # print(v.shape)
        
        print("图片处理完成，准备好用于模型输入")




if __name__=="__main__":
    path=r"C:\Users\28121\Desktop\AI_data\dogs_VS_cat\train\cat.0.jpg"
    load_img(path)
