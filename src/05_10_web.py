# streamlit run 05_10_web.py
# 载入套件
import streamlit as st 
from skimage import io
from skimage.transform import resize
import numpy as np  
import tensorflow as tf

# 模型载入
model = tf.keras.models.load_model('./mnist_model.h5')

# 标题
st.title("上传图片(0~9)辨识")

# 上传图档
uploaded_file = st.file_uploader("上传图片(.png)", type="png")
if uploaded_file is not None:
    # 读取上传图档
    image1 = io.imread(uploaded_file, as_gray=True)
    # 缩小图形为(28, 28)
    image_resized = resize(image1, (28, 28), anti_aliasing=True)    
    # 插入第一维，代表笔数
    X1 = image_resized.reshape(1,28,28,1) 
    # 颜色反转
    X1 = np.abs(1-X1)

    # 预测
    predictions = np.argmax(model.predict(X1), axis=-1)[0]
    # 显示预测结果
    st.write(f'预测结果:{predictions}')
    # 显示上传图档
    st.image(image1)
