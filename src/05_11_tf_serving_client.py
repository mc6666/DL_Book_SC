import json
import numpy as np
import requests
from skimage import io
from skimage.transform import resize

uploaded_file='./myDigits/4.png'
image1 = io.imread(uploaded_file, as_gray=True)
# 缩小图形为(28, 28)
image_resized = resize(image1, (28, 28), anti_aliasing=True)    
# 插入第一维，代表笔数
X1 = image_resized.reshape(1,28,28,1) 
# 颜色反转
X1 = np.abs(1-X1)

# 将预测资料转为 Json 格式
data = json.dumps({
    "instances": X1.tolist()
    })
    
# 呼叫 TensorFlow Serving API    
headers = {"content-type": "application/json"}
json_response = requests.post(
    'http://localhost:8501/v1/models/MLP:predict',
    data=data, headers=headers)
    
# 解析预测结果    
predictions = np.array(json.loads(json_response.text)['predictions'])
print(np.argmax(predictions, axis=-1))
