from tkinter import *
from tkinter import filedialog

from PIL import ImageDraw, Image, ImageGrab
import numpy as np
from skimage import color
from skimage import io
import os
import io

from cnn_class import getData, trainModel, loadModel


class Paint(object):

    # 类别初始化函数
    def __init__(self):
        self.root = Tk()
        
        self.root.title('手写阿拉伯数字辨识')

        #defining Canvas
        self.c = Canvas(self.root, bg='white', width=280, height=280)
        
        self.image1 = Image.new('RGB', (280, 280), color = 'white')
        self.draw = ImageDraw.Draw(self.image1) 

        self.c.grid(row=1, columnspan=6)

        # 建立【辨识】按钮
        self.classify_button = Button(self.root, text='辨识', command=lambda:self.classify(self.c))
        self.classify_button.grid(row=0, column=0, columnspan=2, sticky='EWNS')

        # 建立【清画面】按钮
        self.clear = Button(self.root, text='清画面', command=self.clear)
        self.clear.grid(row=0, column=2, columnspan=2, sticky='EWNS')

        # 建立【存档】按钮
        self.savefile = Button(self.root, text='存档', command=self.savefile)
        self.savefile.grid(row=0, column=4, columnspan=2, sticky='EWNS')

        # 建立【预测】文字框
        self.prediction_text = Text(self.root, height=2, width=10)
        self.prediction_text.grid(row=2, column=4, columnspan=2)

        # self.model = self.loadModel()
        
        # 定义滑鼠事件处理函数
        self.setup()
        
        # 监听事件
        self.root.mainloop()

    # 滑鼠事件处理函数
    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = 15
        self.color = 'black'
        
        # 定义滑鼠事件处理函数，包括移动滑鼠及松开滑鼠按钮
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    # 移动滑鼠 处理函数
    def paint(self, event):
        paint_color = self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
            # 显示设定>100%，抓到的区域会变小
            # 画图同时写到记忆体，避免荧幕字型放大，造成抓到的画布区域不足
            self.draw.line((self.old_x, self.old_y, event.x, event.y), fill='black', width=self.line_width)

        self.old_x = event.x
        self.old_y = event.y

    # 松开滑鼠按钮 处理函数
    def reset(self, event):
        self.old_x, self.old_y = None, None

    # 【清画面】处理函数
    def clear(self):
        self.c.delete("all")
        self.image1 = Image.new('RGB', (280, 280), color = 'white')
        self.draw = ImageDraw.Draw(self.image1) 
        self.prediction_text.delete("1.0", END)

    # 【存档】处理函数
    def savefile(self):
        f = filedialog.asksaveasfilename( defaultextension=".png", filetypes = [("png file",".png")])
        if f is None: # asksaveasfile return `None` if dialog closed with "cancel".
            return
        #print(f)
        self.image1.save(f)

    # 【辨识】处理函数
    def classify(self, widget):
        # self.image1.save('原图.png')
        img = self.image1.resize((28, 28), ImageGrab.Image.ANTIALIAS).convert('L')
        # img.save('缩小.png')
        
        img = np.array(img)
        # Change pixels to work with our classifier
        img = (255 - img) / 255
        
        img2=Image.fromarray(img) 
        #img2.save('2.png')

        img = np.reshape(img, (1, 28, 28, 1))
        
        # Predict digit
        pred = model.predict([img])
        # Get index with highest probability
        pred = np.argmax(pred)
        #print(pred)
        self.prediction_text.delete("1.0", END)
        self.prediction_text.insert(END, pred)


if __name__ == '__main__':
    # 训练模型或载入既有的模型
    if(os.path.exists('mnist_model.h5')):
        print('load model ...')
        model = loadModel()
    else:
        print('train model ...')
        X_train, y_train, X_test, y_test = getData()
        model = trainModel(X_train, y_train, X_test, y_test)

    print(model.summary())
    
    # 显示视窗
    Paint()

