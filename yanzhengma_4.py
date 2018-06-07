#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 15:02:58 2018

@author: wwj
"""

#from keras.layers import Dense, Dropout, Activation, Flatten
#from keras.layers import Conv2D, MaxPooling2D
from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from keras.utils import plot_model
import pandas as pd
#from keras.utils import np_utils
#from PIL import Image
from keras.layers import Dropout,Convolution2D,Conv2D,Flatten,Dense,BatchNormalization,MaxPooling2D,merge
from keras.models import Input,Model
from keras.optimizers import SGD
from keras import optimizers
import string
#设置字符为数字字符
characters = string.digits#+string.ascii_uppercase
#设置产生图片的宽度，高度，验证码字数，字符集长度
width, height, n_len, n_class =120, 80,3, len(characters)
generator = ImageCaptcha(width=width, height=height)
#模型名字
model_name='test_'+str(n_len)+'.h5'
batch_sizes=10000
X = np.zeros((batch_sizes, height, width,3)).astype('float32')
#Y=np.zeros((batch_sizes,int(n_len*n_class)))
Y=[]
for i in range(batch_sizes):
    random_str = ''.join([random.choice(characters) for j in range(n_len)])
    X[i]=generator.generate_image(random_str)
    Y.append(random_str)
#编码码，将字符串格式转换为one-hot转换
def encode_label(label_str_list):
    label_hot=np.zeros((len(label_str_list),int(n_len*n_class)))
    for i in range(len(label_str_list)):
        for j in range(n_len):
            code_index=characters.index(label_str_list[i][j])
            label_hot[i][code_index+j*n_class]=1
    return label_hot
#解码，将one-hot转换为字符串格式
def decode_label(label_hot):
    label=[]
    for i in range(len(label_hot)):
        label_sub=label_hot[i].reshape(n_len,n_class)
        label_str=''
        label_str=''.join([characters[np.argmax(label_sub[j])] for j in range(n_len)])
        label.append(label_str)
    return label
#devide the one-hot
def devide_hot(label_hot):
    label_hot_1=label_hot[:,:n_class]
    label_hot_2=label_hot[:,n_class:n_class*2]
    label_hot_3=label_hot[:,n_class*2:n_class*3]
    
    return label_hot_1,label_hot_2,label_hot_3

#训练集
X_train=X[:int(batch_sizes*0.9)]
#训练集进行归一化
X_train_normalize=X_train/255.0
Y_train=Y[:int(batch_sizes*0.9)]
print(len(Y_train),Y_train[0])
#训练集的标签进行One-hot转换
Y_train_normalize=encode_label(Y_train)

X_test=X[int(batch_sizes*0.9):]
X_test_normalize=X_test/255.0
Y_test=Y[int(batch_sizes*0.9):]
Y_test_normalize=encode_label(Y_test)
print(Y_test[:20])#打印前20个标签

#定义网络结构
def net_structure(input_shape=X_train_normalize.shape[1:]): 
    input_tensor = Input(input_shape)
    x=input_tensor
    print('x:',type(x))
#    x=BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')(x)
    x =Conv2D(16,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform')(x) 
#    x=BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')(x)
    x =Conv2D(16,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform')(x) 
    x =MaxPooling2D(pool_size=(2,2))(x) 
    x =Dropout(0.5)(x)
    x =Conv2D(32,(3,2),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform')(x)  
    x =Dropout(0.5)(x)
    x =Conv2D(32,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform')(x)  
    x =MaxPooling2D(pool_size=(2,2))(x) 
    x =Dropout(0.5)(x)
    x =Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform')(x) 
#    x =Dropout(0.5)(x)
    x =Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform')(x)  
    x =MaxPooling2D(pool_size=(2,2))(x)
    x =Dropout(0.5)(x)
    x = Flatten()(x)
    fc1=Dense(n_class,activation='softmax',name='predict_1')(x)
    fc2=Dense(n_class,activation='softmax',name='predict_2')(x)
    fc3=Dense(n_class,activation='softmax',name='predict_3')(x)
#    x=merge([fc1,fc2],mode='concat',concat_axis=-1)
    model = Model(inputs=input_tensor, outputs=[fc1,fc2,fc3])
    #categorical_crossentropy：亦称作多类的对数损失，注意使用该目标函数时，需要将标签转化为形如(nb_samples, nb_classes)的二值序列
    model.compile(loss='categorical_crossentropy',optimizer='adadelta',
                metrics=['accuracy'])
    return model
#进行训练
def net_train(model,model_name=model_name):    
    Y_train_normalize_1,Y_train_normalize_2,Y_train_normalize_3=devide_hot(Y_train_normalize)
    train_history=model.fit(X_train_normalize, [Y_train_normalize_1,Y_train_normalize_2,Y_train_normalize_3], epochs=100, shuffle=True,batch_size=100,
        validation_split=0.2,verbose=1)
    model.save_weights(model_name)
    #model.save(model_name)
    return model,train_history
#将训练准确率和损失进行图表表示
def show_train_history(train_history,train_1,train_2,train_3,validation_1,validation_2,validation_3):
    plt.plot(train_history.history[validation_1])
    plt.plot(train_history.history[validation_2])
    plt.plot(train_history.history[validation_3])
    plt.plot(train_history.history[train_1])
    plt.plot(train_history.history[train_2])
    plt.plot(train_history.history[train_3])
    plt.title('train_history')
    plt.ylabel(train_1)
    plt.ylabel(train_2)
    plt.ylabel(train_3)
    plt.xlabel('Epoch')
    plt.legend(['train_1','train_2','train_3','validation_1','validation_2','validation_3'],loc='upper left')
    plt.savefig('/home/douqs/wwj/test_4')
#    plt.show()
#测试模型
def test_model(model, X_test, Y_test):
#    model.load_weights(model_name)
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('score',score)
    print('loss',score[0])
    print('Test number_1 score:', score[1])
    print('Test number_2 score:', score[2])
    print('Test number_3 score:', score[3])
    
    
    print('Test number_1 accuracy:', score[4])
    print('Test number_2 accuracy:', score[5])
    print('Test number_3 accuracy:', score[6])
    return score

def plot_imag(img):
    fig=plt.gcf()
    fig.set_size_inches(2,2)
    plt.imshow(img,camp='binary')
    plt.savefig('test')
#    plt.show()

def plot_images_labels_prediction(images,labels,prediction,idx,num=10):
    fig=plt.gcf()
    fig.set_size_inches(60/2,40/2)
    row=0;col=5
    if(num%5!=0):
        row=int(num/5)+1
    elif(num<5):
        row=1
    else:
        row=num/5
        
    if(num<5):
        col=num
#    if num>100: num=100
    for i in range(0,num):
        
        ax=plt.subplot(row,col,1+i)
        ax.imshow(images[idx],cmap='binary')
        title='label='+labels[idx]
        if len(prediction)>0:
            title+=",predict="+prediction[idx]
        ax.set_title(title,fontsize=14)
        ax.set_xticks([]);
        ax.set_yticks([])
        idx+=1
    plt.savefig('/home/douqs/wwj/test/test1')
#    plt.show()
#def plot_confusion_matrix(y_true, y_pred, labels):#建立混淆矩阵
#    import matplotlib.pyplot as plt
#    from sklearn.metrics import confusion_matrix
#    cmap = plt.cm.binary
#    cm = confusion_matrix(y_true, y_pred)
#    tick_marks = np.array(range(len(labels))) + 0.5
#    np.set_printoptions(precision=2)
#    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#    plt.figure(figsize=(10, 8), dpi=120)
#    ind_array = np.arange(len(labels))
#    x, y = np.meshgrid(ind_array, ind_array)
#    intFlag = 0 # 标记在图片中对文字是整数型还是浮点型
#    for x_val, y_val in zip(x.flatten(), y.flatten()):
#        #
#
#        if (intFlag):
#            c = cm[y_val][x_val]
#            plt.text(x_val, y_val, "%d" % (c,), color='red', fontsize=8, va='center', ha='center')
#
#        else:
#            c = cm_normalized[y_val][x_val]
#            if (c > 0.01):
#                #这里是绘制数字，可以对数字大小和颜色进行修改
#                plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
#            else:
#                plt.text(x_val, y_val, "%d" % (0,), color='red', fontsize=7, va='center', ha='center')
#    if(intFlag):
#        plt.imshow(cm, interpolation='nearest', cmap=cmap)
#    else:
#        plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
#    plt.gca().set_xticks(tick_marks, minor=True)
#    plt.gca().set_yticks(tick_marks, minor=True)
#    plt.gca().xaxis.set_ticks_position('none')
#    plt.gca().yaxis.set_ticks_position('none')
#    plt.grid(True, which='minor', linestyle='-')
#    plt.gcf().subplots_adjust(bottom=0.15)
#    plt.title('')
#    plt.colorbar()
#    xlocations = np.array(range(len(labels)))
#    plt.xticks(xlocations, labels, rotation=90)
#    plt.yticks(xlocations, labels)
#    plt.ylabel('Index of True Classes')
#    plt.xlabel('Index of Predict Classes')
#    plt.savefig('confusion_matrix.jpg', dpi=300)
#    plt.show()   
if __name__ == '__main__':
    model=net_structure()
    if os.path.exists(model_name):
        model.load_weights(model_name)
#    plot_model(model, to_file='model1.png',show_shapes=True)
    print(model.summary())#打印网络结构
    Y_test_normalize_1,Y_test_normalize_2,Y_test_normalize_3=devide_hot(Y_test_normalize)
    model,train_history=net_train(model,model_name)


    
    score=test_model(model, X_test_normalize, [Y_test_normalize_1,Y_test_normalize_2,Y_test_normalize_3])#测试集测试
    prediction_1,prediction_2,prediction_3=model.predict(X_test_normalize) #进行预测
    prediction=np.concatenate((prediction_1,prediction_2,prediction_3), axis = 1)
    prediction_str=decode_label(prediction) #转化为字符串类型
#    print(len(Y_test),len(prediction_str),type(Y_test),type(prediction_str),Y_test[:10],prediction_str[:10])
#    CrossTab=pd.crosstab(Y_test,prediction_str,rownames=['test_label'],colnames=['predict'])#建立混淆矩阵
    a = np.array(Y_test, dtype=object)
    b = np.array(prediction_str, dtype=object)
    CrossTab=pd.crosstab(a, b, rownames=['test_label'],colnames=['predict'])
    print(CrossTab) 
    df=pd.DataFrame({'label':Y_test,'prediction_str':prediction_str})
    df_error=df[(df.label!=df.prediction_str)]
    print(df_error)
    print(list(df_error.index))
    print(type(df_error))
#    i=0
#    for index_sub in list(df_error.index):
#        if(i<5):
#            plot_images_labels_prediction(X_test,Y_test,prediction_str,index_sub,1)
#        i=i+1
    CrossTab.to_csv('/home/douqs/wwj/test/CrossTab.csv', encoding = "utf-8")#通过扩展名指定文件存储的数据为csv格式
    plot_images_labels_prediction(X_test,Y_test,prediction_str,idx=10,num=1)#展示测试集中第10个图片
    plot_images_labels_prediction(X_test,Y_test,prediction_str,5,num=50)#举例50个展示
    show_train_history(train_history,'predict_1_acc','predict_2_acc','predict_3_acc','val_predict_1_acc','val_predict_2_acc','val_predict_3_acc')#绘制准确率与损失值变化图


##    

