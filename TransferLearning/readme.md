@[TOC](Transfer Learning 迁移学习（实例+代码）--站在巨人的肩膀上)

# 简介

迁移学习，根据其字面意思便可以理解为将从一种场景中学习到知识迁移到另一种场景中来，这样就可以使得我们不需要从头开始学习一种新的场景。举些简单的例子：当我们学会起自行车的时候，我们再学骑摩托车就没有那么难了；当学会了打羽毛球，再学习打网球也很简单了。这是因为我们通常都是根据已有的经验来处理类似的问题，人的大脑是具备这种迁移学习的能力的，迁移学习可以在很多时候简化我们所处理的问题。

在深度学习领域，我们也面临着很多问题可以用迁移学习的方法来解决。例如，当我们的计算资源有限的时候，我们可以使用别人训练好的模型来微调来解决自己的问题。或者当我们的数据量比较少的时候，这个时候可以将一些在大数据集上训练好的模型来迁移到小数据集上，来缓解数据量不足的问题。



最后附一个[Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)，是由caffe library创建，用来提供给大家分享一些训练好的模型。

# 迁移学习的应用场景
迁移学习的应用场景主要包括以下三个方面：
1. ConvNet as fixed feature extractor：将训练好的模型当作一个特征提取器，例如我们可以将在imageNet上训练好的一个分类模型迁移到一个新的分类问题上，只需要去掉模型的全连接层，剩下的网络便可以作为一个特征提取器，然后提取的特征再经过新的全连接层进行新问题的分类。
2. Fine-tuning the ConvNet：微调训练好的模型。这种方法则是利用新的数据集对原始的网络模型进行重新训练，通过反向传播改变其中某些层或全部层的权重。最常见的是保持前几层的网络的权重不变，而重新训练后几层的权重。这是因为网络的前几层提取的是一些很普遍的低层特征，例如形状、方向、颜色等，然后网络的后几层则是会提取一些对于特定问题的特定特征，所以在进行迁移学习的时候可以保留前面几层的权重，而重新训练后面几层。
3. Pretrained models：把训练好的模型当作一个预训练模型。在一些大数据集上训练模型可能是很花费时间的一个过程，因此有人将一些训练好的模型的checkpoint分享出来，来为别人提供方便。


# 如何进行迁移学习
之前所叙述的应用场景也相当于迁移学习的三种方式。在实践中，我们需要考虑和选择使用哪种方式来进行迁移学习。一般主要从以下两个方面来考虑：
1. 当前数据集的大小
2. 当前数据集和原数据集的相似程度

基于上述两个考虑因素，我们将可能面对的情况分为以下四种：
1. **当前数据集较小但却和原数据集相似度较高**。因为数据集较小，所以不建议使用fine-tune的方式。此外，由于数据集的相似度比较高，那么可以使用 ConvNet as fixed feature extractor 的方式来进行迁移，即保留原始模型的特征提取部分，并重新用数据训练一个分类器来替换掉原始的上层的分类器。
2.  **当前数据集较大且和原数据集相似度较高**。对于这种情况，由于数据量较大，因此不会造成过拟合，所以推荐使用fine-tune的方式来更新模型中的参数。
3. **当前数据集较小且和原数据集相似度较低**。因为当前数据集比较小，所以用fine-tune重新调整整个模型的参数的方法不太可行。因此推荐使用训练一个分类器的方法。但是由于模型中较高的层往往会提取一些和特定数据集相关的特征，而考虑到新数据和原数据相似度较低，因此应该去掉之前模型中较高的层，只保留比较底层的网络结构。将这些底层的层当作一个特征提取器，然后根据提取的特征重新训练一个新的分类器。
4. **当前数据集较大但却和原数据集相似度较低**。因为数据集较大，所以可以直接训练一个新的网络。但是相比于随机初始化的参数，用其他训练好的模型的参数进行初始化会更利于模型的训练。因此，推荐使用fine-tune对整个模型进行参数更新。


# 需要注意的点

* 在使用与训练模型，我们会受网络的结构所牵制，我们不能从随便从网络结构中去掉卷积层或pooling层。
* 在进行fine-tune时，我们使用的学习率应该尽量小一些。因为训练好的模型的权重已经比较平滑，我们不希望它太快的扭曲。
* 此外，图片的大小应该保持一致，包括宽高，还有通道数。

# 实例和代码

假设现有一个训练好的分类模型存放在‘model.hdf5’文件中，现在我们需要将这个模型迁移到一个新的二分类问题，我们将分别使用两种方法来进行迁移。

## 方法一：ConvNet as fixed feature extractor
这种方法是将原始模型当作一个固定的特征提取器，然后再在顶层训练一个分类器。这种方法主要包括三步：

1. 获取basemodel，即从原模型中截取出所需要的层，详见*get_basemodel*函数。在该实例中我们提取从*input layer*到*flatten layer*中的所有层。
2. 加入新的分类器，为了用于新的分类问题，我们需要在basemodel中加入新的层来进行分类，详见*get_model*函数。我们加入了两个全连接层，其中第二个全连接层我们使用*softmax*进行分类。
3. 训练新的模型，详见*main*函数，需要注意的是我们只训练新加入的层，而basemodel中的层不需要重新训练，我们使用如下语句来达到这一目的：
```python
for layer in base_model.layers:
        layer.trainable = False
```
**代码：**
```python
from __future__ import division
import numpy as np
from keras.models import load_model, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense
import utils
from keras.optimizers import SGD

#获取basemodel，加载训练好的模型，然后截取所需的层
def get_basemodel():
    model_path = 'model.hdf5'
    mvgg_model = load_model(model_path)
    base_model = Model(inputs=mvgg_model.input, outputs=mvgg_model.get_layer('flatten_1').output)
    return base_model

#在basemodel后再连接两个全连接层，第二个全连接层用于分类
def get_model(base_model):
    x = base_model.output
    x = Dense(1024, activation='relu')(x)
    prediction = Dense(2, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=prediction)
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    return model

if __name__ == '__main__':
    x_test, y_test = utils.get_data('../test', 2)
    x_train, y_train = utils.get_data('../train', 2)
    y_train = utils.one_hot_encode(y_train)
    y_test = utils.one_hot_encode(y_test)
    base_model = get_basemodel()
    model = get_model(base_model)
    for layer in base_model.layers:
        layer.trainable = False
    early_stop = EarlyStopping(monitor='val_acc', verbose=1, patience=10, mode='max')
    save_best = ModelCheckpoint('mvgg_best_model.hdf5', verbose=1, save_best_only=True)
    model.fit(x_train, y_train, epochs=200, batch_size=64, validation_split=0.1, callbacks=[save_best, early_stop])
    
```


## 方法二：Fine-tuning the ConvNet
这种方法主要是用新的数据重新训练已有的模型，微调权重。这种方法可以重新训练原始模型的所有层，也可以只训练部分层。此外，如何原始分类模型和当前分类问题的类别不一致时，还需要设计一个新的分类器。但是在本实例中，原始模型时二分类模型，新的问题也是二分类问题，因此不需要重新设计分类器。这种方法主要包括两步：

1. 加载model，载入训练好的模型，详见*get_model*函数。
2. 微调模型权重，使用较小的学习率重新训练模型，详见*main*函数。


**代码：**
```python
from __future__ import division
import numpy as np
from keras.models import load_model, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
import utils

#载入模型
def get_model():
    model_path = 'model.hdf5'
    model = load_model(model_path)
    return model

#微调模型
if __name__ == '__main__':
    x_test, y_test = utils.get_data('../test', 2)
    x_train, y_train = utils.get_data('../train', 2)
    y_train = utils.one_hot_encode(y_train)
    y_test = utils.one_hot_encode(y_test)
    model = get_model()
    early_stop = EarlyStopping(monitor='val_acc', verbose=1, patience=10, mode='max')
    save_best = ModelCheckpoint('mvgg_best_model.hdf5', verbose=1, save_best_only=True)
    model.fit(x_train, y_train, epochs=200, batch_size=64, validation_split=0.1, callbacks=[save_best, early_stop])
    
```