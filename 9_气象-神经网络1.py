#!/usr/bin/env python
# coding: utf-8

# ## 9.基于多种神经网络模型的气象时间序列数据预测
# 1.--数据预处理
# 1.1.--导入工具包
# In[1]:
print("In[1]")

import os
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # Seaborn是一个基于matplotlib的数据可视化库，提供了更高级的接口和更美观的图形。
import tensorflow as tf  # TensorFlow是一个开源的机器学习框架，可以用于构建和训练神经网络模型。

from matplotlib.font_manager import FontProperties  # FontProperties类用于设置字体属性，如字体名称、大小、样式等。
font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=15)

import warnings 
warnings.filterwarnings('ignore')  # 警告信息过滤掉，不显示在控制台。

# %matplotlib inline  # 网页版本专用

mpl.rcParams['figure.figsize'] = (16, 10)  # 设置图像大小
mpl.rcParams['figure.dpi'] = 300  # 原600导致WinError 10053
mpl.rcParams['font.size'] = '11'  # #设置字体大小

# 1.2.--读取数据
# 其中第一行数据为各列名称对应的中文解释
# In[2]:
print("In[2]")

df = pd.read_csv('./data/data_climate_detail.csv')
print(df)

# 注: "位温"是指把气块从气压为p处移到海平面处之后,气块所具有的温度。"露点温度"指在固定气压之下,空气中所含的气态水达到饱和而凝结成液态水所需要降至
# 的温度。在这温度时,凝结的水飘浮在空中称为雾、而沾在固体表面上时则称为露,因而得名"露点温度"。"相对空气湿度"指空气中的水汽含量。"饱和差"是指空气
# 在一定温度下的饱和水汽压与当时实际水汽压的差值。“比湿度”是对空气中整体湿气含量的一个测量标准，即每千克干空气中所含水蒸气的数量。“空气密度”是指在
# 一定的温度和压力下，单位体积空气所具有的质量。
# In[3]:
print("In[3]")

df = pd.read_csv('./data/data_climate.csv')
print(df)


# In[4]:
print("In[4]")

print(type(df.iloc[0][1]))  # 查看第一行的p列对应的数据类型，.float32 单精度浮点数，.float64 双精度浮点数


# In[5]:
print("In[5]")

# 将字符型数据改成浮点型数据(原数据为字符型,不利于后续计算)
df[['p', 'T', 'Tpot', 'Tdew', 'rh', 'VPmax', 'VPact', 'VPdef', 'sh', 'H2OC', 'rho', 'wv', 'max. wv', 'wd']] = \
    df[['p', 'T', 'Tpot', 'Tdew', 'rh', 'VPmax', 'VPact', 'VPdef', 'sh', 'H2OC', 'rho', 'wv', 'max. wv', 'wd']].astype(float)

# 1.3.--对数据进行二次取样
# 原数据每十分钟收集一次数据,本案例只需要每小时的数据,因此在原数据集的基础上二次取样:
# In[6]:
print("In[6]")

# 切片[起始点:终止点:步长],从索引5开始,每隔6个记录一次。
df = df[5::6]  # 观察Data Time列,数据已经变为每小时的采样
print(df)

# 1.4.--处理时间列数据
# 处理前：
# In[7]:
print("In[7]")

print(df["Date Time"])  # df数据 Date Time列

# 处理后: (result_climate.csv为完成所有数据预处理工作后的文件,此处读取方便查看结果)
# In[8]:
print("In[8]")
# Day sin 表示一天的正弦值，Day cos 表示一天的余弦值，Year sin 表示一年的正弦值，Year cos 表示一年的余弦值。
# 这些函数通常在计算机编程中使用，特别是在处理时间序列数据时。
result_df = pd.read_csv('./data/result_climate.csv')
print(result_df[['Day sin', 'Day cos', 'Year sin', 'Year cos']])

# 这样做的原因是:如果仅用数字表示时刻,即0表示0点,1表示1点,.23表示23点,那么会造成0点与23点在数字上相差很远,但实际距离很近的问题。
# 事实上时间序列有周期性,如果通过以下的公式将其转换成周期变量,便可解决这个问题。
# 处理方式：
# 步骤一:将'Date Time'列数据从str数据类型转换为日期-时间(datetime)数据类型,保存到新变量date_time中(原df中删除Date Time'列)
# In[9]:
print("In[9]")

# 将名为 "Date Time" 的列从 DataFrame df 中弹出，并将其转换为 datetime 类型。转换时使用的格式为 '%d.%m.%Y %H:%M:%S'。
date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
print(date_time)

# 步骤二：将date_time中的数据转换成时间戳格式的数据
# In[10]:
print("In[10]")

# 时间戳是指从1970年1月1日（UTC/GMT的午夜）开始所经过的秒数，不计入闰秒。在计算机中，时间戳通常以整数形式表示。
print(datetime.datetime.timestamp(date_time[5]))  # 将日期时间对象转换为时间戳


# In[11]:
print("In[11]")

# 使用 map() 函数将 date_time 中的每个日期时间对象应用 datetime.datetime.timestamp 函数
timestamp_s = date_time.map(datetime.datetime.timestamp)
print(timestamp_s)

# 步骤三：将时刻序列映射为正弦曲线序列
# In[12]:
print("In[12]")

day = 24*60*60  # 计算一天有多少秒
year = (365.2425)*day  # 计算一年有多少秒
# df增加四列
df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))  # 计算时间戳timestamp_s在一天内对应的弧度数的正弦值。
df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))  # 计算时间戳timestamp_s在一天内对应的弧度数的余弦值。
df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))  # 计算时间戳timestamp_s在一年内对应的弧度数的正弦值。
df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))  # 计算时间戳timestamp_s在一年内对应的弧度数的余弦值。


# In[13]:
print("In[13]")
# 显示df增加的四列'Day sin', 'Day cos', 'Year sin', 'Year cos'
print(df[['Day sin', 'Day cos', 'Year sin', 'Year cos']])

# 将转换结果可视化：
# In[14]:
print("In[14]")

plt.plot(np.array(df['Day sin'])[:25])
plt.plot(np.array(df['Day cos'])[:25])
plt.xlabel('时间[单位：时]（Time [h]）', fontproperties=font_set)
plt.title('一天中的时间信号（Time of day signal）', fontproperties=font_set)
plt.show()

# 1.5.--处理风向与风速列数据：
# 处理前:用极坐标(风速(m/s)和风向(0-360度) )来描述风的强度和方向;
# 处理后：用正交坐标系的两个维度（X轴和Y轴）上风的强度来描述上述风的强度和方向。
# In[15]:
print("In[15]")

print(df[['wv', 'max. wv', 'wd']])  # 平均风速、最大风速、风向(角度制)


# In[16]:
print("In[16]")

# result_df: 这是一个DataFrame对象，可以想象成一个表格，其中包含多行和多列的数据。
print(result_df[['Wx', 'Wy', 'max Wx', 'max Wy']])  # 转换成风矢量的形式

# 处理方式：
# 步骤一：处理风速列的异常值
# 在用来描述风速的列和max.列中，存在风速为-9999m/s的情况（见下），但是实际上风速应该大于等于0。因此将其替换为零：
# In[17]:
print("In[17]")
# df['wv'] < 0: 这是一个条件表达式，用于选择'wv'列中值小于0的行，的'wv'列的所有值。
print(df[df['wv'] < 0]['wv'])  # 查看平均风速列小于0的异常值


# In[18]:
print("In[18]")
# 选取df中'max. wv'列值小于0的所有行，并返回这些行的'max. wv'列的值。
print(df[df['max. wv'] < 0]['max. wv'])  # 查看最大风速列小于0的异常值  


# In[19]:
print("In[19]")

# 将-9999处的值替换为0
wv = df['wv']
bad_wv = wv == -9999.0
wv[bad_wv] = 0.0

max_wv = df['max. wv']
bad_max_wv = max_wv == -9999.0
max_wv[bad_max_wv] = 0.0


# In[20]:
print("In[20]")

print(df['wv'].min())  # 查看平均风速的最小值为0
print(df['max. wv'].min())  # 查看最大风速的最小值为0

# 步骤二：将风向和风速列数据转换为风矢量，重新存入原数据框中
# 此处我们会绘制2D直方图,目的是通过可视化的方式解释风矢量类型的数据优于原表中风速和风向数据原因。
# 首先,根据原始数据绘制不同的风速与风向组合对应的出现次数图(用2D直方图展示) :
# 横轴为风向,纵轴为风速,风向与风速的交汇点上用颜色的深浅表示该组合出现的次数
# In[21]:
print("In[21]")

# 二维直方图，显示了'wd'和'wv'两个数据列之间的关系。x轴和y轴的50x50的小格子，任何值都不会超过400
plt.hist2d(df['wd'], df['wv'], bins=(50, 50), vmax=400)
plt.colorbar()  # 颜色设置一个标签，这样你可以知道每种颜色所代表的数值。
plt.xlabel('风向 [单位：度]', fontproperties=font_set)
plt.ylabel('风速 [单位：米/秒]', fontproperties=font_set)
plt.show()

# 将风向和风速列转换为风矢量，绘制2D直方图
# In[22]:
print("In[22]")

# 将df中的'wv'列保存到wv中，并从原来的df中删除
# 将df中的'max. wv'列保存到max_wv中，并从原来的df中删除
wv = df.pop('wv')
max_wv = df.pop('max. wv')

# 将df中的'wd (deg)'列由角度制转换为弧度制,保存到wd_rad中,并从原来的df中删除
wd_rad = df.pop('wd')*np.pi / 180

# 计算平均风力的x和y分量，分别保存到df的‘Wx’列和‘Wy’列中
df['Wx'] = wv*np.cos(wd_rad)
df['Wy'] = wv*np.sin(wd_rad)

# 计算最大风力x和y分量，分别保存到df的‘max Wx’列和‘max Wy'列中
df['max Wx'] = max_wv*np.cos(wd_rad)
df['max Wy'] = max_wv*np.sin(wd_rad)


# In[23]:
print("In[23]")
# 二维直方图，显示了'wx'和'wy'两个数据列之间的关系。x轴和y轴的50x50的小格子，任何值都不会超过400
plt.hist2d(df['Wx'], df['Wy'], bins=(50, 50), vmax=400)
plt.colorbar()  # 颜色设置一个标签，这样你可以知道每种颜色所代表的数值。
plt.xlabel('风的X分量[单位：m/s]', fontproperties=font_set)
plt.ylabel('风的Y分量[单位：m/s]', fontproperties=font_set)
plt.show()
 

# 对比两图可见，下图更有利于我们观察风的状况：找到原点（0,0）位置，假设向上方向为北，那么我们可以观察到南方向的风的出现次数较多。
# 此外，我们还可以观察到东北-西南方向的风。
# 1.6.--划分数据集
# 取数据集中70%的数据作为训练集,取数据集中20%的数据作为验证集,取数据集中10%的数据作为测试集
# In[24]:
print("In[24]")

n = len(df)  # 数据框的行数
num_features = df.shape[1]  # df数据形状1是列数

print(n, num_features)


# In[25]:
print("In[25]")

train_df = df[0:int(n*0.7)]  # n是行数，df数据的前70%行。划分训练集
val_df = df[int(n*0.7):int(n*0.9)]  # n是行数，df数据的前70%-90%行。划分验证集
test_df = df[int(n*0.9):]  # n是行数，df数据的前90%-100%行。划分测试集

# 1.7.数据标准化
# 使用训练集的数据来计算平均值和标准差,原因是:因为在训练模型时不能访问验证集和测试集中的数据。而且在训练模型时不能访问训练集中未来时间点的值,
# 而且这种规范化的方法也应该使用移动平均值的方法来完成。但这里并非本案例的重点,且验证集和测试集能确保获得(某种程度上)真实的指标。
# In[26]:
print("In[26]")

train_mean = train_df.mean()  # 计算训练集的均值
train_std = train_df.std()  # 计算训练集的标准差

train_df = (train_df - train_mean) / train_std  # 对训练集进行标准化
val_df = (val_df - train_mean) / train_std  # 对验证集进行标准化
test_df = (test_df - train_mean) / train_std  # 对测试集进行标准化


# In[27]:
print("In[27]")

df_std = (df - train_mean) / train_std  # 对所有集中的数据进行标准化
print(df_std)

# 1.8.--绘制小提琴图
# In[28]:
print("In[28]")

# melt()函数将数据框（DataFrame）进行重塑的操作。具体来说，它将数据框的列转换为行，并将每个列的值作为新行的"Normalized"列的值。
df_std = df_std.melt(var_name='Column', value_name='Normalized')  # 将上表结果变为两列数据，方便绘制小提琴图，列的信息提取到行中
print(df_std)


# In[29]:
print("In[29]")

plt.figure(figsize=(12, 6))
# violinplot函数来绘制一个小提琴图。它根据给定的数据框df_std，将'Column'列作为x轴，'Normalized'列作为y轴进行绘制。
ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
# 下划线 _ 是一个常用的占位符，用于表示我们不关心某个变量的值。
_ = ax.set_xticklabels(df.keys(), rotation=90)  # x轴的刻度标签设置为数据框df的列名，并将标签旋转90度。
plt.show()

# 小提琴图中的每一个小提琴展示了原数据表中的每一列数据的统计特征,例如第二个小提琴表示了温度列数据可能出现的取值,以及这些取值出现的概率。
# 每个小提琴上下端点间的纵坐标值是可能的取值,每个取值在横坐标方向上的宽度表示该取值出现的概率。例如第二个小提琴中,越宽的地方表示该温度出现概率越高。
# 每个小提琴里的矩形的上下端点表示了四分之一和四分之三分位数的位置，白点表示二分之一分位数的位置。
# 2.--数据窗口类与方法定义
# In[30]:
print("In[30]")


# class 语句来创建一个新类， class 之后为类的名称并以冒号结尾:WindowGenerator 是一个类名，用于创建一个窗口生成器。
class WindowGenerator():
    # Python类的初始化方法（__init__），
    def __init__(self, input_width, label_width, shift,
                 train_df=train_df, val_df=val_df, test_df=test_df,
                 label_columns=None):  # label_columns=None 没有明确，是一个参数默认值的设置。
        # 存储原始数据
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # 计算索引
        self.label_columns = label_columns
        # 为标签列设置索引
        if label_columns is not None:
            # 会遍历枚举对象，将每个元素的值作为键，索引作为值，创建一个新的字典。
            # 使用了enumerate()函数来同时遍历label_columns中的索引和元素。
            self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
        # 为所有列设置索引
        self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

        # 计算窗口参数
        self.input_width = input_width  # 输入长度
        self.label_width = label_width  # 标签长度
        self.shift = shift  # 每一步的时间步长

        self.total_window_size = input_width + shift  # 总窗口长度

        # arange函数生成一个从0到self.total_window_size（不包括self.total_window_size）的整数数组，然后取前input_width个元素。
        self.input_indices = np.arange(self.total_window_size)[0:input_width]  # 建立输入的索引

        self.label_start = self.total_window_size - self.label_width  # 标签的起始点
        self.label_indices = np.arange(self.total_window_size)[self.label_start:]  #建立标签的索引

    def __repr__(self):  # 当输出实例化对象时,输出我们想要的信息，__repr__方法，返回
        # 返回一个字符串，其中包含一些关于窗口大小、输入索引、标签索引和标签列名的信息。使用join方法将这个列表中的所有字符串连接起来，
        # 每个字符串之间用换行符（\n）分隔。
        return '\n'.join([
              f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])


# In[31]:
print("In[31]")

def split_window(self, features):  # split_window 是一个方法（函数）的定义，属于某个类（由 self 参数表示）。

    # 从features数组中选择了一个子集，这个子集在第一个维度上包含所有元素，在第二个维度上从第0列到第self.input_width - 1
    # 列，并且在第三个维度（如果有的话）上包含所有元素。
    inputs = features[:, 0:self.input_width, :]  
    labels = features[:, self.label_start:, :]  
    
    if self.label_columns is not None:
        # tf.stack()是TensorFlow中的一个函数，用于将一系列张量沿着一个新的轴堆叠起来。
        labels = tf.stack(
            [labels[:, :, self.column_indices[name]] for name in self.label_columns],
            axis=-1)

    # 将inputs数组的形状设置为有两个维度：第一个维度可以是任意长度，第二个维度的长度为self.input_width，第三个维度也可以是任意长度。
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels


WindowGenerator.split_window = split_window  # 将一个名为 split_window 的函数赋值给 WindowGenerator 类的一个属性或方法。


# In[32]:
print("In[32]")


# 创建一个数据集make_dataset , self 参数,接收一个名为 data 的参数。
def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)  # 将data转换成NumPy的32位浮点数数组。

    # 从一维数组data中创建一个时间序列数据集，其中每个时间序列的长度为self.total_window_size，并且每个批次包含32个样本。
    # tf.keras.preprocessing.timeseries_dataset_from_array是TensorFlow的Keras API中的一个函数，创建时间序列数据集。
    # data: 这是输入数据的一维数组, targets: None，这意味着没有目标值, sequence_length: 这定义了每个时间序列的长度,
    # sequence_stride: 这定义了时间步之间的间隔1。shuffle: 如果设置为True，数据集在创建时会被打乱。batch_size: 每个批次有32个样本。
    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
      data=data,
      targets=None,  
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=32,)

    ds = ds.map(self.split_window)  # 对ds中的每一个元素应用 self.split_window 方法，并返回ds数据集。

    return ds

WindowGenerator.make_dataset = make_dataset  # 将一个名为 make_dataset 的函数赋值给 WindowGenerator 类的一个属性或方法。


# In[33]:
print("In[33]")

# 此处针对训练集、验证集和测试集数据将make_dataset0方法变成属性
@property  # 装饰器，用于将一个方法转换为属性。这样可以让代码更简洁易读，同时还可以对属性的访问进行控制。
# 这个createTrainSet方法的目的是从类的train_df属性中获取数据，并使用make_dataset方法处理这些数据，然后返回处理后的数据集。
def createTrainSet(self):
    return self.make_dataset(self.train_df)

@property
def createValSet(self):
    return self.make_dataset(self.val_df)

@property
def createTestSet(self):
    return self.make_dataset(self.test_df)

@property
def example(self):
    # getattr(self, '_example', None) ：对象、属性名和默认值。获取对象的属性值，如果对象具有指定的属性名，则返回该属性的值；
    # 否则，返回默认值（在本例中为None）。
    result = getattr(self, '_example', None)  
    if result is None:
        # iter() 函数将 self.createTrainSet 转换为一个迭代器对象，然后使用 next() 函数来获取迭代器的下一个元素。
          result = next(iter(self.createTrainSet))
          self._example = result  # 将变量 result 的值赋给对象 elf 的 _example 属性。
    return result


WindowGenerator.createTrainSet = createTrainSet
WindowGenerator.createValSet = createValSet
WindowGenerator.createTestSet = createTestSet
WindowGenerator.example = example


# In[34]:
print("In[34]")

# 定义一个名为plot的函数，self实例，model默认值为None，plot_col默认值'T'，max_subplots默认值3。
def plot(self, model=None, plot_col='T', max_subplots=3):
        inputs, labels = self.example  # 通过使用self.example，解包到inputs和labels变量中。
        plot_col_index = self.column_indices[plot_col]  # 在 self.column_indices 字典中查找键为 plot_col 的值。
        max_n = min(max_subplots, len(inputs))  # 计算 max_subplots 和 inputs 列表长度的较小值。
        # 生成一个从0到max_n-1的整数序列，然后依次遍历这个序列中的每个元素。
        for n in range(max_n):
            plt.subplot(3, 1, n+1)  # 创建子图（3行，1列，n+1 表示当前子图的位置，其中 n 是循环变量，从0开始计数。）
            # y轴标签的内容为"标准化后的"加上变量plot_col的值，使用font_set字体属性来显示标签文本。
            plt.ylabel(f'标准化后的 {plot_col}', fontproperties=font_set)

            # plt.plot()函数来绘制一条线（x轴上的坐标点，inputs数组中提取的数据[子图的索引n，所有行，列的索引plot_col_index]，图例标签为
            # "Inputs"，点号(.)作为标记，-10位于其他线条的后面。）
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                # （self.label_columns_indices）中获取键为plot_col的值。如果字典中不存在该键，则返回None。
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index  # plot_col_index 是一个变量名

            if label_col_index is None:
                continue

            # 绘制散点图(x轴self.label_indices值，y轴[n索引，选择所有行，列的索引label_col_index]，边缘颜色为黑色（'k'），图例的标签为
            # "Labels"，散点的颜色为绿色（'green'），散点的尺寸为64)
            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                    edgecolors='k', label='Labels', c='green', s=64)
            
            
            if model is not None:
                predictions = model(inputs)  # 将输入数据传递给模型并获取输出结果
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='orange', s=64)

            if n == 0:
                plt.legend()  # 图表中添加图例

        plt.xlabel('Time [h]')  # x轴标签为"Time [h]"

WindowGenerator.plot = plot

# 3.--构建窗口数据
# 3.1.--使用WindowGenerator类构建数据窗口
# 创建single-window窗口
# In[35]:
print("In[35]")

# WindowGenerator四个参数(输入宽度6，标签宽度5，移动步长1，标签列“T”)
single_window = WindowGenerator(input_width=6, label_width=5, shift=1, label_columns=['T'])


print(single_window)

# 上述结果解释:输入为6个时间行,标签为5个时间行,预测的时间步长(跨度)为1,且预测未来时刻的温度情况(由1列特征值构成)
# 创建multi-window窗口
# In[36]:
print("In[36]")

# WindowGenerator四个参数(输入宽度6，标签宽度5，移动步长24)
multi_window = WindowGenerator(input_width=6, label_width=5, shift=24)

print(multi_window)

# 上述结果解释:输入为6个时间行，标签为5个时间行,预测的时间步长（跨度)为24，且预测未来时刻所有的天气情况（由19列特征值构成)
# 3.2.--构建训练集、验证集和测试集
# In[37]:
print("In[37]")

print('训练数据:')
print(multi_window.createTrainSet)
print('验证数据：')
print(multi_window.createValSet)
print('测试数据：')
print(multi_window.createTestSet)

# 上述结果解释： createTrainSet()方法内部已经调用了split_window()方法，返回划分出的输入数据和标签数据
# In[38]:
print("In[38]")

# 查看每一批输入模型的数据和标签的形状(以训练集数据举例),使用TensorFlow库中的take()函数来获取训练集的前1个批次。
for train_inputs, train_labels in multi_window.createTrainSet.take(1):
    print(f'Inputs shape (batch, time, features): {train_inputs.shape}')
    print(f'Labels shape (batch, time, features): {train_labels.shape}')

# 上述结果解释：
# (32,6,19)的含义是:全体数据可以拆分为多组训练样本,一组训练样本的特征值包括32个训练样本的特征值,一个训练样本的特征值对应6行数据,
# 一行数据是一个时刻的天气情况(由19列特征值构成) ,第6个时刻记作t_feature
# (32,5,19)的含义是:全体数据可以拆分为多组训练样本,一组训练样本的标签值包括32个训练样本的标签值,一个训练样本的标签值对应5行数据,
# 一行数据是一个时刻的天气情况（由19列特征值构成），第5个时刻记作t_label。
# t_label=t_feature+shift
# 4.--CNN模型预测
# 目标:基于历史6个时间点的天气状况(对应表中6行19列的数据) ,预测经过24小时(shift=24)后,未来的5个时间点天气状况(5行19列的数据) 。
# 4.1.--构建CNN模型
# In[39]:
print("In[39]")

# 使用TensorFlow的Keras API创建的序列模型。这个模型包含以下层：
# 一维卷积层（Conv1D）：有64个过滤器，每个过滤器的大小为3，步长为1，激活函数为ReLU。
# 展平层（Flatten）：将卷积层的输出展平，以便可以连接到全连接层。
# 全连接层（Dense）：有5*19个神经元，权重初始化为0。
# 重塑层（Reshape）：将全连接层的输出重塑为形状为[5, 19]的张量。
multi_conv_model = tf.keras.Sequential([
    
   
  
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, activation='relu'),

  
    
    tf.keras.layers.Flatten(),
    
   
    tf.keras.layers.Dense(5*19, kernel_initializer=tf.initializers.zeros),
    
  
    tf.keras.layers.Reshape([5, 19])
])

# 4.2.--训练CNN模型
# In[40]:
print("In[40]")

# 设置训练的总轮数
MAX_EPOCHS = 20  # 变量MAX_EPOCHS的值设置为20
def compile_and_fit(model, window):
    # 在TensorFlow中设置早停（EarlyStopping）回调的代码。它的作用是在训练过程中，当验证集上的损失不再降低时，停止训练。
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode='min')

    # 编译一个TensorFlow模型的。它使用了均方误差（MeanSquaredError）作为损失函数，Adam优化器作为优化算法，以及平均绝对误差（Mean
    # AbsoluteError）作为评估指标。adam(一种优化的随机梯度下降法) : metrics设置模型检验的方法:平均绝对误差)
    model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    # 训练模型（设置输入：训练数据集：设置训练总轮数：设置验证数据集：设置提前结束训练：
    # verbose设置日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录）
    history = model.fit(window.createTrainSet, epochs=MAX_EPOCHS, validation_data=window.createValSet,
                        callbacks=[early_stopping], verbose=2)
    return history


# In[41]:
print("In[41]")
# compile_and_fit 函数，接收两个参数：multi_conv_model 和 multi_window。
history = compile_and_fit(multi_conv_model, multi_window)


# In[42]:
print("In[42]")

multi_conv_model.summary()  # summary()显示模型multi_conv_model摘要

# 4.3.--评估模型
# In[43]:
print("In[43]")

# 初始化字典
multi_val_performance = {}
multi_test_performance = {}

# 这个函数会使用验证集（multi_window.createValSet）来评估模型的性能，并返回损失值和指标值。其中，verbose=0 表示不输出详细信息。
multi_val_performance['Conv'] = multi_conv_model.evaluate(multi_window.createValSet, verbose=0)
multi_test_performance['Conv'] = multi_conv_model.evaluate(multi_window.createTestSet, verbose=0)
# 输出评估结果到multi performance和multi val performance字典里
# 字典的key对应不同模型名称, value对应不同模型下训练的结果(指标为损失值和MAE)

# In[44]:
print("In[44]")

multi_window.plot(multi_conv_model)
plt.show()

# 上述结果解释：
# 蓝色代表输入的历史6个小时数据。绿色代表待预测的5个小时的真实标签值;橙色代表模型对待预测的5个小时的预测数据。
# 如果模型预测好,橙色会和绿色重合。（图中方便展示只画出了温度的情况，实际其余特征也参与了模型训练）
# 5.--LSTM模型预测
# 5.1.--构建LSTM模型
# In[45]:
print("In[45]")

# 使用TensorFlow的Keras API创建的序列模型。该模型包含以下层：
# LSTM层，具有64个单元，激活函数为tanh，不返回序列。
# Dense层，具有5*19个神经元，权重初始化为全零。
# Reshape层，将输出调整为形状为[5, 19]的张量。
multi_lstm_model = tf.keras.Sequential([
    
    
    tf.keras.layers.LSTM(64, activation='tanh', return_sequences=False),
    
    
    tf.keras.layers.Dense(5*19, kernel_initializer=tf.initializers.zeros),
    
   
    tf.keras.layers.Reshape([5, 19])
])


# In[46]:
print("In[46]")

# 注： tanh 函数（将一个实数映射到（-1，1）的区间）
def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))  # 计算双曲正弦函数（sinh(x)）的值
inputs = np.arange(-10,10,0.1)  # 生成一个从-10到10（不包括10）的等差数列，步长为0.1
outputs = tanh(inputs)  # 计算inputs值的双曲正切（hyperbolic tangent）值。
plt.plot(inputs, outputs)  # 绘制折线图的函数，x=inputs ，y=outputs 。
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# 5.2.--训练LSTM模型
# In[47]:
print("In[47]")

# compile_and_fit 是一个函数In[40]，用于编译和训练一个多层长短期记忆（LSTM）模型。它接受两个参数：multi_lstm_model 和 multi_window。
history = compile_and_fit(multi_lstm_model, multi_window)


# In[48]:
print("In[48]")

print(multi_lstm_model.summary())  # summary() 是一个用于显示深度学习模型结构的方法

# 5.3.--评估模型
# In[49]:
print("In[49]")

# 多层LSTM模型（multi_lstm_model）在验证集（multi_window.createValSet）上的性能。其中，verbose参数设置为0表示不输出详细的评估信息。
multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.createValSet, verbose=0)
multi_test_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.createTestSet, verbose=0)


# In[50]:
print("In[50]")

multi_window.plot(multi_lstm_model)
plt.show()

# 如果return_sequences设置为true
# In[51]:
print("In[51]")

multi_lstm_model2 = tf.keras.models.Sequential([
   
   # LSTM层，return_sequences=True设置输出全部序列,包含6个时间步输出的序列,即每一个Cell的h_t,Shape [32, 6, 19] => [32, 6, 64]
    tf.keras.layers.LSTM(64, activation='tanh',return_sequences=True),
    tf.keras.layers.LSTM(64, activation='tanh',return_sequences=False),
    # dense层，Shape => [32, 95]
    tf.keras.layers.Dense(5*19,kernel_initializer=tf.initializers.zeros),
    
    # 输出层， Shape => [32, 5, 19]
    tf.keras.layers.Reshape([5, 19])
   
])


# In[52]:
print("In[52]")

# 拟合一个多层LSTM模型（multi_lstm_model2）到多窗口数据（multi_window）上。其中，compile函数用于配置模型的优化器、损失函数和评估指标等参数，
# fit函数用于将模型拟合到数据上进行训练。
history = compile_and_fit(multi_lstm_model2, multi_window)


# In[53]:
print("In[53]")

# 打印一个多层LSTM模型（multi_lstm_model2）的摘要信息，包括每一层的输入输出维度、激活函数、参数数量等。
print(multi_lstm_model2.summary())


# In[54]:
print("In[54]")

# 评估一个多层LSTM模型（multi_lstm_model2）在验证集（multi_window.createValSet）上的性能。verbose参数设置为0表示不输出详细的评估信息。
multi_val_performance['LSTM2'] = multi_lstm_model2.evaluate(multi_window.createValSet, verbose=0)
multi_test_performance['LSTM2'] = multi_lstm_model2.evaluate(multi_window.createTestSet, verbose=0)

# 比较CNN模型和LSTM模型预测效果（绘制验证集和测试集在各个模型训练下的平均绝对误差（MAE）对比图）
# In[55]:
print("In[55]")

print(multi_val_performance)  # 展示验证集的评估结果


# In[56]:
print("In[56]")

print(multi_test_performance)  # 展示测试集的评估结果


# In[57]:
print("In[57]")

# 获取一个多层LSTM模型（multi_lstm_model）中测量值MAE（mean_absolute_error）所属的索引。
metric_index = multi_lstm_model.metrics_names.index('mean_absolute_error')  # 找出测量值MAE所属的索引
print(metric_index)


# In[58]:
print("In[58]")

print(multi_val_performance.values())  # 获取一个多窗口验证性能（multi_val_performance）的值。


# In[59]:
print("In[59]")
# 根据MAE的索引遍历验证集的评估结果,返回所有模型的MAE测量值
val_mae = [v[metric_index] for v in multi_val_performance.values()]
print(val_mae)


# In[60]:
print("In[60]")

# 根据MAE的索引遍历测试集的评估结果,返回所有模型的MAE测量值
test_mae = [v[metric_index] for v in multi_test_performance.values()]
print(test_mae)


# In[61]:
print("In[61]")
# 生成一个长度为multi_test_performance列表长度的NumPy数组，数组中的元素从0开始递增。
x = np.arange(len(multi_test_performance))
plt.ylabel('mean_absolute_error')  # 设置Matplotlib绘图库中y轴标签为'mean_absolute_error'。
# 绘制一个柱状图，其中x轴坐标为x-0.17，高度为val_mae，宽度为0.3，标签为'Validation'。
# 该柱状图可以用于展示多窗口测试性能中验证集的mean_absolute_error值。
plt.bar(x=x-0.17, height=val_mae, width=0.3, label='Validation')
plt.bar(x=x+0.17, height=test_mae, width=0.3, label='Test')
plt.xticks(ticks=x, labels=multi_test_performance.keys(), rotation=45)
_ = plt.legend()  # 显示Matplotlib绘图库中绘制的图例。该图例可以用于标识不同颜色或标签所代表的含义，方便读者理解图表。
plt.show()

# 6·--数据窗口类与方法详细分步解析
# 创建WindowGenerator类对象
# In[62]:
print("In[62]")

# WindowGenerator生成滑动窗口，input_width=6 表示每个窗口包含 6 个连续的时间步；label_width=5 表示每个窗口包含 5 个连续的标签；
# shift=1 表示窗口之间的步长为 1；label_columns=['T'] 表示标签列名为 'T'。
single_window = WindowGenerator(input_width=6, label_width=5, shift=1, label_columns=['T'])
print(single_window)

# 当传入input_width=6, label_width=5, shift=1, label_columns=[T]这些参数时,会产生以下结果:
# In[63]:
print("In[63]")

input_width = 6
label_width = 5
shift = 1
label_columns = ['T']

# 创建一个字典 label_columns_indices，将标签列名映射到它们在数据中的索引位置。如果 label_columns 不为空，则遍历 label_columns 列表，
# 使用枚举函数 enumerate() 获取每个标签列名及其对应的索引位置，并将它们添加到字典中。
label_columns_indices = {}
if label_columns is not None:
    for i, name in enumerate(label_columns):
        label_columns_indices[name] = i
print(label_columns_indices)


# In[64]:
print("In[64]")
# 同上
column_indices = {}
for i, name in enumerate(train_df.columns):
    column_indices[name] = i
print(column_indices)  


# In[65]:
print("In[65]")

total_window_size = input_width + shift  # 输入窗口的宽度加上步长
print(total_window_size)


# In[66]:
print("In[66]")

# 会生成一个从0开始，到total_window_size - 1结束的整数数组。从数组中提取从第0个元素到第input_width - 1个元素的部分
input_indices = np.arange(total_window_size)[0:input_width]  # 生成一个长度为 input_width 的一维数组
print(input_indices)


# In[67]:
print("In[67]")

label_start = total_window_size - label_width  # 计算滑动窗口的大小，即 total_window_size 减去标签列的宽度 label_width。
print(label_start)


# In[68]:
print("In[68]")

# arange函数来创建一个从0开始，到total_window_size - 1结束的一维数组。从数组中提取从第label_start个元素开始，到数组末尾的部分。
label_indices = np.arange(total_window_size)[label_start:] 
print(label_indices)


# In[69]:
print("In[69]")

# 将一个字符串列表连接成一个单独的字符串，其中每个字符串之间都插入一个换行符（\n），列表中的字符串格式化:
print('\n'.join([f'Total window size: {total_window_size}',
                 f'Input indices: {input_indices}',
                 f'Label indices: {label_indices}',
                 f'Label column name(s): {label_columns}']))

# 划分数据窗口
# 划分数据窗口只需让刚刚创建好的single_window直接调用createTrainSet()方法或createValSet()方法或createTestSet()方法即可。
# 上述三种方法的区别在于划分对象不同:
# createTrainSet()方法：划分训练集数据
# createValSet()方法：划分验证集数据
# createTestSet()方法：划分测试集数据
# 因此这里以createTrainSet()方法为例，介绍划分数据窗口的整个过程。
# In[70]:
print("In[70]")

print(single_window.createTrainSet)  # 同时划分出输入数据和标签数据


# In[71]:
print("In[71]")

# 在一个滑动窗口中生成训练集的输入和标签。其中，single_window.createTrainSet 是一个函数，用于创建训练集；
# take(1) 表示只取一个样本；train_inputs 和 train_labels 分别表示训练集的输入和标签。
for train_inputs, train_labels in single_window.createTrainSet.take(1):
    print(f'Inputs shape (batch, time, features): {train_inputs.shape}')
    print(f'Labels shape (batch, time, features): {train_labels.shape}')

# 接下来查看createTrainSet()的具体实现过程:
# createTrainSet()方法会返回对象自身调用make_dataset(self.train_df)方法后的结果,如下:
# In[72]:
print("In[72]")

@property  
def createTrainSet(self):  
    return self.make_dataset(self.train_df)  # 调用 make_dataset 方法，将 self.train_df 作为参数传入，生成一个数据集。
WindowGenerator.createTrainSet = createTrainSet

# 因此接下来介绍make_dataset()的实现过程:
# In[73]:
print("In[73]")

data = train_df

data = np.array(train_df, dtype=np.float32)  # 将 train_df 转换为一个NumPy数组，并将数据类型设置为 np.float32。
print(data)


# In[74]:
print("In[74]")

# 使用Keras库中的 timeseries_dataset_from_array 函数将一个一维数组转换为时间序列数据集。其中，data 参数表示输入数据，
# targets 参数表示目标值（可选），sequence_length 参数表示每个样本的时间步长，sequence_stride 参数表示样本之间的步长，
# shuffle 参数表示是否随机打乱数据，batch_size 参数表示每个批次的大小。
ds = tf.keras.preprocessing.timeseries_dataset_from_array(
      data=data,
      targets=None,  
      sequence_length=total_window_size,  # 设置输出序列的长度
      sequence_stride=1,  # #连续输出序列之间的周期
      shuffle=True,  # #设置打乱输出样本的顺序
      batch_size=32,)  # #设置每批中时间序列样本的数量（最后一批除外）
print(ds)


# In[75]:
print("In[75]")

# 使用 as_numpy_iterator() 方法将 ds Dataset 对象转换为一个返回 NumPy 数组的迭代器。
# 然后，使用 list() 函数将这个迭代器转换为一个 Python 列表。
ds_list = list(ds.as_numpy_iterator())
print(len(ds_list), len(ds_list[0]), len(ds_list[0][0]), len(ds_list[0][0][0]))  # 对象的长度

# (1534,32,7,19)的含义是:全体数据可以拆分为1534组训练样本,一组训练样本的值包括32个训练样本的值,一个训练样本的值对应7行数据,
# 一行数据是一个时刻的天气情况（由19列数值构成）
# 为了更好地解释tf.keras.preprocessing.timeseries_dataset_from_array()方法，此处举例说明:
# In[76]:
print("In[76]")

import numpy as np
lst = np.array(list(range(0, 100)))  # list(range(0, 100)) 会生成一个从0到99的整数列表。np.array() 将这个列表转换为一个NumPy数组。
print(lst)


# In[77]:
print("In[77]")

# tf.keras.preprocessing.timeseries_dataset_from_array 是一个用于从数组创建时间序列数据集的函数。
tf_lst = tf.keras.preprocessing.timeseries_dataset_from_array(
         lst, targets=None, sequence_length=10,  # lst：输入数据，targets：目标数据，可选参数，默认为 None。序列长度10。
         sequence_stride=3,  # 序列步长，默认值为 3。
         sampling_rate=2,  # 采样率，默认值为 2。
         batch_size=3,  # 批处理大小，默认值为 3。
         )

temp_lst = list(tf_lst.as_numpy_iterator())  # 将一个TensorFlow张量（tensor）转换为NumPy数组的列表
print(temp_lst)


# In[78]:
print("In[78]")

# temp_lst的第一维长度，temp_lst[0]（即第一维的第一个元素）的长度，temp_lst[0][0]（即第一维第一个元素的第一个元素）的长度。
print(len(temp_lst), len(temp_lst[0]), len(temp_lst[0][0]))

# tf.keras.preprocessing.timeseries_dataset_from_array()方法解释说明结束。
# make_dataset()方法最后一步为：
# ds=ds.map(self.split_window)
# 这一步的作用是:将每个训练样本划分为训练样本的特征值和训练样本的标签值两部分,其中训练样本的特征值对应6行数据,
# 1行数据是一个时刻的天气情况(由19列特征值构成) ,训练样本的标签值对应5行数据, 1行数据是一个时刻的天气情况(由19列特征值构成) 。
# 由于split_window()为自定义方法，因此我们将以其中一个训练样本为例，解释split_window()的实现过程。
# In[79]:
print("In[79]")

# 将数据集（dataset）转换为NumPy数组列表的操作。其中，ds 是一个数据集对象，as_numpy_iterator() 方法用于将数据集转换为一个迭代器，
# 每次迭代返回一个NumPy数组。
ds_list = list(ds.as_numpy_iterator())
example_window = tf.stack([ds_list[0][0]])
print(example_window)

# 查看一个训练样本调用split_window()方法后的结果：
# In[80]:
print("In[80]")

example_inputs, example_labels = single_window.split_window(example_window)
print('训练样本的特征值：', example_inputs)
print('训练样本的标签值：', example_labels)

# split_window()的实现过程如下：
# In[81]:
print("In[81]")

features = example_window


# In[82]:
print("In[82]")

inputs = features[:, 0:input_width, :]  
print(inputs)


# In[83]:
print("In[83]")

labels = features[:, label_start:, :] 
print(labels)


# In[84]:
print("In[84]")

print(column_indices)


# In[85]:
print("In[85]")

if label_columns is not None:
    for name in label_columns:
        labels = tf.stack([labels[:, :, column_indices[name]]], axis=-1)
        
print(labels)

# 查看所有训练样本调用split_window()方法后的结果：
# In[86]:
print("In[86]")

ds = ds.map(single_window.split_window)
print(ds)

# 绘图展示结果
# In[87]:
print("In[87]")

single_window.plot()


# In[88]:
print("In[88]")

plot_col = 'T'
max_subplots = 3
model = None


# In[89]:
print("In[89]")

inputs, labels = single_window.example
plot_col_index = column_indices[plot_col]  
max_n = min(max_subplots, len(inputs)) 

print(inputs)
print('----------------------------------分界线----------------------------------------')
print(labels)
print('----------------------------------分界线----------------------------------------')
print(plot_col_index)


# In[90]:
print("In[90]")

for n in range(max_n):
   
    plt.subplot(3, 1, n+1)       
    plt.ylabel(f'标准化后的 {plot_col}', fontproperties=font_set) 
    
    plt.plot(input_indices, inputs[n, :, plot_col_index], label='Inputs', marker='.', zorder=-10)

   
    if label_columns:  
        label_col_index = label_columns_indices.get(plot_col, None)
   
    else:
        label_col_index = plot_col_index

    if label_col_index is None:
        continue

    
    plt.scatter(label_indices, labels[n, :, label_col_index],edgecolors='k', label='Labels', c='green', s=64) 
            
    
    if model is not None:
        predictions = model(inputs)
        plt.scatter(label_indices, predictions[n, :, label_col_index], marker='X', edgecolors='k', 
                    label='Predictions', c='orange', s=64)

    if n == 0:
        plt.legend()
    plt.xlabel('Time [h]')
plt.show()

# **附注：**
# 附注:
# 以上预测为多步多变量预测，实际上还存在多步单变量预测、单步单变量预测、单步多变量预测
# 注1： 单变量与多变量的区别
# 要预测的天气特征的个数如果为1 (例如仅仅预测未来6小时的温度) ,则定义为单变量预测;
# 要预测的天气特征的个数如果大于1 （例如预测未来6个小时的天气特征，包含温度，风速等等)，则定义为多变量预测。
# 单变量预测的实现：改变数据窗口(label_columns=[T]);改变最后一层输出层，控制输出数据形状为(32, 6, 1)
# 注2： 单步与多步的区别
# 要预测的时间与输入的历史时间间隔了多少时间(shift)：shift>1 定义为多步，shift=1定义为单步
# 单步预测的实现：改变数据窗口（shift=1）；根据数据窗口的形状构建网络架构。
# 为了更好区别,附注中以单步单变量预测为例进行模型预测,并引申了除CNN和LSTM以外的多种模型(baseline, linear, dense)
# ①Baseline基准模型
# In[91]:
print("In[91]")

wide_window = WindowGenerator(
    input_width=6, label_width=6, shift=1,
    label_columns=['T'])

print(wide_window)


# In[92]:
print("In[92]")

for example_inputs, example_labels in wide_window.createTrainSet.take(1):
    print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
    print(f'Labels shape (batch, time, features): {example_labels.shape}')

# 构建Baseline模型
# In[93]:
print("In[93]")

class Baseline(tf.keras.Model):
   
    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index
    
    def call(self, inputs):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]


# In[94]:
print("In[94]")

baseline=Baseline(label_index=column_indices['T'])
baseline.compile(loss=tf.losses.MeanSquaredError(),
                 metrics=[tf.metrics.MeanAbsoluteError()])


# In[95]:
print("In[95]")

print('Input shape:', wide_window.example[0].shape)
print('Output shape:', baseline(wide_window.example[0]).shape)

# 实例化并评估Baseline模型，输出评估结果到performance字典里
# In[96]:
print("In[96]")

baseline = Baseline(label_index=column_indices['T'])

baseline.compile(loss=tf.losses.MeanSquaredError(),
                 metrics=[tf.metrics.MeanAbsoluteError()])

val_performance = {}
performance = {}

val_performance['Baseline'] = baseline.evaluate(wide_window.createValSet)
performance['Baseline'] = baseline.evaluate(wide_window.createTestSet)

# ②linear线性模型构建linear模型
# In[97]:
print("In[97]")

linear = tf.keras.Sequential([
    
    tf.keras.layers.Dense(units=1)
])


# In[98]:
print("In[98]")

print('Input shape:', wide_window.example[0].shape)
print('Output shape:', linear(wide_window.example[0]).shape)

# 上述结果解释:经过线性模型的神经网络,第三个维度被压缩为预测值的一个特征(温度)
# 编译和训练linear模型
# In[99]:
print("In[99]")

history = compile_and_fit(linear, wide_window)

# 上述结果解释:在第13轮的时候停止训练,模型已经达到目前最佳,停止改善
# 评估linear模型,输出评估结果到performance字典里
# In[100]:
print("In[100]")

val_performance['Linear'] = linear.evaluate(wide_window.createValSet)

performance['Linear'] = linear.evaluate(wide_window.createTestSet, verbose=0)

# ③dense模型
# 构建dense模型
# In[101]:
print("In[101]")

dense = tf.keras.Sequential([
  
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=1)
])


# 训练、评估dense模型

# In[102]:
print("In[102]")

history = compile_and_fit(dense, wide_window)

val_performance['Dense'] = dense.evaluate(wide_window.createValSet)
performance['Dense'] = dense.evaluate(wide_window.createTestSet, verbose=0)

# ④CNN卷积神经网络
# 构建数据窗口
# In[103]:
print("In[103]")

CONV_WIDTH = 3

conv_window = WindowGenerator(
    input_width=CONV_WIDTH,
    label_width=1,
    shift=1,
    label_columns=['T'])

print(conv_window)

# 构建CNN模型
# In[104]:
print("In[104]")

conv_model = tf.keras.Sequential([
    
    tf.keras.layers.Conv1D(filters=32,
                           kernel_size=(CONV_WIDTH,),
                           activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
    
])

# 训练和评估CNN模型
# In[105]:
print("In[105]")

history = compile_and_fit(conv_model, conv_window)

val_performance['Conv'] = conv_model.evaluate(conv_window.createValSet)
performance['Conv'] = conv_model.evaluate(conv_window.createTestSet, verbose=0)

# ⑤LSTM模型
# 构建LSTM模型
# In[106]:
print("In[106]")

lstm_model = tf.keras.models.Sequential([
    
    tf.keras.layers.LSTM(32, return_sequences=True),
    
    tf.keras.layers.Dense(units=1)
])


# In[107]:
print("In[107]")

print('Input shape:', wide_window.example[0].shape)
print('Output shape:', lstm_model(wide_window.example[0]).shape)

# 编译训练、评估LSTM模型
# In[108]:
print("In[108]")

history = compile_and_fit(lstm_model, wide_window)

val_performance['LSTM'] = lstm_model.evaluate(wide_window.createValSet)
performance['LSTM'] = lstm_model.evaluate(wide_window.createTestSet, verbose=0)

# 展示各个模型的训练评估效果performance
# In[109]:
print("In[109]")

print(performance)

# 绘制验证集和测试集在各个模型训练下的平均绝对误差(MAE)对比图
# In[110]:
print("In[110]")

x = np.arange(len(performance))
width = 0.3
metric_name = 'mean_absolute_error'
metric_index = lstm_model.metrics_names.index('mean_absolute_error')
val_mae = [v[metric_index] for v in val_performance.values()]
test_mae = [v[metric_index] for v in performance.values()]

plt.ylabel('mean_absolute_error [T (degC), normalized]')

plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(ticks=x, labels=performance.keys(),
           rotation=45)
_ = plt.legend()
print(performance)  # 以便于定位问题。
print(val_performance)  # 以便于定位问题。
plt.show()


# 上述结果解释：LSTM的模型训练效果最佳