# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 16:42:07 2018

@author: Yuzheng.Cui
"""

#1、numpy
#numpy的主要对象是同类型元素的多维数组，这是一个所有的元素都是一种类型、通过一个正整数元组索引的元素表格(通常是元素是数字)。
#在NumPy中维度(dimensions)叫做轴(axes)，轴的个数叫做秩(rank)。
#如下示例中，我们用arr表示numpy数组对象，同时需要做如下引入：
import numpy as np
#1）数据的输入和输出
np.loadtxt(‘file.txt’)#：从文件”file.txt”中导入数据并返回ndarray对象
np.genfromtxt(‘file.csv’,delimiter=’,’)#：从文件”file.csv”中导入数据并返回ndarray对象
np.savetxt(‘file.txt’,arr,delimiter=’’)#：将数组保存为”file.txt”文件
np.savetxt(‘file.csv’,arr,delimiter=’,’)#：将数组保存为”file.csv”文件
#2）创建数组
np.array([1,2,3,4,5,6])#：创建一维数组并返回ndarray对象
np.array([(1,2,3),(,4,5,6)])#：创建二维数组并返回ndarray对象
np.zeros(3)#：创建长度为3的一维数组且数组元素均为0，返回ndarray对象
np.ones((3,4))#：创建3×4的二维数组且数组元素均为1，返回ndarray对象
np.eye(5)#：创建一个5×5的二维数组且对角线元素均为1，其他元素均为0，返回ndarray对象
np.linspace(0,100,6)#：创建一维数组且其元素为0到100之间的6等分数字，返回ndarray对象
np.arange(0,10,3)#：创建一维数组且其元素为以0为起点，步长为3，直到小于10的所有数值，返回ndarray对象
np.full((2,3),8)#：创建2×3二维数组且其所有元素均为8，返回ndarray对象
np.random.rand(4,5)#：创建4×5二维数组且其所有元素为0-1之间的随机小数，返回ndarray对象
np.random.rand(6,7)*100#：创建6×7二维数组且其所有元素为0-100之间的随机小数，返回ndarray对象
np.random.randint(5,size=(2,3))#：创建2×3二维数组且其所有元素为0-4之间的随机整数，返回ndarray对象
#3）获取数组属性
arr.size#：返回数组arr中元素的个数
arr.shape#：返回数组arr的维数（行数和列数）
arr.dtype#：返回数组arr中元素的类型
arr.astype(dtype)#：强制转换数组arr中元素的类型为dtype
arr.tolist()#：将数组arr强制转换为Python列表
np.info(np.eye)#：查看关于np.eye的文档
#4）复制、排序和调整数组
np.copy(arr)#：复制数组arr，返回ndarray对象
arr.view(dtype)#：以指定dtype类型为数组arr每个元素建立视图
arr.sort()#：排序数组，返回ndarray对象
arr.sort(axis=0)#：按指定的轴排序数组，返回ndarray对象
two_d_arr.flatten()#：将二维数组two_d_arr转换为一维数组，返回ndarray对象
arr.T#：返回数组arr的转置（行与列互换）
arr.reshape(3,4)#：将数组arr调整为3行4列，并不改变数据，返回ndarray对象
arr.resize((5,6))#：将数组arr调整为5行6列，空值用0填充，返回ndarray对象
#5）增加、删除元素
np.append(arr,values)#：为数组arr增加元素values，返回ndarray对象
np.insert(arr,2,values)#：为数组arr在索引为2的元素之前插入元素values，返回ndarray对象
np.delete(arr,3,axis=0)#：删除数组arr第3行所有的元素，返回ndarray对象
np.delete(arr,4,axis=1)#：删除数组arr第4列所有的元素，返回ndarray对象
#6）组合与拆分
np.concatenate((arr1,arr2),axis=0)#：将数组arr2按行序拼接在数组arr1的末尾，返回ndarray对象
np.concatenate((arr1,arr2),axis=1)#：将数组arr2按列序拼接在数组arr1的右侧，返回ndarray对象
np.split(arr,3)#：将数组arr拆分为3个子数组，返回ndarray对象
np.hsplit(arr,5)#：将数组arr从索引为5的元素后水平拆分，返回ndarray对象
#7）数组的索引、切片、子集
arr[5]#：返回数组arr的索引为5的元素
arr[2,5]#：返回二维数组arr行索引为2、列索引为5的元素
arr[1]=4#：将数组arr索引为1的元素赋值为4
arr[1,3]=10#：将二维数组arr行索引为1、列索引为3的元素赋值为10
arr[0:3]#：返回数组arr索引从0开始的3个元素，如果arr为二维数组则返回索引从0开始的3行全部元素
arr[0:3,4]#：返回二维数组arr中索引从0开始的3个行中4列的所有元素
arr[:2]#：返回数组arr索引从0开始的2个元素，如果arr为二维数组则返回索引从0开始的2行全部元素
arr[:,1]#：返回数组arr中第1列中所有行的元素
arr<5#：返回布尔型数组，其中数组arr中小于5的元素为True，大于等于5的元素为False
(arr1<3)&(arr2>5)#：返回布尔型数组
~arr#：返回布尔型数组的逆（True变为False、False变为True）
arr[arr<5]#：返回数组中小于5的元素
#8）标量数学
np.add(arr,1)#：为数组arr的每个元素都加1，并返回ndarray对象
np.subtract(arr,2)#：为数组arr的每个元素都减2，并返回ndarray对象
np.multiply(arr,3)#：为数组arr的每个元素都乘3，并返回ndarray对象
np.divide(arr,4)#：为数组arr的每个元素都除4，并返回ndarray对象
np.power(arr,5)#：为数组arr的每个元素都计算5次方，并返回ndarray对象
#9）向量数学
np.add(arr1,arr2)#：将数组arr1和arr2的对应元素相加，并返回ndarray对象
np.subtract(arr1,arr2)#：将数组arr1和arr2的对应元素相减，并返回ndarray对象
nu.multiply(arr1,arr2)#：将数组arr1和arr2的对应元素相乘，并返回ndarray对象
np.divide(arr1,arr2)#：将数组arr1和arr2的对应元素相除，并返回ndarray对象
np.power(arr1,arr2)#：将数组arr2中的元素作为数组arr1中对应元素的指数，并返回ndarray对象
np.array_equal(arr1,arr2)#：判断数组arr1和arr2中对应元素是否相同，并返回ndarray对象的布尔型数组
np.sqrt(arr)#：计算数组arr中每个元素的平方根，并返回ndarray对象
np.sin(arr)#：计算数组arr中每个元素的正弦值(sin())，并返回ndarray对象
np.log(arr)#：计算数组arr中每个元素的自然对数，并返回ndarray对象
np.abs(arr)#：计算数组arr中每个元素的绝对值，并返回ndarray对象
np.ceil(arr)#：对数组arr中每个元素向上取整，并返回ndarray对象
np.floor(arr)#：对数组arr中每个元素向下取整，并返回ndarray对象
np.round(arr)#：对数组arr中每个元素四舍五入取整，并返回ndarray对象
#10）统计数值
np.mean(arr,axis=0)#：计算数组arr中指定轴的均值，并返回ndarray对象
arr.sum()#：返回数组arr中所有元素的和
arr.min()#：返回数组arr中最小的元素
arr.max(axis=0)#：返回数组arr中指定轴的最大元素
np.var(arr)#：返回数组arr中所有元素的方差
np.std(arr,axis=1)#：返回数组arr中指定轴的标准差
arr.corrcoef()#：返回数组arr中所有元素的相关系数
#2、Pandas
#Pandas提供了使我们能够快速便捷地处理结构化数据的大量数据结构和函数，其中两个主要数据结构是DataFrame和Series。
#此处，我们以df代表任意的Pandas DataFrame对象
#，s代表任意的Pandas Series对象，同时我们需要做如下的引入：
import pandas as pd
import numpy as np

#1）导入数据
pd.read_csv(filename)#：从CSV文件导入数据
pd.read_table(filename)#：从限定分隔符的文本文件导入数据
pd.read_excel(filename)#：从Excel文件导入数据
pd.read_sql(query, connection_object)#：从SQL表/库导入数据
pd.read_json(json_string)#：从JSON格式的字符串导入数据
pd.read_html(url)#：解析URL、字符串或者HTML文件，抽取其中的tables表格
pd.read_clipboard()#：从你的粘贴板获取内容，并传给read_table()
pd.DataFrame(dict)#：从字典对象导入数据，Key是列名，Value是数据
#2）导出数据
df.to_csv(filename)#：导出数据到CSV文件
df.to_excel(filename)#：导出数据到Excel文件
df.to_sql(table_name, connection_object)#：导出数据到SQL表
df.to_json(filename)#：以Json格式导出数据到文本文件
df.to_html(filename)#：导出数据到HTML文件
df.to_clipboard()#：导出数据到粘贴板
#3）创建测试对象
pd.DataFrame(np.random.rand(20,5))#：创建20行5列的随机数组成的DataFrame对象
pd.Series(my_list)#：从可迭代对象my_list创建一个Series对象
df.index = pd.date_range('1900/1/30', periods=df.shape[0])#：增加一个日期索引
#4）查看、检查数据
df.head(n)#：查看DataFrame对象的前n行
df.tail(n)#：查看DataFrame对象的最后n行
df.shape()#：查看行数和列数
df.info()#：查看索引、数据类型和内存信息
df.describe()#：查看数值型列的汇总统计
s.value_counts(dropna=False)#：查看Series对象的唯一值和计数
df.apply(pd.Series.value_counts)#：查看DataFrame对象中每一列的唯一值和计数
#5）数据选取
df[col]#：根据列名，并以Series的形式返回列
df[[col1, col2]]#：以DataFrame形式返回多列
s.iloc[0]#：按位置选取数据
s.loc['index_one']#：按索引选取数据
df.iloc[0,:]#：返回第一行
df.iloc[0,0]#：返回第一列的第一个元素
#6）数据清理
df.columns = ['a','b','c']#：重命名列名
pd.isnull()#：检查DataFrame对象中的空值，并返回一个Boolean数组
pd.notnull()#：检查DataFrame对象中的非空值，并返回一个Boolean数组
df.dropna()#：删除所有包含空值的行
df.dropna(axis=1)#：删除所有包含空值的列
df.dropna(axis=1,thresh=n)#：删除所有小于n个非空值的行
df.fillna(x)#：用x替换DataFrame对象中所有的空值
s.fillna(s.mean())#：用均值填充所有的na
s.astype(float)#：将Series中的数据类型更改为float类型
s.replace(1,'one')#：用‘one’代替所有等于1的值
s.replace([1,3],['one','three'])#：用'one'代替1，用'three'代替3
df.rename(columns=lambda x: x + 1)#：批量更改列名
df.rename(columns={'old_name': 'new_ name'})#：选择性更改列名
df.set_index('column_one')#：更改索引列
df.rename(index=lambda x: x + 1)#：批量重命名索引
#7）数据处理：Filter、Sort和GroupBy
df[df[col] > 0.5]#：选择col列的值大于0.5的行
df[(df[col]>0.5)&(df[col]<0.7)]#：选择col列的值大于0.5且小于0.7的行
df.sort_values(col1)#：按照列col1排序数据，默认升序排列
df.sort_values(col2, ascending=False)#：按照列col1降序排列数据
df.sort_values([col1,col2], ascending=[True,False])#：先按列col1升序排列，后按col2降序排列数据
df.groupby(col)#：返回一个按列col进行分组的Groupby对象
df.groupby([col1,col2])#：返回一个按多列进行分组的Groupby对象
df.groupby(col1)[col2].mean()#：返回按列col1进行分组后，列col2的均值
df.pivot_table(index=col1, values=[col2,col3], aggfunc=max)#：创建一个按列col1进行分组，并计算col2和col3的最大值的数据透视表
df.groupby(col1).agg(np.mean)#：返回按列col1分组的所有列的均值
data.apply(np.mean)#：对DataFrame中的每一列应用函数np.mean
data.apply(np.max,axis=1)#：对DataFrame中的每一行应用函数np.max
#8）数据合并
df1.append(df2)#：将df2中的行添加到df1的尾部
pd.concat([df1, df2],axis=1)#：将df2中的列添加到df1的尾部
df1.join(df2,on=col1,how='inner')#：对df1的列和df2的列执行SQL形式的join
pd.merge(df1,df2)#：合并df1和df2
#9）数据统计
df.describe()#：查看数据值列的汇总统计
df.mean()#：返回所有列的均值
df.corr()#：返回列与列之间的相关系数
df.count()#：返回每一列中的非空值的个数
df.max()#：返回每一列的最大值
df.min()#：返回每一列的最小值
df.median()#：返回每一列的中位数
df.std()#：返回每一列的标准差

#3、Matplotlib
# Matplotlib是一个用于创建高质量图表的绘图包（主要是2D图形）。如果需要绘制3D图形，需要使用mplot3d的插件。
#1）准备数据
#一维数据
import numpy as np
x = np.linspace(0, 10, 100)
y = np.cos(x)
z = np.sin(x)
#二维数据或图像
data = 2 * np.random.random((10, 10))
data2 = 3 * np.random.random((10, 10))
Y, X = np.mgrid[-3:3:100j, -3:3:100j]
U = -1 - X**2 + Y
V = 1 + X - Y**2
from matplotlib.cbook import get_sample_data
img = np.load(get_sample_data('axes_grid/bivariate_normal.jpg'))
#2）创建绘图
import matplotlib.pyplot as plt
#图形(figure)
fig = plt.figure()
fig2 = plt.figure(figsize=plt.figaspect(2.0))
#坐标轴
fig.add_axes()
ax1 = fig.add_subplot(221) 
ax3 = fig.add_subplot(212)
fig3, axes = plt.subplots(nrows=2,ncols=2)
fig4, axes2 = plt.subplots(ncols=3)
#3）绘图的范例
#一维数据绘图
fig, ax = plt.subplots()
lines = ax.plot(x,y) 
ax.scatter(x,y) 
axes[0,0].bar([1,2,3],[3,4,5]) 
axes[1,0].barh([0.5,1,2.5],[0,1,2]) 
axes[1,1].axhline(0.45) 
axes[0,1].axvline(0.65) 
ax.fill(x,y,color='blue') 
ax.fill_between(x,y,color='yellow') 
#向量绘图
axes[0,1].arrow(0,0,0.5,0.5) 
axes[1,1].quiver(y,z) 
axes[0,1].streamplot(X,Y,U,V) 
#数据分布图
ax1.hist(y) 
ax3.boxplot(y) 
ax3.violinplot(z) 
#绘二维数据图或图像
fig, ax = plt.subplots()
im = ax.imshow(img,cmap='gist_earth',interpolation='nearest',vmin=-2,vmax=2) 
axes2[0].pcolor(data2) 
axes2[0].pcolormesh(data) 
CS = plt.contour(Y,X,U) 
axes2[2].contourf(data1) 
axes2[2]= ax.clabel(CS) 
#4）定制化绘图
#颜色
plt.plot(x, x, x, x**2, x, x**3)
ax.plot(x, y, alpha = 0.4)
ax.plot(x, y, c='k')
fig.colorbar(im, orientation='horizontal')
im = ax.imshow(img,cmap='seismic')
#标记
fig, ax = plt.subplots()
ax.scatter(x,y,marker=".")
ax.plot(x,y,marker="o")
#线型
plt.plot(x,y,linewidth=4.0)
plt.plot(x,y,ls='solid')
plt.plot(x,y,ls='--')
plt.plot(x,y,'--',x**2,y**2,'-.')
plt.setp(lines,color='r',linewidth=4.0)
#标注
ax.text(1,-2.1,'Example Graph',style='italic')
ax.annotate("Sine",xy=(8,0),xycoords='data',xytext=(10.5,0),textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="arc3"),)
#数字文本
plt.title(r'$sigma_i=15$', fontsize=20)
#坐标范围、比例缩放
ax.margins(x=0.0,y=0.1) 
ax.axis('equal') 
ax.set(xlim=[0,10.5],ylim=[-1.5,1.5]) 
ax.set_xlim(0,10.5) 
#标题
ax.set(title='An Example Axes',ylabel='Y-Axis',xlabel='X-Axis')  
ax.legend(loc='best') 
#标记
ax.xaxis.set(ticks=range(1,5),ticklabels=[3,100,-12,"foo"]) 
ax.tick_params(axis='y',direction='inout',length=10) 
#绘制子图
fig3.subplots_adjust(wspace=0.5,hspace=0.3,left=0.125,right=0.9,top=0.9,
bottom=0.1) 
fig.tight_layout() 
#坐标轴分区
ax1.spines['top'].set_visible(False) 
ax1.spines['bottom'].set_position(('outward',10)) 
#5）保存绘图
#保存图像
plt.savefig('foo.png')
#保存为透明的图像
plt.savefig('foo.png', transparent=True)
#6）显示绘图、清空和关闭窗口
plt.show()
plt.cla() Clear an axis
plt.clf() Clear the entire figure
plt.close() Close a window

#4、Scikit-learn
#Scikit-learn是用于机器学习的开源Python库，它集成了大量的机器学习算法，还包括数据预处理、较差验证和可视化算法等。
#1）加载数据
import numpy as np
X=np.random.random((10,5))
y=np.array(['M','M','F','F','M','F','M','M','F','F','F'])
X[X<0.7]=0
#2）训练样本和测试样本
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)
#3）数据预处理
#标准化（Standardization）
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
standardized_X = scaler.transform(X_train)
standardized_X_test = scaler.transform(X_test)
#归一化（Normalization）
from sklearn.preprocessing import Normalizer
scaler = Normalizer().fit(X_train)
normalized_X = scaler.transform(X_train)
normalized_X_test = scaler.transform(X_test)
#特征二值化（Binarization）
from sklearn.preprocessing import Binarizer
binarizer = Binarizer(threshold=0.0).fit(X)
binary_X = binarizer.transform(X)
#编码分类特征
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
y = enc.fit_transform(y)
#填充缺失值
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values=0, strategy='mean', axis=0)
imp.fit_transform(X_train)
#产生多项式特征
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(5)
poly.fit_transform(X)
#4）构建模型
#有监督机器学习-线性回归
from sklearn.linear_model import LinearRegression
lr = LinearRegression(normalize=True)
#有监督机器学习-支持向量机（SVM）
from sklearn.svm import SVC
svc = SVC(kernel='linear')
#有监督机器学习-朴素贝叶斯
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
#有监督机器学习-K近邻（KNN）
from sklearn import neighbors
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
#无监督机器学习-主成分分析（PCA）
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)
#无监督机器学习-K均值（K-Means）
from sklearn.cluster import KMeans
k_means = KMeans(n_clusters=3, random_state=0)
#5）拟合模型
#有监督学习
lr.fit(X, y)
knn.fit(X_train, y_train)
svc.fit(X_train, y_train)
#无监督学习
k_means.fit(X_train) 
pca_model = pca.fit_transform(X_train) 
#6）预测
#有监督机器学习评估
y_pred = svc.predict(np.random.random((2,5))) 
y_pred = lr.predict(X_test) 
y_pred = knn.predict_proba(X_test) 
#无监督机器学习评估
y_pred = k_means.predict(X_test) 
#7）模型检验
#分类模型检验标准
#（1）准确度分数
knn.score(X_test, y_test) 
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred) 
#（2）分类模型检验报告
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred)) 
#（3）混淆矩阵
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))
#回归模型检验标准
#（1）平均绝对误差
from sklearn.metrics import mean_absolute_error
y_true = [3, -0.5, 2]
mean_absolute_error(y_true, y_pred)
#（2）均方误差
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)
#（3）R2分数
from sklearn.metrics import r2_score
r2_score(y_true, y_pred)
#聚类模型检验标准
#（1）兰德指数（Adjusted Rand Index）
from sklearn.metrics import adjusted_rand_score
adjusted_rand_score(y_true, y_pred)
#（2）因子分布的同质性、均一性（Homogeneity）
from sklearn.metrics import homogeneity_score
homogeneity_score(y_true, y_pred)
#（3）V-Measure（均一性和完整性的加权平均）
from sklearn.cross_validation import cross_val_score
print(cross_val_score(knn, X_train, y_train, cv=4))
print(cross_val_score(lr, X, y, cv=2))
#（4）交叉验证（Cross-Validation）
from sklearn.cross_validation import cross_val_score
print(cross_val_score(knn, X_train, y_train, cv=4))
print(cross_val_score(lr, X, y, cv=2))
#8）模型优化
#（1）网格搜索（Grid Search，一种寻找机器学习模型最优参数的方法）
from sklearn.grid_search import GridSearchCV
params = {"n_neighbors": np.arange(1,3),"metric": ["euclidean", "cityblock"]}
grid = GridSearchCV(estimator=knn,param_grid=params)
grid.fit(X_train, y_train)
print(grid.best_score_)
print(grid.best_estimator_.n_neighbors)
#（2）随机参数最优化（Randomized Parameter Optimization）
from sklearn.grid_search import RandomizedSearchCV
params = {"n_neighbors": range(1,5),"weights": ["uniform", "distance"]}
rsearch = RandomizedSearchCV(estimator=knn,param_distributions=params,
cv=4,n_iter=8,random_state=5)
rsearch.fit(X_train, y_train)
print(rsearch.best_score_)
#9）简单示例
from sklearn import neighbors, datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
iris = datasets.load_iris()
X, y = iris.data[:, :2], iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy_score(y_test, y_pred)

#5、Keras
#Keras是一个强大的、易于使用的深度学习Python库，它提供了高性能深度神经网络的API，可非常方便的用于开发和评估深度学习模型。
#1）数据集（Data Sets）
#使用Keras开发深度学习模型时，应当将数据存储为Numpy数组或Numpy数组列表的形式。
#（1）Keras内置数据集
from keras.datasets import boston_housing,mnist,cifar10,imdb
(x_train,y_train),(x_test,y_test) = mnist.load_data()
(x_train2,y_train2),(x_test2,y_test2) = boston_housing.load_data()
(x_train3,y_train3),(x_test3,y_test3) = cifar10.load_data()
(x_train4,y_train4),(x_test4,y_test4) = imdb.load_data(num_words=20000)
num_classes = 10
#（2）其他数据集
from urllib.request import urlopen
data = np.loadtxt(urlopen("http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"),delimiter=",")
X = data[:,0:8]
y = data [:,8]
#2）数据预处理（Preprocessing）
#（1）顺序填充（Sequence Padding）
from keras.preprocessing import sequence
x_train4 = sequence.pad_sequences(x_train4,maxlen=80)
x_test4 = sequence.pad_sequences(x_test4,maxlen=80)
#（2）One-Hot编码（One-Hot Encoding）
from keras.utils import to_categorical
Y_train = to_categorical(y_train, num_classes)
Y_test = to_categorical(y_test, num_classes)
Y_train3 = to_categorical(y_train3, num_classes)
Y_test3 = to_categorical(y_test3, num_classes)
#（3）标准化/归一化（Standardization/Normalization）
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(x_train2)
standardized_X = scaler.transform(x_train2)
standardized_X_test = scaler.transform(x_test2)
#（4）训练集和测试集（Train and Test Sets）
from sklearn.model_selection import train_test_split
X_train5,X_test5,y_train5,y_test5=train_test_split(X,y,test_size=0.33,random_state=42)
#3）模型架构（Model Architecture）
#（1）序列模型（Sequential Model）
from keras.models import Sequential
model = Sequential()
model2 = Sequential()
model3 = Sequential()
#（2）多层感知器（Multilayer Perceptron）
#二项分类
from keras.layers import Dense
model.add(Dense(12,input_dim=8,kernel_initializer='uniform',activation='relu'))
model.add(Dense(8,kernel_initializer='uniform',activation='relu'))
model.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))
#多类别分类
from keras.layers import Dropout
model.add(Dense(512,activation='relu',input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='softmax'))
#回归分类
model.add(Dense(64,activation='relu',input_dim=train_data.shape[1]))
model.add(Dense(1))
#（3）卷积神经网络（Convolutional Neural Network，CNN）
from keras.layers import Activation,Conv2D,MaxPooling2D,Flatten
model2.add(Conv2D(32,(3,3),padding='same',input_shape=x_train.shape[1:]))
model2.add(Activation('relu'))
model2.add(Conv2D(32,(3,3)))
model2.add(Activation('relu'))
model2.add(MaxPooling2D(pool_size=(2,2)))
model2.add(Dropout(0.25))
model2.add(Conv2D(64,(3,3), padding='same'))
model2.add(Activation('relu'))
model2.add(Conv2D(64,(3, 3)))
model2.add(Activation('relu'))
model2.add(MaxPooling2D(pool_size=(2,2)))
model2.add(Dropout(0.25))
model2.add(Flatten())
model2.add(Dense(512))
model2.add(Activation('relu'))
model2.add(Dropout(0.5))
model2.add(Dense(num_classes))
model2.add(Activation('softmax'))
#（4）递归神经网络（Recurrent Neural Network，RNN）
from keras.klayers import Embedding,LSTM
model3.add(Embedding(20000,128))
model3.add(LSTM(128,dropout=0.2,recurrent_dropout=0.2))
model3.add(Dense(1,activation='sigmoid'))
#4）检查模型（Inspect Model）
model.output_shape Model output shape
model.summary() Model summary representation
model.get_config() Model configuration
model.get_weights() List all weight tensors in the model
#5）编译模型（Compile Model）
#MLP: Binary Classification
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#MLP: Multi-Class Classification
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
#MLP:Regression
model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
#Recurrent Neural Network
model3.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
#6）训练模型（Model Training）
model3.fit(x_train4,y_train4,batch_size=32,epochs=15,verbose=1,validation_data=(x_test4,y_test4))
#7）模型评估
score = model3.evaluate(x_test,y_test,batch_size=32)
#8）预测
model3.predict(x_test4, batch_size=32)
model3.predict_classes(x_test4,batch_size=32)
#9）保存和重新加载模型
from keras.models import load_model
model3.save('model_file.h5')
my_model = load_model('my_model.h5')
#10）模型优化（Model Fine-tuning）
#（1）参数最优化（Optimization Parameters）
from keras.optimizers import RMSprop
opt = RMSprop(lr=0.0001, decay=1e-6)
model2.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
#（2）提前终止（Early Stopping）
from keras.callbacks import EarlyStopping
early_stopping_monitor = EarlyStopping(patience=2)
model3.fit(x_train4,y_train4,batch_size=32,epochs=15,validation_data=(x_test4,y_test4),callbacks=[early_stopping_monitor])
#11）简单示例
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
data = np.random.random((1000,100))
labels = np.random.randint(2,size=(1000,1))
model = Sequential()
model.add(Dense(32,activation='relu',input_dim=100))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(data,labels,epochs=10,batch_size=32)
predictions = model.predict(data)
