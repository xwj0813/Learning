# 逻辑回归（一） 龙芯陈博客
#在数据上做逻辑回归分类的例子

from numpy import loadtxt,where
from pylab import scatter,show ,legend,xlabel,ylabel

#下载数据集
data =loadtxt('data1.txt',delimiter=',')
X=data[:,0:2]
y=data[:,2]

pos=where(y==1)
neg=where(y==0)
scatter(X[pos,0],X[pos,1],marker='o',c='b')
scatter(X[neg,0],X[neg,1],marker='x',c='r')
xlabel('Feature1/Exam 1 sorce')
ylabel('Feature2/Exam 2 score')
legend(['Fail1','Pas'])

show()
