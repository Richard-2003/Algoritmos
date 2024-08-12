import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#read dates
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

train.head(3)
train.columns #see the dates in the colums 
train[['GrLivArea', 'SalePrice']].head()   
train.plot.scatter(x='GrLivArea', y='SalePrice') #make the graphic
#plt.show() #show graphics

#create a line in the graphic
train['GrLivArea'].min()

#parameters from the line, equiation of the line
w = 125
b = 0

x = np.linspace(0,train['GrLivArea'].max(),100)
y = w*x + b

#graphics
train.plot.scatter(x='GrLivArea', y='SalePrice')
plt.plot(x,y, '-r')
plt.ylim(0,train['SalePrice'].max()*1.1)
#plt.show()

#calculas las predicciones 
train['pred'] = train['GrLivArea']*w+b

#calcular la funcion de error
train['diff'] = train['pred']-train['SalePrice']
train['cuad'] = train['diff']**2
train.head()
train['cuad'].mean()
#grid de la funcion de error basado en m que es w, b = 0 se tomara 0 para no darle mas valores a b y no graficar una superficie
w = np.linspace(50,200,50)
grid_error = pd.DataFrame(w, columns=['w'])
grid_error.head()
#print(grid_error)
def sum_error(w, train):
    b=0
    train['pred'] = train['GrLivArea']*w+b
    train['diff'] = train['pred']-train['SalePrice']
    train['cuad'] = train['diff']**2
    return(train['cuad'].mean())

grid_error['error'] = grid_error['w'].apply(lambda x: sum_error(x, train=train))
grid_error.head()

grid_error.plot(x='w', y='error')
plt.show()

#use skelar to know the optimal values
#define input and output

X_train = np.array(train['GrLivArea']).reshape((-1,1))
Y_train = np.array(train['SalePrice'])

#create models 
model = LinearRegression(fit_intercept=False)
model.fit(X_train, Y_train)
#print parameters
print(f"intercepto (b): {model.intercept_} ")
print(f"pendiente (w): {model.coef_} ")



