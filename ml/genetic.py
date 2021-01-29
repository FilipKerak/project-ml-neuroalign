import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from evolutionary_keras.optimizers import CMA
from deepevolution import wrap_keras
from keras.layers import Dropout

wrap_keras()
def transform(seq):
    ret=np.zeros(600).reshape((600))
    dc={'A':1,'T':2,'G':3,'C':4}
    for i in range(len(seq)):
        ret[i]=dc[seq[i]]
    return ret
def evaluate(model,X1,X2,y1,y2):
    suma=len(X1)
    spravne=0
    a=model.predict(X1)
    b=model.predict(X2)
    odchylka=[]
    odchylka_abs=[]
    for i in range(len(a)):
        if (a[i]>b[i])==(y1[i]>y2[i]):
            spravne+=1
            odchylka.append(abs(a[i]-y1[i])/y1[i])
            odchylka.append(abs(a[i]-y2[i])/y2[i])
            odchylka_abs.append(abs(a[i]-y1[i]))
            odchylka_abs.append(abs(a[i]-y2[i]))
    spr=round(spravne/suma*100,2)
    print(f'Spravne by sa dalo priradit organizmu {spr}% citani.')
    relativna=round(np.mean(odchylka)*100,2)
    print(f'Priemerna relativna odchylka je {relativna}%.')
    potrebna=round(np.mean(abs(y1-y2)/y1)*100,2)
    print(f'Aby bolo mozne uvazovat o dobrom priradeni potrebujeme odchylku menej nez {potrebna}%.')
    return spravne/suma

def load_data(file):
    X=[]
    y=[]
    with open(file,'r') as inp:
        for r in inp:
            h=r.split()
            orig=transform(h[0])
            a=transform(h[1])
            sa=int(float(h[3]))

            
            b=transform(h[2])
            sb=int(float(h[4]))
            
            X.append([np.concatenate((orig,a,[1])),np.concatenate((orig,b,[1]))])
            y.append((sa,sb))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_tr=[]
    y_tr=[]
    for i in  range(len(X_train)):
        X_tr.append(X_train[i][0])
        y_tr.append(y_train[i][0])
        X_tr.append(X_train[i][1])
        y_tr.append(y_train[i][1])
    X_test1=[i[0] for i in X_test]
    y_test1=[i[0] for i in y_test]
    X_test2=[i[1] for i in X_test]
    y_test2=[i[1] for i in y_test]
    return np.array(X_tr),np.array(X_test1),np.array(X_test2),np.array(y_tr),np.array(y_test1),np.array(y_test2)

def side_model(X,y):
    md=Sequential()
    md.add(Dense(600,input_dim=len(X[0])))
    md.add(Dropout(0.2))
    
    md.add(Dense(300,activation='elu'))
    md.add(Dropout(0.1))
    
    md.add(Dense(150,activation='elu'))
    md.add(Dropout(0.1))

    md.add(Dense(50,activation='elu'))
    md.add(Dropout(0.1))

    md.add(Dense(25,activation='elu'))
    md.add(Dropout(0.05))
    
    md.add(Dense(10))    
    md.add(Dense(1))
    md.compile(loss='mae')
    '''for i in [50,100,200,400,800,1600,3200]:
        todo=list(range(len(X)))
        np.random.shuffle(todo)
        md.fit(X[todo],y[todo],epochs=3,batch_size=i,validation_split=0.1)'''
    md.fit_evolve(X, y, max_generations=30, population=10, top_k=3,mutation_std=0.05,mutation_rate=0.4)
    return md

X_train,X_test1,X_test2,y_train,y_test1,y_test2=load_data('data_variable')
model=side_model(X_train,y_train)
print(model.evaluate(X_test1,y_test1))
print(model.evaluate(X_test2,y_test2))
evaluate(model,X_test1,X_test2,y_test1,y_test2)
