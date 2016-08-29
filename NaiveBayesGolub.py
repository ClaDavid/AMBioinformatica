# -*- coding: utf-8 -*-
from sklearn.preprocessing import Imputer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as matplot
import numpy as np

def plotar(k, acuracia):
    matplot.plot(k, acuracia)
    matplot.grid(True)
    matplot.xlabel('Quantidade de dados para treinamento (%)')
    matplot.ylabel('Desempenho')
    matplot.savefig('GraficoNaiveBayes.pdf')
    matplot.show()

#main
if __name__ == '__main__':
    
    acuracia = []
	#Lendo o arquivo inteiro do dataset, inclusive com as classes
    ds = np.loadtxt('golub.csv', delimiter=",")

    cancer = ds[:,3051]

    dataset = ds[:,:-1]	

	#Atribuir a parte de treino (2/3 da dataset) e a parte de teste (1/3 da dataset)	
    experimentoTreino = dataset[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,30,31,32,33,34,35,36,37],:]
    experimentoTeste = dataset[[17,18,19,20,21,22,23,24,25,26,27,28,29],:] #desta forma o de teste terá valores da classe 0 e 1

    cancerTreino = cancer[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,30,31,32,33,34,35,36,37]]
    cancerTeste = cancer[[17,18,19,20,21,22,23,24,25,26,27,28,29]]
    
    for i in range(2):

        imp = Imputer()
        
        # trocamos os experimentos usados para treinamento e teste        
        gnb = GaussianNB()
        if(i==0):
            imp.fit(experimentoTreino)       
            experimentoTreinoNpp = imp.transform(experimentoTreino)
            experimentoTesteNpp = imp.transform(experimentoTeste)
            gnb.fit(experimentoTreinoNpp,cancerTreino)
            predito = gnb.predict(experimentoTesteNpp)
            acc = accuracy_score(cancerTeste, predito)
        else:
            imp.fit(experimentoTeste)
            experimentoTreinoNpp = imp.transform(experimentoTeste)
            experimentoTesteNpp = imp.transform(experimentoTreino)
            gnb.fit(experimentoTreinoNpp,cancerTeste)
            predito = gnb.predict(experimentoTesteNpp)
            acc = accuracy_score(cancerTreino, predito)

        acuracia.append(acc)

        print(acc)
        
    k = [0.66, 0.34]
    plotar(k, acuracia)
    
