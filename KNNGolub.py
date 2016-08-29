# coding=utf-8
#IMPORTATE: a linha 1 é extremamente necessaria para a realizacao da leitura do arquivo

#imports
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as matplot

# Funcao para plotar no grafico, recebe duas listas.
def plotar(k, acuracia):
	
    matplot.plot(k, acuracia)
    matplot.grid(True)
    matplot.xlabel('Valores de k')
    matplot.ylabel('Valores de acuracia')
    matplot.savefig('GraficoKNN.pdf')
    matplot.show()

#main
if __name__ == '__main__':
	
	#Lendo o arquivo inteiro do dataset, inclusive com as classes
	ds = np.loadtxt('golub.csv', delimiter=",")

	cancer = ds[:,3051]

	dataset = ds[:,:-1]	

	#Atribuir a parte de treino (2/3 da dataset) e a parte de teste (1/3 da dataset)	
	experimentoTreino = dataset[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,30,31,32,33,34,35,36,37],:]
	experimentoTeste = dataset[[17,18,19,20,21,22,23,24,25,26,27,28,29],:] #desta forma o de teste terá valores da classe 0 e 1

	cancerTreino = cancer[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,30,31,32,33,34,35,36,37]]
	cancerTeste = cancer[[17,18,19,20,21,22,23,24,25,26,27,28,29]]

	imp = Imputer()
	imp.fit(experimentoTreino)

	experimentoTreinoNpp = imp.transform(experimentoTreino)
	experimentoTesteNpp = imp.transform(experimentoTeste)

	acuracia = []
	k = []

	for i in range(1, 6): #de k=1 ate k=5
		# realizacao do knn, depois faz alguma funcaozinha automatica pra passar varios valores de vizinhos
		knn = KNeighborsClassifier(n_neighbors=i)
		knn.fit(experimentoTreinoNpp,cancerTreino)
		predito = knn.predict(experimentoTesteNpp)
		acc = accuracy_score(cancerTeste, predito)
		acuracia.append(acc)
		k.append(i)		

		print "Valor de k: ", i
		print(acc)

	# Grafico mostrando
	plotar(k, acuracia)
	

	

