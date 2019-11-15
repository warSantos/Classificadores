from sys import argv
from utils import carregar_base, seleciona_variaveis_SFS
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import matplotlib.pyplot as plt



def busca_melhores_features(model, dados):
    n_features = len(dados.columns)-40
    sfs = SFS(model, k_features=n_features, forward=True,
		floating=False, verbose=2, scoring='accuracy', cv=5,
		n_jobs=-1)
    seleciona_variaveis_SFS(model, dados, sfs)
#    lista = list()
#    for i in range(1, 31):
    #    lista.append(sfs.subsets_[i]['avg_score'])
#    return lista
    return sfs.subsets_
#    plt.plot(sfs.subsets_[:][1], range(1, 31))
#    plt.show()




if __name__=='__main__':
    if len(argv) < 3:
        print("Parametros incorretos.")
        print("ex.: usage: python3 classificadores.py ../../dados/base_dados.csv 1")
        exit(1)
    base = carregar_base(argv[1])
    modelo = ''
    if argv[2] == '1':
        print("Random Forest.")
        modelo = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')
    elif argv[2] == '2':
        print("SVM.")
        modelo = SVC(C=1, gamma='scale')
    elif argv[2] == '3':
        print("Naive Bayes.")
        modelo = GaussianNB()
    busca_melhores_features(modelo, base)
