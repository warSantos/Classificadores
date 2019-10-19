from sys import argv
from utils import carregar_base, seleciona_variaveis_SFS
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

def busca_melhores_features(model, dados):
    n_features = 30
    seleciona_variaveis_SFS(model, dados, n_features)

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
