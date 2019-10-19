from sys import argv
from utils import carregar_base, seleciona_variaveis_SFS, validacao_cruzada
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

def busca_melhores_features(model, dados):
    n_features = 30
    seleciona_variaveis_SFS(model, dados, n_features)

if __name__=='__main__':
    base = carregar_base(argv[1])
    svm_modelo = SVC(C=1, gamma='scale')
    busca_melhores_features(svm_modelo, base)