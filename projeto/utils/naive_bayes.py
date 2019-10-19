from sys import argv
from utils import carregar_base, seleciona_variaveis_SFS
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectFromModel

def busca_melhores_features(model, dados):
    n_features = 31
    seleciona_variaveis_SFS(model, dados, n_features)

if __name__=='__main__':
    base = carregar_base(argv[1])
    nb_modelo = GaussianNB()
    busca_melhores_features(nb_modelo, base)