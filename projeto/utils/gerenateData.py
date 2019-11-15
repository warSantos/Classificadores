from sys import argv
from utils import carregar_base, seleciona_variaveis_SFS
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from classificadores import busca_melhores_features
import matplotlib.pyplot as plt
import pickle
'''
if __name__=='__main__':
    base = carregar_base('../../dados/base_dados.csv')
    arqNb = open('arqNb.txt', 'w')
    arqRfc = open('arqRfc.txt', 'w')
    arqSvm = open('arqSvm.txt', 'w')



    rfc = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')
    SVM = SVC(C=1, gamma='scale')
    nb = GaussianNB()

    scores_NB = busca_melhores_features(nb, base)
#    pickle.dump(scores_NB, arqNb)
    arqNb.write(str(scores_NB)+'\n')

    scores_RFC = busca_melhores_features(rfc, base)
    arqRfc.write(str(scores_RFC) + '\n')

    scores_SVM = busca_melhores_features(SVM, base)

    arqSvm.write(str(scores_SVM) + '\n')


    fclose(arqN)
'''
if __name__=='__main__':
    base = carregar_base('../../dados/base_dados.csv')
    arqNb = open('arqNb2.txt', 'w')
    #arqRfc = open('arqRfc.txt', 'w')
    #arqSvm = open('arqSvm.txt', 'w')



#    rfc = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')
    #SVM = SVC(C=1, gamma='scale')
    nb = GaussianNB()

    scores_NB = busca_melhores_features(nb, base)
#    pickle.dump(scores_NB, arqNb)
    arqNb.write(str(scores_NB)+'\n')

    scores_RFC = busca_melhores_features(rfc, base)
    arqRfc.write(str(scores_RFC) + '\n')

    scores_SVM = busca_melhores_features(SVM, base)

    arqSvm.write(str(scores_SVM) + '\n')


    fclose(arqN)
