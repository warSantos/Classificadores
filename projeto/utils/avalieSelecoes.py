from scipy.io import arff
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, precision_score, accuracy_score,recall_score, f1_score, confusion_matrix
import numpy as np
from mlxtend.feature_selection import SequentialFeatureSelector as SFS


def RankList(dic_sfs, df):
    rankList = list()
    rankListName = list()
    name = df.columns
    x = dic_sfs.subsets_
    for i in range(1, 49):
        candidates = x[i]['feature_idx']
        for j in range(0, i):
            if(candidates[j] not in rankList):
                rankList.append(candidates[j])
                rankListName.append(name[candidates[j]])
    return rankListName


def sfsListSelect(model, df):
    n_features = len(df.columns)-1
    sfs = SFS(model, k_features=n_features, forward=True,
            floating=False, verbose=2, scoring='accuracy', cv=5,
            n_jobs=-1)
    X = df.drop('CLASS_LABEL',axis=1)
    Y = df['CLASS_LABEL'].astype('int32')
    sfs.fit(X, Y)
    return RankList(sfs, df)


def avaliaModelLista(rankList, model, df):
    X = df.drop('CLASS_LABEL',axis=1)
    Y = df['CLASS_LABEL'].astype('int32')
    y_pred = cross_val_predict(model, X, Y, cv=5)
    allMetrics = list()
    acuracy = list()
    allMetrics.append(calculateMetrics(y_pred, Y))
    acuracy.append(allMetrics[0][0])

    for feature in rankList:
        X = X.drop(feature, axis = 1)
        varNumber += 1
        allMetrics.append(calculateMetrics(y_pred, Y))
        acuracy.append(allMetrics[varNumber][0])

    return allMetrics, acuracy

def treeListSelect(df):
    X = df.drop('CLASS_LABEL',axis=1)
    Y = df['CLASS_LABEL'].astype('int32')
    rfc = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')
    rfc.fit(X, Y)
    columList = df.columns
    rankListRFC = list()
    featuresImportance = rfc.feature_importances_
    for i in range(len(featuresImportance)):
        index = np.argmax(featuresImportance)
        rankListRFC.append(columList[index])
        featuresImportance[index] = - 1000


def plotGrafico(acc_RFC, acc_Tree, acc_SVM, path):
    x = range(1, 49)

    plt.plot(x, acc_Tree, 'o', color = 'blue' )
    plt.plot(x, acc_Tree, label = 'Decision-Tree', color = 'blue')

    plt.plot(x, acc_RFC, 'o', color = 'red')
    plt.plot(x, acc_RFC, label = 'RFC', color = 'red')

    plt.plot(x, acc_SVM, 'o', color = 'green')
    plt.plot(x, acc_SVM, label = 'SVM',color = 'green')

    plt.xlabel('Número de variáveis')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.savefig(path)

def seleciona_variaveis_RFE_Metrics(model, dados, num_variaveis):
    X = dados.drop('CLASS_LABEL',axis=1).astype('int32')
    Y = dados['CLASS_LABEL'].astype('int32')
    rfe = RFE(model, step=1, n_features_to_select = num_variaveis).fit(X, Y)

    # gera lista de variaveis para serem retiradas
    featuresNames = X.iloc[0].index
    featuresDrops = list()

    for i in range(0,len(featuresNames)):
        if(not rfe.support_[i]):
            FeaturesDrops.append((i, featuresNames[i]))

    # gera novos conjuntos de treinamento excluindo as variaveis
    X_new = X
    for i in range(0, len(featuresDrops)):
        X_new = X_new.drop(featuresNames[i], axis = 1)
    return calculateMetrics(y_pred, Y)


if __name__=='__main__':

    path = '../../dados/Phishing_Legitimate_full.arff'
    data = arff.loadarff(path)
    df = pd.DataFrame(data[0])
    RFC = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')
    tree = tree.DecisionTreeClassifier()
    SVM = SVC(C=1, gamma='scale', kernel = 'linear')

    rankListRFC = sfsListSelect(RFC, df)
    rankListTree = sfsListSelect(tree, df)
    rankListSVM = sfsListSelect(SVM, df)

    allMetricsRFC, acc_RFC = avaliaModelLista(rankListRFC, RFC, df)
    allMetricsTree, acc_Tree = avaliaModelLista(rankListTree, tree, df)
    allMetricsSVM, acc_SVM = avaliaModelLista(rankListSVM, SVM, df)

    plotGrafico(acc_RFC, acc_Tree, acc_SVM, 'GraficoSVS.png')


    rfc1 = open('rfcsfs.txt', w)
    for e in allMetricsRFC:
        rfc1.write(str(e) + '\n')

    tree1 = open('treefs.txt', w)
    for e in allMetricsTree:
        tree1.write(str(e) + '\n')

    svm1 = open('svmsfs.txt', w)
    for e in allMetricsSVM:
        svm1.write(str(e) + '\n')


    # rfe
    allMetricsRFC2 = list()
    acc_RFC2 = list()
    for i in range(0, 48):
        k = seleciona_variaveis_RFE_Metrics(RFC, df, 48-i)
        allMetricsRFC2.append(k);
        acc_RFC2.append(allMetricsRFC2[i][0])

    allMetricsTree2 = list()
    acc_Tree2 = list()
    for i in range(0, 48):
        k = seleciona_variaveis_RFE_Metrics(tree, df, 48-i)
        allMetricsTree2.append(k);
        acc_Tree2.append(allMetricsTree2[i][0])

    allMetricsSVM2 = list()
    acc_SVM2 = list()
    for i in range(0, 48):
        k = seleciona_variaveis_RFE_Metrics(SVM, df, 48-i)
        allMetricsSVM2.append(k);
        acc_SVM2.append(allMetricsSVM2[i][0])

    plotGrafico(acc_RFC2, acc_Tree2, acc_SVM2, 'GraficoRFE.png')
    #salva em arquivos
    rfc2 = open('rfcrfe.txt', w)
    for e in allMetricsRFC2:
        rfc2.write(str(e) + '\n')

    tree2 = open('treerfe.txt', w)
    for e in allMetricsTree2:
        tree2.write(str(e) + '\n')

    svm2 = open('svmrfe.txt', w)
    for e in allMetricsSVM2:
        svm2.write(str(e) + '\n')

    # rank por random forest
    rankListImportanceRFC3 = treeListSelect(df)
    allMetricsRFC3, acc_RFC3 = avaliaModelLista(rankListImportanceRFC, RFC, df)
    allMetricsTree3, acc_Tree3 = avaliaModelLista(rankListImportanceRFC, tree, df)
    allMetricsSVM3, acc_SVM3 = avaliaModelLista(rankListImportanceRFC, SVM, df)

    plotGrafico(acc_RFC3, acc_Tree3, acc_SVM3, 'GraficoRFCSelect.png')


    rfc3 = open('rfcES.txt', w)
    for e in allMetricsRFC3:
        rfc3.write(str(e) + '\n')

    tree3 = open('treeES.txt', w)
    for e in allMetricsTree3:
        tree3.write(str(e) + '\n')

    svm3 = open('svmES.txt', w)
    for e in allMetricsSVM3:
        svm3.write(str(e) + '\n')
