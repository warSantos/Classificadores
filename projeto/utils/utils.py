import pandas as pd
from sys import argv, exit
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

def carregar_base(caminho):
	base_dados = pd.read_csv(caminho)
	return base_dados

def validacao_cruzada(X, Y, model):
    scores_dt = cross_val_score(model, X, Y, scoring='accuracy', cv=5)
    return scores_dt.mean()

# retorna conjunto de dados (x, y) para treinamento, (x, y) para test de acordo com o modelo e os dados
def seleciona_variaveis_RNE(model, dados, num_variaveis):
	X = dados.drop('Result',axis=1).astype('int32')
	Y = dados['Result'].astype('int32')
	rfe = RFE(model, step=1, n_features_to_select = num_variaveis).fit(X, Y)

	# gera lista de variaveis para serem retiradas
	featuresNames = X.iloc[0].index
	featuresDrops = list()
	for i in range(0,len(featuresNames)):
		if(not rfe.support_[i]):
			featuresDrops.append(featuresNames[i])

	# gera novos conjuntos de treinamento excluindo as variaveis
	X_new = X
	for i in range(0, len(featuresDrops)):
		X_new = X_new.drop(featuresNames[i], axis = 1)

	print("Variaveis Retiradas: ", featuresDrops)
	return X_new, Y, featuresDrops

def seleciona_variaveis_SFS(model, dados, num_variaveis):
	
	X = dados.drop('Result',axis=1).astype('int32')
	Y = dados['Result'].astype('int32')
	sfs = SFS(model, k_features=num_variaveis, forward=True,
		floating=False, verbose=2, scoring='accuracy', cv=4,
		n_jobs=-1).fit(X, Y)
	
	# gera lista de variaveis para serem retiradas
	featuresNames = X.iloc[0].index
	featuresDrops = list()
	for i in range(0,len(featuresNames)):
		if not sfs.k_feature_names_:
			featuresDrops.append(featuresNames[i])

	# gera novos conjuntos de treinamento excluindo as variaveis
	X_new = X
	for i in range(0, len(featuresDrops)):
		X_new = X_new.drop(featuresNames[i], axis = 1)

	print("Variaveis Retiradas: ", featuresDrops)
	return X_new, Y, featuresDrops

if __name__=='__main__':
	base = carregar_base(argv[1])