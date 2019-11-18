from sys import argv, exit
from utils import carregar_base
from FCBF_master.src.fcbf import fcbf_wrapper
from sklearn.feature_selection import VarianceThreshold, chi2, SelectKBest, f_regression

CLASS_LABEL = 'CLASS_LABEL'

def variance_threshold(dados):

	# Removendo coluna de resultados.
	dados = dados.drop(columns = CLASS_LABEL)

	# Aplicando seletor de características.
	seletor = VarianceThreshold()
	caracteristicas = seletor.fit_transform(dados)
	print(caracteristicas)

def kbest(dados):

	# Removendo coluna de resultados.
	dados = dados.drop(columns = CLASS_LABEL)
	# Trocando -1 (não phishing) por 2.
	dados.replace(-1, 2)

def fcbf(caminho):
	
	print(fcbf_wrapper(caminho, 0.01, delim=',', header=False))

def cfs(dados):
	print("Olá.")

def chi_square(dados):
	X = dados.drop(columns=CLASS_LABEL).replace(-1,2)
	Y = dados[-1:].replace(-1,2)
	SelectKBest(f_regression, k=20).fit_transform(X.shape, Y.shape)

if __name__=='__main__':

	dados = carregar_base(argv[1])
	chi_square(dados)