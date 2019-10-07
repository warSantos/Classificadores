from sys import argv, exit
from utils import carregar_base
from FCBF_master.src.fcbf import fcbf_wrapper
from sklearn.feature_selection import VarianceThreshold, chi2

def variance_threshold(dados):

	# Removendo coluna de resultados.
	dados = dados.drop(columns = "Result")

	# Aplicando seletor de características.
	seletor = VarianceThreshold()
	caracteristicas = seletor.fit_transform(dados)
	print(caracteristicas)

def kbest(dados):

	# Removendo coluna de resultados.
	dados = dados.drop(columns = "Result")
	# Trocando -1 (não phishing) por 2.
	dados.replace(-1, 2)

def fcbf(caminho):
	
	print(fcbf_wrapper(caminho, 0.01, delim=',', header=False))

def cfs(dados):

	

if __name__=='__main__':

	dados = carregar_base(argv[1])
	#variance_threshold(dados)
	#fcbf(argv[1])