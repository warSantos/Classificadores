from sys import argv, exit
from utils import carregar_base
from random import randint

# Constroi tabela de probailidades para predição
def treino(arquivo):

	dados = carregar_base(arquivo)
	# Obtendo a quantidade de páginas de phishing e legítmas normalizado.
	classes_prob = dados['Result'].value_counts(normalize=True)
	
	# Separando as páginas de phishing de legitmas e removendo a classe.
	dt_phishing = dados[dados['Result'] == 1].drop(columns='Result')
	dt_legitimo = dados[dados['Result'] == -1].drop(columns='Result')
	
	# Construindo a tabela de probabilidade para cada variável.
	phishing = {}
	legitimo = {}
	for variavel in dt_phishing.keys():
		phishing[variavel] = dt_phishing[variavel].value_counts(normalize=True)
		legitimo[variavel] = dt_legitimo[variavel].value_counts(normalize=True)
	
	return classes_prob, phishing, legitimo

def predicao(classes_prob, phishing, legitimo, alvo):

	p_prob = 1
	l_prob = 1
	# Para cada variável do alvo.
	for variavel in phishing.keys():
		# Multiplicando pela probabilidade da variável em relação a classe phishing.
		p_prob *= phishing[variavel][alvo[variavel]]
		# Multiplicando pela probabilidade da variável em relação a classe legitimo.
		l_prob *= legitimo[variavel][alvo[variavel]]
	p_prob *= classes_prob[1]
	l_prob *= classes_prob[-1]
	
	print("Probabilidade de ser phishing: ", p_prob/(p_prob + l_prob))
	print("Probabilidade de ser legítima: ", l_prob/(p_prob + l_prob))

def funcao_teste():
	dados = carregar_base(argv[1])
	# Treinando o algoritmo.
	classes_prob, phishing, legitimo = treino(argv[1])
	# Pegando uma amostra para teste.
	indice = randint(0, len(dados.index))
	alvo = dados.iloc[indice][:-1]
	# Pegando a classe da amostra.
	classe = dados.iloc[indice][-1:]
	predicao(classes_prob, phishing, legitimo, alvo)
	print("Classe real: ", classe)

if __name__=='__main__':
	funcao_teste()