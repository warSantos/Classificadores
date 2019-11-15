from sys import argv, exit
from utils import carregar_base
from random import randint

# Constroi tabela de probailidades para predição
def treino(dados):

	# Obtendo a quantidade de páginas de phishing e legítmas normalizado.
	prob_classes = dados['Result'].value_counts(normalize=True)

	# Separando as páginas de phishing de legitmas e removendo a classe.
	dt_phishing = dados[dados['Result'] == 1]
	dt_legitimo = dados[dados['Result'] == -1]
	
	# Construindo a tabela de probabilidade para cada variável.
	phishing = {}
	legitimo = {}
	variaveis = list(dt_phishing.keys())
	variaveis.remove('Result')
	for variavel in variaveis:
		phishing[variavel] = dt_phishing[variavel].value_counts(normalize=True)
		legitimo[variavel] = dt_legitimo[variavel].value_counts(normalize=True)
	
	return prob_classes, phishing, legitimo

def predicao(prob_classes, phishing, legitimo, alvo):

	p_prob = 1
	l_prob = 1
	# Para cada variável do alvo.
	for variavel in phishing.keys():
		try:
			# Multiplicando pela probabilidade da variável em relação a classe phishing.
			p_prob *= phishing[variavel][alvo[variavel]]
			# Multiplicando pela probabilidade da variável em relação a classe legitimo.
			l_prob *= legitimo[variavel][alvo[variavel]]
		except Exception as err:
			#print(str(err), variavel)
			pass
	# Multiplicando pela probabilidade a priori das classes.
	p_prob *= prob_classes[1]
	l_prob *= prob_classes[-1]
	# Normalizando as probabilidades.
	pf = p_prob/(p_prob + l_prob)
	pl = l_prob/(p_prob + l_prob)
	classe = 0
	if pf > pl:
		classe = 1
	else:
		classe = -1
	return pf, pl, classe
	#print("Probabilidade de ser phishing: ", p_prob/(p_prob + l_prob))
	#print("Probabilidade de ser legítima: ", l_prob/(p_prob + l_prob))

def validacao_cruzada(arquivo, lotes=4):
	
	base = carregar_base(argv[1])
	positivos = 0
	negativos = 0
	f_positivos = 0
	f_negativos = 0
	iteracoes = 0
	total_testes = 0
	# Calculando quantas instâncias seram utilizadas para treino e teste.
	total = len(base.index)
	ntreino = total - int(total/lotes)
	while iteracoes < lotes:
		dados = base.sample(frac=1)
		# Selecionando espaço de treino.
		dt_treino = dados.head(ntreino)
		# Selecionando espaço de teste.
		dt_teste = dados.tail(total - ntreino)
		# Treinando o algoritmo.
		prob_classes, phishing, legitimo = treino(dt_treino)
		# Testando o algoritmo.
		for index, row in dt_teste.iterrows():
			pf, pl, classe = predicao(prob_classes, phishing, legitimo, row)
			# Contabilizando negativos.
			if classe == -1 and row['Result'] == -1:
				negativos += 1
			# Contabilizando falsos negativos.
			elif classe == -1 and row['Result'] == 1:
				f_negativos += 1
			# Contabilizando positivos.
			elif classe == 1 and row['Result'] == 1:
				positivos += 1
			# Contabilizando falso positivos.
			else:
				f_positivos += 1
			total_testes += 1
		iteracoes += 1
	
	print("Positivos: ", positivos, " | Falso Positivos: ", f_positivos)
	print("Negativos: ", negativos, " | Falso Negativos: ", f_negativos)
	print("-"*30)
	acuracia = (positivos + negativos)/total_testes
	print("Acurácia: ", acuracia)
	recall = positivos/(positivos + f_negativos)
	print("Recall: ", recall)
	precisao = positivos/(f_positivos + positivos)
	print("Precisão: ", precisao)
	print("F1 Score: ", 2 * recall * precisao / (precisao + recall))

def funcao_teste():
	dados = carregar_base(argv[1])
	# Treinando o algoritmo.
	prob_classes, phishing, legitimo = treino(dados)
	# Pegando uma amostra para teste.
	indice = randint(0, len(dados.index))
	alvo = dados.iloc[indice]
	# Pegando a classe da amostra.
	predicao(prob_classes, phishing, legitimo, alvo)
	print("Classe real: ", alvo['Result'])

if __name__=='__main__':
	#funcao_teste()
	validacao_cruzada(argv[1])