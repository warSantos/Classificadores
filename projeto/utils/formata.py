from sys import argv

if __name__=='__main__':

	pt = open(argv[1], 'r')
	linha = 0
	escolha = 0
	acc_max = 0
	scores = list()
	for p in pt:
		p = p.replace('(', '')
		p = p.replace(')', '')
		p = p.replace(' ', '')
		valores = p.split(',')
		valores.pop()
		valores.pop()
		valores.pop()
		matrix = valores.pop()
		if float(valores[0]) > acc_max:
			escolha = linha
			acc_max = float(valores[0])
			del scores
			scores = list()
			for v in valores:
				scores.append(round(float(v),4)*100)
		linha += 1



	print("Quantidade de variáveis: ", 30 - escolha)
	print("Acurácia máxima: ", acc_max)
	print(scores)
	"""
	(Aucuracia, roc_auc, f1-score, recall, precision, array[ tn, fp, fn, tp])
	"""