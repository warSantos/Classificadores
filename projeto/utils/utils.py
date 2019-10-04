import pandas as pd
from sys import argv, exit

def carregar_base(caminho):
	base_dados = pd.read_csv(caminho)
	return base_dados

if __name__=='__main__':
	carregar_base(argv[1])