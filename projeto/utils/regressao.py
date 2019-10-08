from sys import argv
from utils import carregar_base, seleciona_variaveis_RNE, validacao_cruzada
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

def busca_melhores_features(model, dados):
    print('\nBuscando a melhor seleção de variavel utilizando recursive feature elimination para RFC')
    melhor_score = 0.0
    melhor_i = 30
    for i in range(0, 30):
        print('\nITERAÇÃO ',i,'------------------------------------------------------------------------')

        X, Y, features_excuidas = seleciona_variaveis_RNE(model, dados, 30-i)
        atual_score = validacao_cruzada(X, Y, model)
        print("Acurácia: ", atual_score, "| Número de Features: ", 30-i, " --- ", X.shape)
        if(atual_score > melhor_score):
            melhor_score = atual_score
            melhor_i = i
            melhores_exclusoes = features_excuidas


    return melhor_score, melhor_i, melhores_exclusoes

if __name__=='__main__':
    base = carregar_base(argv[1])
    rl_model = LogisticRegression(solver='newton-cg')
    score, num_variaveis = busca_melhores_features(rl_model, base)
    print("\n\nMelhor acurácia: ", score, 'Número de váriveis', 30 - num_variaveis)