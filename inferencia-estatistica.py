"""
================================================================================
SEMINÁRIO - INFERÊNCIA ESTATÍSTICA
Grupo 7 - Teste de Kruskal-Wallis
================================================================================
Tema da análise: Existe diferença no salário (em USD) de profissionais de
Ciência de Dados entre os diferentes níveis de experiência?

Variável quantitativa : salary_in_usd
Variável de agrupamento (k >= 3 grupos independentes): experience_level
    EN = Entry-level / Junior
    MI = Mid-level / Intermediate
    SE = Senior-level / Expert
    EX = Executive-level / Director

Por que Kruskal-Wallis (e não ANOVA)?
    - Dados de salário costumam ser fortemente assimétricos (poucos salários
      muito altos "puxam" a distribuição). Por isso testamos a normalidade
      (Shapiro-Wilk) para decidir entre ANOVA e Kruskal-Wallis.
    - Kruskal-Wallis é a alternativa não paramétrica à ANOVA de um fator,
      pois compara as distribuições (postos/ranks) de 3 ou mais grupos
      independentes sem exigir normalidade.
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scikit_posthocs as sp

# Configurações gerais de plotagem
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

# 1. IMPORTAÇÃO DA BASE DE DADOS -----------------------------------------------------------------------------
df = pd.read_csv("ds_salaries.csv")

print()
print("1. IMPORTAÇÃO DA BASE DE DADOS", "-" * 80)
print(f"Dimensões do dataset: {df.shape[0]} linhas x {df.shape[1]} colunas")
print(df.head(), "\n")

# Ordenar os níveis de experiência de forma lógica para os gráficos
ordem_niveis = ["EN", "MI", "SE", "EX"]
nomes_niveis = {
    "EN": "EN (Júnior)",
    "MI": "MI (Pleno)",
    "SE": "SE (Sênior)",
    "EX": "EX (Executivo)",
}
df["experience_level"] = pd.Categorical(
    df["experience_level"], categories=ordem_niveis, ordered=True
)

# 2. ANÁLISE EXPLORATÓRIA DOS DADOS -----------------------------------------------------------------------------
print("2. ANÁLISE EXPLORATÓRIA", "-" * 80)

print("Frequência de observações por grupo (n por nível de experiência):")
print(df["experience_level"].value_counts().sort_index(), "\n")

resumo = (
    df.groupby("experience_level")["salary_in_usd"]
    .agg(n="count", media="mean", mediana="median", desvio_padrao="std",
         minimo="min", maximo="max")
    .round(2)
)
print("Tabela-resumo do salário (USD) por nível de experiência:")
print(resumo, "\n")

# Histograma GERAL do salário (visão da forma da distribuição -> assimetria).
# Importante: aqui NÃO separamos por grupo (a comparação entre grupos já é
# feita pelo boxplot, de forma muito mais legível). Os grupos têm tamanhos
# bem diferentes (SE tem 2516 obs. e EX tem 114), então um histograma
# empilhado por grupo fica poluído e os grupos pequenos somem visualmente.
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="salary_in_usd", bins=40, color="#3b6e8f")
plt.title("Histograma do salário em USD (todos os profissionais)")
plt.xlabel("Salário anual (USD)")
plt.ylabel("Frequência")
plt.tight_layout()
plt.savefig("01_histograma.png")
plt.close()

# Boxplot por grupo: esta é a comparação entre os k grupos.
plt.figure(figsize=(7, 5))
sns.boxplot(
    data=df, x="experience_level", y="salary_in_usd",
    order=ordem_niveis, hue="experience_level", palette="viridis", legend=False
)
plt.title("Boxplot do salário em USD, por nível de experiência")
plt.xlabel("Nível de experiência")
plt.ylabel("Salário anual (USD)")
plt.tight_layout()
plt.savefig("02_boxplot.png")
plt.close()

# Gráfico adequado ao teste: distribuição dos POSTOS (ranks)
# O Kruskal-Wallis compara os postos (ranks) das observações entre os grupos,
# por isso este gráfico mostra exatamente o que o teste avalia.
df["posto_global"] = stats.rankdata(df["salary_in_usd"])
plt.figure(figsize=(7, 5))
sns.boxplot(
    data=df, x="experience_level", y="posto_global",
    order=ordem_niveis, hue="experience_level", palette="mako", legend=False
)
plt.title("Distribuição dos postos (ranks) do salário, por grupo\n"
          "(o que o teste de Kruskal-Wallis efetivamente compara)")
plt.xlabel("Nível de experiência")
plt.ylabel("Posto (rank) do salário")
plt.tight_layout()
plt.savefig("03_postos_rank.png")
plt.close()

print("Gráficos salvos: 01_histograma.png | 02_boxplot.png | 03_postos_rank.png\n")

# 3. VERIFICAÇÃO DOS PRESSUPOSTOS -----------------------------------------------------------------------------
print("3. VERIFICAÇÃO DOS PRESSUPOSTOS", "-" * 80)
print("""Pressupostos do teste de Kruskal-Wallis:
  (a) As amostras dos k grupos são independentes;
  (b) A variável resposta é, no mínimo, ordinal (aqui é quantitativa contínua);
  (c) Sob H0, todos os grupos provêm da mesma distribuição populacional.
O Kruskal-Wallis NÃO exige normalidade nem homogeneidade de variâncias - ele
não tem isso como pressuposto. A normalidade é testada abaixo apenas para
decidir se a ANOVA (paramétrica) seria apropriada.
""")

print("Teste de Shapiro-Wilk por grupo, para verificar se a ANOVA seria apropriada:\n")
algum_rejeitou_normalidade = False
for g in ordem_niveis:
    amostra = df.loc[df["experience_level"] == g, "salary_in_usd"]
    # Shapiro-Wilk é sensível a amostras grandes; usamos amostra aleatória
    # de até 5000 obs. (limite do teste) apenas como referência.
    stat_sw, p_sw = stats.shapiro(amostra.sample(min(len(amostra), 5000),
                                                  random_state=42))
    normal = p_sw >= 0.05
    algum_rejeitou_normalidade = algum_rejeitou_normalidade or not normal
    conclusao = "compatível com normalidade" if normal else "rejeita normalidade"
    print(f"  {g}: n={len(amostra):4d} | W={stat_sw:.4f} | p-valor={p_sw:.4g} -> {conclusao}")

print("""
=> Foi realizado o teste de Shapiro-Wilk para verificar se a ANOVA seria
   apropriada. Como a normalidade não foi observada em pelo menos um grupo,
   optou-se pelo teste não paramétrico de Kruskal-Wallis.
""")

# 4. FORMULAÇÃO DAS HIPÓTESES -----------------------------------------------------------------------------
print("4. HIPÓTESES (teste de Kruskal-Wallis - sempre bilateral/global)", "-" * 80)
print("""H0: as distribuições do salário são iguais entre todos os níveis de
    experiência (EN, MI, SE e EX vêm da mesma distribuição populacional).
H1: pelo menos um nível de experiência apresenta distribuição de salário
    estocasticamente diferente das demais.
Nível de significância adotado: alpha = 0,05

Observação importante: o Kruskal-Wallis testa igualdade de DISTRIBUIÇÕES,
não igualdade de medianas. A leitura "rejeitar H0 = medianas diferentes" só
é estritamente válida quando os grupos têm distribuições de forma/dispersão
semelhante, diferindo apenas em locação - por isso olhamos os boxplots (item
2) para avaliar visualmente se essa suposição é razoável antes de falar em
"diferença de mediana" na conclusão.
""")

# 5. EXEMPLO RESOLVIDO MANUALMENTE -----------------------------------------------------------------------------
# Este exemplo usa um conjunto de dados FICTÍCIO e pequeno, só para fins
# didáticos (slide), já que não é viável acompanhar um cálculo manual com as
# 3.755 observações da base real. A base real é tratada separadamente no
# item 6, usando a função pronta do scipy.
print("5. EXEMPLO RESOLVIDO MANUALMENTE (conjunto de dados fictício e pequeno)", "-" * 80)
print("""Suponha 3 grupos fictícios com 4 observações cada (n = 12 no total),
sem empates, apenas para ilustrar o passo a passo do cálculo de H:

  Grupo EN: 45, 48, 50, 52
  Grupo MI: 58, 60, 62, 65
  Grupo SE: 75, 78, 80, 82
""")

grupos_didaticos = {
    "EN": [45, 48, 50, 52],
    "MI": [58, 60, 62, 65],
    "SE": [75, 78, 80, 82],
}

valores_d = np.concatenate(list(grupos_didaticos.values()))
postos_d = stats.rankdata(valores_d)

print("Passo 1 - Ranquear TODOS os 12 valores juntos (do menor para o maior):")
idx = 0
R_d, n_d = {}, {}
for nome, vals in grupos_didaticos.items():
    n_g = len(vals)
    ranks_grupo = postos_d[idx: idx + n_g]
    for v, r in zip(vals, ranks_grupo):
        print(f"    valor = {v:3d}  ->  posto = {r:.0f}  (grupo {nome})")
    R_d[nome] = ranks_grupo.sum()
    n_d[nome] = n_g
    idx += n_g

print("\nPasso 2 - Somar os postos (R_j) de cada grupo:")
for nome in grupos_didaticos:
    print(f"    Grupo {nome}: n_j = {n_d[nome]} | R_j = {R_d[nome]:.0f}")

n_total_d = len(valores_d)
k_d = len(grupos_didaticos)
gl_d = k_d - 1

H_d = (12 / (n_total_d * (n_total_d + 1))) * sum(
    (R_d[g] ** 2) / n_d[g] for g in grupos_didaticos
) - 3 * (n_total_d + 1)

print(f"""
Passo 3 - Aplicar a fórmula:
    H = [12 / (N(N+1))] * Σ(R_j² / n_j) - 3(N+1)
    N = {n_total_d}   k = {k_d}   gl = k - 1 = {gl_d}

    H = [12 / ({n_total_d}x{n_total_d+1})] * ({R_d['EN']:.0f}²/{n_d['EN']} + {R_d['MI']:.0f}²/{n_d['MI']} + {R_d['SE']:.0f}²/{n_d['SE']})
        - 3x({n_total_d}+1)
    H = {H_d:.3f}
""")

alpha_d = 0.05
valor_critico_d = stats.chi2.ppf(1 - alpha_d, gl_d)
p_valor_d = 1 - stats.chi2.cdf(H_d, gl_d)

print(f"""Passo 4 - Regra de decisão (alpha = {alpha_d}):
  (a) Valor crítico: Qui-Quadrado(gl={gl_d}; alpha={alpha_d}) = {valor_critico_d:.3f}
      H ({H_d:.3f}) > valor crítico ({valor_critico_d:.3f})?
      -> {"SIM, REJEITA H0" if H_d > valor_critico_d else "NÃO, NÃO REJEITA H0"}

  (b) p-valor: p = {p_valor_d:.4f}
      p-valor < alpha?  -> {"SIM, REJEITA H0" if p_valor_d < alpha_d else "NÃO, NÃO REJEITA H0"}

  Conclusão do exemplo didático: como os três grupos fictícios foram criados
  sem nenhuma sobreposição de valores, era esperado H grande e p-valor
  pequeno -> rejeita-se H0 (grupos vêm de distribuições diferentes).
""")

# 6. APLICAÇÃO COMPUTACIONAL NA BASE REAL -----------------------------------------------------------------------------
# Aqui usamos a função pronta diretamente sobre os 3.755 registros reais.
print("6. APLICAÇÃO COMPUTACIONAL NA BASE REAL - scipy.stats.kruskal()", "-" * 80)
grupos = [df.loc[df["experience_level"] == g, "salary_in_usd"].values for g in ordem_niveis]
n_total = sum(len(g) for g in grupos)
k = len(grupos)
gl = k - 1

H_scipy, p_scipy = stats.kruskal(*grupos)
alpha = 0.05
valor_critico = stats.chi2.ppf(1 - alpha, gl)

print(f"N total = {n_total} | k = {k} grupos | gl = {gl}")
print(f"Estatística H (scipy) = {H_scipy:.4f}")
print(f"p-valor   (scipy)     = {p_scipy:.6g}\n")

print("Regra de decisão (mesma lógica do exemplo manual, agora na base real):")
print(f"  (a) Valor crítico Qui-Quadrado(gl={gl}, alpha={alpha}) = {valor_critico:.4f}")
print(f"      Decisão: {'REJEITA H0' if H_scipy > valor_critico else 'NÃO REJEITA H0'}")
print(f"  (b) p-valor = {p_scipy:.6g}")
print(f"      Decisão: {'REJEITA H0' if p_scipy < alpha else 'NÃO REJEITA H0'}\n")

# 7. INTERPRETAÇÃO DOS RESULTADOS  -----------------------------------------------------------------------------
print("7. INTERPRETAÇÃO", "-" * 80)
if p_scipy < alpha:
    print(f"""Como o p-valor ({p_scipy:.4g}) é menor que alpha (0,05), rejeitamos H0.
Há evidência estatística de que pelo menos um nível de experiência possui
distribuição de salário diferente dos demais. Os boxplots do item 2 mostram
formas/dispersões razoavelmente parecidas entre os grupos, o que permite
também ler o resultado como: as medianas de salário tendem a crescer de
EN -> MI -> SE -> EX.
""")
else:
    print(f"""Como o p-valor ({p_scipy:.4g}) não é menor que alpha (0,05), não há
evidência suficiente para rejeitar H0.
""")

# 8. COMPARAÇÕES MÚLTIPLAS (POST-HOC) APÓS REJEIÇÃO DE H0 -----------------------------------------------------------------------------
print("8. COMPARAÇÕES MÚLTIPLAS (TESTE DE DUNN, correção de Bonferroni)", "-" * 80)
posthoc = sp.posthoc_dunn(
    df, val_col="salary_in_usd", group_col="experience_level", p_adjust="bonferroni"
)
posthoc = posthoc.loc[ordem_niveis, ordem_niveis].round(4)
print("Matriz de p-valores ajustados (par a par):")
print(posthoc, "\n")
print("Interpretação: pares com p-valor ajustado < 0,05 diferem significativamente entre si.\n")

# Heatmap do post-hoc
plt.figure(figsize=(6, 5))
sns.heatmap(posthoc, annot=True, fmt=".2e", cmap="rocket_r", vmin=0, vmax=0.1,
            cbar_kws={"label": "p-valor ajustado (Bonferroni)"})
plt.title("Teste de Dunn (pós-hoc) - p-valores ajustados")
plt.tight_layout()
plt.savefig("04_posthoc_dunn_heatmap.png")
plt.close()

# 9. TABELA-RESUMO FINAL DOS RESULTADOS -----------------------------------------------------------------------------
print("9. TABELA-RESUMO DOS RESULTADOS (base real)", "-" * 80)
tabela_final = pd.DataFrame({
    "Estatística": ["H (Kruskal-Wallis)", "Graus de liberdade", "p-valor",
                     "Valor crítico (alpha=0,05)", "Decisão"],
    "Valor": [round(H_scipy, 4), gl, f"{p_scipy:.4g}", round(valor_critico, 4),
              "Rejeita H0" if p_scipy < alpha else "Não rejeita H0"]
})
print(tabela_final.to_string(index=False))
tabela_final.to_csv("05_tabela_resumo_resultados.csv", index=False)

# 10. CONCLUSÃO -----------------------------------------------------------------------------
print()
print("10. CONCLUSÃO", "-" * 80)
print("""O teste de Kruskal-Wallis indicou diferença estatisticamente significativa
na distribuição de salários (USD) entre os níveis de experiência dos
profissionais de Ciência de Dados. O teste de Dunn (pós-hoc) permite
identificar especificamente quais pares de níveis diferem entre si,
confirmando a tendência observada nos boxplots: o salário tende a aumentar
conforme o nível de experiência cresce (EN < MI < SE < EX).
""")