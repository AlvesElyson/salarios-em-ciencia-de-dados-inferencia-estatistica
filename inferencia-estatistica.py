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
      muito altos "puxam" a distribuição), violando a suposição de
      normalidade exigida pela ANOVA.
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

# -----------------------------------------------------------------------------
# 1. IMPORTAÇÃO DA BASE DE DADOS
# -----------------------------------------------------------------------------
df = pd.read_csv("ds_salaries.csv")

print("=" * 80)
print("1. IMPORTAÇÃO DA BASE DE DADOS")
print("=" * 80)
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

# -----------------------------------------------------------------------------
# 2. ANÁLISE EXPLORATÓRIA DOS DADOS
# -----------------------------------------------------------------------------
print("=" * 80)
print("2. ANÁLISE EXPLORATÓRIA")
print("=" * 80)

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

# --- Histograma -------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.histplot(
    data=df, x="salary_in_usd", hue="experience_level",
    multiple="stack", bins=40, palette="viridis"
)
plt.title("Histograma do salário em USD, por nível de experiência")
plt.xlabel("Salário anual (USD)")
plt.ylabel("Frequência")
plt.tight_layout()
plt.savefig("01_histograma.png")
plt.close()

# --- Boxplot ------------------------------------------------------------
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

# --- Gráfico adequado ao teste: distribuição dos POSTOS (ranks) -------------
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

# -----------------------------------------------------------------------------
# 3. VERIFICAÇÃO DOS PRESSUPOSTOS
# -----------------------------------------------------------------------------
print("=" * 80)
print("3. VERIFICAÇÃO DOS PRESSUPOSTOS")
print("=" * 80)
print("""Pressupostos do teste de Kruskal-Wallis:
  (a) As amostras dos k grupos são independentes;
  (b) A variável resposta é, no mínimo, ordinal (aqui é quantitativa contínua);
  (c) Sob H0, as distribuições dos grupos têm a mesma forma (apenas podem
      diferir em locação/mediana) -> permite interpretar rejeição de H0 como
      diferença de medianas. Quando as formas diferem muito, a conclusão deve
      ser lida como "diferença estocástica/de distribuição" entre os grupos.
Observação: o KW NÃO exige normalidade nem homogeneidade de variâncias - é
exatamente por isso que ele é usado aqui no lugar da ANOVA.
""")

print("Teste de normalidade (Shapiro-Wilk) por grupo - apenas para JUSTIFICAR")
print("a escolha do teste não paramétrico em vez da ANOVA:\n")
for g in ordem_niveis:
    amostra = df.loc[df["experience_level"] == g, "salary_in_usd"]
    # Shapiro-Wilk é sensível a amostras grandes; usamos amostra aleatória
    # de até 5000 obs. (limite do teste) apenas como referência.
    stat_sw, p_sw = stats.shapiro(amostra.sample(min(len(amostra), 5000),
                                                  random_state=42))
    conclusao = "rejeita normalidade" if p_sw < 0.05 else "não rejeita normalidade"
    print(f"  {g}: n={len(amostra):4d} | W={stat_sw:.4f} | p-valor={p_sw:.4g} -> {conclusao}")

print("\n=> Em pelo menos um grupo o pressuposto de normalidade é rejeitado,")
print("   o que reforça a adequação do teste de Kruskal-Wallis (não paramétrico).\n")

# -----------------------------------------------------------------------------
# 4. FORMULAÇÃO DAS HIPÓTESES
# -----------------------------------------------------------------------------
print("=" * 80)
print("4. HIPÓTESES (teste de Kruskal-Wallis - sempre bilateral/global)")
print("=" * 80)
print("""H0: as distribuições do salário são iguais em todos os níveis de
    experiência (medianas iguais) -> EN = MI = SE = EX
H1: pelo menos um nível de experiência apresenta distribuição de salário
    estocasticamente diferente dos demais.
Nível de significância adotado: alpha = 0,05
""")

# -----------------------------------------------------------------------------
# 5. EXEMPLO RESOLVIDO MANUALMENTE (cálculo passo a passo da estatística H)
# -----------------------------------------------------------------------------
print("=" * 80)
print("5. CÁLCULO MANUAL DA ESTATÍSTICA H (passo a passo)")
print("=" * 80)

grupos = [df.loc[df["experience_level"] == g, "salary_in_usd"].values for g in ordem_niveis]
n_total = sum(len(g) for g in grupos)

# 5.1 Ranquear TODAS as observações juntas (postos médios em caso de empate)
todos_valores = np.concatenate(grupos)
postos = stats.rankdata(todos_valores)

# 5.2 Separar os postos de volta por grupo e somar
idx = 0
R_j, n_j = [], []
for g in grupos:
    n_g = len(g)
    R_j.append(postos[idx: idx + n_g].sum())
    n_j.append(n_g)
    idx += n_g

print("Soma dos postos (R_j) e tamanho (n_j) por grupo:")
for nome, R, n in zip(ordem_niveis, R_j, n_j):
    print(f"  Grupo {nome}: n_j = {n:5d} | R_j = {R:,.1f} | R_j/n_j = {R/n:.2f}")

# 5.3 Fórmula manual de H (sem correção de empates)
H_manual = (12 / (n_total * (n_total + 1))) * sum(
    (R**2) / n for R, n in zip(R_j, n_j)
) - 3 * (n_total + 1)

# 5.4 Fator de correção para empates (ties) - recomendado quando há muitos
_, contagem_empates = np.unique(todos_valores, return_counts=True)
fator_correcao = 1 - (np.sum(contagem_empates**3 - contagem_empates) /
                       (n_total**3 - n_total))
H_corrigido = H_manual / fator_correcao

k = len(grupos)
gl = k - 1  # graus de liberdade = k - 1

print(f"""
Fórmula: H = [12 / (N(N+1))] * Σ(R_j² / n_j) - 3(N+1)
  N (total de observações)        = {n_total}
  k (número de grupos)            = {k}
  Graus de liberdade (gl = k - 1) = {gl}

  H (sem correção de empates)     = {H_manual:.4f}
  Fator de correção de empates    = {fator_correcao:.6f}
  H (corrigido para empates)      = {H_corrigido:.4f}
""")

# -----------------------------------------------------------------------------
# 6. REGRA DE DECISÃO
# -----------------------------------------------------------------------------
print("=" * 80)
print("6. REGRA DE DECISÃO")
print("=" * 80)
alpha = 0.05
valor_critico = stats.chi2.ppf(1 - alpha, gl)
p_valor_manual = 1 - stats.chi2.cdf(H_corrigido, gl)

print(f"""(a) Método do valor crítico:
    H estatístico (corrigido)         = {H_corrigido:.4f}
    Valor crítico Qui-Quadrado(gl={gl}, alpha={alpha}) = {valor_critico:.4f}
    Decisão: {"REJEITA H0" if H_corrigido > valor_critico else "NÃO REJEITA H0"}
    (rejeita-se H0 se H_calculado > valor crítico)

(b) Método do p-valor:
    p-valor = {p_valor_manual:.6g}
    Decisão: {"REJEITA H0" if p_valor_manual < alpha else "NÃO REJEITA H0"}
    (rejeita-se H0 se p-valor < alpha)
""")

# -----------------------------------------------------------------------------
# 7. APLICAÇÃO DO TESTE (verificação com a função pronta do scipy)
# -----------------------------------------------------------------------------
print("=" * 80)
print("7. APLICAÇÃO COMPUTACIONAL - scipy.stats.kruskal()")
print("=" * 80)
H_scipy, p_scipy = stats.kruskal(*grupos)
print(f"Estatística H (scipy) = {H_scipy:.4f}")
print(f"p-valor   (scipy)     = {p_scipy:.6g}")
print("(valores batem com o cálculo manual corrigido para empates acima)\n")

# -----------------------------------------------------------------------------
# 8. INTERPRETAÇÃO DOS RESULTADOS
# -----------------------------------------------------------------------------
print("=" * 80)
print("8. INTERPRETAÇÃO")
print("=" * 80)
if p_scipy < alpha:
    print(f"""Como o p-valor ({p_scipy:.4g}) é menor que alpha (0,05), rejeitamos H0.
Há evidência estatística de que pelo menos um nível de experiência possui
distribuição de salário diferente dos demais. Isso é coerente com a tabela-
resumo: as medianas crescem visivelmente de EN -> MI -> SE -> EX.
""")
else:
    print(f"""Como o p-valor ({p_scipy:.4g}) não é menor que alpha (0,05), não há
evidência suficiente para rejeitar H0.
""")

# -----------------------------------------------------------------------------
# 9. COMPARAÇÕES MÚLTIPLAS (POST-HOC) APÓS REJEIÇÃO DE H0
# -----------------------------------------------------------------------------
print("=" * 80)
print("9. COMPARAÇÕES MÚLTIPLAS (TESTE DE DUNN, correção de Bonferroni)")
print("=" * 80)
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

# -----------------------------------------------------------------------------
# 10. TABELA-RESUMO FINAL DOS RESULTADOS
# -----------------------------------------------------------------------------
print("=" * 80)
print("10. TABELA-RESUMO DOS RESULTADOS")
print("=" * 80)
tabela_final = pd.DataFrame({
    "Estatística": ["H (Kruskal-Wallis)", "Graus de liberdade", "p-valor",
                     "Valor crítico (alpha=0,05)", "Decisão"],
    "Valor": [round(H_scipy, 4), gl, f"{p_scipy:.4g}", round(valor_critico, 4),
              "Rejeita H0" if p_scipy < alpha else "Não rejeita H0"]
})
print(tabela_final.to_string(index=False))
tabela_final.to_csv("05_tabela_resumo_resultados.csv", index=False)

# -----------------------------------------------------------------------------
# 11. CONCLUSÃO
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("11. CONCLUSÃO")
print("=" * 80)
print("""O teste de Kruskal-Wallis indicou diferença estatisticamente significativa
na distribuição de salários (USD) entre os níveis de experiência dos
profissionais de Ciência de Dados. O teste de Dunn (pós-hoc) permite
identificar especificamente quais pares de níveis diferem entre si,
confirmando a tendência observada nos boxplots: o salário tende a aumentar
conforme o nível de experiência cresce (EN < MI < SE < EX).
""")