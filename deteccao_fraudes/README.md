# Detecção de Fraude em Cartão de Crédito

Classificador de fraude sobre uma base real de transações de cartão de crédito, construído com foco no que de fato importa em risco: **capturar fraude sem afogar a operação em falsos alarmes** — e tornar cada decisão auditável.

O projeto trata o problema como ele aparece num banco de verdade, não como um exercício de classificação genérico: base extremamente desbalanceada, custo assimétrico entre os tipos de erro, necessidade de validação honesta e de explicabilidade.

---

## O problema

A base tem **284.807 transações, das quais apenas 492 (0,17%) são fraudes**. Esse desbalanceamento define tudo:

- **Accuracy é inútil aqui.** Um modelo que classifica *toda* transação como legítima acerta 99,83% das vezes — e não pega uma fraude sequer. Qualquer avaliação baseada em accuracy é enganosa.
- **Os erros custam diferente.** Deixar passar uma fraude (falso negativo) gera perda financeira direta; bloquear um cliente legítimo (falso positivo) gera atrito e custo de análise. O ponto de operação do modelo é, no fundo, uma decisão de negócio.

## A abordagem

| Decisão | Por quê |
|---|---|
| **AUC-PR, recall e precision da classe fraude** como métricas | Refletem o desempenho na classe rara. AUC-PR é a referência para dados desbalanceados. |
| **Split temporal** (treina no passado, testa no futuro) | Fraude evolui no tempo. Embaralhar os dados vazaria o futuro no treino e inflaria as métricas. |
| **Escalonamento ajustado só no treino** | Ajustar no dataset inteiro vazaria estatísticas do teste para o treino. |
| **`class_weight="balanced"`** | Penaliza mais os erros na classe minoritária sem descartar dados (ao contrário do undersampling). |
| **Ajuste de threshold pela curva PR** | O corte de 0,5 é arbitrário. O threshold é a alavanca que adapta o modelo à política de risco. |
| **Explicabilidade com SHAP** | Em serviços financeiros, decisão de fraude precisa ser justificável — inclusive por exigência regulatória. |

## Modelos comparados

Regressão Logística, Random Forest, Gradient Boosting e SVM Linear, avaliados na validação por métricas de fraude.

Um achado ilustrativo: as **AUC-ROC** ficaram todas coladas em ~0,97, enquanto as **AUC-PR** se espalharam de ~0,55 a ~0,79. É a demonstração prática de por que ROC engana em base desbalanceada — quem separa os modelos é a métrica focada na classe rara. O **Random Forest** entregou o melhor equilíbrio e foi o modelo levado adiante.

## Resultado no conjunto de teste

O mesmo modelo, sob duas políticas de corte, no conjunto de teste (20% mais recentes, nunca vistos):

| Política | Recall | Precision | Fraudes pegas | Falsos alarmes | Fraudes perdidas |
|---|---|---|---|---|---|
| Corte padrão (0,50) | 0,79 | 0,63 | 59 de 75 | 34 | 16 |
| Corte ajustado (0,62) | 0,75 | 0,86 | 56 de 75 | 9 | 19 |

Mudar o threshold não muda o modelo — muda a política. O corte mais alto reduz falsos alarmes de 34 para 9, ao custo de perder 3 fraudes a mais. Qual é o "certo" depende do custo que o negócio atribui a cada erro. O papel do modelo é tornar esse trade-off explícito e mensurável.

## Explicabilidade

O SHAP decompõe cada previsão e mostra quais features mais empurram o score para "fraude". Um punhado de componentes concentra o poder discriminante. Como a base é anonimizada por PCA, os nomes reais não estão disponíveis — mas em um projeto interno, esse mesmo gráfico apontaria diretamente para as variáveis de negócio que mais sinalizam fraude, guiando regras e investigação.

---

## Como rodar

```bash
pip install -r requirements.txt
jupyter notebook Deteccao_Fraude_Cartao.ipynb
```

O dataset (`creditcard.csv`) está disponível no [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) — não incluído no repositório por tamanho.

## Stack

Python · scikit-learn · SHAP · pandas · matplotlib

## Próximos passos

- Testar SMOTE como alternativa de balanceamento
- Servir o modelo como uma **API (FastAPI)** que recebe uma transação e retorna score + faixa de risco + top fatores SHAP
- Monitorar *drift* em produção com PSI (Population Stability Index)
