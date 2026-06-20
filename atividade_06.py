# ==========================================
# TRABALHO 6 - MINERAÇÃO DE DADOS
# Opção 01: IterativeImputer + HistGradientBoostingRegressor
# ==========================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import HistGradientBoostingRegressor

print("1. Carregando o conjunto de dados...")
url = "https://raw.githubusercontent.com/gakudo-ai/open-datasets/refs/heads/main/employees_dataset_with_missing.csv"
df = pd.read_csv(url)

print("\nValores nulos presentes no dataset original:")
print(df.isnull().sum())

df = df.dropna(subset=['income'])

print("\nValores nulos APÓS limpar o alvo (Y):")
print(df.isnull().sum())

# 2. Estratégia de Colunas
# Alvo (Y): income
# Preditoras (X): age, education_years, experience, credit_score
X = df[['age', 'education_years', 'experience', 'credit_score']]
y = df['income']

# 3. Divisão em Treino e Teste (80% para treinar, 20% para testar)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Construção da Pipeline "A Melhor"
# O imputer usa inteligência para preencher os dados, o regressor faz a previsão final
pipeline_opcao_01 = Pipeline([
    ('imputer', IterativeImputer(random_state=42, max_iter=10)),
    ('regressor', HistGradientBoostingRegressor(random_state=42))
])

# 5. Execução e Treinamento
print("\n2. Treinando a pipeline (Imputação Iterativa + Gradient Boosting)...")
pipeline_opcao_01.fit(X_train, y_train)
print("Treinamento concluído com sucesso!")

# 6. Realizando Previsões na base de Teste
y_pred = pipeline_opcao_01.predict(X_test)

# 7. Avaliação Final do Modelo
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n===============================")
print("    RESULTADOS DA AVALIAÇÃO    ")
print("===============================")
print(f"Erro Médio Absoluto (MAE): ${mae:.2f}")
print(f"Coeficiente de Determinação (R²): {r2:.4f}")
print("-------------------------------")
print("Primeiras 5 previsões salariais geradas:")
for real, pred in zip(y_test[:5], y_pred[:5]):
    print(f"Real: ${real:.2f} | Previsto: ${pred:.2f}")
print("===============================")