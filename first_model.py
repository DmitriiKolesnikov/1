import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib

# Загрузка данных
df = pd.read_excel('/Users/jimsgood/Desktop/мисис _первая_модель.xlsx')

# Обработка пропущенных данных
df = df.drop(columns=['ФИО', 'Адрес по прописке'])
df.fillna(df.median(), inplace=True)

# Проверка столбцов, значения которых отличны от нуля
columns_to_check = ['Английский язык', 'Информатика и ИКТ', 'История', 'Математика',
                    'Немецкий язык', 'Обществознание', 'Русский язык', 'Физика',
                    'Французский язык', 'Химия', 'Полугодие_1', 'Полугодие_2',
                    'Полугодие_3', 'Полугодие_4', 'Полугодие_5', 'Полугодие_6',
                    'Полугодие_7', 'Полугодие_8', 'Полугодие_9', 'Полугодие_10',
                    'Полугодие_11', 'Полугодие_12']

# Создание нового DataFrame, содержащего только те строки, где значения в указанных столбцах не равны нулю
df_filtered = df.loc[:, columns_to_check]
mask = (df_filtered != 0).any(axis=1)
df_filtered = df_filtered[mask]

# Добавление остальных нужных столбцов
df_filtered = pd.concat([df_filtered, df[['Отчислен']]], axis=1)

# Выделение признаков и целевой переменной
X = df_filtered.drop(['Отчислен'], axis=1)
y = df_filtered['Отчислен']

# Балансировка классов с помощью ресемплинга
df_minority = df_filtered[df_filtered['Отчислен'] == 1]
df_majority = df_filtered[df_filtered['Отчислен'] == 2]

df_minority_upsampled = resample(df_minority,
                                 replace=True,     # замена данных
                                 n_samples=len(df_majority),    # до размера мажоритарного класса
                                 random_state=123) # для воспроизводимости

df_resampled = pd.concat([df_majority, df_minority_upsampled])

X_resampled = df_resampled.drop(['Отчислен'], axis=1)
y_resampled = df_resampled['Отчислен']

# Разделение на обучающую и тестовую выборку
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Обработка категориальных данных и масштабирование числовых данных
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])

# Создание пайплайна
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Поиск лучших гиперпараметров
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [4, 6, 8],
    'classifier__max_features': ['sqrt', 'log2', 0.2, 0.5],
    'classifier__criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Лучшие параметры
print("Лучшие параметры:", grid_search.best_params_)

# Предсказание на тестовом наборе
y_pred = grid_search.predict(X_test)

# Оценка модели
print("Точность модели:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))

# Подсчет вероятностей отчисления для всех студентов
y_proba = grid_search.predict_proba(X)[:, 1]

# Создание таблицы с вероятностями отчисления
df_result = df_filtered.copy()
df_result['Вероятность НЕ отчисления'] = y_proba

# Сохранение таблицы в Excel
df_result.to_excel('/Users/jimsgood/Desktop/результаты_студентов.xlsx', index=False)

# # Матрица путаницы
# conf_matrix = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(10, 7))
# sns.heatmap(conf_matrix, annot=True, fmt='d')
# plt.title('Матрица путаницы')
# plt.xlabel('Предсказано')
# plt.ylabel('Истинное значение')
# plt.show()

# Сохранение модели
joblib.dump(grid_search.best_estimator_, 'best_model.pkl')
print("Модель успешно сохранена в 'best_model.pkl'")

