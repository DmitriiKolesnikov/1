import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint

# Путь к файлу
file_path = '/Users/jimsgood/Desktop/мисис_и_маргу_вместе.xlsx'
output_file_path = '/Users/jimsgood/Desktop/мисис_и_маргу_с_предсказаниями.xlsx'

# Загрузка данных из Excel файла
try:
    df = pd.read_excel(file_path)
except FileNotFoundError:
    print(f"Файл по пути {file_path} не найден.")
    exit(1)
except Exception as e:
    print(f"Ошибка при загрузке файла: {e}")
    exit(1)


# Функция для предобработки данных
def preprocess_data(df):
    try:
        # Исключаем столбец ID_студента из анализа
        if 'ID_студента' in df.columns:
            df.drop(columns=['ID_студента'], inplace=True)

        # Заменяем значения ниже 2 в колонках Полугодия на NaN
        semester_columns = [col for col in df.columns if 'Полугодие' in col]
        for col in semester_columns:
            df[col] = df[col].apply(lambda x: np.nan if x < 2.9999 else x)

    except Exception as e:
        print(f"Ошибка при предобработке данных: {e}")
        exit(1)

    return df

# Обработка данных
df = preprocess_data(df)

# Проверка наличия необходимых столбцов в исходном DataFrame
required_columns = [
    'Возраст', 'Пол', 'Курс', 'Отчислен', 'Уровень подготовки',
    'Формат обучения', 'Балл_ЕГЭ_1', 'Балл_ЕГЭ_2', 'Балл_ЕГЭ_3', 'Балл_ЕГЭ_4'
]
semester_columns = [col for col in df.columns if 'Полугодие' in col]

# Стандартизация данных
scaler = StandardScaler()
df[required_columns + semester_columns] = scaler.fit_transform(df[required_columns + semester_columns])

# Функция для предсказания недостающих значений
def predict_missing_values(df, columns):
    for col in columns:
        try:
            # Выделяем обучающие и тестовые данные
            train_data = df[df[col].notna()]
            test_data = df[df[col].isna()]

            if test_data.empty:
                continue

            X_train = train_data.drop(columns=columns)
            y_train = train_data[col]
            X_test = test_data.drop(columns=columns)

            # Разделение данных на обучение и тест
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

            # Настройка гиперпараметров
            param_dist = {
                'n_estimators': randint(50, 200),
                'max_depth': randint(10, 50),
                'min_samples_split': randint(2, 10),
                'min_samples_leaf': randint(1, 10)
            }

            # Использование RandomizedSearchCV для оптимизации гиперпараметров
            model = RandomForestRegressor(random_state=42)
            random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=100, cv=3, random_state=42, n_jobs=-1)
            random_search.fit(X_train, y_train)

            # Обучение модели с лучшими параметрами
            best_model = random_search.best_estimator_
            best_model.fit(X_train, y_train)

            # Валидация модели
            y_pred = best_model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            print(f"Ошибка для {col}: {mse}, Точность: {100 - (mse * 100):.2f}%")

            # Предсказание недостающих значений
            df.loc[df[col].isna(), col] = best_model.predict(X_test)

        except Exception as e:
            print(f"Ошибка при предсказании для {col}: {e}")
            continue

# Вызов функции для предсказания недостающих значений
predict_missing_values(df, semester_columns)

# Обратное преобразование стандартизированных данных
df[required_columns + semester_columns] = scaler.inverse_transform(df[required_columns + semester_columns])

# Сохранение данных в новый Excel файл
df.to_excel(output_file_path, index=False)
print(f"Файл сохранен по пути {output_file_path}")