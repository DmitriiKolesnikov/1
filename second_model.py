import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Загрузка данных из файла Excel
file_path = '/Users/jimsgood/Desktop/мисис_и_маргу_вместе.xlsx'
data = pd.read_excel(file_path)

semesters = [
    'Полугодие_1', 'Полугодие_2', 'Полугодие_3', 'Полугодие_4', 'Полугодие_5',
    'Полугодие_6', 'Полугодие_7', 'Полугодие_8', 'Полугодие_9', 'Полугодие_10',
    'Полугодие_11', 'Полугодие_12'
]

ege_scores = ['Балл_ЕГЭ_1', 'Балл_ЕГЭ_2', 'Балл_ЕГЭ_3', 'Балл_ЕГЭ_4']

common_data = ['Возраст', 'Пол', 'Курс', 'Отчислен',  'Уровень подготовки', 'Формат обучения']


# Функция для вычисления среднего балла
def calculate_avg(grades):
    non_zero_grades = [grade for grade in grades if grade > 2.3 and not np.isnan(grade)]
    avg_grade = np.mean(non_zero_grades) if non_zero_grades else 0
    return avg_grade


# Функция для подсчета количества пятерок (оценка >= 4.5)
def count_high_grades(grades):
    return sum(1 for grade in grades if not np.isnan(grade) and grade >= 4.5)


# Функция для подсчета количества нулевых оценок
def count_zeros(grades):
    return sum(1 for grade in grades if grade == 0)


# Функция для подсчета количества семестров со средней оценкой >= 4.75
def count_semesters_with_high_avg(grades):
    return sum(1 for grade in grades if grade >= 4.75)


# Добавляем столбцы со средним баллом, количеством пятерок, количеством нулей и количеством высоко оценённых семестров
data['Average_Score'] = data[semesters].apply(calculate_avg, axis=1)
data['High_Grades_Count'] = data[semesters].apply(count_high_grades, axis=1)
data['Zero_Count'] = data[semesters].apply(count_zeros, axis=1)
data['High_Avg_Semesters_Count'] = data[semesters].apply(count_semesters_with_high_avg, axis=1)

# Задаем целевой параметр y
y = (data['Average_Score'] >= 4.75).astype(int)

# Отбираем входные параметры X, включая баллы ЕГЭ
X = data[common_data + ege_scores + semesters + ['High_Grades_Count', 'Zero_Count', 'High_Avg_Semesters_Count']]

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Масштабируем числовые признаки
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Обучаем модель логистической регрессии
model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(X_train_scaled, y_train)

# Делаем предсказания и вычисляем точность
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f'Точность модели: {accuracy}')

# Прогнозируем вероятность красного диплома для всех студентов
data['Probability'] = model.predict_proba(scaler.transform(X))[:, 1]


# Функция корректировки вероятности на основе различных факторов
def calculate_red_diploma_probability(avg_score, high_grades_count, zero_count, high_avg_semesters_count, ege_scores,
                                      high_grades_weight, zero_weight, ege_weight, default_prob):
    # Если средний балл слишком низкий, устанавливаем минимальную вероятность 0.1
    if avg_score < 2.5:
        return 0.1

    # Исключительный случай с большим количеством семестров с высокой средней оценкой
    if high_avg_semesters_count >= 8:
        return 1.0

    # Обрабатываем баллы ЕГЭ
    valid_ege_scores = [score for score in ege_scores if score > 0 and not np.isnan(score)]
    ege_avg = np.mean(valid_ege_scores) if valid_ege_scores else 0

    # Корректируем вероятность, применяя веса и логистическую функцию для нормализации
    weighted_sum = (
        default_prob +
        high_grades_weight * high_grades_count -
        zero_weight * zero_count +
        ege_weight * ege_avg
    )

    # Применяем логистическую функцию для нормализации вероятности
    adjusted_prob = 1 / (1 + np.exp(-weighted_sum))

    # Дополнительно корректируем вероятность в зависимости от количества высоко оценённых семестров
    adjusted_prob *= (1 + high_avg_semesters_count / 8)

    # Нормализуем вероятность, чтобы она находилась в диапазоне [0.1, 1.0]
    adjusted_prob = max(min(adjusted_prob, 1.0), 0.1)

    return adjusted_prob


# Определяем веса для факторов
high_grades_weight = 0.1
zero_weight = 0.1
ege_weight = 0.02

# Обновляем столбец 'Red_Diploma' на основе вычисленных вероятностей
data['Red_Diploma'] = data.apply(
    lambda row: calculate_red_diploma_probability(
        row['Average_Score'],
        row['High_Grades_Count'],
        row['Zero_Count'],
        row['High_Avg_Semesters_Count'],
        row[ege_scores].values,
        high_grades_weight,
        zero_weight,
        ege_weight,
        row['Probability']
    ), axis=1)

# Сохраняем результаты в Excel-файл
output_file_path = '/Users/jimsgood/Desktop/red_diploma_probabilities.xlsx'
data[['ID_студента', 'Average_Score', 'Red_Diploma']].to_excel(output_file_path, index=False)

# Сохраняем модель и масштабировщик
joblib.dump(model, 'logistic_regression_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Результаты успешно сохранены.")
print(X, type(X))