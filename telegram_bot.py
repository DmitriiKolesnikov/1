import logging
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.utils import executor
import joblib
import numpy as np
import pandas as pd


# Задайте ваш токен бота
API_TOKEN = '6301511057:AAF4TtPx-8ryeZSunYihqNi6CgeTGGxKOF0'

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Инициализация бота и диспетчера
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)
dp.middleware.setup(LoggingMiddleware())

# Загрузка моделей и масштабировщика
red_diploma_model = joblib.load('logistic_regression_model.pkl')
dropout_model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

# Список вопросов для красного диплома
red_diploma_questions = [
    f"введите ваш <b>возраст</b>:\n\nВаш возраст должен находится в пределах <b>16 и 80</b>",
    f"введите ваш <b>пол</b>:\n\nПри вводе данных вы должны учитывать, что <b>1'Мужской': 1</b>\n"
    f"<b>'Женский': 2</b>. ",
    f"введите ваш <b>курс</b>:\n\nПри вводе данных вы должны учитывать, что <b>'Первый'</b>: 1\n"
    f"<b>'Второй'</b>: 2\n"
    f"<b>'Третий'</b>: 3\n"
    f"<b>'Четвертый'</b>: 4\n"
    f"<b>'Пятый'</b>: 5\n"
    f"<b>'Шестой'</b>: 6\n"
    f"<b>'Седьмой'</b>: 7",
    f"укажите, были ли вы <b>отчислены</b>:\n\nПри вводе данных вы должны учитывать, что <b>'Нет': 1</b>\n"
    f"<b>'Да: 2'</b>",
    f"введите ваш <b>уровень подготовки</b>:\n\nПри вводе данных вы должны учитывать, что <b>'Бакалавриат'</b>: 1\n"
    f"<b>'Академический бакалавриат'</b>: 2\n"
    f"<b>'СПО'</b>: 3\n"
    f"<b>'Специалитет'</b>: 4\n"
    f"<b>'Магистратура'</b>: 5",
    f"<b>Введите формат обучения</b>:\n\nПри вводе данных вы должны учитывать, что <b>'Очная'</b>: 1\n"
    "<b>'Заочная'</b>: 2\n"
    "<b>'Очно-заочная'</b>: 3",
    f"введите ваши баллы за <b>ЕГЭ 1</b>:",
    f"введите ваши баллы за <b>ЕГЭ 2</b>:",
    f"введите ваши баллы за <b>ЕГЭ 3</b>:",
    f"введите ваши баллы за <b>ЕГЭ 4</b>:",
    f"введите <b>среднюю оценку</b> по всем предметам за <b>'Первое полугодие'</b>:",
    f"введите <b>среднюю оценку</b> по всем предметам за <b>'Второе полугодие'</b>:",
    f"введите <b>среднюю оценку</b> по всем предметам за <b>'Третье полугодие'</b>:",
    f"введите <b>среднюю оценку</b> по всем предметам за <b>'Четвертое полугодие'</b>:",
    f"введите <b>среднюю оценку</b> по всем предметам за <b>'Пятое полугодие'</b>:",
    f"введите <b>среднюю оценку</b> по всем предметам за <b>'Шестое полугодие'</b>:",
    f"введите <b>среднюю оценку</b> по всем предметам за <b>'Седьмое полугодие'</b>:",
    f"введите <b>среднюю оценку</b> по всем предметам за <b>'Восьмое полугодие'</b>:",
    f"введите <b>среднюю оценку</b> по всем предметам за <b>'Девятое полугодие'</b>:",
    f"введите <b>среднюю оценку</b> по всем предметам за <b>'Десятое полугодие'</b>:",
    f"введите <b>среднюю оценку</b> по всем предметам за <b>'Одиннадцатое полугодие'</b>:",
    f"введите <b>среднюю оценку</b> по всем предметам за <b>'Двенадцатое полугодие'</b>:",
]

# Список вопросов для вероятности не отчисления
dropout_questions = [
    f"введите ваш <b>возраст</b>т:\n\nВаш возраст должен находится в пределах <b>16 и 80</b>",
    f"введите ваш <b>пол<b>:\n\nПри вводе данных вы должны учитывать, что <b>1'Мужской': 1</b>\n"
    f"<b>'Женский': 2</b>. ",
    "введите,<b>работаете ли вы по специальности</b>:\n\nПри вводе данных вы должны учитывать, что <b>1'Нет': 1</b>\n"
    f"<b>'Да': 2</b>. ",
    "укажите, в каких <b>родственных отношениях</b> вы состоите:\n\nПри вводе данных вы должны учитывать, "
    "что <b>'Мать'</b>: 1\n"
    "<b>'Отец'</b>: 2\n"
    "<b>'Опекун'</b>: 4\n"
    "<b>'Брат'</b>: 5\n"
    "<b>'Иные физические лица'</b>: 6\n"
    "<b>'Сестра'</b>: 7\n"
    "<b>'Супруга'</b>: 8\n"
    "<b>'Супруг'</b>: 9\n"
    "<b>'Дочь'</b>: 10\n"
    "<b>'Сын'</b>: 11,\n"
    "<b>'Отчим'</b>: 12 ",
    "укажите, обладаете ли вы какими - либо <b>льготами/особыми отметками</b>:\n\nПри вводе данных вы должны учитывать,"
    " что <b>'Инвалид с детства'</b>: 1\n"
    "<b>'Средний балл аттестата'</b>: 3\n"
    "<b>'Призер олимпиады'</b>: 4\n"
    "<b>'Направление Минобрнауки России'</b>: 5\n"
    "<b>'Победитель олимпиады'</b>: 6\n"
    "<b>'Лица из числа детей-сирот'</b>: 7\n"
    "<b>'Средний балл диплома'</b>: 8\n"
    "<b>'Дети-инвалиды'</b>: 9\n"
    "<b>'Лица, получившие государственную социальную помощь'</b>: 10\n"
    "<b>'Дети, оставшиеся без попечения родителей'</b>: 11\n"
    "<b>'Лица из числа детей, оставшихся без попечения родителей' 'Целевой прием'</b>: 12\n"
    "'<b>Дети военнослужащих, сотрудников, направленных в другие государства за исключением погибших, получивших увечье"
    " или заболевание'</b>: 13\n"
    "<b>'Граждане, которые подверглись воздействию радиации вследствие катастрофы на Чернобыльской АЭС'</b>: 14'\n"
    "<b>'Лица, имеющие право на получение гос. соц. помощи (малоимущие)'</b>: 15\n"
    "<b>'Дети лиц, принимавших участие в СВО за исключением погибших, получивших увечье или заболевание'</b>: 16\n"
    "</b>'Ветеран боевых действий'</b>: 17\n"
    "<b>'Дети военнослужащих и сотрудников, за исключением погибших'</b>: 18\n"
    "<b>'Лица, потерявшие в период обучения обоих родителей или единственного родителя'</b>: 19\n"
    "<b>'Дети лиц, принимавших участие в СВО погибших, получивших увечье или заболевание'</b>: 20\n"
    "<b>'Дети военнослужащих, погибших при исполнении ими обязанностей военной службы'</b>: 21\n"
    "<b>'Военнослужащие, проходящие военную службу по контракту не менее 3 лет'</b>: 22\n"
    "<b>'Инвалид 1-ой группы'</b: 23\n"
    "<b>'Инвалид 2-ой группы'</b>: 24\n"
    "<b>'Лица из числа детей, оставшихся без попечения родителей'</b>: 25\n"
    "<b>'Дети-сироты'</b>: 26\n"
    "<b>'Целевой прием'</b>: 27",
    f"введите ваш <b>курс</b>:\n\nПри вводе данных вы должны учитывать, что <b>'Первый'</b>: 1\n"
    f"<b>'Второй'</b>: 2\n"
    f"<b>'Третий'</b>: 3\n"
    f"<b>'Четвертый'</b>: 4\n"
    f"<b>'Пятый'</b>: 5\n"
    f"<b>'Шестой'</b>: 6\n"
    f"<b>'Седьмой'</b>: 7",
    "введите ваш <b>уровень подготовки<b>:\n\nПри вводе данных вы должны учитывать, что <b>'Бакалавриат'</b>: 1\n"
    "<b>'Академический бакалавриат'</b>: 2\n"
    "<b>'СПО'</b>: 3\n"
    "<b>'Специалитет'</b>: 4\n"
    "<b>'Магистратура'</b>: 5",
    "введите вашу <b>форму обучения</b>:\n\nПри вводе данных вы должны учитывать, что <b>'Очная'</b>: 1\n"
    "<b>'Заочная'</b>: 2\n"
    "<b>'Очно-заочная'</b>: 3",
    "введите ваше <b>состояние внутри ВУЗа</b>:\n\nПри вводе данных вы должны учитывать, что <b>'Студент'</b>: 1\n"
    "<b>'АКО'</b>: 2\n"
    "<b>'Обучается за границей'</b>: 3",
    "укажите, являетесь ли вы <b>призером олимпиад</b>:\n\nПри вводе данных вы должны учитывать, что <b>'Нет'</b>: 1\n"
    "<b>'Да'</b>: 2",
    "укажите <b>результаты ЕГЭ по Английскому языку</b>:\n\nЕсли вы не писали данный экзамен, то в качестве ответа "
    "пришлите '0'",
    "Укажите <b>результаты ЕГЭ по Информатике и ИКТ</b>:\n\nЕсли вы не писали данный экзамен, то в качестве ответа "
    "пришлите '0'",
    "укажите <b>результаты ЕГЭ по Истории</b>:\n\nЕсли вы не писали данный экзамен, то в качестве ответа пришлите '0'",
    "укажите <b>результаты ЕГЭ по Математике</b>:\n\nЕсли вы не писали данный экзамен, "
    "то в качестве ответа пришлите '0'",
    "укажите <b>результаты ЕГЭ по Немецкому языку</b>:\n\nЕсли вы не писали данный экзамен, "
    "то в качестве ответа пришлите '0'",
    "укажите <b>результаты ЕГЭ по Обществознанию</b>:\n\nЕсли вы не писали данный экзамен, "
    "то в качестве ответа пришлите '0'",
    "укажите <b>результаты ЕГЭ по Русскому языку</b>:\n\nЕсли вы не писали данный экзамен, "
    "то в качестве ответа пришлите '0'",
    "укажите <b>результаты ЕГЭ по Физике</b>:\n\nЕсли вы не писали данный экзамен, то в качестве ответа пришлите '0'",
    "укажите <b>результаты ЕГЭ по Французскому языку</b>:\n\nЕсли вы не писали данный экзамен, то в качестве ответа "
    "пришлите '0'",
    "укажите <b>результаты ЕГЭ по Химии<b>:\n\nЕсли вы не писали данный экзамен, то в качестве ответа пришлите '0'",
    f"введите <b>среднюю оценку</b> по всем предметам за <b>'Первое полугодие'</b>:",
    f"введите <b>среднюю оценку</b> по всем предметам за <b>'Второе полугодие'</b>:",
    f"введите <b>среднюю оценку</b> по всем предметам за <b>'Третье полугодие'</b>:",
    f"введите <b>среднюю оценку</b> по всем предметам за <b>'Четвертое полугодие'</b>:",
    f"введите <b>среднюю оценку</b> по всем предметам за <b>'Пятое полугодие'</b>:",
    f"введите <b>среднюю оценку</b> по всем предметам за <b>'Шестое полугодие'</b>:",
    f"введите <b>среднюю оценку</b> по всем предметам за <b>'Седьмое полугодие'</b>:",
    f"введите <b>среднюю оценку</b> по всем предметам за <b>'Восьмое полугодие'</b>:",
    f"введите <b>среднюю оценку</b> по всем предметам за <b>'Девятое полугодие'</b>:",
    f"введите <b>среднюю оценку</b> по всем предметам за <b>'Десятое полугодие'</b>:",
    f"введите <b>среднюю оценку</b> по всем предметам за <b>'Одиннадцатое полугодие'</b>:",
    f"введите <b>среднюю оценку</b> по всем предметам за <b>'Двенадцатое полугодие'</b>:"
]

user_data = {}

main_kb = ReplyKeyboardMarkup(resize_keyboard=True).add(
    KeyboardButton("Красный диплом"),
    KeyboardButton('Вероятность отчисления'),
    KeyboardButton('Инструкция')
)


@dp.message_handler(commands=['start'])
async def start_handler(message: types.Message):
    await bot.send_photo(chat_id=message.from_user.id,
                         photo="https://i.postimg.cc/LXDr83qH/image.jpg",
                         caption=f"<b>{message.from_user.full_name}</b> "
                         f"добро пожаловать в наш Telegram бот! \n\nЭтот бот создан специально для того, "
                         f"чтобы помочь вам лучше понять ваши академические перспективы.",
                         parse_mode="HTML")
    await bot.send_message(chat_id=message.from_user.id,
                           text=f"Наш бот обладает "
                           f"двумя основными функциями:\n\n"
                           f"Рассчитать вероятность получения красного диплома: используя современные методы "
                           f"машинного обучения, мы проанализируем ваши учебные достижения и предскажем, "
                           f"насколько высока вероятность того, что вы получите красный диплом. Этот диплом — "
                           f"символ ваших выдающихся успехов в учебе и упорства.\n\n"
                           f"<b>Рассчитать вероятность отчисления</b>: узнайте, каковы ваши риски не закончить "
                           f"учебу. "
                           f"Наша модель оценит ваши текущие результаты и может помочь вам понять, где "
                           f"необходимо приложить дополнительные усилия, чтобы избежать неприятных ситуаций "
                           f"и успешно продолжить обучение.\n\n"
                           f"Для того чтобы получить максимально точные результаты, пожалуйста, <b>внимательно"
                           f" прочитайте инструкции при вводе данных</b>. От качества и точности введенной "
                           f"информации зависит корректность наших прогнозов. Правильный ввод данных "
                           f"поможет нашей модели машинного обучения предоставить <b>наиболее точные</b> и "
                           f"персонализированные рекомендации\n\n"
                           f"Желаем вам <b>успехов в учебе и яркого студенческого времени!</b>",
                           parse_mode="HTML",
                           reply_markup=main_kb)
    await message.delete()


@dp.message_handler(lambda message: message.text == "Красный диплом")
async def red_diploma_handler(message: types.Message):
    user_data[float(message.from_user.id)] = {
        'step': 0,
        'answers': [],
        'type': 'red_diploma'
    }
    await bot.send_message(chat_id=message.from_user.id,
                           text=f'Уважаемый <b>{message.from_user.full_name}</b>, {red_diploma_questions[0]}',
                           parse_mode="HTML")
    await message.delete()


@dp.message_handler(lambda message: message.text == "Инструкция")
async def instruction_handler(message: types.Message):
    await bot.send_message(chat_id=message.from_user.id,
                           text=f"Уважаемый <b>{message.from_user.full_name}</b>!\n\n"
                                f"Вот что вам необходимо сделать:\n"
                                f"<b>Подготовьте все необходимые данные</b>: убедитесь, что у вас под рукой имеются все "
                                f"ваши академические достижения, такие как оценки, текущий средний балл, количество "
                                f"пропущенных занятий, а также любая другая значимая информация."
                                f"<b>Вводите данные аккуратно</b>: малейшая ошибка может повлиять на результаты. "
                                f"Пожалуйста, "
                                f"вводите информацию точно и корректно.\n"
                                f"<b>Следуйте инструкциям</b>: наш бот будет задавать вопросы шаг за шагом. Обязательно "
                                f"читайте каждую инструкцию перед тем, как отвечать на вопросы."
                                f"Мы надеемся, что наш бот станет вашим надежным помощником на пути к академическому "
                                f"успеху. Мы непрерывно работаем над улучшением наших прогнозных моделей, чтобы "
                                f"предоставлять вам наиболее точную и полезную информацию.\n\n"
                                f"Если у вас возникнут вопросы или проблемы при использовании бота, "
                                f"вы всегда можете обратиться к нашей службе поддержки.",
                           parse_mode="HTML")
    await message.delete()


@dp.message_handler(lambda message: message.text == "Вероятность отчисления")
async def dropout_handler(message: types.Message):
    user_data[float(message.from_user.id)] = {
        'step': 0,
        'answers': [],
        'type': 'dropout'
    }
    await bot.send_message(chat_id=message.from_user.id,
                           text=f'Уважаемый <b>{message.from_user.full_name}</b>, {dropout_questions[0]}',
                           parse_mode='HTML')
    await message.delete()


@dp.message_handler(lambda message: message.from_user.id in user_data)
async def process_question(message: types.Message):
    user_id = message.from_user.id
    user_info = user_data[user_id]

    user_info['answers'].append(message.text)
    questions = red_diploma_questions if user_info['type'] == 'red_diploma' else dropout_questions

    if user_info['step'] < len(questions) - 1:
        user_info['step'] += 1
        await message.reply(questions[user_info['step']])
    else:
        if user_info['type'] == 'red_diploma':
            await calculate_red_diploma_probability(message, user_info['answers'])
        else:
            await calculate_dropout_probability(message, user_info['answers'])
        del user_data[user_id]


async def calculate_red_diploma_probability(message, answers):
    try:
        # Название колонок для ДатаСета
        columns = [
            'Возраст', 'Пол', 'Курс', 'Отчислен', 'Уровень подготовки', 'Формат обучения',
            'Балл_ЕГЭ_1', 'Балл_ЕГЭ_2', 'Балл_ЕГЭ_3', 'Балл_ЕГЭ_4', 'Полугодие_1',
            'Полугодие_2', 'Полугодие_3', 'Полугодие_4', 'Полугодие_5', 'Полугодие_6',
            'Полугодие_7', 'Полугодие_8', 'Полугодие_9', 'Полугодие_10', 'Полугодие_11',
            'Полугодие_12'
        ]

        # Названия колонок по их принадлежности

        semesters = [
            'Полугодие_1', 'Полугодие_2', 'Полугодие_3', 'Полугодие_4', 'Полугодие_5',
            'Полугодие_6', 'Полугодие_7', 'Полугодие_8', 'Полугодие_9', 'Полугодие_10',
            'Полугодие_11', 'Полугодие_12'
        ]

        ege_scores = ['Балл_ЕГЭ_1', 'Балл_ЕГЭ_2', 'Балл_ЕГЭ_3', 'Балл_ЕГЭ_4']

        common_data = ['Возраст', 'Пол', 'Курс', 'Отчислен', 'Уровень подготовки', 'Формат обучения']

        # Преобразование ответов в числовые значения
        numeric_answers = [float(f) for f in answers]

        # Создание ДатаСета
        df = pd.DataFrame([numeric_answers], columns=columns)

        # Выборка оценок (предполагается, что они начинаются с третьего элемента)
        grades = numeric_answers[6:]

        # Рассчет дополнительных переменных

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

        df['Average_Score'] = df[semesters].apply(calculate_avg, axis=1)
        df['High_Grades_Count'] = df[semesters].apply(count_high_grades, axis=1)
        df['Zero_Count'] = df[semesters].apply(count_zeros, axis=1)
        df['High_Avg_Semesters_Count'] = df[semesters].apply(count_semesters_with_high_avg, axis=1)

        Average = calculate_avg(grades[11:])
        High_Grades_Count = count_high_grades(grades[11:])
        Zero_Count = count_zeros(grades[11:])
        High_Avg_Semesters_Count = count_semesters_with_high_avg(grades[11:])

        # Добавление дополнительных переменных в список данных
        extended_answers = numeric_answers + [High_Grades_Count, Zero_Count, High_Avg_Semesters_Count]

        # Преобразование данных в np.array
        features = np.array(extended_answers, dtype=np.float64).reshape(1, -1)
    except ValueError as e:
        await message.reply(f"Возникла ошибка при преобразовании введённых данных: {e}")
        return

    # Предсказание вероятности получения красного диплома
    probability = red_diploma_model.predict_proba(features)[0, 1]
    df['Probability'] = probability

    def red_diploma_probability(avg_score, high_grades_count, zero_count, high_avg_semesters_count, ege_scores,
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
    df['Red_Diploma'] = df.apply(
        lambda row: red_diploma_probability(
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

    red_diploma_value = df['Red_Diploma'][0]
    red_diploma_value = float(red_diploma_value) * 100
    print(red_diploma_value)
    await message.reply(f"Вероятность получения красного диплома: {red_diploma_value:}%")


async def calculate_dropout_probability(message, answers):
    try:
        formatted_answers = {}
        for idx, answer in enumerate(answers):
            col_name = dropout_questions[idx]
            formatted_answers[col_name] = [answer]

        df_user = pd.DataFrame.from_dict(formatted_answers)

        # Убедимся, что в DataFrame есть все необходимые столбцы
        required_cols = dropout_model.feature_names_in_
        for col in required_cols:
            if col not in df_user.columns:
                df_user[col] = 0

        y_proba = dropout_model.predict_proba(df_user)[:, 1]
        await message.reply(f"Вероятность НЕ отчисления: {y_proba[0]:.2%}")

    except ValueError as e:
        await message.reply(f"Возникла ошибка при формировании введённых данных: {e}")


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)