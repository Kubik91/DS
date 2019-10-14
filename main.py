import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # инструмент для создания и обучения модели
from sklearn import metrics  # инструменты для оценки точности модели


def main():
    df = pd.read_csv("files/main_task.csv")

    # Разбиваем датафрейм на части, необходимые для обучения и тестирования модели
    # x - данные с информацией о ресторанах, у - целевая переменная (рейтинги ресторанов)
    x = df.drop(['Restaurant_id', 'Rating'], axis=1)
    y = df['Rating']

    x = x.drop(['City', 'Reviews', 'URL_TA', 'ID_TA'], axis=1)
    x['Price Range'] = x['Price Range'].apply(
        lambda i: i if type(i) != str else sum([len(j.strip()) for j in i.split(' - ')]) / len([len(j.strip()) for j in i.split(' - ')]))
    #
    x['Cuisine Style'] = x['Cuisine Style'].apply(
        lambda i: len([j.strip() for j in eval(i)]) if type(i) == str else 0)
    #
    x = x.fillna({'Number of Reviews': x['Number of Reviews'].mean(), 'Price Range': x['Price Range'].mean()})

    # Наборы данных с меткой "train" будут использоваться для обучения модели, "test" - для тестирования.
    # Для тестирования мы будем использовать 25% от исходного датасета.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # Создаём модель
    regr = RandomForestRegressor(n_estimators=200)

    # Обучаем модель на тестовом наборе данных
    regr.fit(x_train, y_train)

    # Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.
    # Предсказанные значения записываем в переменную y_pred
    y_pred = regr.predict(x_test)
    print('MAE:', metrics.mean_absolute_error(y_test, y_pred))


if __name__ == '__main__':
    main()
