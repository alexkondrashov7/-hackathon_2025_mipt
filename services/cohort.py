import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from operator import attrgetter

# Загрузка данных


def load_data():
    customers = pd.read_csv("../research/clean_data/customers.csv")
    geolocation = pd.read_csv("../research/clean_data/geolocation.csv")
    order_pay = pd.read_csv("../research/clean_data/order_payments.csv")
    reviews = pd.read_csv("../research/clean_data/order_reviews.csv")
    orders = pd.read_csv("../research/clean_data/orders.csv")
    item = pd.read_csv("../research/clean_data/orders_items.csv")
    category_name = pd.read_csv(
        "../research/clean_data/product_category_name_translation.csv")
    products = pd.read_csv("../research/clean_data/products.csv")
    sellers = pd.read_csv("../research/clean_data/sellers.csv")
    return customers, geolocation, order_pay, reviews, orders, item, category_name, products, sellers


# Объединение данных
def merge_data(orders, item, order_pay, reviews, products, customers, sellers, category_name):
    df = orders.merge(item, on='order_id', how='left')
    df = df.merge(order_pay, on='order_id', how='outer', validate='m:m')
    df = df.merge(reviews, on='order_id', how='outer')
    df = df.merge(products, on='product_id', how='outer')
    df = df.merge(customers, on='customer_id', how='outer')
    df = df.merge(sellers, on='seller_id', how='outer')
    df = df.merge(category_name, on="product_category_name", how="left")
    return df

# Очистка данных: удаление строк без customer_unique_id


def filter_customers(df):
    return df[~df["customer_unique_id"].isna()]


# Главная функция пайплайна
def main_pipeline():
    # Шаг 1: Загрузка данных
    customers, geolocation, order_pay, reviews, orders, item, category_name, products, sellers = load_data()

    # Шаг 2: Объединение данных
    df = merge_data(orders, item, order_pay, reviews, products,
                    customers, sellers, category_name)

    # Шаг 3: Фильтрация данных
    df = filter_customers(df)

    final_data = df

    return final_data


def data_preprocessing():
    data = main_pipeline()
    data.dropna(inplace=True)

    # преобразуем дату в формат datetime
    data['order_purchase_timestamp'] = pd.to_datetime(
        data['order_purchase_timestamp'])

    # определим дату первого заказа для каждого клиента
    data['cohort_month'] = data.groupby('customer_unique_id')[
        'order_purchase_timestamp'].transform('min').dt.to_period('M')

    # отдельная колонка: месяц покупки (для расчёта времени от когорты)
    data['order_month'] = data['order_purchase_timestamp'].dt.to_period('M')

    # cчитаем разницу между месяцем заказа и месяцем когорты (возраст когорты в месяцах)
    data['cohort_index'] = (data['order_month'] -
                            data['cohort_month']).apply(attrgetter('n'))

    # создаём когортную таблицу
    cohort_data = data.groupby(['cohort_month', 'cohort_index'])[
        'customer_unique_id'].nunique().reset_index()
    cohort_pivot = cohort_data.pivot(
        index='cohort_month', columns='cohort_index', values='customer_unique_id')

    # нормализуем по первому месяцу когорты (retention)
    cohort_size = cohort_pivot.iloc[:, 0]
    retention = cohort_pivot.divide(cohort_size, axis=0)

    # cтроим когортную таблицу по количеству пользователей
    cohort_data = data.groupby(['cohort_month', 'cohort_index'])[
        'customer_unique_id'].nunique().unstack(1)

    # группируем пользователей по региону, когортному месяцу и индексу
    regional_cohort = data.groupby(['customer_state', 'cohort_month', 'cohort_index'])[
        'customer_unique_id'].nunique().reset_index()

    # теперь по одному региону строим сводную таблицу
    state_list = data['customer_state'].unique()

    return state_list, retention, cohort_data, regional_cohort


def cohort_heatmap_churn(retention):
    plt.figure(figsize=(12, 8))
    sns.heatmap(retention, annot=True, fmt='.0%', cmap='Blues')
    plt.title('Когортный анализ: удержание по месяцам')
    plt.ylabel('Когорта (месяц первой покупки)')
    plt.xlabel('Месяцы с момента первой покупки')
    plt.savefig('./results/plots/cohort/cohort_churn_by_month.png')


def cohort_heatmap_customers(cohort_data):
    plt.figure(figsize=(12, 6))
    sns.heatmap(cohort_data, annot=True, fmt='g', cmap='Blues')
    plt.title('Когортный анализ: количество уникальных покупателей')
    plt.ylabel('Когорта (месяц первой покупки)')
    plt.xlabel('Месяц с момента первой покупки')
    plt.savefig('./results/plots/cohort/cohort_unique_users.png')


def cohort_churn_by_region(state_list, regional_cohort):
    for state in state_list:
        cohort_state = regional_cohort[regional_cohort['customer_state'] == state]
        cohort_pivot = cohort_state.pivot(
            index='cohort_month', columns='cohort_index', values='customer_unique_id')

        if cohort_pivot.empty or cohort_pivot.shape[1] < 2:
            continue

        # рассчитаем retention (%)
        cohort_size = cohort_pivot.iloc[:, 0]
        retention = cohort_pivot.divide(cohort_size, axis=0)

        # построим тепловую карту
        plt.figure(figsize=(10, 6))
        sns.heatmap(retention, annot=True, fmt='.0%', cmap='YlGnBu')
        plt.title(f'Когортный анализ удержания - регион {state}')
        plt.ylabel('Когорта (месяц первой покупки)')
        plt.xlabel('Месяцы с момента первой покупки')
        plt.tight_layout()
        plt.savefig(
            f'./results/plots/cohort/region/cohort_churn_by_region_{state}.png')


def cohort_churn_by_month(state_list, regional_cohort):
    for state in state_list:
        cohort_state = regional_cohort[regional_cohort['customer_state'] == state]
        cohort_pivot = cohort_state.pivot(
            index='cohort_month', columns='cohort_index', values='customer_unique_id')

        if cohort_pivot.empty or cohort_pivot.shape[1] < 2:
            continue

        # построение heatmap без нормализации - просто количество покупателей
        plt.figure(figsize=(10, 6))
        sns.heatmap(cohort_pivot, annot=True, fmt='.0f', cmap='YlGnBu')
        plt.title(f'Когортный анализ (в числах) - регион {state}')
        plt.ylabel('Когорта (месяц первой покупки)')
        plt.xlabel('Месяцы с момента первой покупки')
        plt.tight_layout()
        plt.savefig(
            f'./results/plots/cohort/month/cohort_churn_by_month_{state}.png')


if __name__ == "__main__":
    state_list, retention, cohort_data, regional_cohort = data_preprocessing()
    cohort_heatmap_churn(retention)
    cohort_heatmap_customers(cohort_data)
    cohort_churn_by_region(state_list, regional_cohort)
    cohort_churn_by_month(state_list, regional_cohort)
