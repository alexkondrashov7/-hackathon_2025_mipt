import os
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.filterwarnings("ignore")

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
    customers, geolocation, order_pay, reviews, orders, item, category_name, products, sellers = load_data()
    df = merge_data(orders, item, order_pay, reviews, products,
                    customers, sellers, category_name)
    df = filter_customers(df)
    return df

# Обработка и анализ данных


def process_data(df):
    # Приводим столбцы к нужному типу данных
    df['order_purchase_timestamp'] = pd.to_datetime(
        df['order_purchase_timestamp'])
    df['customer_unique_id'] = df['customer_unique_id'].astype(str)

    # Убираем строки с отсутствующими customer_id или order_purchase_timestamp
    df = df.dropna()

    # RFM
    current_date = df['order_purchase_timestamp'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('customer_unique_id').agg({
        'order_purchase_timestamp': lambda x: (current_date - x.max()).days,
        'order_id': 'count',
        'payment_value': 'sum'
    }).rename(columns={
        'order_purchase_timestamp': 'recency',
        'order_id': 'frequency',
        'payment_value': 'monetary'
    }).reset_index()

    # Применяем квантильный анализ
    rfm['r_quartile'] = pd.qcut(
        rfm['recency'], 4, labels=False, duplicates='drop')
    rfm['f_quartile'] = pd.qcut(
        rfm['frequency'], 4, labels=False, duplicates='drop')
    rfm['m_quartile'] = pd.qcut(
        rfm['monetary'], 4, labels=False, duplicates='drop')
    rfm['rfm_score'] = rfm[['r_quartile',
                            'f_quartile', 'm_quartile']].sum(axis=1)

    weights = {'R': 0.5, 'F': 0.3, 'M': 0.2}
    rfm['RFM_Weighted'] = (rfm['r_quartile'] * weights['R'] +
                           rfm['f_quartile'] * weights['F'] +
                           rfm['m_quartile'] * weights['M'])

    rfm['Churn_Risk'] = pd.qcut(rfm['RFM_Weighted'], q=[0, 0.25, 0.75, 1], labels=[
                                'Высокий риск', 'Средний риск', 'Низкий риск'])

    # ABC
    rfm = rfm.sort_values('monetary', ascending=False)
    rfm['cumulative_value'] = rfm['monetary'].cumsum()
    total_value = rfm['monetary'].sum()
    rfm['cumulative_percent'] = rfm['cumulative_value'] / total_value * 100

    def assign_abc_category(row):
        if row['cumulative_percent'] <= 80:
            return 'A'
        elif row['cumulative_percent'] <= 95:
            return 'B'
        else:
            return 'C'

    rfm['abc_class'] = rfm.apply(assign_abc_category, axis=1)

    # XYZ
    rfm['std_dev'] = df.groupby('customer_unique_id')[
        'payment_value'].transform(lambda x: x.std())
    rfm['x_category'] = pd.cut(
        rfm['std_dev'], bins=[-1, 0.01, 50, float('inf')], labels=['X', 'Y', 'Z'])
    rfm['x_category'] = rfm['x_category'].cat.add_categories(
        ['Single Purchase'])
    rfm['x_category'] = rfm['x_category'].fillna('Single Purchase')

    new_categories = ['X', 'Y', 'Z', 'Single Purchase']
    rfm['x_category'] = pd.Categorical(
        rfm['x_category'], categories=new_categories)

    # Сегментация клиентов
    segments = rfm.copy()
    segments['abc_class'] = segments['abc_class'].astype(str)
    segments['x_category'] = segments['x_category'].astype(str)
    segments['segment'] = segments['abc_class'] + "_" + segments['x_category']

    segment_descriptions = {
        'A_Single Purchase': 'Клиенты с одной покупкой, высокий денежный объем.',
        'B_Single Purchase': 'Клиенты с одной покупкой, средний денежный объем.',
        'C_Single Purchase': 'Клиенты с одной покупкой, низкий денежный объем.',
        'A_X': 'Высокоприбыльные клиенты, низкая вариативность.',
        'B_X': 'Клиенты со средней частотой и объемом, низкая вариативность.',
        'C_X': 'Клиенты с низким объемом, низкая вариативность.',
        'A_Y': 'Высокоприбыльные клиенты с разумной вариативностью.',
        'B_Y': 'Средние клиенты с некоторой вариативностью.',
        'C_Y': 'Клиенты с низким объемом и частотой, но с вариативностью.',
        'A_Z': 'Высокоприбыльные клиенты с большой вариативностью.',
        'B_Z': 'Средние клиенты с высокой вариативностью.',
        'C_Z': 'Низкие клиенты с высокой вариативностью.',
    }

    segments['segment_description'] = segments['segment'].map(
        segment_descriptions)

    return segments


def save_plots(segments):
    # Визуализация распределения по сегментам
    plt.figure(figsize=(10, 6))
    sns.countplot(data=segments, x='segment_description',
                  order=segments['segment_description'].value_counts().index, palette='viridis')
    plt.title('Distribution of Customer Segments')
    plt.xlabel('Segment')
    plt.ylabel('Number of Customers')
    plt.xticks(rotation=90)
    plt.savefig('./results/plots/customer_segments_distribution.png')
    plt.close()

    # Визуализация распределения по категориям риска
    plt.figure(figsize=(10, 6))
    sns.countplot(data=segments, x='Churn_Risk', palette='viridis')
    plt.title('Distribution of Churn Risk Segments')
    plt.xlabel('Churn Risk')
    plt.ylabel('Number of Customers')
    plt.savefig('./results/plots/churn_risk_distribution.png')
    plt.close()


def time_plot(df, feat):
    df_time = df.copy()
    # Преобразуем дату в datetime, если не было сделано ранее
    df_time['order_purchase_timestamp'] = pd.to_datetime(
        df['order_purchase_timestamp'])
    # Установите индекс на order_purchase_timestamp
    df_time.set_index('order_purchase_timestamp', inplace=True)

    # Группировка данных
    grouped_1 = df_time.groupby(
        [pd.Grouper(freq='M'), f'{feat}']).size().unstack(fill_value=0)

    # Преобразуем обратно в длинный формат
    grouped_1 = grouped_1.reset_index()
    grouped_melted = pd.melt(grouped_1, id_vars='order_purchase_timestamp',
                             var_name=f'{feat}', value_name='count')

    # Построение графика с использованием Seaborn
    plt.figure(figsize=(20, 10))
    sns.barplot(data=grouped_melted, x='order_purchase_timestamp',
                y='count', hue=f'{feat}', palette='muted')
    plt.title('Количество заказов по сегментам по месяцам')
    plt.xlabel('Дата')
    plt.ylabel('Количество заказов')
    plt.xticks(rotation=45)
    plt.legend(title='Сегмент')
    plt.tight_layout()
    plt.savefig(f'./results/plots/churn_risk_by_time_via_{feat}.png')


def save_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data.to_dict(orient='records'),
                  f, ensure_ascii=False, indent=4)


# Основная функция
if __name__ == "__main__":
    data = main_pipeline()

    data_time = data[["customer_unique_id", "order_purchase_timestamp"]]
    data['order_purchase_timestamp'] = pd.to_datetime(
        data['order_purchase_timestamp'])

    processed_data = process_data(data)
    data_labels_time = processed_data[[
        "customer_unique_id", "segment_description", "Churn_Risk"]]

    work_df = data_time.merge(
        data_labels_time, on="customer_unique_id", how="left")

    time_plot(work_df, feat="segment_description")
    time_plot(work_df, feat="Churn_Risk")

    save_to_json(processed_data, './results/labels/customer_segments.json')
    save_plots(processed_data)
