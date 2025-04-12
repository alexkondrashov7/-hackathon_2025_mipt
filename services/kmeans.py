import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os
from IPython.display import display

def load_data():
    """Загрузка всех исходных данных"""
    data_path = "../research/clean_data/"
    return {
        'customers': pd.read_csv(f"{data_path}customers.csv"),
        'orders': pd.read_csv(f"{data_path}orders.csv"),
        'items': pd.read_csv(f"{data_path}orders_items.csv"),
        'payments': pd.read_csv(f"{data_path}order_payments.csv"),
        'reviews': pd.read_csv(f"{data_path}order_reviews.csv"),
        'products': pd.read_csv(f"{data_path}products.csv"),
        'sellers': pd.read_csv(f"{data_path}sellers.csv"),
        'category_translation': pd.read_csv(f"{data_path}product_category_name_translation.csv")
    }

def merge_datasets(data):
    """Объединение данных в единый датафрейм"""
    df = data['orders'].merge(data['items'], on='order_id', how='left')
    df = df.merge(data['payments'], on='order_id', how='outer')
    df = df.merge(data['reviews'], on='order_id', how='outer')
    df = df.merge(data['products'], on='product_id', how='outer')
    df = df.merge(data['customers'], on='customer_id', how='outer')
    df = df.merge(data['sellers'], on='seller_id', how='outer')
    return df.merge(data['category_translation'], 
                   on='product_category_name', 
                   how='left')

def preprocess_data(df):
    """Очистка и первичная обработка данных"""
    df = df[~df['customer_unique_id'].isna()].dropna()
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    return df

def calculate_order_metrics(df):
    """Расчет метрик заказов"""
    return df.groupby('customer_unique_id').agg(
        total_orders=('order_id', 'nunique'),
        avg_payment=('payment_value', 'mean'),
        max_installments=('payment_installments', 'max'),
        total_spent=('price', 'sum'),
        avg_review=('review_score', 'mean')
    ).reset_index()

def calculate_time_features(df):
    """Расчет временных характеристик"""
    time_df = df.groupby('customer_unique_id')['order_purchase_timestamp'].agg([
        ('first_order', 'min'),
        ('last_order', 'max')
    ]).reset_index()
    
    time_df['order_duration'] = (time_df['last_order'] - time_df['first_order']).dt.total_seconds()
    return time_df

def calculate_categorical_features(df):
    """Определение категориальных признаков"""
    def get_mode(series):
        return series.mode()[0] if not series.mode().empty else None
    
    return df.groupby('customer_unique_id').agg(
        favorite_category=('product_category_name_english', get_mode),
        main_city=('customer_city', get_mode)
    ).reset_index()

def encode_features(df, cat_columns):
    """Кодирование категориальных признаков"""
    encoders = {}
    encoded_df = df.copy()
    
    for col in cat_columns:
        le = LabelEncoder()
        encoded_df[col] = le.fit_transform(encoded_df[col].astype(str))
        encoders[col] = le
    
    return encoded_df, encoders

def build_client_profiles():
    """Главный пайплайн обработки данных"""
    # Загрузка и объединение данных
    data = load_data()
    merged_df = merge_datasets(data)
    
    # Предобработка
    cleaned_df = preprocess_data(merged_df)
    
    # Расчет признаков
    order_metrics = calculate_order_metrics(cleaned_df)
    time_features = calculate_time_features(cleaned_df)
    categorical_features = calculate_categorical_features(cleaned_df)
    
    # Объединение результатов
    final_df = order_metrics.merge(time_features, on='customer_unique_id')
    final_df = final_df.merge(categorical_features, on='customer_unique_id')
    
    # Кодирование категорий
    encoded_df, _ = encode_features(final_df, ['favorite_category', 'main_city'])
    
    # Финализация данных
    return encoded_df.drop(columns=['first_order', 'last_order'])

def clustarisation(data):
    k_means_whole = KMeans(n_clusters=3, random_state=42)
    k_means_whole.fit(data.iloc[:,1:])

    data["labels"] = k_means_whole.labels_
    final_df = data.copy()[["customer_unique_id", "labels"]]
    
    return final_df



def save_labels(label_data, filename='labels/labels.json'):
    try:
        # Создаем директорию если нужно
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        label_data.to_json(filename, index=False)

    except Exception as e:
        print(f'Ошибка при сохранении лэйблов: {str(e)}')
    pass


# Запуск пайплайна
if __name__ == "__main__":
    client_data = build_client_profiles()
    clusters = clustarisation(client_data)
    save_labels(clusters, "results/labels/labels_kmeans.json")
    print("Job is done")