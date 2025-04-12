import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from IPython.display import display

import warnings
warnings.filterwarnings("ignore")


# Загрузка данных (в будущем этот и следующий шаг можно заменить на выгрузку sql)
def load_data():
    """Загрузка всех исходных данных"""
    data_path = "../research/clean_data/"
    return {
        'customers': pd.read_csv(f"{data_path}customers.csv"),
        'geolocation': pd.read_csv(f"{data_path}geolocation.csv"),
        'payments': pd.read_csv(f"{data_path}order_payments.csv"),
        'reviews': pd.read_csv(f"{data_path}order_reviews.csv"),
        'orders': pd.read_csv(f"{data_path}orders.csv"),
        'items': pd.read_csv(f"{data_path}orders_items.csv"),
        'products': pd.read_csv(f"{data_path}products.csv"),
        'sellers': pd.read_csv(f"{data_path}sellers.csv"),
        'category_translation': pd.read_csv(f"{data_path}product_category_name_translation.csv")
    }


# Объединение данных
def merge_data(data):
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
    # return df[~df["customer_unique_id"].isna()]


def calculate_rfm(df):
    """Расчет RFM-метрик с обработкой исключений"""
    try:
        df['order_purchase_timestamp'] = pd.to_datetime(
            df['order_purchase_timestamp'])
        current_date = df['order_purchase_timestamp'].max()

        return df.groupby('customer_unique_id').agg(
            Recency=(
                'order_purchase_timestamp',
                lambda x: (current_date - x.max()).days),
            Frequency=('order_id', 'nunique'),
            Monetary=('product_id', 'count')
        ).reset_index()
    except KeyError as e:
        print(f"Ошибка: Отсутствует необходимая колонка {e}")
        return pd.DataFrame()


def add_rfm_segments(rfm_data):
    """Добавление RFM-сегментов с весами"""
    quantiles = rfm_data[['Recency', 'Frequency', 'Monetary']].quantile(
        [0.25, 0.5, 0.75]).to_dict()

    def rank_rfm(x, metric):
        """
        Параметры:
        x : float - значение метрики (Recency, Frequency, Monetary) для клиента
        metric : str - название метрики ('Recency', 'Frequency', 'Monetary')
        quantiles : dict - словарь с квартилями для всех метрик

        Возвращает:
        int - ранг от 1 до 4, где для Recency 4 - лучший, для Frequency/Monetary 4 - худший

        Примечание: 
        Для Recency используется обратная шкала (меньше дней = лучше)
        Для Frequency и Monetary - прямая шкала (больше значений = лучше)
        Квартили рассчитываются для всей популяции клиентов

        """

        # Логика ранжирования для Recency
        if metric == 'Recency':
            # Чем меньше дней прошло с последней покупки (ниже Recency), тем лучше
            if x <= quantiles[metric][0.25]:
                return 4  # Топ-25% самых активных (покупали недавно)
            elif x <= quantiles[metric][0.5]:
                return 3  # 25-50% - выше среднего
            elif x <= quantiles[metric][0.75]:
                return 2  # 50-75% - ниже среднего
            else:
                return 1  # Худшие 25% - давно не покупали

        # Логика ранжирования для Frequency и Monetary
        else:
            # Чем больше покупок/сумма (выше значения), тем лучше
            if x <= quantiles[metric][0.25]:
                return 1  # Худшие 25% - редко/мало покупают
            elif x <= quantiles[metric][0.5]:
                return 2  # 25-50% - ниже среднего
            elif x <= quantiles[metric][0.75]:
                return 3  # 50-75% - выше среднего
            else:
                return 4  # Топ-25% - самые частые/крупные покупатели

    for metric in ['Recency', 'Frequency', 'Monetary']:
        rfm_data[f'{metric[0]}_rank'] = rfm_data[metric].apply(
            rank_rfm, metric=metric)

    # Взвешенная оценка
    weights = {'R': 0.5, 'F': 0.3, 'M': 0.2}
    rfm_data['RFM_Weighted'] = sum(
        rfm_data[f'{k}_rank']*v for k, v in weights.items())

    # Автоматическая классификация
    rfm_data['Churn_Risk'] = pd.qcut(
        rfm_data['RFM_Weighted'],
        q=[0, 0.25, 0.75, 1],
        labels=['3_high', '2_medium', '1_low']  # Формат для сортировки
    )
    return rfm_data

def save_churn_plot(rfm_data, filename='plots/churn_risk_distribution.png'):
    """Сохранение графика распределения рисков оттока"""
    try:
        # Создаем директорию если нужно
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        plt.figure(figsize=(12, 6))

        # Используем фактические метки из данных
        plot_order = ['3_high', '2_medium', '1_low']
        palette = {'3_high': 'red', '2_medium': 'orange', '1_low': 'green'}

        ax = sns.countplot(
            x='Churn_Risk',
            data=rfm_data,
            order=plot_order,
            palette=palette
        )

        # Добавляем аннотации
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.0f}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center',
                        va='center',
                        xytext=(0, 5),
                        textcoords='offset points')

        # Русские подписи для визуализации
        plt.title('Распределение клиентов по риску оттока')
        plt.xlabel('Категория риска')
        plt.ylabel('Количество клиентов')
        plt.xticks(
            ticks=range(len(plot_order)),
            labels=['Высокий риск', 'Средний риск', 'Низкий риск'],
            rotation=45
        )

        # Сохраняем и закрываем график
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'График сохранен в {filename}')

    except Exception as e:
        print(f'Ошибка при сохранении графика: {str(e)}')


def save_labels(label_data, filename='labels/labels.json'):
    try:
        # Создаем директорию если нужно
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        label_data.to_json(filename, index=False)

    except Exception as e:
        print(f'Ошибка при сохранении лэйблов: {str(e)}')
    pass


def main_pipeline():
    """Главный конвейер обработки данных"""
    # Шаг 1: Загрузка данных
    data = load_data()

    # Шаг 2: Объединение данных
    df = merge_data(data)

    # Шаг 3: Фильтрация пропусков
    data = preprocess_data(df)

    # Шаг 4: Расчет rfm
    rfm_raw = calculate_rfm(data)

    # Шаг 5: Объединение данных
    rfm_segment = add_rfm_segments(rfm_raw)

    # Шаг 6: Сохраняем график для дашборда
    save_churn_plot(rfm_segment, 'results/plots/churn_distribution_rfm.png')

    # Шаг 7: Вычленяем лэйблы для классификации
    label_data = rfm_segment.copy()[['customer_unique_id', 'Churn_Risk']]
    risk_mapping = {'3_high': 3, '2_medium': 2, '1_low': 1}
    label_data['Churn_Risk'] = label_data['Churn_Risk'].map(risk_mapping)

    # Шаг 8: Сохранение лэйблов для обучения классификатора
    save_labels(label_data, 'results/labels/labels.json')

    return rfm_segment, label_data


# Запуск пайплайна и вывод результатов
if __name__ == "__main__":
    rfm_result, label_data = main_pipeline()
    print("Job is done!")