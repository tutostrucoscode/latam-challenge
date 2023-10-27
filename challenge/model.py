from typing import Tuple, Union, List
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report
from xgboost import plot_importance
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


def get_period_day(date):
    date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
    morning_min = datetime.strptime("05:00", '%H:%M').time()
    morning_max = datetime.strptime("11:59", '%H:%M').time()
    afternoon_min = datetime.strptime("12:00", '%H:%M').time()
    afternoon_max = datetime.strptime("18:59", '%H:%M').time()
    evening_min = datetime.strptime("19:00", '%H:%M').time()
    evening_max = datetime.strptime("23:59", '%H:%M').time()
    night_min = datetime.strptime("00:00", '%H:%M').time()
    night_max = datetime.strptime("4:59", '%H:%M').time()

    if (date_time > morning_min and date_time < morning_max):
        return 'mañana'
    elif (date_time > afternoon_min and date_time < afternoon_max):
        return 'tarde'
    elif (
        (date_time > evening_min and date_time < evening_max) or
        (date_time > night_min and date_time < night_max)
    ):
        return 'noche'


def is_high_season(fecha):
    fecha_año = int(fecha.split('-')[0])
    fecha = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
    range1_min = datetime.strptime('15-Dec', '%d-%b').replace(year=fecha_año)
    range1_max = datetime.strptime('31-Dec', '%d-%b').replace(year=fecha_año)
    range2_min = datetime.strptime('1-Jan', '%d-%b').replace(year=fecha_año)
    range2_max = datetime.strptime('3-Mar', '%d-%b').replace(year=fecha_año)
    range3_min = datetime.strptime('15-Jul', '%d-%b').replace(year=fecha_año)
    range3_max = datetime.strptime('31-Jul', '%d-%b').replace(year=fecha_año)
    range4_min = datetime.strptime('11-Sep', '%d-%b').replace(year=fecha_año)
    range4_max = datetime.strptime('30-Sep', '%d-%b').replace(year=fecha_año)

    if ((fecha >= range1_min and fecha <= range1_max) or
        (fecha >= range2_min and fecha <= range2_max) or
        (fecha >= range3_min and fecha <= range3_max) or
            (fecha >= range4_min and fecha <= range4_max)):
        return 1
    else:
        return 0


def get_min_diff(data):
    fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
    fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
    min_diff = ((fecha_o - fecha_i).total_seconds())/60
    return min_diff


def get_rate_from_column(data, column):
    delays = {}
    for _, row in data.iterrows():
        if row['delay'] == 1:
            if row[column] not in delays:
                delays[row[column]] = 1
            else:
                delays[row[column]] += 1
    total = data[column].value_counts().to_dict()

    rates = {}
    for name, total in total.items():
        if name in delays:
            rates[name] = round(total / delays[name], 2)
        else:
            rates[name] = 0

    return pd.DataFrame.from_dict(data=rates, orient='index', columns=['Tasa (%)'])


class DelayModel:

    def __init__(self):
        self._model = None  # Model should be saved in this attribute.
        ##data = pd.read_csv('../data/data.csv')
        self.data = None
        self.data.info()
        features, target = self.preprocess(self.data, target_column='delay')
        self.fit(features, target)

    # Prepare raw data for training or predict.
    # Args:
    # data (pd.DataFrame): raw data.
    # target_column (str, optional): if set, the target is returned.
    # Returns:
    # Tuple[pd.DataFrame, pd.DataFrame]: features and target.
    # or
    # pd.DataFrame: features.
    def preprocess(self, data: pd.DataFrame, target_column: str = None) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        self.data = data
        
        self.data['period_day'] = self.data['Fecha-I'].apply(get_period_day)
        self.data['high_season'] = self.data['Fecha-I'].apply(is_high_season)
        self.data['min_diff'] = data.apply(get_min_diff(self.data), axis=1)
        threshold_in_minutes = 15
        self.data['delay'] = np.where(self.data['min_diff'] > threshold_in_minutes, 1, 0)
        data.columns

        self.data_analysis(self.data)
        self.calculate_delay_rate_by_column(self.data)

        features = pd.concat([
            pd.get_dummies(self.data['OPERA'], prefix='OPERA'),
            pd.get_dummies(self.data['TIPOVUELO'], prefix='TIPOVUELO'),
            pd.get_dummies(self.data['MES'], prefix='MES')],
            axis=1
        )

        expected_columns = [
            "OPERA_Latin American Wings",
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air"
        ]
        features = features[expected_columns]

        if target_column:
            target = self.data[target_column].to_frame()
            return (features, target)
        else:
            return features

    # Fit model with preprocessed data.
    # Args:
    # features (pd.DataFrame): preprocessed data.
    # target (pd.DataFrame): target.
    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        # Data Split (Training and Validation)
        training_data = shuffle(features, random_state=111)
        x_train, x_test, y_train, y_test = train_test_split(
            features, target, test_size=0.33, random_state=42)
        n_y0 = len(target[target == 0])
        n_y1 = len(target[target == 1])
        scale = n_y0 / n_y1

        # XGBoost

        xgb_model = xgb.XGBClassifier(
            random_state=1, learning_rate=0.01, scale_pos_weight=scale)
        xgb_model.fit(x_train, y_train)
        xgboost_y_preds = xgb_model.predict(x_test)

        best_threshold = 0.5  # Inicializa con 0.5 que es el valor estándar
        best_recall = 0

        for threshold in np.arange(0.1, 1.0, 0.1):  # Itera de 0.1 a 0.9
            xgboost_y_preds = (xgb_model.predict_proba(
                x_test)[:, 1] > threshold).astype(int)
            report = classification_report(
                y_test, xgboost_y_preds, output_dict=True)
            # Asume que '0' es la etiqueta de la clase que te interesa
            recall = report['0']['recall']
            if recall > best_recall:
                best_recall = recall
                best_threshold = threshold

        xgboost_y_preds = (xgb_model.predict_proba(x_test)[
                           :, 1] > best_threshold).astype(int)
        xgboost_y_preds = [1 if y_pred >
                           0.5 else 0 for y_pred in xgboost_y_preds]

        # Logistic Regression
        reg_model_2 = LogisticRegression(class_weight={1: float(
            np.sum(target == 0))/len(y_train), 0: np.sum(target == 1)/len(y_train)})
        reg_model = LogisticRegression()
        reg_model.fit(x_train, y_train)
        reg_y_preds = reg_model.predict(x_test)

        # Se elige el modelo que mas optimo.:

        self.evaluate_model(y_test, xgboost_y_preds)
        self._model = xgb_model
        plot_importance(self._model)

        return

    # Predict delays for new flights.
    # Args:
    # features (pd.DataFrame): preprocessed data.
    # Returns:
    # (List[int]): predicted targets.
    def predict(self, features: pd.DataFrame) -> List[int]:
        predictions = self._model.predict(features)
        return [int(prediction) for prediction in predictions]

    def evaluate_model(self, y_true, y_pred):
        print(confusion_matrix(y_true, y_pred))
        print(classification_report(y_true, y_pred))

    def data_analysis(self, data: pd.DataFrame) -> None:
        flights_by_airline = data['OPERA'].value_counts()
        plt.figure(figsize=(10, 2))
        sns.set(style="darkgrid")
        sns.barplot(x=flights_by_airline.index,
                    y=flights_by_airline.values, alpha=0.9)
        plt.title('Flights by Airline')
        plt.ylabel('Flights', fontsize=12)
        plt.xlabel('Airline', fontsize=12)
        plt.xticks(rotation=90)

        flights_by_day = data['DIA'].value_counts()
        plt.figure(figsize=(10, 2))
        sns.set(style="darkgrid")
        sns.barplot(x=flights_by_day.index, y=flights_by_day.values,
                    color='lightblue', alpha=0.8)
        plt.title('Flights by Day')
        plt.ylabel('Flights', fontsize=12)
        plt.xlabel('Day', fontsize=12)
        plt.xticks(rotation=90)

        flights_by_month = data['MES'].value_counts()
        plt.figure(figsize=(10, 2))
        sns.set(style="darkgrid")
        sns.barplot(x=flights_by_month.index,
                    y=flights_by_month.values, color='lightblue', alpha=0.8)
        plt.title('Flights by Month')
        plt.ylabel('Flights', fontsize=12)
        plt.xlabel('Month', fontsize=12)
        plt.xticks(rotation=90)

        flights_by_day_in_week = data['DIANOM'].value_counts()
        days = [
            flights_by_day_in_week.index[2],
            flights_by_day_in_week.index[5],
            flights_by_day_in_week.index[4],
            flights_by_day_in_week.index[1],
            flights_by_day_in_week.index[0],
            flights_by_day_in_week.index[6],
            flights_by_day_in_week.index[3]
        ]
        values_by_day = [
            flights_by_day_in_week.values[2],
            flights_by_day_in_week.values[5],
            flights_by_day_in_week.values[4],
            flights_by_day_in_week.values[1],
            flights_by_day_in_week.values[0],
            flights_by_day_in_week.values[6],
            flights_by_day_in_week.values[3]
        ]
        plt.figure(figsize=(10, 2))
        sns.set(style="darkgrid")
        sns.barplot(x=days, y=values_by_day, color='lightblue', alpha=0.8)
        plt.title('Flights by Day in Week')
        plt.ylabel('Flights', fontsize=12)
        plt.xlabel('Day in Week', fontsize=12)
        plt.xticks(rotation=90)

        flights_by_type = data['TIPOVUELO'].value_counts()
        sns.set(style="darkgrid")
        plt.figure(figsize=(10, 2))
        sns.barplot(x=flights_by_type.index,
                    y=flights_by_type.values, alpha=0.9)
        plt.title('Flights by Type')
        plt.ylabel('Flights', fontsize=12)
        plt.xlabel('Type', fontsize=12)

        flight_by_destination = data['SIGLADES'].value_counts()
        plt.figure(figsize=(10, 2))
        sns.set(style="darkgrid")
        sns.barplot(x=flight_by_destination.index,
                    y=flight_by_destination.values, color='lightblue', alpha=0.8)
        plt.title('Flight by Destination')
        plt.ylabel('Flights', fontsize=12)
        plt.xlabel('Destination', fontsize=12)
        plt.xticks(rotation=90)

        plt.show()
        return

    def calculate_delay_rate_by_column(self, data: pd.DataFrame) -> None:
        destination_rate = get_rate_from_column(data, 'SIGLADES')
        destination_rate_values = data['SIGLADES'].value_counts().index
        plt.figure(figsize=(20, 5))
        sns.set(style="darkgrid")
        sns.barplot(x=destination_rate_values,
                    y=destination_rate['Tasa (%)'], alpha=0.75)
        plt.title('Delay Rate by Destination')
        plt.ylabel('Delay Rate [%]', fontsize=12)
        plt.xlabel('Destination', fontsize=12)
        plt.xticks(rotation=90)

        airlines_rate = get_rate_from_column(data, 'OPERA')
        airlines_rate_values = data['OPERA'].value_counts().index
        plt.figure(figsize=(20, 5))
        sns.set(style="darkgrid")
        sns.barplot(x=airlines_rate_values,
                    y=airlines_rate['Tasa (%)'], alpha=0.75)
        plt.title('Delay Rate by Airline')
        plt.ylabel('Delay Rate [%]', fontsize=12)
        plt.xlabel('Airline', fontsize=12)
        plt.xticks(rotation=90)

        month_rate = get_rate_from_column(data, 'MES')
        month_rate_value = data['MES'].value_counts().index
        plt.figure(figsize=(20, 5))
        sns.set(style="darkgrid")
        sns.barplot(x=month_rate_value,
                    y=month_rate['Tasa (%)'], color='blue', alpha=0.75)
        plt.title('Delay Rate by Month')
        plt.ylabel('Delay Rate [%]', fontsize=12)
        plt.xlabel('Month', fontsize=12)
        plt.xticks(rotation=90)
        plt.ylim(0, 10)

        days_rate = get_rate_from_column(data, 'DIANOM')
        days_rate_value = data['DIANOM'].value_counts().index
        sns.set(style="darkgrid")
        plt.figure(figsize=(20, 5))
        sns.barplot(x=days_rate_value,
                    y=days_rate['Tasa (%)'], color='blue', alpha=0.75)
        plt.title('Delay Rate by Day')
        plt.ylabel('Delay Rate [%]', fontsize=12)
        plt.xlabel('Days', fontsize=12)
        plt.xticks(rotation=90)
        plt.ylim(0, 7)

        high_season_rate = get_rate_from_column(data, 'high_season')
        high_season_rate_values = data['high_season'].value_counts().index
        plt.figure(figsize=(5, 2))
        sns.set(style="darkgrid")
        sns.barplot(x=["no", "yes"], y=high_season_rate['Tasa (%)'])
        plt.title('Delay Rate by Season')
        plt.ylabel('Delay Rate [%]', fontsize=12)
        plt.xlabel('High Season', fontsize=12)
        plt.xticks(rotation=90)
        plt.ylim(0, 6)

        flight_type_rate = get_rate_from_column(data, 'TIPOVUELO')
        flight_type_rate_values = data['TIPOVUELO'].value_counts().index
        plt.figure(figsize=(5, 2))
        sns.set(style="darkgrid")
        sns.barplot(x=flight_type_rate_values, y=flight_type_rate['Tasa (%)'])
        plt.title('Delay Rate by Flight Type')
        plt.ylabel('Delay Rate [%]', fontsize=12)
        plt.xlabel('Flight Type', fontsize=12)
        plt.ylim(0, 7)

        period_day_rate = get_rate_from_column(data, 'period_day')
        period_day_rate_values = data['period_day'].value_counts().index
        plt.figure(figsize=(5, 2))
        sns.set(style="darkgrid")
        sns.barplot(x=period_day_rate_values, y=period_day_rate['Tasa (%)'])
        plt.title('Delay Rate by Period of Day')
        plt.ylabel('Delay Rate [%]', fontsize=12)
        plt.xlabel('Period', fontsize=12)
        plt.ylim(3, 7)

        plt.show()
