import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
from typing import Tuple, Union, List, Any
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report


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

    if date_time > morning_min and date_time < morning_max:
        return 'mañana'
    elif date_time > afternoon_min and date_time < afternoon_max:
        return 'tarde'
    elif (
        date_time > evening_min and date_time < evening_max or
        date_time > night_min and date_time < night_max
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


def get_min_diff(row_data):
    fecha_o = datetime.strptime(row_data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
    fecha_i = datetime.strptime(row_data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
    min_diff = ((fecha_o - fecha_i).total_seconds())/60
    return min_diff


class DelayModel:

    top_10_features = [
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

    def __init__(self):
        self._model = None  # Model should be saved in this attribute.
        self._data = None

    def preprocess(self, data: pd.DataFrame, target_column: str = None) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        data['period_day'] = data['Fecha-I'].apply(get_period_day)
        data['high_season'] = data['Fecha-I'].apply(is_high_season)
        data['min_diff'] = data.apply(get_min_diff, axis=1)
        threshold_in_minutes = 15
        data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)

        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix='OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'),
            pd.get_dummies(data['MES'], prefix='MES'),
            pd.get_dummies(data['period_day'], prefix='PERIODO_dia'),
            pd.get_dummies(data['AÑO'], prefix='AÑO')
        ], axis=1)

        features = features[self.top_10_features]

        self._data = data
        features.info()
        features.columns
        data.info()
        data.columns
        if target_column:
            target = data[target_column].to_frame()
            return features, target

        return features

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        x_train, x_test, y_train, y_test, xgb_model = self.prepare_data_and_model(
            features, target)
        xgb_model.fit(x_train, y_train)

        self._model = xgb_model
        return

    def predict(self, features: pd.DataFrame) -> List[int]:
        if self._model is None:
            return [0] * len(features)

        return list(self._model.predict(features))

    def prepare_data_and_model(self, features: pd.DataFrame, target: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, xgb.XGBClassifier]:
        x_train, x_test, y_train, y_test = train_test_split(
            features[self.top_10_features], target, test_size=0.33, random_state=42)
        n_y0 = len(y_train[y_train == 0])
        n_y1 = len(y_train[y_train == 1])
        scale = n_y0 / n_y1
        xgb_model = xgb.XGBClassifier(
            random_state=1, learning_rate=0.01, scale_pos_weight=scale)
        return x_train, x_test, y_train, y_test, xgb_model
