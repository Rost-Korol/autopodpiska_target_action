import pandas as pd
import pickle
import dill
import datetime

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score



def filter_data(df):
    columns_to_drop = [
        'session_id',
        'client_id',
        'visit_date',
        'visit_time',
        'device_model',
        # 'utm_keyword',
    ]

    return df.drop(columns_to_drop, axis=1)


def fill_gap(df):
    df = df.copy()
    features_to_fill = [
        'utm_source',
        'device_brand',
        'utm_campaign',
        'utm_adcontent',
        'utm_keyword'
    ]

    def is_pc(sample):
        if sample.device_brand == '' and sample.device_category == 'desktop':
            return 'pc'
        return sample.device_brand

    def find_os(data):
        if data.device_os is None:
            if data.device_category == 'desktop':
                if data.device_brand == 'Apple' or data.device_browser == 'Safari':
                    return 'Macintosh'
                else:
                    return 'other'
            else:
                if data.device_browser == 'Safari' or data.device_brand == 'Apple':
                    return 'iOS'
                else:
                    return 'Android'
        else:
            return data.device_os

    df['device_brand'] = df.apply(is_pc, axis=1)
    df['device_os'] = df.apply(find_os, axis=1)

    for feat in features_to_fill:
        df[feat] = df[feat].fillna('other')

    return df


def reduce_rare(df):
    df = df.copy()

    def visit_count(visit):
        if visit <= 5:
            return "visit_" + str(visit)
        else:
            return "visit_over_5"

    df['visit_number'] = df.visit_number.apply(visit_count)

    return df


def cat_faetures_gen(df):
    df = df.copy()

    organic_traf_list = [
        'organic',
        'referal',
        '(none)'
    ]

    sm_ad_list = [
        'QxAxdyPLuQMEcrdZWdWb',
        'MvfHsxITijuriZxsqZqt',
        'ISrKoXQCxqqYvAZICvjs',
        'IZEXUFLARCUMynmHNBGo',
        'PlbkrSYoHuZBWfYjYnfw',
        'gVRrcxiDQubJiljoTbGm'
    ]

    df['is_organic'] = df.utm_medium.apply(lambda x: 1 if x in organic_traf_list else 0)
    df['is_sm_ad'] = df.utm_source.apply(lambda x: 1 if x in sm_ad_list else 0)

    return df


def num_features_gen(df):
    df = df.copy()

    # Количество пикселей (pixels)

    def pixels_outliers(pixels):
        if pixels > (3840 * 2160):
            return (3840 * 2160)
        elif pixels < (360 * 640):
            return (360 * 640)
        else:
            return pixels

    df['pixels'] = df.device_screen_resolution.apply(lambda x: int(x.split('x')[0]) * int(x.split('x')[1]))
    df['pixels'] = df.pixels.apply(pixels_outliers)

    # Соотношение сторон (aspect_ratio)

    def find_aspect_ratio(data):
        diagonals = data.split('x')
        if int(diagonals[1]) == 0:
            return 1
        else:
            return int(diagonals[0]) / int(diagonals[1])

    df['aspect_ratio'] = df.device_screen_resolution.apply(find_aspect_ratio)

    return df.drop(columns='device_screen_resolution')


def main():
    with open('../data/join_data.pkl', 'rb') as file:
        df = pickle.load(file)

    x = df.drop(columns='target')
    y = df.target

    numerical_transformer = Pipeline(steps=[
        ('scaler', MinMaxScaler())
    ])

    categorial_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(min_frequency=0.0001, handle_unknown='infrequent_if_exist'))
    ])

    feat_transformer = ColumnTransformer(transformers=[
        ('cat_features', categorial_transformer, make_column_selector(dtype_include=object)),
        ('num_features', numerical_transformer, make_column_selector(dtype_include=['int64', 'float64']))

    ])

    preprocessor = Pipeline(steps=[
        ('filter', FunctionTransformer(filter_data)),
        ('filling', FunctionTransformer(fill_gap)),
        ('visit_reduce', FunctionTransformer(reduce_rare)),
        ('cat_gen', FunctionTransformer(cat_faetures_gen)),
        ('num_gen', FunctionTransformer(num_features_gen))
    ])

    model_logreg = LogisticRegression(
        random_state=42,
        max_iter=10000,
        class_weight='balanced',
        C=0.1,
        multi_class='ovr',
        n_jobs=-1
    )

    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('feat_transformer', feat_transformer),
        ('model', model_logreg)
    ])

    score = cross_val_score(pipe, x, y, cv=5, scoring='roc_auc')
    print(f'model: Logistic regression, roc_auc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}')

    pipe.fit(x, y)
    with open('models/logreg_v2.pkl', 'wb') as file:
        dill.dump({
            'model': pipe,
            'metadata': {
                'name': 'Sber autopodpiska convehsion predict',
                'author': 'Korol Rostislav',
                'version': 2,
                'date': datetime.datetime.now(),
                'type': 'Logistic regression (C=0.1)',
                'roc_auc': score.mean()
            }
        }, file)




if __name__ == '__main__':
    main()

















