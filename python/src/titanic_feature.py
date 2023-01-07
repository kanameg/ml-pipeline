# -------------------------------------------------------------------
# create feature of second hand apartments
# -------------------------------------------------------------------
import argparse
import category_encoders as ce
from contextlib import contextmanager
from feature import Feature
import glob
from logger import Logger
import numpy as np
import os
import pandas as pd
import time
import yaml


# -------------------------------------------------------------------
# read configuration YAML file
# -------------------------------------------------------------------
CONFIG_FILE = '../configs/config.yaml'

with open(CONFIG_FILE) as file:
    yml = yaml.safe_load(file)

LOG_DIR = yml['SETTING']['LOG_DIR']
RAW_DATA_DIR = yml['SETTING']['RAW_DATA_DIR']
FEATURE_DIR = yml['SETTING']['FEATURE_DIR']
Feature.dir = FEATURE_DIR  # set feature file directory to base class

# -------------------------------------------------------------------
# feature class name to retrieve
# -------------------------------------------------------------------
features = [
    'PassengerId', 'Pclass',
]


class PassengerId(Feature):
    """
    PassengerIdをそのまま利用
    """
    def create_feature(self):
        self.feature_train[self.name] = self.train['PassengerId']
        self.feature_test[self.name] = self.test['PassengerId']
        self.create_memo('PassengerIdは乗船者のID。特徴量としては利用しない。')

class Area(Feature):
    """
    面積(m^3)
    """
    def create_feature(self):
        self.feature_train[self.name] = self.train['面積（㎡）'].replace('2000㎡以上', 2000).astype(float)
        self.feature_test[self.name] = self.test['面積（㎡）'].replace('2000㎡以上', 2000).astype(float)
        self.create_memo('面積(m^3)')

class AreaLog(Feature):
    """
    面積(m^3) Log10
    """
    def create_feature(self):
        self.feature_train[self.name] = np.log10(self.train['面積（㎡）'].replace('2000㎡以上', 2000).astype(float))
        self.feature_test[self.name] = np.log10(self.test['面積（㎡）'].replace('2000㎡以上', 2000).astype(float))
        self.create_memo('面積(m^3)をlog10したもの')


class NearestStationMinute(Feature):
    """
    最寄駅：距離（分）
    """
    def create_feature(self):
        str_to_minute = {
            '30分?60分': 45,
            '1H?1H30': 75,
            '1H30?2H': 105,
            '2H?': 120
        }
        self.feature_train[self.name] = self.train['最寄駅：距離（分）'].replace(str_to_minute).astype(float)
        self.feature_test[self.name] = self.test['最寄駅：距離（分）'].replace(str_to_minute).astype(float)
        self.create_memo('最寄駅：距離（分）で文字で入っている30?60分などは中間の45とかに変換したモノ')


class BuildYear(Feature):
    """
    建築年（西暦）
    """
    def create_feature(self):
        from jeraconv import jeraconv
        j2w = jeraconv.J2W()

        tt = pd.concat([self.train, self.test], axis=0)
        jera_to_ad = {}
        for j_year in tt['建築年'].value_counts().keys():
            if j_year == '戦前':
                ac_year = j2w.convert('昭和20年') # 戦前は昭和20年で計算
            else:
                ac_year = j2w.convert(j_year)
            jera_to_ad[j_year] = ac_year
        
        self.feature_train[self.name] = self.train['建築年'].replace(jera_to_ad)
        self.feature_test[self.name] = self.test['建築年'].replace(jera_to_ad)
        self.create_memo('建築年（西暦）で和暦を西暦に変換したもの')


class SalesYear(Feature):
    """
    取引年（西暦）小数点あり
    """
    def create_feature(self):
        q_to_num = {
            '年第1四半期': '.0',
            '年第2四半期': '.25',
            '年第3四半期': '.5',
            '年第4四半期': '.75',
        }

        tt = pd.concat([self.train, self.test], axis=0)
        q_to_year = {}
        for year_q in tt['取引時点'].value_counts().keys():
            for q, num in q_to_num.items():
                if q in year_q:
                    year = year_q.replace(q, num)
                    q_to_year[year_q] = year

        self.feature_train[self.name] = self.train['取引時点'].replace(q_to_year).astype(float)
        self.feature_test[self.name] = self.test['取引時点'].replace(q_to_year).astype(float)
        self.create_memo('取引年（西暦）で四半期を0.25で小数点づけしたもの')


class BuildAgeSaleYear(Feature):
    """
    取引時の築年数
    """
    def create_feature(self):
        # 取得済みのfeatureをロード
        build_year = BuildYear()
        sales_year = SalesYear()
        build_year.load_csv()
        sales_year.load_csv()

        self.train[build_year.name] = build_year.feature_train[build_year.name]
        self.test[build_year.name] = build_year.feature_test[build_year.name]
        self.train[sales_year.name] = sales_year.feature_train[sales_year.name]
        self.test[sales_year.name] = sales_year.feature_test[sales_year.name]

        self.feature_train[self.name] = self.train[sales_year.name] - self.train[build_year.name]
        self.feature_test[self.name] = self.test[sales_year.name] - self.test[build_year.name]
        self.create_memo('取引時点での築年数で、取引年から築年を引いたもの')
        

class FloorSpace(Feature):
    """
    床面積
    """
    def create_feature(self):
        # 取得済みのfeatureをロード
        area = Area()
        area.load_csv()
        self.train[area.name] = area.feature_train[area.name]
        self.test[area.name] = area.feature_test[area.name]

        # 床面積を計算
        self.feature_train[self.name] = self.train[area.name] * self.train['容積率（％）'] / 100.0
        self.feature_test[self.name] = self.test[area.name] * self.test['容積率（％）'] / 100.0
        self.create_memo('計算した総床面積で、面積に容積率をかけて計算')


class FloorSpaceLog(Feature):
    """
    床面積
    """
    def create_feature(self):
        # 取得済みのfeatureをロード
        floor_space = FloorSpace()
        floor_space.load_csv()
        self.train[floor_space.name] = floor_space.feature_train[floor_space.name]
        self.test[floor_space.name] = floor_space.feature_test[floor_space.name]

        # 床面積のLog10を計算
        self.feature_train[self.name] = np.log10(self.train[floor_space.name])
        self.feature_test[self.name] = np.log10(self.test[floor_space.name])
        self.create_memo('計算した総床面積をlog10を取ったもの')


class Renovation(Feature):
    """
    改装あり (あり:1, なし:0) 欠損値は0:改装なし
    """
    def create_feature(self):
        tt = pd.concat([self.train, self.test], axis=0)
        oh_enc = ce.OneHotEncoder(use_cat_names=True)
        tt[self.name] = oh_enc.fit_transform(tt['改装'])['改装_改装済']

        # trainとtestに分離して必要な特徴量だけ取り出し
        self.feature_train[self.name] = tt.loc[tt['train'] == True][self.name].reset_index(drop=True)
        self.feature_test[self.name] = tt.loc[tt['train'] == False][self.name].reset_index(drop=True)
        self.create_memo('改装ありが1で、改装なしや欠損値は0で計算')


class RoomNum(Feature):
    """
    部屋数 間取りから算出
    """
    def create_feature(self):
        room_num = {
            'オープンフロア': 1, 'メゾネット': 3, 'スタジオ': 1,
            '１Ｋ': 2, '１ＤＫ': 2, '１ＬＤＫ': 2, '１Ｒ': 1, '１ＬＤＫ＋Ｓ': 3, '１Ｌ': 2, '１ＤＫ＋Ｓ': 3,
            '１ＤＫ＋Ｋ': 3, '１ＬＫ': 2, '１Ｒ＋Ｓ': 2, '１ＬＤ＋Ｓ': 3, '１Ｌ＋Ｓ': 3, '１Ｋ＋Ｓ': 3, '１ＬＫ＋Ｓ': 3,
            '１ＬＤＫ＋Ｋ': 3,
            '２ＬＤＫ': 3, '２ＤＫ': 3, '２ＬＤＫ＋Ｓ': 4, '２ＬＫ': 3, '２ＤＫ＋Ｓ': 4,'２Ｋ': 3, '２Ｌ': 3,
            '２ＬＤＫ＋Ｋ': 4, '２Ｋ＋Ｓ': 4, '２ＬＫ＋Ｓ': 4, '２ＬＤ＋Ｓ': 4, '２Ｌ＋Ｓ': 4, '２Ｄ': 3, '２ＬＤ': 3,
            '３ＬＤＫ': 4, '３ＬＤＫ＋Ｓ': 5, '３ＤＫ': 4, '３ＬＤ': 4, '３Ｋ': 4, '３ＤＫ＋Ｓ': 5, '３Ｋ＋Ｓ': 5,
            '３ＬＤＫ＋Ｋ': 5, '３Ｄ': 4, '３ＬＤ＋Ｓ': 5, '３ＬＫ＋Ｓ': 5, '３ＬＫ': 4,
            '４ＬＤＫ': 5, '４ＤＫ': 5, '４ＬＤＫ＋Ｓ': 6, '４Ｋ': 5, '４Ｌ＋Ｋ': 6, '４ＤＫ＋Ｓ': 6, '４ＬＤＫ＋Ｋ': 6,
            '４Ｄ': 5, '４Ｌ': 5,
            '５ＬＤＫ': 6, '５ＬＤＫ＋Ｓ': 7, '５ＤＫ': 6, '５Ｋ': 6, '５ＬＤＫ＋Ｋ': 7,
            '６ＬＤＫ': 7, '６ＬＤＫ＋Ｓ': 8, '６ＤＫ': 7,
            '７ＬＤＫ': 8, '７ＬＤＫ＋Ｓ': 9, '７ＤＫ': 8,
            '８ＬＤＫ': 9, '８ＬＤＫ＋Ｓ': 10,
        }
        self.feature_train[self.name] = self.train['間取り'].replace(room_num)
        self.feature_test[self.name] = self.test['間取り'].replace(room_num)
        self.create_memo('間取りから部屋数を計算したモノ LDKは1部屋で計算')


class Prefecture(Feature):
    """
    都道府県名をOrdinalエンコード
    """
    def create_feature(self):
        tt = pd.concat([self.train, self.test], axis=0)
        or_enc = ce.OrdinalEncoder()
        tt[self.name] = or_enc.fit_transform(tt['都道府県名'])

        # trainとtestに分離して必要な特徴量だけ取り出し
        self.feature_train[self.name] = tt.loc[tt['train'] == True][self.name].reset_index(drop=True)
        self.feature_test[self.name] = tt.loc[tt['train'] == False][self.name].reset_index(drop=True)      
        self.create_memo('都道府県名をOrdinalエンコードしたモノ')


class PrefectureName(Feature):
    """
    都道府県名をそのまま利用
    """
    def create_feature(self):
        self.feature_train[self.name] = self.train['都道府県名']
        self.feature_test[self.name] = self.test['都道府県名']
        self.create_memo('都道府県名そのまま（文字列なので利用しない）')


class City(Feature):
    """
    市区町村名 コードの部分をそのまま利用
    """
    def create_feature(self):
        self.feature_train[self.name] = self.train['市区町村コード']
        self.feature_test[self.name] = self.test['市区町村コード']
        self.create_memo('市区町村名 コードの部分をそのまま利用')


class CityName(Feature):
    """
    市区町村名をそのまま利用
    """
    def create_feature(self):
        self.feature_train[self.name] = self.train['市区町村名']
        self.feature_test[self.name] = self.test['市区町村名']
        self.create_memo('市区町村名そのまま（文字列なので利用しない）')


class NearestStation(Feature):
    """
    最寄駅：名称をOrdinalエンコード
    """
    def create_feature(self):
        tt = pd.concat([self.train, self.test], axis=0)
        or_enc = ce.OrdinalEncoder()
        tt[self.name] = or_enc.fit_transform(tt['最寄駅：名称'])

        # trainとtestに分離して必要な特徴量だけ取り出し
        self.feature_train[self.name] = tt.loc[tt['train'] == True][self.name].reset_index(drop=True)
        self.feature_test[self.name] = tt.loc[tt['train'] == False][self.name].reset_index(drop=True)
        self.create_memo('最寄駅：名称をOrdinalエンコードしたモノ')


class NearestStationName(Feature):
    """
    最寄駅：名称
    """
    def create_feature(self):
        self.feature_train[self.name] = self.train['最寄駅：名称']
        self.feature_test[self.name] = self.test['最寄駅：名称']
        self.create_memo('最寄駅：名称そのまま（文字列なので利用しない）')

# class PricePreFloorSpaceLog(Feature):
#     """
#     単位床面積(m3)当たりの取引額
#     """
#     def create_feature(self):
#         # 取得済みの床面積をロード
#         floor_space_log = FloorSpaceLog()
#         floor_space_log.load_csv()
#         self.train[floor_space_log.name] = floor_space_log.feature_train[floor_space_log.name]
#         self.test[floor_space_log.name] = floor_space_log.feature_test[floor_space_log.name]

#         self.feature_train[self.name] = self.train['取引価格（総額）_log'] - self.train[floor_space_log.name]
#         self.feature_test[self.name] = self.test['取引価格（総額）_log'] - self.test[floor_space_log.name]


class PriceLog(Feature):
    """
    取引価格のLog10
    """
    def create_feature(self):
        self.feature_train[self.name] = self.train['取引価格（総額）_log']
        self.feature_test[self.name] = np.nan
        self.create_memo('目的変数の取引価格のLog10をそのまま')


class Price(Feature):
    """
    取引価格
    """
    def create_feature(self):
        self.feature_train[self.name] = np.round(np.power(10, self.train['取引価格（総額）_log']))  # 四捨五入
        self.feature_test[self.name] = np.nan
        self.create_memo('目的変数の取引価格のLog10を円に戻したもの')


class FururePurpose(Feature):
    """
    今後の利用目的
    """
    def create_feature(self):
        tt = pd.concat([self.train, self.test], axis=0)
        or_enc = ce.OrdinalEncoder()
        tt[self.name] = or_enc.fit_transform(tt['今後の利用目的'])

        # trainとtestに分離して必要な特徴量だけ取り出し
        self.feature_train[self.name] = tt.loc[tt['train'] == True][self.name].reset_index(drop=True)
        self.feature_test[self.name] = tt.loc[tt['train'] == False][self.name].reset_index(drop=True)
        self.create_memo('「今後の利用目的」をOrdinalエンコードしたもの')


class Defect(Feature):
    """
    瑕疵物件
    """
    def create_feature(self):
        self.train['defect'] = 0
        self.train.loc[self.train['取引の事情等'].str.contains('瑕疵') == True, 'defect'] = 1
        self.test['defect'] = 0
        self.test.loc[self.test['取引の事情等'].str.contains('瑕疵') == True, 'defect'] = 1

        self.feature_train[self.name] = self.train['defect']
        self.feature_test[self.name] = self.test['defect']
        self.create_memo('「取引の事情等」に瑕疵の文字が入っているものを1、それ以外は0')


class Auction(Feature):
    """
    競売物件
    """
    def create_feature(self):
        self.train['auction'] = 0
        self.train.loc[self.train['取引の事情等'].str.contains('競売') == True, 'auction'] = 1
        self.test['auction'] = 0
        self.test.loc[self.test['取引の事情等'].str.contains('競売') == True, 'auction'] = 1

        self.feature_train[self.name] = self.train['auction']
        self.feature_test[self.name] = self.test['auction']
        self.create_memo('「取引の事情等」に競売の文字が入っているものを1、それ以外は0')

    
class Circumstances(Feature):
    """
    取引の事情等をOrdinalエンコード
    """
    def create_feature(self):
        tt = pd.concat([self.train, self.test], axis=0)
        or_enc = ce.OrdinalEncoder()
        tt[self.name] = or_enc.fit_transform(tt['取引の事情等'])

        # trainとtestに分離して必要な特徴量だけ取り出し
        self.feature_train[self.name] = tt.loc[tt['train'] == True][self.name].reset_index(drop=True)
        self.feature_test[self.name] = tt.loc[tt['train'] == False][self.name].reset_index(drop=True)
        self.create_memo('「取引の事情等」をOrdinalエンコードしたもの')


class CityPlanning(Feature):
    """
    都市計画をOrdinalエンコード
    """
    def create_feature(self):
        tt = pd.concat([self.train, self.test], axis=0)
        or_enc = ce.OrdinalEncoder()
        tt[self.name] = or_enc.fit_transform(tt['都市計画'])

        # trainとtestに分離して必要な特徴量だけ取り出し
        self.feature_train[self.name] = tt.loc[tt['train'] == True][self.name].reset_index(drop=True)
        self.feature_test[self.name] = tt.loc[tt['train'] == False][self.name].reset_index(drop=True)
        self.create_memo('「都市計画」をOrdinalエンコードしたもの')


class Usage(Feature):
    """
    用途をOrdinalエンコード
    """
    def create_feature(self):
        tt = pd.concat([self.train, self.test], axis=0)
        or_enc = ce.OrdinalEncoder()
        tt[self.name] = or_enc.fit_transform(tt['用途'])

        # trainとtestに分離して必要な特徴量だけ取り出し
        self.feature_train[self.name] = tt.loc[tt['train'] == True][self.name].reset_index(drop=True)
        self.feature_test[self.name] = tt.loc[tt['train'] == False][self.name].reset_index(drop=True)
        self.create_memo('「用途」をOrdinalエンコードしたもの')

        
class FloorPlan(Feature):
    """
    間取りをOrdinalエンコード
    """
    def create_feature(self):
        tt = pd.concat([self.train, self.test], axis=0)
        or_enc = ce.OrdinalEncoder()
        tt[self.name] = or_enc.fit_transform(tt['間取り'])

        # trainとtestに分離して必要な特徴量だけ取り出し
        self.feature_train[self.name] = tt.loc[tt['train'] == True][self.name].reset_index(drop=True)
        self.feature_test[self.name] = tt.loc[tt['train'] == False][self.name].reset_index(drop=True)
        self.create_memo('「間取り」をOrdinalエンコードしたもの')


class District(Feature):
    """
    地区名をOrdinalエンコード
    """
    def create_feature(self):
        tt = pd.concat([self.train, self.test], axis=0)
        or_enc = ce.OrdinalEncoder()
        tt[self.name] = or_enc.fit_transform(tt['地区名'])

        # trainとtestに分離して必要な特徴量だけ取り出し
        self.feature_train[self.name] = tt.loc[tt['train'] == True][self.name].reset_index(drop=True)
        self.feature_test[self.name] = tt.loc[tt['train'] == False][self.name].reset_index(drop=True)
        self.create_memo('「地区名」をOrdinalエンコードしたもの')


class DealTerm(Feature):
    """
    取引時点をOrdinalエンコード
    """
    def create_feature(self):
        tt = pd.concat([self.train, self.test], axis=0)
        or_enc = ce.OrdinalEncoder()
        tt[self.name] = or_enc.fit_transform(tt['取引時点'])

        # trainとtestに分離して必要な特徴量だけ取り出し
        self.feature_train[self.name] = tt.loc[tt['train'] == True][self.name].reset_index(drop=True)
        self.feature_test[self.name] = tt.loc[tt['train'] == False][self.name].reset_index(drop=True)
        self.create_memo('「取引時点」をOrdinalエンコードしたもの')


class Structure(Feature):
    """
    建物の構造をOrdinalエンコード
    """
    def create_feature(self):
        tt = pd.concat([self.train, self.test], axis=0)
        or_enc = ce.OrdinalEncoder()
        tt[self.name] = or_enc.fit_transform(tt['建物の構造'])

        # trainとtestに分離して必要な特徴量だけ取り出し
        self.feature_train[self.name] = tt.loc[tt['train'] == True][self.name].reset_index(drop=True)
        self.feature_test[self.name] = tt.loc[tt['train'] == False][self.name].reset_index(drop=True)
        self.create_memo('「建物の構造」をOrdinalエンコードしたもの')


class FloorAreaRatio(Feature):
    """
    容積率（％）
    """
    def create_feature(self):
        self.feature_train[self.name] = self.train['容積率（％）']
        self.feature_test[self.name] = self.test['容積率（％）']
        self.create_memo('「容積率（％）」をそのまま')


class BuildingCoverageRatio(Feature):
    """
    建ぺい率（％）
    """
    def create_feature(self):
        self.feature_train[self.name] = self.train['建ぺい率（％）']
        self.feature_test[self.name] = self.test['建ぺい率（％）']
        self.create_memo('「建ぺい率（％）」をそのまま')




from abc import ABCMeta, abstractmethod

class Cleanse(metaclass=ABCMeta):
    dir = '../data/raw/'

    """
    abstruct class cleanse data
    """
    def __init__(self, train, test) -> None:
        train['train'] = True
        test['train'] = False
        self.all = pd.concat([train, test], axis=0)
        pass
    
    @abstractmethod
    def clean(self):
        """
        do cleansing train and test data
        """
        pass

    
    def save(self):
        """
        split all data into training and test data and save to files
        """
        train = self.all.loc[self.all['train'] == True]
        test = self.all.loc[self.all['train'] == False]
        train.to_csv(dir + 'train_clean.csv', index=False)
        test.to_csv(dir + 'test_clean.csv', index=False)
        pass


@contextmanager
def timer(name):
    """
    timer for mesuring execute time
    """ 
    t0 = time.time()
    logger.info(f"[{name}] start!")
    yield
    logger.info(f"[{name}] done in {time.time() - t0:.3f} s")


def cleanse_nearest_station_minute(df):
    """
    cleaning minutes from nearest station('最寄駅：距離（分）') string format to numeric
    """
    str_to_minute = {
        '30分?60分': 45,
        '1H?1H30': 75,
        '1H30?2H': 105,
        '2H?': 120
    }
    df['最寄駅：距離（分）'] = df['最寄駅：距離（分）'].replace(str_to_minute).astype(float)


def fillna_district_nearest_station(df):
    """
    cleansing district name and nearest station name
    """
    # fill the district name with the most common district name among the properties with the same prefecture and city name.
    df['prefecture_city'] = df['都道府県名'].str.cat(df['市区町村名'])
    prefecture_city_to_district = df.groupby(['prefecture_city'])['地区名'].apply(lambda x: x.mode()).reset_index(1)['地区名'].to_dict()
    df['地区名'].fillna(
        df.loc[(df['地区名'].isnull()) & (df['最寄駅：名称'].isnull())]['prefecture_city'].map(prefecture_city_to_district),
        inplace=True
    )
    # fill the nearest station name with the most common station name among the properties with the same prefecture city and district name.
    df['prefecture_city_district'] = df['prefecture_city'].str.cat(df['地区名'])
    prefecture_city_district_to_station = df.groupby(['prefecture_city_district'])['最寄駅：名称'].apply(lambda x: x.mode()).reset_index(1)['最寄駅：名称'].to_dict()
    df['最寄駅：名称'].fillna(
        df.loc[(df['地区名'].notnull()) & (df['最寄駅：名称'].isnull())]['prefecture_city_district'].map(prefecture_city_district_to_station), inplace=True
    )

    # fill the district name with the most common district name among the properies with the same nearest station name.
    station_to_district = df.groupby(['最寄駅：名称'])['地区名'].apply(lambda x: x.mode()).reset_index(1)['地区名'].to_dict()
    df['地区名'].fillna(
        df.loc[(df['地区名'].isnull()) & (df['最寄駅：名称'].notnull())]['最寄駅：名称'].map(station_to_district), inplace=True
    )

def fillna_nearest_station_minutes(df):
    """
    cleansing minutes from nearest station
    """

    df['prefecture'] = df['都道府県名']
    df['prefecture_city'] = df['都道府県名'].str.cat(df['市区町村名'])
    df['prefecture_city_district'] = df['prefecture_city'].str.cat(df['地区名'])
    df['prefecture_city_district_station'] = df['prefecture_city_district'].str.cat(df['最寄駅：名称'])

    prefecture_city_district_station_to_distance = df.groupby(['prefecture_city_district_station'])['最寄駅：距離（分）'].mean().to_dict()
    df['最寄駅：距離（分）'].fillna(
        df.loc[(df['最寄駅：名称'].notnull()) & (df['最寄駅：距離（分）'].isnull())]['prefecture_city_district_station'].map(prefecture_city_district_station_to_distance),
        inplace=True
    )

    prefecture_city_district_to_distance = df.groupby(['prefecture_city_district'])['最寄駅：距離（分）'].agg('mean').to_dict()
    df['最寄駅：距離（分）'].fillna(
        df.loc[(df['最寄駅：名称'].notnull()) & (df['最寄駅：距離（分）'].isnull())]['prefecture_city_district'].map(prefecture_city_district_to_distance),
        inplace=True
    )

    prefecture_city_to_distance = df.groupby(['prefecture_city'])['最寄駅：距離（分）'].agg('mean').to_dict()
    df['最寄駅：距離（分）'].fillna(
        df.loc[(df['最寄駅：名称'].notnull()) & (df['最寄駅：距離（分）'].isnull())]['prefecture_city'].map(prefecture_city_to_distance),
        inplace=True
    )

    prefecture_to_distance = df.groupby(['prefecture'])['最寄駅：距離（分）'].agg('mean').to_dict()
    df['最寄駅：距離（分）'].fillna(
        df.loc[(df['最寄駅：名称'].notnull()) & (df['最寄駅：距離（分）'].isnull())]['prefecture'].map(prefecture_to_distance),
        inplace=True
    )
    print(df.loc[(df['地区名'].notnull()) & (df['最寄駅：名称'].notnull()) & (df['最寄駅：距離（分）'].isnull())].info())
    print(df.loc[(df['最寄駅：距離（分）'].isnull()) | (df['地区名'].isnull()) | (df['最寄駅：名称'].isnull())]['市区町村名'].unique())
    print(df.loc[df['最寄駅：距離（分）'].isnull()].info())



if __name__ == '__main__':
    # -------------------------------------------------------------------
    # setting logger
    # -------------------------------------------------------------------
    # log_filename = os.path.splitext(__file__)[0] + ".log"
    # logger = get_feature_logger(LOG_DIR + log_filename)
    logger = Logger('feature')

    # -------------------------------------------------------------------
    # get option parameter [-f force recreate feature parameter]
    # -------------------------------------------------------------------
    argp = argparse.ArgumentParser(description='default')
    argp.add_argument('--force', '-f', action='store_true', help='Overwrite existing files')
    args = argp.parse_args()
    logger.info(args)

    # -------------------------------------------------------------------
    # read training data and submit data
    # -------------------------------------------------------------------
    files = []
    files.extend(glob.glob(RAW_DATA_DIR + "train/??.csv"))
    logger.debug(files)
    logger.info(f"number of raw train files: {len(files)}")

    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    train = pd.concat(dfs, axis=0, ignore_index=True)
    train['train'] = True
    logger.info(f"train shape = {train.shape}")

    test = pd.read_csv(RAW_DATA_DIR + 'test.csv')
    test['train'] = False
    logger.info(f"test shape = {test.shape}")

    # -------------------------------------------------------------------
    # cleansing raw train and test data
    # -------------------------------------------------------------------
    all = pd.concat([train, test], axis=0)

    cleanse_nearest_station_minute(all)     # 最寄駅からの時間の一部文字列データの部分を数値に変換

    fillna_district_nearest_station(all)    # 地区名と最寄駅名の欠損値を補完
    fillna_nearest_station_minutes(all)     # 最寄駅からの時間の欠損値を補完

    print(all.loc[(all['最寄駅：距離（分）'].isnull()) | (all['地区名'].isnull()) | (all['最寄駅：名称'].isnull())]['市区町村名'].unique())

    train = all.loc[all['train']==True].copy()
    test = all.loc[all['train']==False].copy()

    # -------------------------------------------------------------------
    # create feature parameter in the order of feature array
    # -------------------------------------------------------------------
    for feature in features:
        f = eval(feature)(train, test)
        if f.check_csv() and not args.force:
            print(f.name, 'was skipped')
        else:
            f.run().save_csv()
            f.load_csv()
    