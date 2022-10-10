# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 18:51:25 2021

@author: dell
"""


import requests
import json
import datetime
import os
import pickle

import decimal
from tqdm import tqdm

import pandas as pd
import numpy as np

import uqer

import warnings






# import sys
# reload(sys)
# sys.setdefaultencoding('utf8')

# from conn import SessionInfluxDB
from influxdb import InfluxDBClient

warnings.filterwarnings("ignore")

LIST_DATABASE_SUNTIME = ['t_factor_value_all', 't_factor_profit_all', 't_factor_growth_all', 't_factor_scale_all', 't_factor_emotion_all', 'stock_score_all']

LIST_DATABASE_WIND = ["ashareearningest", "ashareconsensusindex", "ashareconsensusrollingdata"]

def CurrentTime(a, b, c, d=0, e=0, f=0):
    # return str(datetime.datetime.utcnow().isoformat("T"))
    return datetime.datetime.strftime(datetime.datetime(a, b, c, d, e, f) - datetime.timedelta(hours=8), "%Y-%m-%dT%H:%M:%SZ")

def ConnInfluxdb(ip, username, password, database, timeout=10000):
    return InfluxDBClient(ip, 12086, username, password, database, timeout=timeout)


class SessionInfluxDB(object):

    def __init__(self):
        self.client = self.ConnInfluxdb("175.25.50.120", "xtech", "xtech123", "factor")

    @staticmethod
    def ConnInfluxdb(ip, username, password, database, timeout=10000):
        return InfluxDBClient(ip, 12086, username, password, database, timeout=timeout)

    def Write(self, body, n=1000):
        for i in range(int((len(body) + n - 1) / n)):
            h = i * n
            t = min((i + 1) * n, len(body))
            self.client.write_points(body[h:t])


class DatabaseConnMysql(object):

    def __init__(self, ip, username, password):
        self.ip = ip
        self.username = username
        self.password = password
        self.database = None
        self.connect = None
        self.cursor = None
        self.session = SessionInfluxDB()

    def SqlExecute(self, sql):
        try:
            self.cursor.execute(sql)
            results = self.cursor.fetchall()
            return results
        except:
            print("Error: unable to execute sql %s" % sql)
            return None

    def ConnectClose(self):
        self.cursor = None
        try:
            self.connect.close()
        except:
            print("Error: cant close %s" % self.connect)

    def SetDatabase(self, database="wind"):
        self.database = database
        self.cursor = None
        self.connect = None

    def Connect(self):
        self.connect = MySQLdb.connect(self.ip, self.username, self.password, self.database, charset="utf8")
        self.cursor = self.connect.cursor()

    def GetDataHistory(self, table_name):
        sql = "SHOW COLUMNS FROM %s" % table_name
        resultsColumns = self.SqlExecute(sql)
        if resultsColumns is None:
            return
        else:
            sql = "SELECT * FROM %s" % table_name
            results = self.SqlExecute(sql)
            if results is None:
                return
            elif len(results) == 0:
                print("Warning: no data is returned from < %s >" % sql)
                return
            else:
                lenColumns = len(resultsColumns)
                body = []
                for rows in results:
                    bodyDict = {
                        "measurement": "%s_%s" % (self.database, table_name),
                        # "time": CurrentTime(),
                        # "tags": {"unique_datetime": str(datetime.datetime.now())},
                        "tags": {},
                        "fields": {},
                    }
                    for i in range(lenColumns):
                        tmp = None
                        if type(rows[i]) is int or type(rows[i]) is decimal.Decimal:
                            tmp = float(rows[i])
                        elif type(rows[i]) is datetime.datetime:
                            tmp = rows[i].strftime('%Y-%m-%d %H:%M:%S')
                        elif rows[i] is not None:
                            tmp = str(rows[i])
                        else:
                            tmp = rows[i]
                        bodyDict["fields"][resultsColumns[i][0]] = tmp
                    t = bodyDict["fields"]["OPDATE"]
                    tmpYear = int(t[0:4])
                    tmpMonth = int(t[5:7])
                    tmpDay = int(t[8:10])
                    # bodyDict["fields"]["year"] = tmpYear
                    # bodyDict["fields"]["month"] = tmpMonth
                    # bodyDict["fields"]["day"] = tmpDay
                    # bodyDict["fields"]["ymd"] = int(tmpYear * 10000 + tmpMonth * 100 + tmpDay)
                    bodyDict["time"] = CurrentTime(tmpYear, tmpMonth, tmpDay)
                    if table_name == "ashareconsensusindex":
                        bodyDict["tags"]["unique_code"] = bodyDict["fields"]["S_INFO_COMPCODE"]
                    else:
                        bodyDict["tags"]["unique_code"] = bodyDict["fields"]["S_INFO_WINDCODE"]
                    bodyDict["tags"]["unique_object"] = bodyDict["fields"]["OBJECT_ID"][7]
                    body.append(bodyDict)
                # print(body[0])
                print("data length =", len(body))
                self.session.Write(body)

    def GetDataNew(self, table_name):
        sql = "SHOW COLUMNS FROM %s" % table_name
        resultsColumns = self.SqlExecute(sql)
        if resultsColumns is None:
            return
        else:
            dateToday = str(datetime.date.today())
            # dateToday = "2020-12-30"
            print(dateToday)
            # dateToday = "2020-10-30"
            sql = "SELECT * FROM %s WHERE OPDATE REGEXP '^%s'" % (table_name, dateToday)
            results = self.SqlExecute(sql)
            if results is None:
                return
            elif len(results) == 0:
                print("Warning: no data is returned from %s" % table_name)
                return
            else:
                lenColumns = len(resultsColumns)
                body = []
                for rows in results:
                    bodyDict = {
                        "measurement": "%s_%s" % (self.database, table_name),
                        # "time": CurrentTime(),
                        # "tags": {"unique_datetime": str(datetime.datetime.now())},
                        "tags": {},
                        "fields": {},
                    }
                    for i in range(lenColumns):
                        tmp = None
                        if type(rows[i]) is int or type(rows[i]) is decimal.Decimal:
                            tmp = float(rows[i])
                        elif type(rows[i]) is datetime.datetime:
                            tmp = rows[i].strftime('%Y-%m-%d %H:%M:%S')
                        elif rows[i] is not None:
                            tmp = str(rows[i])
                        else:
                            tmp = rows[i]
                        bodyDict["fields"][resultsColumns[i][0]] = tmp
                    t = bodyDict["fields"]["OPDATE"]
                    tmpYear = int(t[0:4])
                    tmpMonth = int(t[5:7])
                    tmpDay = int(t[8:10])
                    # bodyDict["fields"]["year"] = tmpYear
                    # bodyDict["fields"]["month"] = tmpMonth
                    # bodyDict["fields"]["day"] = tmpDay
                    # bodyDict["fields"]["ymd"] = int(tmpYear * 10000 + tmpMonth * 100 + tmpDay)
                    bodyDict["time"] = CurrentTime(tmpYear, tmpMonth, tmpDay)
                    if table_name == "ashareconsensusindex":
                        bodyDict["tags"]["unique_code"] = bodyDict["fields"]["S_INFO_COMPCODE"]
                    else:
                        bodyDict["tags"]["unique_code"] = bodyDict["fields"]["S_INFO_WINDCODE"]
                    bodyDict["tags"]["unique_object"] = bodyDict["fields"]["OBJECT_ID"][7]
                    body.append(bodyDict)
                if len(body) > 0:
                    print("data length =", len(body))
                    # print(body[0])
                    self.session.Write(body)


class DatabaseConnSsymmetry(object):

    def __init__(self):
        self.username = '384465609@qq.com'
        self.password = 'qinghua123'
        self.id = None
        self.token = None
        self.authorization = None
        self.id_eastmoney_day = None
        self.info = None
        self.session = SessionInfluxDB()

        self.GetLogin()
        print("Got Login")
        self.GetMappings()
        print("Got Mappings")
        self.GetInfo()
        print("Got Info")

    def GetLogin(self):
        result = requests.post(
            url='https://api.ssymmetry.com/api/v2/user/login',
            data=json.dumps({
                'username': self.username,
                'password': self.password,
            }),
            headers={
                'Content-type': 'application/json',
            },
            verify=False
        ).json()
        self.id = result['data']['id']
        self.token = result['data']['token']
        self.authorization = 'user-%s-%s' % (self.id, self.token)

    def GetMappings(self):
        result = requests.post(
            url='https://api.ssymmetry.com/api/v2/individualStock/mappings',
            headers={
                'Authorization': self.authorization,
                'Content-type': 'application/json',
            },
            verify=False
        ).json()
        self.id_eastmoney_day = ''
        for i in result['data']:
            if i['name'] == u'天级数据':
                for j in i['list']:
                    if j['name'] == u'东方财富':
                        self.id_eastmoney_day = j['id']
                        break
                break

    def GetInfo(self):
        result = requests.post(
            url='https://api.ssymmetry.com/api/v2/individualStock/info',
            headers={
                'Authorization': self.authorization,
                'Content-type': 'application/json',
            },
            verify=False
        ).json()
        self.info = result['data']

    def GetIndividualStockSingleDay(self, date):
        result = requests.post(
            url='https://api.ssymmetry.com/api/v2/individualStock/day/singleDay',
            data=json.dumps({
                'id': self.id_eastmoney_day,
                'date': date,
            }),
            headers={
                'Authorization': self.authorization,
                'Content-type': 'application/json',
            },
            verify=False
        ).json()
        
        return result

    def GetIndividualDaySingleStock(self, stock, start, end):
        result = requests.post(
            url='https://api.ssymmetry.com/api/v2/individualStock/day/singleStock',
            data=json.dumps({
                'id': self.id_eastmoney_day,
                'code': stock,
                'startDate': start,
                'endDate': end,
            }),
            headers={
                'Authorization': self.authorization,
                'Content-type': 'application/json',
            },
            verify=False
        ).json()

        return result

    def GetData(self, start='2019-01-01', end='2020-10-30'):
        num = 0
        datestart = datetime.datetime.strptime(start, '%Y-%m-%d')
        dateend = datetime.datetime.strptime(end, '%Y-%m-%d')
        while datestart <= dateend:
            result = self.GetIndividualStockSingleDay(str(datestart)[0:10])
            # print(result)
            if result['data'] is not None:
                body = []
                for data in result['data']:
                    tmpYear = int(data["date"][0:4])
                    tmpMonth = int(data["date"][5:7])
                    tmpDay = int(data["date"][8:10])
                    # print(tmpYear, tmpMonth, tmpDay)
                    # data["year"] = tmpYear
                    # data["month"] = tmpMonth
                    # data["day"] = tmpDay
                    # data["ymd"] = int(tmpYear * 10000 + tmpMonth * 100 + tmpDay)
                    bodyDict = {
                        "measurement": "ssymmetry_guba",
                        "time": CurrentTime(tmpYear, tmpMonth, tmpDay),
                        # "tags": {"unique_datetime": str(datetime.datetime.now())},
                        "tags": {
                            "unique_code": data["code"],
                        },
                        "fields": data,
                    }
                    body.append(bodyDict)
                num += len(body)
                print("date =", datestart)
                print("data length =", len(body))
                # print(body[0])
                self.session.Write(body)
            datestart += datetime.timedelta(days=1)
        return num


class DatabaseConnUqer(object):

    def __init__(self):
        self.client = uqer.Client(token=u'18266a7c0ac9f8cdbe00f9b2ecb65f42316a5f78d9cc22ebabcbd923593356e4')
        self.date_begin = '2010-01-01'
        self.date_end = '2020-07-31'
        self.date_begin0 = '20100101'
        self.date_end0 = '20200731'
        if not os.path.exists('SecID.pkl'):
            self.GetSecID()
        if not os.path.exists('TradeCal.pkl'):
            self.GetTradeCal()
        SecID = pd.read_pickle('SecID.pkl')
        TradeCal = pd.read_pickle('TradeCal.pkl')
        tmp = TradeCal[TradeCal['exchangeCD'] == 'XSHE']
        tmp = tmp[tmp['isOpen'] == 1]['calendarDate']
        self.index = tmp
        self.columns = SecID['secID']
        self.mode = pd.DataFrame(index=tmp, columns=SecID['secID'])
        self.session = SessionInfluxDB()

    @staticmethod
    def Dataframe2Body(df, database):
        body = []
        for rec in df.to_dict(orient="records"):
            for i in rec:
                if type(rec[i]) is int:
                    if np.isnan(rec[i]):
                        rec[i] = None
                    else:
                        rec[i] = float(rec[i])
                elif type(rec[i]) is float:
                    if np.isnan(rec[i]):
                        rec[i] = None
                elif rec[i] is not None:
                    rec[i] = str(rec[i])
            bodyDict = {
                "measurement": "uqer_%s" % database,
                # "time": CurrentTime(),
                "tags": {},
                "fields": rec,
            }
            body.append(bodyDict)
        return body

    @staticmethod
    def Dataframe2Body4Code(df, database):
        body = []
        for rec in df.to_dict(orient="records"):
            for i in rec:
                if type(rec[i]) is int:
                    if np.isnan(rec[i]):
                        rec[i] = None
                    else:
                        rec[i] = float(rec[i])
                elif type(rec[i]) is float:
                    if np.isnan(rec[i]):
                        rec[i] = None
                elif rec[i] is not None:
                    rec[i] = str(rec[i])
            bodyDict = {
                "measurement": "uqer_%s_%s.%s" % (database, rec["ticker"], rec["exchangeCD"]),
                # "time": CurrentTime(),
                "tags": {},
                "fields": rec,
            }
            body.append(bodyDict)
        return body

    @staticmethod
    def AddDate2Fields(body, word):
        for i in body:
            t = i["fields"][word]
            if t is not None:
                tmpYear = int(t[0:4])
                tmpMonth = int(t[5:7])
                tmpDay = int(t[8:10])
                tmpYMD = int(tmpYear * 10000 + tmpMonth * 100 + tmpDay)
                i["fields"]["year_%s" % word] = tmpYear
                i["fields"]["month_%s" % word] = tmpMonth
                i["fields"]["day_%s" % word] = tmpDay
                i["fields"]["ymd_%s" % word] = tmpYMD
        return body

    @staticmethod
    def AddDateTime2Fields(body, word):
        for i in body:
            t = i["fields"][word]
            tmpYear = int(t[0:4])
            tmpMonth = int(t[5:7])
            tmpDay = int(t[8:10])
            tmpYMD = int(tmpYear * 10000 + tmpMonth * 100 + tmpDay)
            tmpHour = int(t[11:13])
            tmpMinute = int(t[14:16])
            tmpSecond = int(t[17:19])
            tmpHMS = int(tmpHour * 10000 + tmpMinute * 100 + tmpSecond)
            i["fields"]["year_%s" % word] = tmpYear
            i["fields"]["month_%s" % word] = tmpMonth
            i["fields"]["day_%s" % word] = tmpDay
            i["fields"]["ymd_%s" % word] = tmpYMD
            i["fields"]["hour_%s" % word] = tmpHour
            i["fields"]["minute_%s" % word] = tmpMinute
            i["fields"]["second_%s" % word] = tmpSecond
            i["fields"]["hms_%s" % word] = tmpHMS
        return body

    @staticmethod
    def AddDate2Time(body, word):
        for i in body:
            t = i["fields"][word]
            if t is None:
                continue
            tmpYear = int(t[0:4])
            tmpMonth = int(t[5:7])
            tmpDay = int(t[8:10])
            # tmpYMD = int(tmpYear * 10000 + tmpMonth * 100 + tmpDay)
            # i["fields"]["year"] = tmpYear
            # i["fields"]["month"] = tmpMonth
            # i["fields"]["day"] = tmpDay
            # i["fields"]["ymd"] = tmpYMD
            i["time"] = CurrentTime(tmpYear, tmpMonth, tmpDay)
        return body

    @staticmethod
    def AddDateTime2Time(body, word):
        for i in body:
            t = i["fields"][word]
            tmpYear = int(t[0:4])
            tmpMonth = int(t[5:7])
            tmpDay = int(t[8:10])
            # tmpYMD = int(tmpYear * 10000 + tmpMonth * 100 + tmpDay)
            tmpHour = int(t[11:13])
            tmpMinute = int(t[14:16])
            tmpSecond = int(t[17:19])
            # tmpHMS = int(tmpHour * 10000 + tmpMinute * 100 + tmpSecond)
            # i["fields"]["year"] = tmpYear
            # i["fields"]["month"] = tmpMonth
            # i["fields"]["day"] = tmpDay
            # i["fields"]["ymd"] = tmpYMD
            # i["fields"]["hour"] = tmpHour
            # i["fields"]["minute"] = tmpMinute
            # i["fields"]["second"] = tmpSecond
            # i["fields"]["hms"] = tmpHMS
            i["time"] = CurrentTime(tmpYear, tmpMonth, tmpDay, tmpHour, tmpMinute, tmpSecond)
        return body

    @staticmethod
    def AddCode2Tags(body, word):
        for i in body:
            i["tags"]["unique_%s" % word] = i["fields"][word]
        return body

    def GetSecID(self):
        test = uqer.DataAPI.SecIDGet(partyID=u"", ticker=u"", cnSpell=u"", assetClass=u"E",
                                     field=['secID', 'ticker', 'exchangeCD', 'listStatusCD', 'listDate',
                                            'transCurrCD'], pandas="1")
        test = test[test['transCurrCD'] == 'CNY']
        test = test[test['listStatusCD'] != 'UN']
        test_xshg = test[test['exchangeCD'] == 'XSHG']
        test_xshe = test[test['exchangeCD'] == 'XSHE']
        SecID = pd.concat([test_xshe, test_xshg])
        SecID = SecID[SecID['listStatusCD'] != 'O']
        _dict_SecID = SecID.to_dict(orient='index')
        _drop_key = []
        for key in _dict_SecID:
            # print(_dict_SecID[key])
            if len(_dict_SecID[key]['ticker']) != 6:
                _drop_key.append(key)
        SecID = SecID.drop(index=_drop_key)
        SecID.to_pickle('SecID.pkl')
        return SecID

    def GetTradeCal(self):
        test = uqer.DataAPI.TradeCalGet(exchangeCD=u"XSHG,XSHE", beginDate=self.date_begin0, endDate=self.date_end0,
                                        field=u"", pandas="1")
        # test_xshg = test[test['exchangeCD'] == 'XSHG']['isOpen'].values
        # test_xshe = test[test['exchangeCD'] == 'XSHE']['isOpen'].values
        TradeCal = test
        TradeCal.to_pickle('TradeCal.pkl')
        return TradeCal

    def GetMktEqud(self, b, e):
        print("GetMktEqud")
        db = datetime.datetime.strptime(b, '%Y-%m-%d')
        de = datetime.datetime.strptime(e, '%Y-%m-%d')
        while db <= de:
            print(db)
            result = uqer.DataAPI.MktEqudGet(secID=u"", ticker=u"", tradeDate=str(db)[0:10], beginDate=u"",
                                         endDate=u"", isOpen="", field=u"", pandas="1")
            body = self.Dataframe2Body(result, 'MktEqud')
            if len(body) > 0:
                body = self.AddDate2Time(body, "tradeDate")
                body = self.AddCode2Tags(body, "secID")
                print("data length =", len(body))
                # print(body[0])
                self.session.Write(body)
            db += datetime.timedelta(days=1)

    def GetFdmtBS(self, b, e):
        print("GetFdmtBS")
        # for all ticker
        # for s in self.columns.tolist():
        result = uqer.DataAPI.FdmtBSGet(ticker="", secID=self.columns.tolist(), reportType=u"", endDate=u"", beginDate=u"",
                                        publishDateEnd=e, publishDateBegin=b, endDateRep="",
                                        beginDateRep="",
                                        beginYear="", endYear="", fiscalPeriod="", field=u"", pandas="1")
        body = self.Dataframe2Body(result, 'FdmtBS')
        if len(body) > 0:
            body = self.AddDateTime2Fields(body, "updateTime")
            body = self.AddDate2Time(body, "publishDate")
            body = self.AddCode2Tags(body, "endDate")
            body = self.AddDate2Fields(body, "endDateRep")
            body = self.AddCode2Tags(body, "secID")
            print("data length =", len(body))
            # print(body[0])
            self.session.Write(body)

    def GetFdmtCF(self, b, e):
        print("GetFdmtCF")
        result = uqer.DataAPI.FdmtCFGet(ticker="", secID=self.columns.tolist(), reportType=u"", endDate=u"", beginDate=u"",
                                        publishDateEnd=e, publishDateBegin=b, endDateRep="", beginDateRep="",
                                        beginYear="", endYear="", fiscalPeriod="", field=u"", pandas="1")
        body = self.Dataframe2Body(result, 'FdmtCF')
        if len(body) > 0:
            body = self.AddDateTime2Fields(body, "updateTime")
            body = self.AddDate2Time(body, "publishDate")
            body = self.AddCode2Tags(body, "endDate")
            body = self.AddDate2Fields(body, "endDateRep")
            body = self.AddCode2Tags(body, "secID")
            print("data length =", len(body))
            # print(body[0])
            self.session.Write(body)

    def GetFdmtIS(self, b, e):
        print("GetFdmtIS")
        result = uqer.DataAPI.FdmtISGet(ticker="", secID=self.columns.tolist(), reportType=u"", endDate=u"", beginDate=u"",
                                        publishDateEnd=e, publishDateBegin=b, endDateRep="", beginDateRep="",
                                        beginYear="", endYear="", fiscalPeriod="", field=u"", pandas="1")
        body = self.Dataframe2Body(result, 'FdmtIS')
        if len(body) > 0:
            body = self.AddDateTime2Fields(body, "updateTime")
            body = self.AddDate2Time(body, "publishDate")
            body = self.AddCode2Tags(body, "endDate")
            body = self.AddDate2Fields(body, "endDateRep")
            body = self.AddCode2Tags(body, "secID")
            print("data length =", len(body))
            # print(body[0])
            self.session.Write(body)

    def GetFdmtDerPit(self, b, e):
        print("GetFdmtDerPit")
        r = uqer.DataAPI.FdmtDerPitGet(ticker="", secID=self.columns.tolist(), beginDate="",endDate="",beginYear=u"",endYear=u"",reportType=u"",publishDateEnd=e,publishDateBegin=b,field=u"",pandas="1")
        body = self.Dataframe2Body(r, "FdmtDerPit")
        if len(body) > 0:
            body = self.AddDateTime2Fields(body, "actPubtime")
            body = self.AddDate2Time(body, "publishDate")
            body = self.AddCode2Tags(body, "endDate")
            body = self.AddCode2Tags(body, "secID")
            print("data length =", len(body))
            # print(body[0])
            self.session.Write(body)

    def GetHKshszHold(self, b, e):
        print("GetHKshszHold")
        # for all date
        result = uqer.DataAPI.HKshszHoldGet(secID=u"", ticker=u"", tradeCD=u"1", ticketCode=u"", partyName=u"",
                                            beginDate=b, endDate=e, field=u"", pandas="1")
        body1 = self.Dataframe2Body(result, 'HKshszHold')
        result = uqer.DataAPI.HKshszHoldGet(secID=u"", ticker=u"", tradeCD=u"2", ticketCode=u"", partyName=u"",
                                            beginDate=b, endDate=e, field=u"", pandas="1")
        body2 = self.Dataframe2Body(result, 'HKshszHold')
        if len(body1) > 0:
            body1 = self.AddDateTime2Time(body1, "updateTime")
            body1 = self.AddDate2Fields(body1, "endDate")
            body1 = self.AddCode2Tags(body1, "secID")
            print("data length =", len(body1))
            # print(body1[0])
            self.session.Write(body1)
        if len(body2) > 0:
            body2 = self.AddDateTime2Time(body2, "updateTime")
            body2 = self.AddDate2Fields(body2, "endDate")
            body2 = self.AddCode2Tags(body2, "secID")
            print("data length =", len(body2))
            # print(body2[0])
            self.session.Write(body2)

    def GetHKshszDetl(self, b, e):
        db = datetime.datetime.strptime(b, '%Y-%m-%d')
        de = datetime.datetime.strptime(e, '%Y-%m-%d')
        while db <= de:
            print(db)
            result = uqer.DataAPI.HKshszDetlGet(beginDate=db,endDate=db,secID=self.columns.tolist(),ticker=u"",partyName=u"",tradeCD="",shcID=u"",field=u"",pandas="1")
            # result.to_csv('002594.csv')
            body = self.Dataframe2Body(result, 'HKshszDetl')
            body = self.AddCode2Tags(body, "secID")
            body = self.AddCode2Tags(body, "shcID")
            body = self.AddDateTime2Time(body, "updateTime")
            body = self.AddDate2Fields(body, "endDate")
            if len(body) > 0:
                print("data length =", len(body))
                print(body[0])
                self.session.Write(body)
            db += datetime.timedelta(days=1)
            # for i in body:
            #     print(i["tags"])

    def GetMktEquFlowOrder(self, b, e):
        print("GetMktEquFlowOrder")
        result = uqer.DataAPI.MktEquFlowOrderGet(secID=u"", ticker=u"", beginDate=b,
                                                 endDate=e, field=u"", pandas="1")
        body = self.Dataframe2Body(result, 'MktEquFlowOrder')
        if len(body) > 0:
            body = self.AddDate2Time(body, "tradeDate")
            body = self.AddCode2Tags(body, "secID")
            print("data length =", len(body))
            # print(body[0])
            self.session.Write(body)

    def GetFund(self):
        print("GetFund")
        result = uqer.DataAPI.FundGet(secID=u"", ticker=u"", etfLof=u"", listStatusCd=['L', 'S', 'DE', 'UN'], category=u"",
                                      idxID=u"", idxTicker=u"", operationMode=u"", field=u"", pandas="1")
        body = self.Dataframe2Body(result, 'Fund')
        body = self.AddDate2Time(body, "establishDate")
        body = self.AddCode2Tags(body, "secID")
        print("data length =", len(body))
        # print(body[0])
        self.session.Write(body)

    def GetFundHoldings(self, new=False):
        print("GetFundHoldings")
        r = uqer.DataAPI.SecIDGet(partyID=u"",ticker=u"",cnSpell=u"",assetClass=u"F",field=u"",pandas="1")['secID'].tolist()
        # print(r)
        # db = datetime.datetime.strptime(b, '%Y%m%d')
        # de = datetime.datetime.strptime(e, '%Y%m%d')
        # while db <= de:
        #     result = uqer.DataAPI.FundHoldingsGet(secID=u"", ticker=u"", reportDate="20200930", beginDate=u"", endDate=u"",
        #                                         secType="", field=u"", pandas="1")
        #     body = self.Dataframe2Body(result, 'FundHoldings')
        #     print(len(body))
        #     db += datetime.timedelta(days=1)
        # print(len(r))
        for s in r:
            result = uqer.DataAPI.FundHoldingsGet(secID=s, ticker=u"", reportDate=u"", beginDate='', endDate='',
                                                secType="", field=u"", pandas="1")
            if new:
                result = result[result['publishDate']==str(datetime.date.today())]
            body = self.Dataframe2Body(result, 'FundHoldings')
            if len(body) > 0:
                body = self.AddDate2Fields(body, "reportDate")
                body = self.AddDate2Time(body, "publishDate")
                body = self.AddCode2Tags(body, "secID")
                print("data length =", len(body))
                # print(body[0])
                self.session.Write(body)

    def GetEquShTen(self, b, e):
        print("GetEquShTen")
        b1 = datetime.datetime.strptime(b, "%Y-%m-%d") + datetime.timedelta(days=-31)
        result = uqer.DataAPI.EquShTenGet(secID=self.columns.tolist(), ticker=u"", beginDate=b1, endDate=e, field=u"",
                                          pandas="1")
        body = self.Dataframe2Body(result, 'EquShTen')
        if len(body) > 0:
            body = self.AddDateTime2Time(body, "updateTime")
            body = self.AddDate2Fields(body, "endDate")
            body = self.AddDate2Fields(body, "publishDate")
            body = self.AddCode2Tags(body, "secID")
            print("data length =", len(body))
            # print(body[0])
            self.session.Write(body)

    def GetEquFloatShTen(self, b, e):
        print("GetEquFloatShTen")
        b1 = datetime.datetime.strptime(b, "%Y-%m-%d") + datetime.timedelta(days=-31)
        result = uqer.DataAPI.EquFloatShTenGet(secID=self.columns.tolist(), ticker=u"", beginDate=b1, endDate=e,
                                               field=u"", pandas="1")
        body = self.Dataframe2Body(result, 'EquFloatShTen')
        if len(body) > 0:
            body = self.AddDateTime2Fields(body, "updateTime")
            body = self.AddDate2Fields(body, "endDate")
            body = self.AddDate2Time(body, "publishDate")
            body = self.AddCode2Tags(body, "secID")
            body = self.AddCode2Tags(body, "shNum")
            print("data length =", len(body))
            # print(body[0])
            self.session.Write(body)

    def GetEquShareholderNum(self, b, e):
        print("GetEquShareholderNum")
        result = uqer.DataAPI.EquShareholderNumGet(secID=self.columns.tolist(), ticker=u"", beginDate=u"", endDate=u"",
                                                   beginPublishDate=b, endPublishDate=e, field=u"", pandas="1")
        body = self.Dataframe2Body(result, 'EquShareholderNum')
        if len(body) > 0:
            body = self.AddDateTime2Time(body, "updateTime")
            body = self.AddDate2Fields(body, "endDate")
            body = self.AddDate2Fields(body, "publishDate")
            body = self.AddCode2Tags(body, "secID")
            print("data length =", len(body))
            # print(body[0])
            self.session.Write(body)

    def GetMktStockFactorsDateRangePro(self, b, e):
        print("GetMktStockFactorsDateRangePro")
        for s in self.columns.tolist():
            result = uqer.DataAPI.MktStockFactorsDateRangeProGet(secID=s, ticker=u"", beginDate=b,
                                                                endDate=e, field=["secID", "ticker", "tradeDate", "TVMA20", "BBI", "Variance20", "VEMA26", "Rank1M", "BIAS60"], pandas="1")
            # print(result)
            body = self.Dataframe2Body(result, 'MktStockFactorsDateRangePro')
            if len(body) > 0:
                body = self.AddDate2Time(body, "tradeDate")
                body = self.AddCode2Tags(body, "secID")
                # print("data length =", len(body))
                # print(body[0])
                self.session.Write(body)

    def GetMktBarHistOneDay(self, b, e):
        print("GetMktBarHistOneDay")
        db = datetime.datetime.strptime(b, '%Y-%m-%d')
        de = datetime.datetime.strptime(e, '%Y-%m-%d')
        while de >= db:
            print(de)
            r = uqer.DataAPI.MktBarHistOneDayGet(securityID=self.columns.tolist(),date=str(de)[0:10],startTime=u"",endTime=u"",unit="",field=u"",pandas="1")
            body = self.Dataframe2Body4Code(r, "MktBarHistOneDay")
            if len(body) > 0:
                # body = self.AddDate2Fields(body, "dataDate")
                for i in body:
                    t = i["fields"]["dataDate"]
                    a = int(t[0:4])
                    b = int(t[5:7])
                    c = int(t[8:10])
                    t = i["fields"]["barTime"]
                    d = int(t[0:2])
                    e = int(t[3:5])
                    # tmpSecond = 0
                    # tmpHMS = int(tmpHour * 10000 + tmpMinute * 100)
                    # i["fields"]["hour"] = tmpHour
                    # i["fields"]["minute"] = tmpMinute
                    # i["fields"]["second"] = tmpSecond
                    # i["fields"]["hms"] = tmpHMS
                    i["time"] = CurrentTime(a, b, c, d, e)
                    i["tags"]["unique_code"] = "%s.%s" % (i["fields"]["ticker"], i["fields"]["exchangeCD"])
                print("data length =", len(body))
                # print(body[0])
                self.session.Write(body)
            de += datetime.timedelta(days=-1)


    def GetNewsRelatedScore(self, b, e):
        print("GetNewsRelatedScore")
        r = uqer.DataAPI.NewsRelatedScoreGet(assetClass=u"E",beginDate=b,endDate=e,newsID=u"",secID=u"",ticker=u"",partyID=u"2",equType=u"",field=u"",pandas="1")
        body = self.Dataframe2Body(r, "NewsRelatedScore")
        if len(body) > 0:
            body = self.AddDateTime2Fields(body, "newsPublishTime")
            body = self.AddDateTime2Time(body, "newsInsertTime")
            body = self.AddCode2Tags(body, "secID")
            print("data length =", len(body))
            # print(body[0])
            self.session.Write(body)

    def GetRecInfosMeta(self, b, e):
        print("GetRecInfosMeta")
        _len_body = 0
        client = Client()
        client.init('8711870acd70997d943f707252f5893935089002b5cd7d48e3cd9000dc1a706a')
        db = datetime.datetime.strptime(b, '%Y-%m-%d')
        de = datetime.datetime.strptime(e, '%Y-%m-%d')
        d_b = str(db)[0:10].replace("-", "")
        d_e = str(de)[0:10].replace("-", "")
        # while de >= db:
        #     print(de)
        tickerList = uqer.DataAPI.SecIDGet(partyID=u"",ticker=u"",cnSpell=u"",assetClass=u"E",exchangeCD="XSHG,XSHE",field=u"",pandas="1")['ticker'].tolist()
        print("length of r =", len(tickerList))
        for s in tickerList:
            # print(s)
            # d = str(de)[0:10].replace("-", "")
            url1='/api/subject/getRecInfosMeta.json?field=&ticker=%s&employer=&workLocation=&beginDate=%s&endDate=%s' % (s, d_b, d_e)
            code, result = client.getData(url1)#调用getData函数获取数据，数据以字符串的形式返回
            if code==200:
                # if len(result.decode('utf-8')) == 0:
                #     continue
                body = []
                try:
                    r = eval(result.decode("utf-8"))['data']
                except:
                    print(result.decode('utf-8'))
                    continue
                tag = 0
                latestDate = None
                for i in r:
                    try:
                        if type(i["aSalaRangeStart"]) is float:
                            i["aSalaRangeStart"] = int(i["aSalaRangeStart"])
                    except:
                        pass
                    try:
                        if type(i["aSalaRangeEnd"]) is float:
                            i["aSalaRangeEnd"] = int(i["aSalaRangeEnd"])
                    except:
                        pass
                    i#f latestDate != i["publishDate"] or (latestDate is not None and not i.__contains__("publishDate")):
                        #tag = 0
                      #  latestDate = i["publishDate"]
                    #else:
                    tag += 1
                    bodyDict = {
                        "measurement": "uqer_RecInfosMeta_lastest",
                        # "time": CurrentTime(),
                        "tags": {
                            "unique_ticker": i["ticker"] if i.__contains__("ticker") else None,
                            "unique_employer": i["employer"] if i.__contains__("employer") else None,
                            "unique_source": i["source"] if i.__contains__("source") else None,
                            #"unique_title": i["title"] if i.__contains__("title") else None,
                            "unique_publishDate": i["publishDate"][:10] if i.__contains__("publishDate") else None,
                            "unique_tag": tag,
                        },
                        "fields": i,
                    }
                    body.append(bodyDict)
                    # print("tag =", tag)
                if len(body) > 0:
                    body = self.AddDate2Time(body, "publishDate")
                    print("data length =", len(body))
                    _len_body += len(body)
                    # print(body[0])
                    self.session.Write(body)
            de += datetime.timedelta(days=-1)
        print("# DATA =", _len_body)

    def GetMktIdx(self, b, e):
        print("GetMktIdx")
        lr = uqer.DataAPI.SecIDGet(partyID=u"",ticker=u"",cnSpell=u"",assetClass=u"IDX",field=u"",pandas="1")['secID'].tolist()
        # print(len(lr))
        r = uqer.DataAPI.MktIdxdGet(indexID=lr,ticker=u"",tradeDate="",beginDate=b,endDate=e,exchangeCD=u"XSHE,XSHG",field=u"",pandas="1")
        body = self.Dataframe2Body(r, "MktIdx")
        if len(body) > 0:
            # body = self.AddDateTime2Fields(body, "newsPublishTime")
            body = self.AddDate2Time(body, "tradeDate")
            body = self.AddCode2Tags(body, "indexID")
            print("data length =", len(body))
            # print(body[0])
            self.session.Write(body)

    def GetEquIndustry(self):
        print("GetEquIndustry")
        r = uqer.DataAPI.EquIndustryGet(secID=self.columns.tolist(),ticker="",industryVersionCD=u"010301",industry=u"",industryID=u"",industryID1=u"",industryID2=u"",industryID3=u"",intoDate=u"",equTypeID=u"",field=u"",pandas="1")
        body = self.Dataframe2Body(r, "EquIndustry")
        if len(body) > 0:
            body = self.AddCode2Tags(body, "secID")
            body = self.AddDate2Time(body, "intoDate")
            body = self.AddDate2Fields(body, "outDate")
            print("data length =", len(body))
            # print(body[0])
            self.session.Write(body)


class DatabaseConnSuntime(object):

    def __init__(self):
        self.api = 'ggservice.go-goal.cn'
        self.public_key = 'DdTPHiFNPdurhcV'
        self.private_key = 'WhhWHvcgGFfDmgA76zSNZzYd9GgI3eE8'

        self.resource_name = ['t_factor_value_all', 't_factor_profit_all', 't_factor_growth_all', 't_factor_scale_all',
                              't_factor_emotion_all', 'stock_score_all']

        self.uqer = DatabaseConnUqer()
        self.uqer.GetSecID()
        # print("test1.5")
        self.session = SessionInfluxDB()

    @staticmethod
    def Dataframe2Body(df, database):
        body = []
        for rec in df.to_dict(orient="records"):
            for i in rec:
                if type(rec[i]) is float:
                    if np.isnan(rec[i]):
                        rec[i] = None
                elif type(rec[i]) is int:
                    if np.isnan(rec[i]):
                        rec[i] = None
                    else:
                        rec[i] = float(rec[i])
                elif rec[i] is not None:
                    rec[i] = str(rec[i])
            bodyDict = {
                "measurement": "suntime_%s" % database,
                # "time": CurrentTime(),
                # "tags": {"unique_datetime": str(datetime.datetime.now())},
                "tags": {
                    "unique_code": rec["stock_code"],
                },
                "fields": rec,
            }
            body.append(bodyDict)
        return body

    def GetFactorLib(self, begin='2005-01-01', end='2020-09-30'):
        sec = pd.read_pickle('SecID.pkl')
        for dataIdx in LIST_DATABASE_SUNTIME:
            print(dataIdx)
            result = FactorLib.Factor_Get(self.api, self.public_key, self.private_key, dataIdx, begin, end, 
                                            ['all'], sec['secID'].apply(lambda x: x[0:6]).tolist(), frequency=0, pandas=1)
            body = self.Dataframe2Body(result, dataIdx)
            for i in body:
                t = i["fields"]["con_date"]
                tmpYear = int(t[0:4])
                tmpMonth = int(t[5:7])
                tmpDay = int(t[8:10])
                # i["fields"]["year"] = tmpYear
                # i["fields"]["month"] = tmpMonth
                # i["fields"]["day"] = tmpDay
                # i["fields"]["ymd"] = int(tmpYear * 10000 + tmpMonth * 100 + tmpDay)
                i["time"] = CurrentTime(tmpYear, tmpMonth, tmpDay)
            if len(body) > 0:
                print("data length =", len(body))
                # print(body[0])
                self.session.Write(body)


class DatabaseConnJoinquant(object):

    def __init__(self):
        self.session = SessionInfluxDB()
        self.username = "15210597532"
        self.password = "jin00904173"
        auth(self.username, self.password)
        # print(is_auth())
        print(get_query_count())
        self.uqer = DatabaseConnUqer()
        self.uqer.GetSecID()
        self.sec = pd.read_pickle('SecID.pkl')['secID'].tolist()
        self.factors = get_all_factors()
    
    @staticmethod
    def Dataframe2Body(df, database):
        body = []
        for rec in df.to_dict(orient="records"):
            for i in rec:
                if type(rec[i]) is float:
                    if np.isnan(rec[i]):
                        rec[i] = None
                elif type(rec[i]) is int:
                    if np.isnan(rec[i]):
                        rec[i] = None
                    else:
                        rec[i] = float(rec[i])
                elif rec[i] is not None:
                    rec[i] = str(rec[i])
            bodyDict = {
                "measurement": "joinquant_%s" % database,
                # "time": CurrentTime(),
                # "tags": {"unique_datetime": str(datetime.datetime.now())},
                "tags": {},
                "fields": rec,
            }
            body.append(bodyDict)
        return body

    @staticmethod
    def Dict2Body(dict, database):
        body_list = []
        for key, value in dict.items():
            body = []
            for key1, rec in value.to_dict(orient="index").items():
                for i in rec:
                    if type(rec[i]) is float:
                        if np.isnan(rec[i]):
                            rec[i] = None
                    elif type(rec[i]) is int:
                        if np.isnan(rec[i]):
                            rec[i] = None
                        else:
                            rec[i] = float(rec[i])
                    elif rec[i] is not None:
                        rec[i] = str(rec[i])
                rec['date'] = str(key1)
                tmpYear = int(str(key1)[0:4])
                tmpMonth = int(str(key1)[5:7])
                tmpDay = int(str(key1)[8:10])
                # rec["year"] = tmpYear
                # rec["month"] = tmpMonth
                # rec["day"] = tmpDay
                # rec["ymd"] = int(tmpYear * 10000 + tmpMonth * 100 + tmpDay)
                bodyDict = {
                    "measurement": "joinquant_%s_%s" % (database, key),
                    "time": CurrentTime(tmpYear, tmpMonth, tmpDay),
                    # "tags": {"unique_datetime": str(datetime.datetime.now())},
                    "tags": {},
                    "fields": rec,
                }
                body.append(bodyDict)
            body_list.append(body)
        return body_list

    def GetFundamentals(self, begin='2019-01-01', end='2020-10-31'):
        # print(self.sec)
        datestart = datetime.datetime.strptime(begin, '%Y-%m-%d')
        dateend = datetime.datetime.strptime(end, '%Y-%m-%d')
        while datestart <= dateend:
            print("date =", datestart)
            df = get_fundamentals(query(
                    valuation
                ).filter(
                    valuation.code.in_(self.sec)
                ), date=datestart)
            body = self.Dataframe2Body(df, "fundamentals_valuation")
            for i in body:
                t = i["fields"]["day"]
                tmpYear = int(t[0:4])
                tmpMonth = int(t[5:7])
                tmpDay = int(t[8:10])
                # i["tags"] = {
                #     "year": tmpYear,
                #     "month": tmpMonth,
                #     "day": tmpDay,
                #     "ymd": int(tmpYear * 10000 + tmpMonth * 100 + tmpDay),
                # }
                # i["fields"]["year"] = tmpYear
                # i["fields"]["month"] = tmpMonth
                # i["fields"]["day"] = tmpDay
                # i["fields"]["ymd"] = int(tmpYear * 10000 + tmpMonth * 100 + tmpDay)
                i["time"] = CurrentTime(tmpYear, tmpMonth, tmpDay)
                i["tags"]["unique_code"] = i["fields"]["code"]
            if len(body) > 0:
                print("data length =", len(body))
                # print(body[0])
                self.session.Write(body)
            datestart += datetime.timedelta(days=1)

    def GetFinance(self, begin='2019-01-01', end='2020-10-30'):
        for code in self.sec:
            df = finance.run_query(query(finance.STK_INCOME_STATEMENT).filter(finance.STK_INCOME_STATEMENT.code==code, finance.STK_INCOME_STATEMENT.pub_date>=begin,
            finance.STK_INCOME_STATEMENT.pub_date<=end))
            body = self.Dataframe2Body(df, "finance_stk_income_statement")
            for i in body:
                t = i["fields"]["pub_date"]
                tmpYear = int(t[0:4])
                tmpMonth = int(t[5:7])
                tmpDay = int(t[8:10])
                # i["tags"] = {
                #     "year": tmpYear,
                #     "month": tmpMonth,
                #     "day": tmpDay,
                #     "ymd": int(tmpYear * 10000 + tmpMonth * 100 + tmpDay),
                # }
                # i["fields"]["year"] = tmpYear
                # i["fields"]["month"] = tmpMonth
                # i["fields"]["day"] = tmpDay
                # i["fields"]["ymd"] = int(tmpYear * 10000 + tmpMonth * 100 + tmpDay)
                i["time"] = CurrentTime(tmpYear, tmpMonth, tmpDay)
                i["tags"]["unique_code"] = i["fields"]["code"]
                i["tags"]["unique_end_date"] = i["fields"]["end_date"]
            if len(body) > 0:
                print("data length =", len(body))
                # print(body[0])
                self.session.Write(body)
            
            df = finance.run_query(query(finance.STK_BALANCE_SHEET).filter(finance.STK_BALANCE_SHEET.code==code,finance.STK_BALANCE_SHEET.pub_date>=begin,
            finance.STK_BALANCE_SHEET.pub_date<=end))
            body = self.Dataframe2Body(df, "finance_stk_balance_sheet")
            for i in body:
                t = i["fields"]["pub_date"]
                tmpYear = int(t[0:4])
                tmpMonth = int(t[5:7])
                tmpDay = int(t[8:10])
                # i["tags"] = {
                #     "year": tmpYear,
                #     "month": tmpMonth,
                #     "day": tmpDay,
                #     "ymd": int(tmpYear * 10000 + tmpMonth * 100 + tmpDay),
                # }
                # i["fields"]["year"] = tmpYear
                # i["fields"]["month"] = tmpMonth
                # i["fields"]["day"] = tmpDay
                # i["fields"]["ymd"] = int(tmpYear * 10000 + tmpMonth * 100 + tmpDay)
                i["time"] = CurrentTime(tmpYear, tmpMonth, tmpDay)
                i["tags"]["unique_code"] = i["fields"]["code"]
                i["tags"]["unique_end_date"] = i["fields"]["end_date"]
            if len(body) > 0:
                print("data length =", len(body))
                # print(body[0])
                self.session.Write(body)

    def GetFactorValues(self, begin='2019-01-01', end='2020-10-30'):
        datestart = datetime.datetime.strptime(begin, '%Y-%m-%d')
        dateend = datetime.datetime.strptime(end, '%Y-%m-%d')
        while datestart <= dateend:
            print("date =", datestart)
            # for f in self.factors['factor'].tolist():
                # print(f)
            f = ["margin_stability", "cash_rate_of_sales", "EBIT", "net_operate_cash_flow_to_asset", "roa_ttm_8y", "operating_profit_per_share"]
            dict = get_factor_values(self.sec, f, datestart, datestart)
            body_list = self.Dict2Body(dict, "factor_values")
            for body in body_list:
                if len(body) > 0:
                    # print(body[0])
                    print("data length =", len(body))
                    self.session.Write(body)
            datestart += datetime.timedelta(days=1)


if __name__ == '__main__':
    # dbConnSsy = DatabaseConnSsymmetry()
    # dbConnSsy.GetInfo()
    # dateStart = datetime.date(2007, 1, 3)
    # dateEnd = datetime.date(2020, 10, 20)
    # day = '2020-10-21'
    # body = []
    # print(day)
    # j = 0
    # for stock in dbConnSsy.info:
    #     result = dbConnSsy.GetIndividualDaySingleStock(stock['code'], day, day)
    #     if result['data'] is not None:
    #         bodyDict = {
    #             "measurement": "SsymmetryGuba",
    #             "time": CurrentTime(),
    #             "tags": {},
    #             "fields": result['data'][0],
    #         }
    #         body.append(bodyDict)
    #     j += 1
    #     if j > 5:
    #         break
    # print(body)
    # uqer.Client(token='8711870acd70997d943f707252f5893935089002b5cd7d48e3cd9000dc1a706a')
    # dbConnUqer = DatabaseConnUqer()
    # dbConnUqer.
    # res = dbConnUqer.GetSecID()
    # print(res)

    # dbConnSuntime = DatabaseConnSuntime()
    # dbConnSuntime.GetFactorLib(end=str(datetime.date.today()))
    # sec = pd.read_pickle('SecID.pkl')
    # for i in sec.itertuples():
    #     print(getattr(i, 'secID'))

    client = InfluxDBClient(host="175.25.50.120", port=12086, username="xtech", password="xtech123", database="alternative")
    body = [{
        "measurement": "test",
        "time": CurrentTime(2020, 12, 30),
        "tags": {},
        "fields": {"test": "测试"},
    }]
    client.write_points(body)
    result = client.query("select * from test;")
    print(result)
    # print(len(list(result.get_points())))
    # try:
    #     client.query("drop measurement suntime_t_factor_value_all")
    # except:
    #     pass
    # conn = DatabaseConnUqer()
    # conn.GetHKshszDetl()
