#coding: UTF-8
#Pythonsのライブラリから日経平均10年分のデータを取得して予測するプログラム(2010.01.01からデータあり)
#前日から４日分のデータから翌日の株価を予測する
#直近３００日間のデータを使用
#全データの前半75%を訓練に使用、直近25%でテスト実施。
import time

from scikit-learn import svm
from pandas_datareader import data as web
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import csv
import streamlit as st

st.set_page_config(
  page_title="nikkei_predict app",
  page_icon="🚁",
)

st.title('日経平均予測アプリ(Scikit-Learn)')
st.markdown('## 概要及び注意事項')
st.write("当アプリでは、翌営業日の日経平均の終値が前日終値よりも上昇するか、下落するかを過去データによりScikit-Learn（サキットラーン）を使用して予測します。ただし本結果により投資にいかなる損失が生じても、当アプリでは責任を取りません。あくまで参考程度にご利用ください。")

if st.button('予測開始'):
    try:
        comment = st.empty()
        comment.write('予測を開始しました。')
        t1 = time.time()
        n225 = web.DataReader("NIKKEI225", "fred")
        n225 = n225.dropna()
        lastday = n225[-1:]
        lastday = lastday.index.tolist()
        lastday = map(str, lastday)
        lastday = ''.join(lastday)
        lastday = lastday.rstrip("00:00:00")
        n225.to_csv("nikkei_price.csv")

        #データフレーム型からリスト型に変換して、価格のみを取り出す
        n225 = n225.values.tolist()
        #2Dを１Dに変換する
        n225arr = np.array(n225)
        n225arr = n225arr.ravel()
        n225 = n225arr.tolist()

        #データ数を直近の何日分にするか設定する
        stock_data_close = n225[-3000:]

        #予測する月日を示す
        st.write(f'{str(lastday)}の翌営業日の予測')
        #print(str(lastday) + "の翌営業日の予測")
        # データの確認
        # print (stock_data)
        count_s = len(stock_data_close)
        #print ('データ量:' + str(count_s) + '日分')

        # 株価の上昇率を算出、おおよそ-1.0-1.0の範囲に収まるように調整
        modified_data = []
        for i in range(1, count_s):
            modified_data.append(float(stock_data_close[i] - stock_data_close[i-1])/float(stock_data_close[i-1]) * 20)

        count_m = len(modified_data)

        # 前日までの4連続の上昇率のデータ
        successive_data = []
        # 正解値 価格上昇: 1 価格下落: 0
        answers = []
        for i in range(4, count_m):
            successive_data.append([modified_data[i-4], modified_data[i-3], modified_data[i-2], modified_data[i-1]])
            if modified_data[i] > 0:
                answers.append(1)
            else:
                answers.append(0)

        # データ数
        n = len(successive_data)

        m = len(answers)

        # 線形サポートベクターマシーン
        clf = svm.LinearSVC()
        # サポートベクターマシーンによる訓練 （データの75%を訓練に使用）
        clf.fit(successive_data[:int(n*750/1000)], answers[:int(n*750/1000)])

        # テスト用データ
        # 正解
        expected = answers[int(-n*250/1000):]
        # 予測
        predicted = clf.predict(successive_data[int(-n*250/1000):])

        # 末尾の10個を比較
        #print ('正解:' + str(expected[-10:]))
        #print ('予測:' + str(list(predicted[-10:])))

        # 正解率の計算
        correct = 0.0
        wrong = 0.0
        for i in range(int(n*250/1000)):
            if expected[i] == predicted[i]:
                correct += 1
            else:
                wrong += 1
                
        #print('正解数： ' + str(int(correct)))
        #print('不正解数： ' + str(int(wrong)))

        Positive_Solution_Rate = str(round(correct / (correct+wrong) * 100,  2))
        #print ("正解率: " + str(round(correct / (correct+wrong) * 100,  2)) + "%")

        successive_data.append([modified_data[count_m-4], modified_data[count_m-3], modified_data[count_m-2], modified_data[count_m-1]])
        predicted = clf.predict(successive_data[-1:])
        #print ('翌営業日の予測:' + str(list(predicted)) + ' 1:上昇,　0:下落')
        if str(list(predicted)) == str([1]):
            st.write('「上昇」するでしょう。')
            #print('翌営業日の日経平均株価は「上昇」するでしょう。')
        else:
            st.write('「下落」するでしょう。')
            #print('翌営業日の日経平均株価は「下落」するでしょう。')

        st.write(f'データ量：{str(count_s)}日分')
        st.write(f'正解率：{Positive_Solution_Rate}%')

        t2 = time.time()
        elapsed_time = t2- t1
        elapsed_time = round(elapsed_time, 2)
        st.write(f'プログラム処理時間： {str(elapsed_time)}秒')
        #print('プログラム処理時間： ' + str(elapsed_time) + '秒')
        comment.write('完了しました！')
    except:
        st.error('エラーが生じました。申し訳ありません。もうしばらくして、再度実行してください。')
