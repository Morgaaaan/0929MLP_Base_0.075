# 0929MLP_Base_0.075
<b>使用指南</b>
acct_transaction.csv檔案太大無法傳上來，請先丟進/dataset後，單獨執行split_transactions_by_alert_robust.py。<br>
將會生成/sort_data，之後執行main.py即可。<br>

產出預測請執行inference.py，閥值於檔案開頭調整。<br>
結果於/output中predict_raw.csv為浮點值，prediction.csv為提交檔。<br>

<b>簡介</b>
MLP測試第一版，alert相關交易紀錄全提出，非alert隨機提10000筆ID的交易紀錄。<br>
輸入層基本權輸入，沒做額外特徵，剩下黑盒讓他跑，Threshold判定0或1閥值下調至0.25，試跑一次public test為0.075。<br>
(註)此prediction.csv不為0.075那份<br>
