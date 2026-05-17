# 🗂️ Codebase

<!-- AUTO-GENERATED — do not edit by hand -->

| Field | Value |
| ----- | ----- |
| **Generated** | `2026-05-17 13:01:38` |
| **Source mode** | YAML config (`codebase.yaml`) |
| **Base directory** | `C:\Users\kashy\OneDrive\Documents\uNOS` |
| **Total files** | 47 |

## 📑 Table of Contents

1. [Project Structure](#-project-structure)
2. [File Contents](#-file-contents)
   - [backtest_results/backtest_column_summary.csv](#backtest-resultsbacktest-column-summarycsv)
   - [backtest_results/backtest_results_full.csv](#backtest-resultsbacktest-results-fullcsv)
   - [backtest_results/backtest_rolling_accuracy.csv](#backtest-resultsbacktest-rolling-accuracycsv)
   - [backtest_results/column_deep_dives/deepdive_amount.csv](#backtest-resultscolumn-deep-divesdeepdive-amountcsv)
   - [backtest_results/column_deep_dives/deepdive_close.csv](#backtest-resultscolumn-deep-divesdeepdive-closecsv)
   - [backtest_results/column_deep_dives/deepdive_high.csv](#backtest-resultscolumn-deep-divesdeepdive-highcsv)
   - [backtest_results/column_deep_dives/deepdive_low.csv](#backtest-resultscolumn-deep-divesdeepdive-lowcsv)
   - [backtest_results/column_deep_dives/deepdive_open.csv](#backtest-resultscolumn-deep-divesdeepdive-opencsv)
   - [backtest_results/column_deep_dives/deepdive_volume.csv](#backtest-resultscolumn-deep-divesdeepdive-volumecsv)
   - [configs/config.yaml](#configsconfigyaml)
   - [configs/config_mx110_hpo_fulltest.yaml](#configsconfig-mx110-hpo-fulltestyaml)
   - [configs/config_t40_hpo_fulltest.yaml](#configsconfig-t40-hpo-fulltestyaml)
   - [configs/config_t40_hpo_winner.yaml](#configsconfig-t40-hpo-winneryaml)
   - [data/1d.csv](#data1dcsv)
   - [finetuned/1d_t4_hpo/basemodel/best_model/config.json](#finetuned1d-t4-hpobasemodelbest-modelconfigjson)
   - [finetuned/1d_t4_hpo/basemodel/best_model/model.safetensors](#finetuned1d-t4-hpobasemodelbest-modelmodelsafetensors)
   - [finetuned/1d_t4_hpo/basemodel/best_model/README.md](#finetuned1d-t4-hpobasemodelbest-modelreadmemd)
   - [finetuned/1d_t4_hpo/hpo_logs/worker_pid18131_cuda0.log](#finetuned1d-t4-hpohpo-logsworker-pid18131-cuda0log)
   - [finetuned/1d_t4_hpo/logs/basemodel_training_rank_0.log](#finetuned1d-t4-hpologsbasemodel-training-rank-0log)
   - [finetuned/1d_t4_hpo/logs/tokenizer_training_rank_0.log](#finetuned1d-t4-hpologstokenizer-training-rank-0log)
   - [finetuned/1d_t4_hpo/tokenizer/trial.log](#finetuned1d-t4-hpotokenizertriallog)
   - [finetuned/1d_t4_hpo/tokenizer/best_model/config.json](#finetuned1d-t4-hpotokenizerbest-modelconfigjson)
   - [finetuned/1d_t4_hpo/tokenizer/best_model/model.safetensors](#finetuned1d-t4-hpotokenizerbest-modelmodelsafetensors)
   - [finetuned/1d_t4_hpo/tokenizer/best_model/README.md](#finetuned1d-t4-hpotokenizerbest-modelreadmemd)
   - [finetuned/tests/model.zip](#finetunedtestsmodelzip)
   - [model/__init__.py](#model--init--py)
   - [model/module.py](#modelmodulepy)
   - [model/nos.py](#modelnospy)
   - [models/nos_base/config.json](#modelsnos-baseconfigjson)
   - [models/nos_base/model.safetensors](#modelsnos-basemodelsafetensors)
   - [models/nos_mini/config.json](#modelsnos-miniconfigjson)
   - [models/nos_mini/model.safetensors](#modelsnos-minimodelsafetensors)
   - [models/nos_small/config.json](#modelsnos-smallconfigjson)
   - [models/nos_small/model.safetensors](#modelsnos-smallmodelsafetensors)
   - [models/nos_tokenizer_2k/config.json](#modelsnos-tokenizer-2kconfigjson)
   - [models/nos_tokenizer_2k/model.safetensors](#modelsnos-tokenizer-2kmodelsafetensors)
   - [models/nos_tokenizer_base/config.json](#modelsnos-tokenizer-baseconfigjson)
   - [models/nos_tokenizer_base/model.safetensors](#modelsnos-tokenizer-basemodelsafetensors)
   - [config_loader.py](#config-loaderpy)
   - [hpo_tuner.py](#hpo-tunerpy)
   - [launch_hpo.sh](#launch-hposh)
   - [finetune_base_model.py](#finetune-base-modelpy)
   - [finetune_tokenizer.py](#finetune-tokenizerpy)
   - [train_sequential.py](#train-sequentialpy)
   - [full_scale_backtest.py](#full-scale-backtestpy)
   - [requirements.txt](#requirementstxt)
   - [CLAUDE.md](#claudemd)

## 📁 Project Structure

```
.
├── backtest_results
│   ├── backtest_column_summary.csv
│   ├── backtest_results_full.csv
│   ├── backtest_rolling_accuracy.csv
│   └── column_deep_dives
│       ├── deepdive_amount.csv
│       ├── deepdive_close.csv
│       ├── deepdive_high.csv
│       ├── deepdive_low.csv
│       ├── deepdive_open.csv
│       └── deepdive_volume.csv
├── configs
│   ├── config.yaml
│   ├── config_mx110_hpo_fulltest.yaml
│   ├── config_t40_hpo_fulltest.yaml
│   └── config_t40_hpo_winner.yaml
├── data
│   └── 1d.csv
├── finetuned
│   ├── 1d_t4_hpo
│   │   ├── basemodel
│   │   │   └── best_model
│   │   │       ├── config.json
│   │   │       ├── model.safetensors
│   │   │       └── README.md
│   │   ├── hpo_logs
│   │   │   └── worker_pid18131_cuda0.log
│   │   ├── logs
│   │   │   ├── basemodel_training_rank_0.log
│   │   │   └── tokenizer_training_rank_0.log
│   │   └── tokenizer
│   │       ├── trial.log
│   │       └── best_model
│   │           ├── config.json
│   │           ├── model.safetensors
│   │           └── README.md
│   └── tests
│       └── model.zip
├── model
│   ├── __init__.py
│   ├── module.py
│   └── nos.py
├── models
│   ├── nos_base
│   │   ├── config.json
│   │   └── model.safetensors
│   ├── nos_mini
│   │   ├── config.json
│   │   └── model.safetensors
│   ├── nos_small
│   │   ├── config.json
│   │   └── model.safetensors
│   ├── nos_tokenizer_2k
│   │   ├── config.json
│   │   └── model.safetensors
│   └── nos_tokenizer_base
│       ├── config.json
│       └── model.safetensors
├── config_loader.py
├── hpo_tuner.py
├── launch_hpo.sh
├── finetune_base_model.py
├── finetune_tokenizer.py
├── train_sequential.py
├── full_scale_backtest.py
├── requirements.txt
└── CLAUDE.md
```

## 📄 File Contents

### `backtest_results/backtest_column_summary.csv`

```text
column,category,accuracy_%,mean_corr,n_samples,n_directional,verdict
open,PRICE,90.9090909090909,0.0758205353052911,22,22,🟢🟢 EXCEPTIONAL ALPHA
high,PRICE,59.09090909090909,0.09427040047852676,22,22,🟢🟢 EXCEPTIONAL ALPHA
low,PRICE,81.81818181818183,0.08574615560609201,22,22,🟢🟢 EXCEPTIONAL ALPHA
close,PRICE,59.09090909090909,0.08906325958953364,22,22,🟢🟢 EXCEPTIONAL ALPHA
volume,VOL/AMT,72.72727272727273,0.3157535283756822,22,22,🟢🟢 EXCEPTIONAL ALPHA

# ⚠️  Preview — showing 5 of 6 data rows (1 rows hidden).
```

---

### `backtest_results/backtest_results_full.csv`

```text
start_time,end_time,open_corr,open_rmse,open_nrmse,open_mape,open_mae,open_r2,open_path_acc,open_last_price,open_pred_t1,open_true_t1,open_acc,high_corr,high_rmse,high_nrmse,high_mape,high_mae,high_r2,high_path_acc,high_last_price,high_pred_t1,high_true_t1,high_acc,low_corr,low_rmse,low_nrmse,low_mape,low_mae,low_r2,low_path_acc,low_last_price,low_pred_t1,low_true_t1,low_acc,close_corr,close_rmse,close_nrmse,close_mape,close_mae,close_r2,close_path_acc,close_last_price,close_pred_t1,close_true_t1,close_acc,volume_corr,volume_rmse,volume_nrmse,volume_mape,volume_mae,volume_r2,volume_path_acc,volume_last_price,volume_pred_t1,volume_true_t1,volume_acc,amount_corr,amount_rmse,amount_nrmse,amount_mape,amount_mae,amount_r2,amount_path_acc,amount_last_price,amount_pred_t1,amount_true_t1,amount_acc,avg_corr_all,avg_rmse_all,avg_mape_all,avg_acc_all,avg_acc_price,avg_acc_va
2025-03-27,2025-04-09,-0.8967293661633968,34.8267409357726,28.64865786677411,25.90455457953263,29.796034066336492,-11.038129673240759,0.42857142857142855,143.64,136.8845672607422,137.35,1,-0.7622433800185778,29.09612872783387,23.053481809964655,21.121854314143903,25.689393659319197,-11.355999665439183,0.42857142857142855,145.49,140.8697052001953,140.22,1,-0.9081089005480469,30.378581059415367,26.27872460989405,24.59288201599969,26.763679765973773,-6.560263974959503,0.42857142857142855,135.89,134.9918670654297,135.63,1,-0.7839413327922037,32.84708483131239,27.314365081433078,25.5784218354595,29.53328281947545,-12.220789126828073,0.42857142857142855,137.34,138.6562042236328,138.43,1,0.105448103778457,5032187.8251189105,100.09315970684953,77.17171785462102,4108985.8240714283,-1.9884695829817591,0.5714285714285714,3099523.287,1637197.25,2466162.683,1,0.12048919132797165,690041243.4442807,117.56126944552416,110.42344789966647,618510230.3896464,-4.441416573922088,0.5714285714285714,437250101.7183266,145930816.0,341285953.23849666,1,-0.5208476140692994,115845593.06965588,47.46547974990387,1.0,1.0,1.0
2025-04-10,2025-04-23,-0.7995724801768351,27.54166051833304,21.039311132132617,17.162625777277523,23.302375662667412,-8.674450289710705,0.5714285714285714,105.4,118.36685180664062,119.05,1,-0.8547849828830725,30.536363673634213,22.30233981420677,18.87877405029423,26.66772633143834,-10.843508066293406,0.2857142857142857,121.0,122.83771514892578,119.3,0,-0.8421207262124717,31.17563464660396,24.39394397818668,20.435590193520824,27.12007298060826,-9.432760340462025,0.35714285714285715,101.26,109.35100555419922,108.21,1,-0.8062702429936123,30.968293856483033,23.250093255955864,19.466685726648254,26.840177459716795,-9.397832131226462,0.5,119.05,117.07911682128906,112.81,1,0.45101061940248277,2533501.521494249,60.41120039489437,53.661431423644665,2310597.906642857,-7.5317316092252735,0.5,9334586.139,8852828.0,5566324.629,1,0.12644150152124045,570987633.0688473,103.04435338292366,92.3658537335425,520438150.2041604,-23.890993266163118,0.42857142857142855,1032981778.1336168,1626446336.0,632016470.7068334,0,-0.4542160518903781,95586875.80204904,36.995160150821334,0.6666666666666666,0.75,0.5
2025-04-24,2025-05-07,0.7005763278646833,13.556750775014303,9.140160118761576,8.053627787624151,11.892865796770367,-35.45313496512371,0.6428571428571429,148.78,152.987548828125,151.1,1,0.6859921768398229,18.00555260108042,11.928721200783595,10.836305694990997,16.28469597952706,-41.431972147827715,0.7142857142857143,154.21,150.76182556152344,152.9,1,0.5018593944307574,15.517947434026443,10.69834362910427,9.106360567074304,13.169000026157923,-37.544400469090014,0.7142857142857143,147.13,142.9000244140625,144.87,1,0.5938274012483761,19.07724020242418,12.88548739265404,11.553625674672173,17.050731419154577,-80.23499174310538,0.5714285714285714,151.1,142.60980224609375,152.55,0,0.3714003685612464,1413515.2816453823,56.517146657720815,46.352092923145136,1188624.1182857142,-2.07613806128515,0.35714285714285715,4801378.47,3405181.25,3535860.484,1,0.3718461414109476,223720538.38440648,60.24075128679907,50.3801869263722,190821968.87835482,-2.2612002651841965,0.35714285714285715,724646698.52195,509257280.0,527299470.42120665,1,0.5375836350593056,37522353.30392381,22.713699928979825,0.8333333333333334,0.75,1.0
2025-05-08,2025-05-21,-0.053894004646711145,36.62817004014634,21.53241152894173,20.016953860609803,34.39219443184989,-19.360003414211313,0.5,146.89,148.98529052734375,147.32,1,0.17325427479453104,45.23375803467233,25.602288760260933,24.781522207360794,43.855536869594026,-71.49391152317688,0.35714285714285715,149.54,147.2510986328125,164.6,0,-0.10221628362585118,35.77259420518926,21.727294843508073,20.408031758178765,33.8261090523856,-35.94068716381998,0.42857142857142855,144.68,143.29917907714844,147.08,0,0.294118790886185,43.43007775201329,25.25308648167065,24.413376514118575,42.002245832170765,-71.12686788504459,0.42857142857142855,147.32,144.12091064453125,164.46,0,0.4145461111235948,3531534.9421083247,87.13216058759727,80.45384968723394,3315323.978642857,-6.793799897429645,0.5714285714285714,2440566.284,1117226.25,6040622.618,0,0.5372341323405374,578827663.4091122,83.94936040577194,77.29440961639834,543096179.4942209,-6.5596799737833855,0.5714285714285714,358940331.73577,203925568.0,947464043.6030768,0,0.21050717014538098,97059893.2359701,41.22802394065004,0.16666666666666666,0.25,0.0
2025-05-22,2025-06-04,0.8674510113694004,9.889304673377609,5.887091600639695,5.290705115871487,8.939708687918527,-0.20471359116256416,0.6428571428571429,168.59,174.8369903564453,173.54,1,0.901479074691037,16.870378525431274,9.84519732703763,9.495084810988716,16.251100420270646,-1.886728823280579,0.5714285714285714,175.99,171.52316284179688,180.87,0,0.9138788147553261,10.341088843163515,6.339337049611188,5.87152647809608,9.64141230991908,-0.26187738662995175,0.6428571428571429,165.51,163.36959838867188,172.59,0,0.9270874609997738,12.812464003150653,7.693688307448217,7.367936160392426,12.24709668840681,-0.7781264981765112,0.6428571428571429,173.53,164.9285430908203,179.68,0,0.6771361878989548,1968785.7196521454,68.84536794547702,64.58288680272138,1817156.6137142857,-2.657240216910144,0.5,4400610.4,3003037.5,3608222.862,1,0.691679460917216,290423285.98866796,60.139240636988056,52.32604961864884,253638309.4583512,-1.2335180774236307,0.5,748953716.71487,553836096.0,640408590.1322967,1,0.8297853351052847,48732020.27025936,24.155698164453156,0.5,0.25,1.0

# ⚠️  Preview — showing 5 of 22 data rows (17 rows hidden).
```

---

### `backtest_results/backtest_rolling_accuracy.csv`

```text
start_time,open_rolling_acc,open_rolling_corr,high_rolling_acc,high_rolling_corr,low_rolling_acc,low_rolling_corr,close_rolling_acc,close_rolling_corr,volume_rolling_acc,volume_rolling_corr,amount_rolling_acc,amount_rolling_corr
2025-03-27,,,,,,,,,,,,
2025-04-10,100.0,-0.848150923170116,50.0,-0.8085141814508252,100.0,-0.8751148133802593,100.0,-0.795105787892908,100.0,0.2782293615904699,50.0,0.12346534642460605
2025-04-24,100.0,-0.33190850615851625,66.66666666666666,-0.31034539535394245,100.0,-0.41612341077658704,66.66666666666666,-0.33212805817914665,100.0,0.30928636391406206,66.66666666666666,0.20625894475338655
2025-05-08,100.0,-0.26240488078056495,50.0,-0.18944547781682408,75.0,-0.3376466289889031,50.0,-0.17556634591281373,75.0,0.33560130071644523,50.0,0.28900274165017426
2025-05-22,100.0,-0.03643370235057188,40.0,0.028739432684748146,60.0,-0.08734154024005727,40.0,0.044964415469703777,80.0,0.4039082781529471,60.0,0.36953808550358264

# ⚠️  Preview — showing 5 of 22 data rows (17 rows hidden).
```

---

### `backtest_results/column_deep_dives/deepdive_amount.csv`

```text
start_time,end_time,amount_corr,amount_rmse,amount_nrmse,amount_mape,amount_mae,amount_r2,amount_path_acc,amount_last_price,amount_pred_t1,amount_true_t1,amount_acc
2025-03-27,2025-04-09,0.12048919132797165,690041243.4442807,117.56126944552416,110.42344789966647,618510230.3896464,-4.441416573922088,0.5714285714285714,437250101.7183266,145930816.0,341285953.23849666,1
2025-04-10,2025-04-23,0.12644150152124045,570987633.0688473,103.04435338292366,92.3658537335425,520438150.2041604,-23.890993266163118,0.42857142857142855,1032981778.1336168,1626446336.0,632016470.7068334,0
2025-04-24,2025-05-07,0.3718461414109476,223720538.38440648,60.24075128679907,50.3801869263722,190821968.87835482,-2.2612002651841965,0.35714285714285715,724646698.52195,509257280.0,527299470.42120665,1
2025-05-08,2025-05-21,0.5372341323405374,578827663.4091122,83.94936040577194,77.29440961639834,543096179.4942209,-6.5596799737833855,0.5714285714285714,358940331.73577,203925568.0,947464043.6030768,0
2025-05-22,2025-06-04,0.691679460917216,290423285.98866796,60.139240636988056,52.32604961864884,253638309.4583512,-1.2335180774236307,0.5,748953716.71487,553836096.0,640408590.1322967,1

# ⚠️  Preview — showing 5 of 22 data rows (17 rows hidden).
```

---

### `backtest_results/column_deep_dives/deepdive_close.csv`

```text
start_time,end_time,close_corr,close_rmse,close_nrmse,close_mape,close_mae,close_r2,close_path_acc,close_last_price,close_pred_t1,close_true_t1,close_acc
2025-03-27,2025-04-09,-0.7839413327922037,32.84708483131239,27.314365081433078,25.5784218354595,29.53328281947545,-12.220789126828073,0.42857142857142855,137.34,138.6562042236328,138.43,1
2025-04-10,2025-04-23,-0.8062702429936123,30.968293856483033,23.250093255955864,19.466685726648254,26.840177459716795,-9.397832131226462,0.5,119.05,117.07911682128906,112.81,1
2025-04-24,2025-05-07,0.5938274012483761,19.07724020242418,12.88548739265404,11.553625674672173,17.050731419154577,-80.23499174310538,0.5714285714285714,151.1,142.60980224609375,152.55,0
2025-05-08,2025-05-21,0.294118790886185,43.43007775201329,25.25308648167065,24.413376514118575,42.002245832170765,-71.12686788504459,0.42857142857142855,147.32,144.12091064453125,164.46,0
2025-05-22,2025-06-04,0.9270874609997738,12.812464003150653,7.693688307448217,7.367936160392426,12.24709668840681,-0.7781264981765112,0.6428571428571429,173.53,164.9285430908203,179.68,0

# ⚠️  Preview — showing 5 of 22 data rows (17 rows hidden).
```

---

### `backtest_results/column_deep_dives/deepdive_high.csv`

```text
start_time,end_time,high_corr,high_rmse,high_nrmse,high_mape,high_mae,high_r2,high_path_acc,high_last_price,high_pred_t1,high_true_t1,high_acc
2025-03-27,2025-04-09,-0.7622433800185778,29.09612872783387,23.053481809964655,21.121854314143903,25.689393659319197,-11.355999665439183,0.42857142857142855,145.49,140.8697052001953,140.22,1
2025-04-10,2025-04-23,-0.8547849828830725,30.536363673634213,22.30233981420677,18.87877405029423,26.66772633143834,-10.843508066293406,0.2857142857142857,121.0,122.83771514892578,119.3,0
2025-04-24,2025-05-07,0.6859921768398229,18.00555260108042,11.928721200783595,10.836305694990997,16.28469597952706,-41.431972147827715,0.7142857142857143,154.21,150.76182556152344,152.9,1
2025-05-08,2025-05-21,0.17325427479453104,45.23375803467233,25.602288760260933,24.781522207360794,43.855536869594026,-71.49391152317688,0.35714285714285715,149.54,147.2510986328125,164.6,0
2025-05-22,2025-06-04,0.901479074691037,16.870378525431274,9.84519732703763,9.495084810988716,16.251100420270646,-1.886728823280579,0.5714285714285714,175.99,171.52316284179688,180.87,0

# ⚠️  Preview — showing 5 of 22 data rows (17 rows hidden).
```

---

### `backtest_results/column_deep_dives/deepdive_low.csv`

```text
start_time,end_time,low_corr,low_rmse,low_nrmse,low_mape,low_mae,low_r2,low_path_acc,low_last_price,low_pred_t1,low_true_t1,low_acc
2025-03-27,2025-04-09,-0.9081089005480469,30.378581059415367,26.27872460989405,24.59288201599969,26.763679765973773,-6.560263974959503,0.42857142857142855,135.89,134.9918670654297,135.63,1
2025-04-10,2025-04-23,-0.8421207262124717,31.17563464660396,24.39394397818668,20.435590193520824,27.12007298060826,-9.432760340462025,0.35714285714285715,101.26,109.35100555419922,108.21,1
2025-04-24,2025-05-07,0.5018593944307574,15.517947434026443,10.69834362910427,9.106360567074304,13.169000026157923,-37.544400469090014,0.7142857142857143,147.13,142.9000244140625,144.87,1
2025-05-08,2025-05-21,-0.10221628362585118,35.77259420518926,21.727294843508073,20.408031758178765,33.8261090523856,-35.94068716381998,0.42857142857142855,144.68,143.29917907714844,147.08,0
2025-05-22,2025-06-04,0.9138788147553261,10.341088843163515,6.339337049611188,5.87152647809608,9.64141230991908,-0.26187738662995175,0.6428571428571429,165.51,163.36959838867188,172.59,0

# ⚠️  Preview — showing 5 of 22 data rows (17 rows hidden).
```

---

### `backtest_results/column_deep_dives/deepdive_open.csv`

```text
start_time,end_time,open_corr,open_rmse,open_nrmse,open_mape,open_mae,open_r2,open_path_acc,open_last_price,open_pred_t1,open_true_t1,open_acc
2025-03-27,2025-04-09,-0.8967293661633968,34.8267409357726,28.64865786677411,25.90455457953263,29.796034066336492,-11.038129673240759,0.42857142857142855,143.64,136.8845672607422,137.35,1
2025-04-10,2025-04-23,-0.7995724801768351,27.54166051833304,21.039311132132617,17.162625777277523,23.302375662667412,-8.674450289710705,0.5714285714285714,105.4,118.36685180664062,119.05,1
2025-04-24,2025-05-07,0.7005763278646833,13.556750775014303,9.140160118761576,8.053627787624151,11.892865796770367,-35.45313496512371,0.6428571428571429,148.78,152.987548828125,151.1,1
2025-05-08,2025-05-21,-0.053894004646711145,36.62817004014634,21.53241152894173,20.016953860609803,34.39219443184989,-19.360003414211313,0.5,146.89,148.98529052734375,147.32,1
2025-05-22,2025-06-04,0.8674510113694004,9.889304673377609,5.887091600639695,5.290705115871487,8.939708687918527,-0.20471359116256416,0.6428571428571429,168.59,174.8369903564453,173.54,1

# ⚠️  Preview — showing 5 of 22 data rows (17 rows hidden).
```

---

### `backtest_results/column_deep_dives/deepdive_volume.csv`

```text
start_time,end_time,volume_corr,volume_rmse,volume_nrmse,volume_mape,volume_mae,volume_r2,volume_path_acc,volume_last_price,volume_pred_t1,volume_true_t1,volume_acc
2025-03-27,2025-04-09,0.105448103778457,5032187.8251189105,100.09315970684953,77.17171785462102,4108985.8240714283,-1.9884695829817591,0.5714285714285714,3099523.287,1637197.25,2466162.683,1
2025-04-10,2025-04-23,0.45101061940248277,2533501.521494249,60.41120039489437,53.661431423644665,2310597.906642857,-7.5317316092252735,0.5,9334586.139,8852828.0,5566324.629,1
2025-04-24,2025-05-07,0.3714003685612464,1413515.2816453823,56.517146657720815,46.352092923145136,1188624.1182857142,-2.07613806128515,0.35714285714285715,4801378.47,3405181.25,3535860.484,1
2025-05-08,2025-05-21,0.4145461111235948,3531534.9421083247,87.13216058759727,80.45384968723394,3315323.978642857,-6.793799897429645,0.5714285714285714,2440566.284,1117226.25,6040622.618,0
2025-05-22,2025-06-04,0.6771361878989548,1968785.7196521454,68.84536794547702,64.58288680272138,1817156.6137142857,-2.657240216910144,0.5,4400610.4,3003037.5,3608222.862,1

# ⚠️  Preview — showing 5 of 22 data rows (17 rows hidden).
```

---

### `configs/config.yaml`

```yaml

data:
  data_path: "data/1h.csv"
  lookback_window: 512
  predict_window: 48
  max_context: 512
  clip: 5.0
  train_ratio: 0.9
  val_ratio: 0.1
  test_ratio: 0.0

training:
  tokenizer_epochs: 30
  basemodel_epochs: 20
  batch_size: 32
  log_interval: 50
    # num_workers: DataLoader subprocess count per GPU process.
  # hpo_tuner.py overrides this dynamically based on CPU count and number
  # of concurrent GPU workers. The value here is used for standalone
  # (non-HPO) training runs only.
  # Formula used by HPO: floor(cpu_count * 0.8 / n_gpu_workers), capped at 4.
  num_workers: 4

  # pin_memory: Pins DataLoader output tensors to page-locked (pinned) CPU
  # memory, enabling faster async CPU→GPU transfers via DMA. Always true
  # when training on GPU.
  pin_memory: true

  # persistent_workers: Keeps DataLoader worker processes alive between
  # epochs instead of restarting them. Eliminates the worker process
  # spawn overhead at the start of each epoch (~0.5-2s per epoch on large
  # datasets). Requires num_workers > 0.
  persistent_workers: true
  seed: 42

  tokenizer_learning_rate: 0.0002
  predictor_learning_rate: 0.000001

  adam_beta1: 0.9
  adam_beta2: 0.95
  adam_weight_decay: 0.1

  accumulation_steps: 1

  # Scheduler params (previously hardcoded)
  tokenizer_pct_start: 0.03
  tokenizer_div_factor: 10.0
  basemodel_pct_start: 0.03
  basemodel_div_factor: 10.0

  # Gradient clipping.
  # Both spellings are supported in config and HPO search space.
  # The training code reads tokenizer_grad_clip / basemodel_grad_clip.
  # config_loader.py aliases tokenizer_max_grad_norm → tokenizer_grad_clip
  # automatically, so either key works in YAML and in HPO search_space.
  tokenizer_max_grad_norm: 2.0   # alias → tokenizer_grad_clip
  basemodel_max_grad_norm: 3.0   # alias → basemodel_grad_clip
  tokenizer_grad_clip: 2.0
  basemodel_grad_clip: 3.0

  # Dropout overrides (optional — leave null to use pretrained values)
  ffn_dropout_p: null
  attn_dropout_p: null
  resid_dropout_p: null
  token_dropout_p: null

  # BSQ loss weight overrides (optional — leave null to use pretrained values)
  bsq_beta: null
  bsq_gamma0: null
  bsq_gamma: null
  bsq_zeta: null

model_paths:
  pretrained_tokenizer: "models/nos_tokenizer_2k"
  pretrained_predictor: "models/nos_mini"
  exp_name: "test_1h"
  base_path: "finetuned"
  base_save_path: ""
  finetuned_tokenizer: ""
  tokenizer_save_name: "tokenizer"
  basemodel_save_name: "basemodel"

experiment:
  name: "Nos_custom_finetune"
  description: "Custom finetune for 1h stock data"
  use_comet: false
  train_tokenizer: true
  train_basemodel: true
  skip_existing: false

device:
  use_cuda: true
  device_id: 0

# ── Hyperparameter Search Space ──────────────────────────────────
hpo:
  enabled: false                    # Set true to activate HPO
  n_trials: 30                      # Optuna trials
  direction: "minimize"             # minimize val_loss
  sampler: "tpe"                    # tpe | random | cmaes
  pruner: "median"                  # median | hyperband | none
  # SQLite URI with 60-second busy timeout.
  # The ?timeout=60 parameter prevents "database is locked" errors when
  # 8 GPU workers finish trials simultaneously. WAL journal mode is applied
  # programmatically by hpo_tuner.py via a SQLAlchemy connection hook,
  # which allows concurrent reads while one worker writes.
  #
  # For production clusters with >8 GPUs, switch to PostgreSQL:
  #   storage: "postgresql://user:password@localhost:5432/nos_hpo"
  #
  # For single-GPU development (in-memory, no persistence):
  #   storage: null

  # TASK 2.1: Local SQLite path to prevent OneDrive lock corruption
  # The ?timeout=60 parameter prevents "database is locked" errors when
  # 8 GPU workers finish trials simultaneously.
  storage: "sqlite:///C:/Temp/nos_hpo.db?timeout=60"
  study_name: "nos_finetune_hpo"

  # What to optimize
  optimize_tokenizer: true
  optimize_basemodel: true

  # Fast evaluation mode during HPO (reduced epochs)
  hpo_tokenizer_epochs: 5
  hpo_basemodel_epochs: 3

  search_space:
    # ── Tokenizer ──────────────────────────────────────────────
    tokenizer_learning_rate:
      type: float
      low: 1.0e-5
      high: 5.0e-3
      log: true

    # ── Predictor ──────────────────────────────────────────────
    predictor_learning_rate:
      type: float
      low: 1.0e-7
      high: 1.0e-4
      log: true

    # ── Shared Optimizer ───────────────────────────────────────
    adam_weight_decay:
      type: float
      low: 0.001
      high: 0.3
      log: true

    adam_beta1:
      type: float
      low: 0.85
      high: 0.95
      log: false

    adam_beta2:
      type: float
      low: 0.90
      high: 0.999
      log: false

    # ── Batch & Accumulation ───────────────────────────────────
    batch_size:
      type: categorical
      choices: [16, 32, 64, 128]

    accumulation_steps:
      type: categorical
      choices: [1, 2, 4]

    # ── Scheduler ─────────────────────────────────────────────
    tokenizer_pct_start:
      type: float
      low: 0.01
      high: 0.15
      log: false

    basemodel_pct_start:
      type: float
      low: 0.01
      high: 0.15
      log: false

    tokenizer_div_factor:
      type: categorical
      choices: [5.0, 10.0, 25.0, 50.0]

    basemodel_div_factor:
      type: categorical
      choices: [5.0, 10.0, 25.0, 50.0]

    # ── Gradient Clipping ─────────────────────────────────────
    tokenizer_max_grad_norm:
      type: float
      low: 0.5
      high: 5.0
      log: false

    basemodel_max_grad_norm:
      type: float
      low: 0.5
      high: 5.0
      log: false

    # ── Dropout (finetuning regularization) ────────────────────
    ffn_dropout_p:
      type: float
      low: 0.0
      high: 0.4
      log: false

    attn_dropout_p:
      type: float
      low: 0.0
      high: 0.2
      log: false

    resid_dropout_p:
      type: float
      low: 0.0
      high: 0.3
      log: false

    token_dropout_p:
      type: float
      low: 0.0
      high: 0.2
      log: false

    # ── BSQ Loss Weights (careful tuning) ─────────────────────
    bsq_beta:
      type: float
      low: 0.01
      high: 0.2
      log: true

    bsq_gamma0:
      type: float
      low: 0.5
      high: 2.0
      log: false

    bsq_gamma:
      type: float
      low: 0.8
      high: 2.0
      log: false

    bsq_zeta:
      type: float
      low: 0.01
      high: 0.2
      log: true

    # ── Data Params ────────────────────────────────────────────
    clip:
      type: float
      low: 3.0
      high: 10.0
      log: false
```

---

### `configs/config_mx110_hpo_fulltest.yaml`

```yaml
data:
  data_path: "data/1d.csv"
  lookback_window: 3
  predict_window: 2
  max_context: 5
  clip: 5.0
  train_ratio: 0.9
  val_ratio: 0.1
  test_ratio: 0.0

training:
  # Base epochs kept at 1 for the final "apply best" retraining phase
  tokenizer_epochs: 2
  basemodel_epochs: 2
  
  # CRITICAL for 2GB VRAM: keep base batch_size tiny
  batch_size: 512
  log_interval: 10

  num_workers: 2

  pin_memory: true
  persistent_workers: true
  seed: 42

  tokenizer_learning_rate: 0.0002
  predictor_learning_rate: 0.000001

  adam_beta1: 0.9
  adam_beta2: 0.95
  adam_weight_decay: 0.1

  accumulation_steps: 1

  tokenizer_pct_start: 0.03
  tokenizer_div_factor: 10.0
  basemodel_pct_start: 0.03
  basemodel_div_factor: 10.0

  tokenizer_max_grad_norm: 2.0   
  basemodel_max_grad_norm: 3.0   
  tokenizer_grad_clip: 2.0
  basemodel_grad_clip: 3.0

  ffn_dropout_p: null
  attn_dropout_p: null
  resid_dropout_p: null
  token_dropout_p: null

  bsq_beta: null
  bsq_gamma0: null
  bsq_gamma: null
  bsq_zeta: null

model_paths:
  pretrained_tokenizer: "models/nos_tokenizer_2k"
  pretrained_predictor: "models/nos_mini"
  exp_name: "smoke_test_full_hpo"
  base_path: "finetuned"
  base_save_path: ""
  finetuned_tokenizer: ""
  tokenizer_save_name: "tokenizer"
  basemodel_save_name: "basemodel"

experiment:
  name: "Nos_smoke_test_full"
  description: "2GB VRAM Full HPO Structure Test"
  use_comet: false
  train_tokenizer: true
  train_basemodel: true
  skip_existing: false

device:
  use_cuda: true
  device_id: 0

# ── Hyperparameter Search Space ──────────────────────────────────
hpo:
  enabled: true                    
  # 5 trials is enough for TPE to sample a wide variety of combinations
  # without taking all day on a local GPU.
  n_trials: 2                      
  direction: "minimize"            
  sampler: "tpe"                   
  pruner: "median"                 
  
  # TASK 2.1: Local SQLite path to prevent OneDrive lock corruption
  # The ?timeout=60 parameter prevents "database is locked" errors when
  # 8 GPU workers finish trials simultaneously.
  storage: "sqlite:///C:/Temp/nos_mx110_hpo_fulltest.db?timeout=60"
  study_name: "nos_finetune_hpo_full"

  optimize_tokenizer: true
  optimize_basemodel: true

  # 1 epoch per trial to burn through the 5 trials rapidly
  hpo_tokenizer_epochs: 1
  hpo_basemodel_epochs: 1

  search_space:
    # ── Tokenizer ──────────────────────────────────────────────
    tokenizer_learning_rate:
      type: float
      low: 1.0e-5
      high: 5.0e-3
      log: true

    # ── Predictor ──────────────────────────────────────────────
    predictor_learning_rate:
      type: float
      low: 1.0e-7
      high: 1.0e-4
      log: true

    # ── Shared Optimizer ───────────────────────────────────────
    adam_weight_decay:
      type: float
      low: 0.001
      high: 0.3
      log: true

    adam_beta1:
      type: float
      low: 0.85
      high: 0.95
      log: false

    adam_beta2:
      type: float
      low: 0.90
      high: 0.999
      log: false

    # ── Batch & Accumulation ───────────────────────────────────
    # CRITICAL VRAM GUARD: Limits batch to what a 2GB card can survive.
    batch_size:
      type: categorical
      choices: [1, 2, 4]

    # Accumulation doesn't cost extra VRAM, so we can test larger sizes.
    accumulation_steps:
      type: categorical
      choices: [1, 2, 4]

    # ── Scheduler ─────────────────────────────────────────────
    tokenizer_pct_start:
      type: float
      low: 0.01
      high: 0.15
      log: false

    basemodel_pct_start:
      type: float
      low: 0.01
      high: 0.15
      log: false

    tokenizer_div_factor:
      type: categorical
      choices: [5.0, 10.0, 25.0, 50.0]

    basemodel_div_factor:
      type: categorical
      choices: [5.0, 10.0, 25.0, 50.0]

    # ── Gradient Clipping ─────────────────────────────────────
    tokenizer_max_grad_norm:
      type: float
      low: 0.5
      high: 5.0
      log: false

    basemodel_max_grad_norm:
      type: float
      low: 0.5
      high: 5.0
      log: false

    # ── Dropout (finetuning regularization) ────────────────────
    ffn_dropout_p:
      type: float
      low: 0.0
      high: 0.4
      log: false

    attn_dropout_p:
      type: float
      low: 0.0
      high: 0.2
      log: false

    resid_dropout_p:
      type: float
      low: 0.0
      high: 0.3
      log: false

    token_dropout_p:
      type: float
      low: 0.0
      high: 0.2
      log: false

    # ── BSQ Loss Weights (careful tuning) ─────────────────────
    bsq_beta:
      type: float
      low: 0.01
      high: 0.2
      log: true

    bsq_gamma0:
      type: float
      low: 0.5
      high: 2.0
      log: false

    bsq_gamma:
      type: float
      low: 0.8
      high: 2.0
      log: false

    bsq_zeta:
      type: float
      low: 0.01
      high: 0.2
      log: true

    # ── Data Params ────────────────────────────────────────────
    clip:
      type: float
      low: 3.0
      high: 10.0
      log: false
```

---

### `configs/config_t40_hpo_fulltest.yaml`

```yaml
data:
  data_path: "data/1d.csv"
  lookback_window: 90
  predict_window: 14
  max_context: 90
  clip: 5.0
  train_ratio: 0.8
  val_ratio: 0.2
  test_ratio: 0.0

training:
  # Increased epochs because larger batch sizes mean fewer steps per epoch
  tokenizer_epochs: 50
  basemodel_epochs: 30
  
  # Base default pushed higher for T4 16GB VRAM
  batch_size: 128
  log_interval: 10

  # T4 has strong DMA, keeping 4 workers is optimal for data feeding
  num_workers: 4
  pin_memory: true
  persistent_workers: true
  seed: 42

  tokenizer_learning_rate: 0.0005
  predictor_learning_rate: 0.0001

  adam_beta1: 0.9
  adam_beta2: 0.95
  adam_weight_decay: 0.1

  accumulation_steps: 1

  tokenizer_pct_start: 0.05
  tokenizer_div_factor: 10.0
  basemodel_pct_start: 0.05
  basemodel_div_factor: 10.0

  tokenizer_max_grad_norm: 2.0   
  basemodel_max_grad_norm: 3.0   

model_paths:
  pretrained_tokenizer: "models/nos_tokenizer_2k"
  pretrained_predictor: "models/nos_mini"
  exp_name: "1d_t4_hpo"
  base_path: "finetuned"
  base_save_path: ""
  finetuned_tokenizer: ""
  tokenizer_save_name: "tokenizer"
  basemodel_save_name: "basemodel"

experiment:
  name: "Nos_1d_T4_HPO"
  description: "T4 16GB Optimized HPO for 5-Year Daily Data"
  use_comet: false
  train_tokenizer: true
  train_basemodel: true
  skip_existing: false

device:
  use_cuda: true
  device_id: 0

# ── Hyperparameter Search Space ──────────────────────────────────
hpo:
  enabled: true                    
  n_trials: 10                      
  direction: "minimize"            
  sampler: "tpe"                   
  pruner: "hyperband"

  # Local SQLite path to prevent OneDrive lock corruption
  storage: "sqlite:///C:/Temp/nos_t40_hpo.db?timeout=60"
  study_name: "nos_finetune_1d_hpo"

  optimize_tokenizer: true
  optimize_basemodel: true

  # 1D datasets pass quickly; 5 epochs gives Hyperband enough data to prune accurately
  hpo_tokenizer_epochs: 5
  hpo_basemodel_epochs: 5

  search_space:
    # ── Tokenizer ──────────────────────────────────────────────
    tokenizer_learning_rate:
      type: float
      low: 5.0e-5    # Shifted up to accommodate larger batch sizes
      high: 5.0e-3
      log: true

    # ── Predictor ──────────────────────────────────────────────
    predictor_learning_rate:
      type: float
      low: 5.0e-6    # Shifted up to accommodate larger batch sizes
      high: 5.0e-4
      log: true

    # ── Shared Optimizer ───────────────────────────────────────
    adam_weight_decay:
      type: float
      low: 0.01      # Raised floor to force stronger regularization on small dataset
      high: 0.3
      log: true

    # ── Batch & Accumulation ───────────────────────────────────
    batch_size:
      type: categorical
      # T4 can easily handle 256 on nos_mini. 
      # 256 is the absolute ceiling for a 1,800 row dataset (~7 steps/epoch)
      choices: [64, 128, 256]

    accumulation_steps:
      type: categorical
      # Keep at 1 or 2 to avoid reducing steps-per-epoch too severely
      choices: [1, 2]

    # ── Scheduler ─────────────────────────────────────────────
    tokenizer_pct_start:
      type: float
      low: 0.05
      high: 0.20
      log: false

    basemodel_pct_start:
      type: float
      low: 0.05
      high: 0.20
      log: false

    tokenizer_div_factor:
      type: categorical
      choices: [10.0, 25.0, 50.0]

    basemodel_div_factor:
      type: categorical
      choices: [10.0, 25.0, 50.0]

    # ── Dropout (Crucial for smaller 5-yr datasets) ────────────
    ffn_dropout_p:
      type: float
      low: 0.1       # Force at least 10% dropout to prevent memorization
      high: 0.4
      log: false

    resid_dropout_p:
      type: float
      low: 0.1       # Force at least 10% dropout to prevent memorization
      high: 0.3
      log: false

    # ── BSQ Loss Weights ──────────────────────────────────────
    bsq_beta:
      type: float
      low: 0.01
      high: 0.2
      log: true

    bsq_gamma:       # ADDED: Crucial for preventing codebook collapse in T4 high-batch mode
      type: float
      low: 0.8
      high: 2.0
      log: false

    bsq_zeta:
      type: float
      low: 0.01
      high: 0.2
      log: true
```

---

### `configs/config_t40_hpo_winner.yaml`

```yaml
data:
  data_path: data/1d.csv
  lookback_window: 90
  predict_window: 14
  max_context: 90
  clip: 5.0
  train_ratio: 0.8
  val_ratio: 0.2
  test_ratio: 0.0
training:
  tokenizer_epochs: 50
  basemodel_epochs: 30
  batch_size: 128
  log_interval: 10
  num_workers: 4
  pin_memory: true
  persistent_workers: true
  seed: 42
  tokenizer_learning_rate: 0.0021627162139822797
  predictor_learning_rate: 0.0003784146030083808
  adam_beta1: 0.9
  adam_beta2: 0.95
  adam_weight_decay: 0.20978213384576816
  accumulation_steps: 1
  tokenizer_pct_start: 0.10064227571054421
  tokenizer_div_factor: 10.0
  basemodel_pct_start: 0.09879954961448967
  basemodel_div_factor: 50.0
  tokenizer_max_grad_norm: 2.0
  basemodel_max_grad_norm: 3.0
  bsq_beta: 0.08215779086695435
  bsq_gamma: 1.2363555228551528
  bsq_zeta: 0.1837882061421475
  ffn_dropout_p: 0.2070259980080768
  resid_dropout_p: 0.15618690193747614
model_paths:
  pretrained_tokenizer: models/nos_tokenizer_2k
  pretrained_predictor: models/nos_mini
  exp_name: 1d_t4_hpo
  base_path: finetuned
  base_save_path: ''
  finetuned_tokenizer: ''
  tokenizer_save_name: tokenizer
  basemodel_save_name: basemodel
experiment:
  name: Nos_1d_T4_HPO
  description: T4 16GB Optimized HPO for 5-Year Daily Data
  use_comet: false
  train_tokenizer: true
  train_basemodel: true
  skip_existing: false
  hpo_applied: true
  hpo_applied_timestamp: '2026-05-17T07:11:32.136801'
device:
  use_cuda: true
  device_id: 0
hpo:
  enabled: true
  n_trials: 10
  direction: minimize
  sampler: tpe
  pruner: hyperband
  storage: sqlite:///C:/Temp/nos_t40_hpo.db?timeout=60
  study_name: nos_finetune_1d_hpo
  optimize_tokenizer: true
  optimize_basemodel: true
  hpo_tokenizer_epochs: 5
  hpo_basemodel_epochs: 5
  search_space:
    tokenizer_learning_rate:
      type: float
      low: 5.0e-05
      high: 0.005
      log: true
    predictor_learning_rate:
      type: float
      low: 5.0e-06
      high: 0.0005
      log: true
    adam_weight_decay:
      type: float
      low: 0.01
      high: 0.3
      log: true
    batch_size:
      type: categorical
      choices:
      - 64
      - 128
      - 256
    accumulation_steps:
      type: categorical
      choices:
      - 1
      - 2
    tokenizer_pct_start:
      type: float
      low: 0.05
      high: 0.2
      log: false
    basemodel_pct_start:
      type: float
      low: 0.05
      high: 0.2
      log: false
    tokenizer_div_factor:
      type: categorical
      choices:
      - 10.0
      - 25.0
      - 50.0
    basemodel_div_factor:
      type: categorical
      choices:
      - 10.0
      - 25.0
      - 50.0
    ffn_dropout_p:
      type: float
      low: 0.1
      high: 0.4
      log: false
    resid_dropout_p:
      type: float
      low: 0.1
      high: 0.3
      log: false
    bsq_beta:
      type: float
      low: 0.01
      high: 0.2
      log: true
    bsq_gamma:
      type: float
      low: 0.8
      high: 2.0
      log: false
    bsq_zeta:
      type: float
      low: 0.01
      high: 0.2
      log: true

```

---

### `data/1d.csv`

```text
timestamps,open,high,low,close,volume,amount
2020-08-11,2.85,3.5208,2.8433,3.2985,1552384.78,4941872.660548333
2020-08-12,3.2985,3.9289,3.08,3.7558,1737042.95,6176664.409993667
2020-08-13,3.75,4.1387,3.5003,3.73,1685759.24,6446323.368316334
2020-08-14,3.7207,3.7676,3.321,3.4099,1474161.79,5210030.644981333
2020-08-15,3.4181,3.74,3.15,3.173,1070233.2,3656268.640716667

# ⚠️  Preview — showing 5 of 1999 data rows (1994 rows hidden).
```

---

### `finetuned/1d_t4_hpo/basemodel/best_model/config.json`

```json
{
  "attn_dropout_p": 0.0,
  "d_model": 256,
  "ff_dim": 512,
  "ffn_dropout_p": 0.2,
  "learn_te": true,
  "n_heads": 4,
  "n_layers": 4,
  "resid_dropout_p": 0.2,
  "s1_bits": 10,
  "s2_bits": 10,
  "token_dropout_p": 0.0
}
```

---

### `finetuned/1d_t4_hpo/basemodel/best_model/model.safetensors`

```text
[File too large to display: 15.7 MB]
```

---

### `finetuned/1d_t4_hpo/basemodel/best_model/README.md`

```markdown
---
tags:
- model_hub_mixin
- pytorch_model_hub_mixin
---

This model has been pushed to the Hub using the [PytorchModelHubMixin](https://huggingface.co/docs/huggingface_hub/package_reference/mixins#huggingface_hub.PyTorchModelHubMixin) integration:
- Code: [More Information Needed]
- Paper: [More Information Needed]
- Docs: [More Information Needed]
```

---

### `finetuned/1d_t4_hpo/hpo_logs/worker_pid18131_cuda0.log`

```text
2026-05-17 07:03:15 [INFO    ] [__main__] NosHPOTuner initialised | config=configs/config_t40_hpo_fulltest.yaml | device=cuda:0 | worker_tag=pid18131_cuda0
2026-05-17 07:03:15 [INFO    ] [__main__] DataLoader num_workers: 1 (CPU cores=2, GPU workers=1)
2026-05-17 07:03:15 [INFO    ] [__main__] Starting tokenizer HPO: 10 trials | device=cuda:0
2026-05-17 07:03:15 [INFO    ] [__main__] Creating/loading study 'nos_finetune_1d_hpo_tokenizer' | storage=sqlite:///C:/Temp/nos_t40_hpo.db?timeout=60
2026-05-17 07:03:16 [INFO    ] [__main__] SQLite WAL mode, NullPool, and busy_timeout=300s applied via connection hook.
2026-05-17 07:03:16 [INFO    ] [__main__] Study ready: 0 existing trials loaded.
2026-05-17 07:03:16 [INFO    ] [__main__.TokenizerObjective.pid18131] Trial 0 | Sampled overrides: {'tokenizer_learning_rate': 0.0002805758207667253, 'adam_weight_decay': 0.2536999076681772, 'batch_size': 64, 'accumulation_steps': 1, 'tokenizer_pct_start': 0.1799264218662403, 'tokenizer_div_factor': 25.0, 'bsq_beta': 0.18276027831785724, 'bsq_gamma': 1.798931168960506, 'bsq_zeta': 0.018891200276189388, 'tokenizer_epochs': 5, 'log_interval': 999999, 'num_workers': 1}
2026-05-17 07:03:16 [DEBUG   ] [filelock] Attempting to acquire lock 139497524456208 on data/1d.csv.train.lock
2026-05-17 07:03:16 [DEBUG   ] [filelock] Lock 139497524456208 acquired on data/1d.csv.train.lock
2026-05-17 07:03:16 [DEBUG   ] [filelock] Attempting to release lock 139497524456208 on data/1d.csv.train.lock
2026-05-17 07:03:16 [DEBUG   ] [filelock] Lock 139497524456208 released on data/1d.csv.train.lock
2026-05-17 07:03:16 [DEBUG   ] [filelock] Attempting to acquire lock 139497524493008 on data/1d.csv.val.lock
2026-05-17 07:03:16 [DEBUG   ] [filelock] Lock 139497524493008 acquired on data/1d.csv.val.lock
2026-05-17 07:03:16 [DEBUG   ] [filelock] Attempting to release lock 139497524493008 on data/1d.csv.val.lock
2026-05-17 07:03:16 [DEBUG   ] [filelock] Lock 139497524493008 released on data/1d.csv.val.lock
2026-05-17 07:03:38 [INFO    ] [__main__.TokenizerObjective.pid18131] Trial 0 COMPLETE | val_loss=0.055199 | overrides={'tokenizer_learning_rate': 0.0002805758207667253, 'adam_weight_decay': 0.2536999076681772, 'batch_size': 64, 'accumulation_steps': 1, 'tokenizer_pct_start': 0.1799264218662403, 'tokenizer_div_factor': 25.0, 'bsq_beta': 0.18276027831785724, 'bsq_gamma': 1.798931168960506, 'bsq_zeta': 0.018891200276189388, 'tokenizer_epochs': 5, 'log_interval': 999999, 'num_workers': 1}
2026-05-17 07:03:39 [INFO    ] [__main__.TokenizerObjective.pid18131] Trial 1 | Sampled overrides: {'tokenizer_learning_rate': 0.00011551009439226474, 'adam_weight_decay': 0.018659959624904916, 'batch_size': 128, 'accumulation_steps': 2, 'tokenizer_pct_start': 0.07092407909780628, 'tokenizer_div_factor': 50.0, 'bsq_beta': 0.10508421338691762, 'bsq_gamma': 1.0396085385900318, 'bsq_zeta': 0.04666963767236924, 'tokenizer_epochs': 5, 'log_interval': 999999, 'num_workers': 1}
2026-05-17 07:03:53 [INFO    ] [__main__.TokenizerObjective.pid18131] Trial 1 COMPLETE | val_loss=0.100841 | overrides={'tokenizer_learning_rate': 0.00011551009439226474, 'adam_weight_decay': 0.018659959624904916, 'batch_size': 128, 'accumulation_steps': 2, 'tokenizer_pct_start': 0.07092407909780628, 'tokenizer_div_factor': 50.0, 'bsq_beta': 0.10508421338691762, 'bsq_gamma': 1.0396085385900318, 'bsq_zeta': 0.04666963767236924, 'tokenizer_epochs': 5, 'log_interval': 999999, 'num_workers': 1}
2026-05-17 07:03:54 [INFO    ] [__main__.TokenizerObjective.pid18131] Trial 2 | Sampled overrides: {'tokenizer_learning_rate': 0.000765242606091573, 'adam_weight_decay': 0.011711509955524094, 'batch_size': 64, 'accumulation_steps': 2, 'tokenizer_pct_start': 0.1712596022174692, 'tokenizer_div_factor': 50.0, 'bsq_beta': 0.03738105868191797, 'bsq_gamma': 0.9464458818137347, 'bsq_zeta': 0.04407984038169244, 'tokenizer_epochs': 5, 'log_interval': 999999, 'num_workers': 1}
2026-05-17 07:04:10 [INFO    ] [__main__.TokenizerObjective.pid18131] Trial 2 COMPLETE | val_loss=0.049402 | overrides={'tokenizer_learning_rate': 0.000765242606091573, 'adam_weight_decay': 0.011711509955524094, 'batch_size': 64, 'accumulation_steps': 2, 'tokenizer_pct_start': 0.1712596022174692, 'tokenizer_div_factor': 50.0, 'bsq_beta': 0.03738105868191797, 'bsq_gamma': 0.9464458818137347, 'bsq_zeta': 0.04407984038169244, 'tokenizer_epochs': 5, 'log_interval': 999999, 'num_workers': 1}
2026-05-17 07:04:10 [INFO    ] [__main__.TokenizerObjective.pid18131] Trial 3 | Sampled overrides: {'tokenizer_learning_rate': 5.8579686961535335e-05, 'adam_weight_decay': 0.22038218939289875, 'batch_size': 128, 'accumulation_steps': 2, 'tokenizer_pct_start': 0.07772816832882906, 'tokenizer_div_factor': 10.0, 'bsq_beta': 0.1459476894296684, 'bsq_gamma': 1.5174799745733023, 'bsq_zeta': 0.15826541904647565, 'tokenizer_epochs': 5, 'log_interval': 999999, 'num_workers': 1}
2026-05-17 07:04:23 [INFO    ] [__main__.TokenizerObjective.pid18131] Trial 4 | Sampled overrides: {'tokenizer_learning_rate': 7.515450322528411e-05, 'adam_weight_decay': 0.01947558230629543, 'batch_size': 256, 'accumulation_steps': 2, 'tokenizer_pct_start': 0.1035129990040384, 'tokenizer_div_factor': 25.0, 'bsq_beta': 0.11058146376563001, 'bsq_gamma': 0.8894607724157251, 'bsq_zeta': 0.19229567074543377, 'tokenizer_epochs': 5, 'log_interval': 999999, 'num_workers': 1}
2026-05-17 07:04:44 [INFO    ] [__main__.TokenizerObjective.pid18131] Trial 5 | Sampled overrides: {'tokenizer_learning_rate': 0.0017516992455793442, 'adam_weight_decay': 0.019657448966046126, 'batch_size': 128, 'accumulation_steps': 2, 'tokenizer_pct_start': 0.06110669776011356, 'tokenizer_div_factor': 50.0, 'bsq_beta': 0.06470376604234768, 'bsq_gamma': 1.1970776298231791, 'bsq_zeta': 0.012097379927033842, 'tokenizer_epochs': 5, 'log_interval': 999999, 'num_workers': 1}
2026-05-17 07:05:00 [INFO    ] [__main__.TokenizerObjective.pid18131] Trial 5 COMPLETE | val_loss=0.053635 | overrides={'tokenizer_learning_rate': 0.0017516992455793442, 'adam_weight_decay': 0.019657448966046126, 'batch_size': 128, 'accumulation_steps': 2, 'tokenizer_pct_start': 0.06110669776011356, 'tokenizer_div_factor': 50.0, 'bsq_beta': 0.06470376604234768, 'bsq_gamma': 1.1970776298231791, 'bsq_zeta': 0.012097379927033842, 'tokenizer_epochs': 5, 'log_interval': 999999, 'num_workers': 1}
2026-05-17 07:05:11 [INFO    ] [__main__.TokenizerObjective.pid18131] Trial 6 | Sampled overrides: {'tokenizer_learning_rate': 0.00020937973594503152, 'adam_weight_decay': 0.030222834756467344, 'batch_size': 256, 'accumulation_steps': 1, 'tokenizer_pct_start': 0.15698671808344927, 'tokenizer_div_factor': 50.0, 'bsq_beta': 0.043897812845882046, 'bsq_gamma': 1.4272793952583929, 'bsq_zeta': 0.035995125264172326, 'tokenizer_epochs': 5, 'log_interval': 999999, 'num_workers': 1}
2026-05-17 07:05:17 [INFO    ] [__main__.TokenizerObjective.pid18131] Trial 7 | Sampled overrides: {'tokenizer_learning_rate': 5.6209310478965295e-05, 'adam_weight_decay': 0.01443340240633889, 'batch_size': 128, 'accumulation_steps': 2, 'tokenizer_pct_start': 0.08739383437233125, 'tokenizer_div_factor': 25.0, 'bsq_beta': 0.012593695017526326, 'bsq_gamma': 1.1477017434965218, 'bsq_zeta': 0.01620890700720353, 'tokenizer_epochs': 5, 'log_interval': 999999, 'num_workers': 1}
2026-05-17 07:05:41 [INFO    ] [__main__.TokenizerObjective.pid18131] Trial 8 | Sampled overrides: {'tokenizer_learning_rate': 0.003617139922832709, 'adam_weight_decay': 0.156203869845265, 'batch_size': 128, 'accumulation_steps': 2, 'tokenizer_pct_start': 0.13090133628734762, 'tokenizer_div_factor': 25.0, 'bsq_beta': 0.013905315749737282, 'bsq_gamma': 1.07352219505033, 'bsq_zeta': 0.03594843964212239, 'tokenizer_epochs': 5, 'log_interval': 999999, 'num_workers': 1}
2026-05-17 07:05:58 [INFO    ] [__main__.TokenizerObjective.pid18131] Trial 9 | Sampled overrides: {'tokenizer_learning_rate': 0.0021627162139822797, 'adam_weight_decay': 0.18681142751959703, 'batch_size': 128, 'accumulation_steps': 1, 'tokenizer_pct_start': 0.10064227571054421, 'tokenizer_div_factor': 10.0, 'bsq_beta': 0.08215779086695435, 'bsq_gamma': 1.2363555228551528, 'bsq_zeta': 0.1837882061421475, 'tokenizer_epochs': 5, 'log_interval': 999999, 'num_workers': 1}
2026-05-17 07:06:15 [INFO    ] [__main__.TokenizerObjective.pid18131] Trial 9 COMPLETE | val_loss=0.045195 | overrides={'tokenizer_learning_rate': 0.0021627162139822797, 'adam_weight_decay': 0.18681142751959703, 'batch_size': 128, 'accumulation_steps': 1, 'tokenizer_pct_start': 0.10064227571054421, 'tokenizer_div_factor': 10.0, 'bsq_beta': 0.08215779086695435, 'bsq_gamma': 1.2363555228551528, 'bsq_zeta': 0.1837882061421475, 'tokenizer_epochs': 5, 'log_interval': 999999, 'num_workers': 1}
2026-05-17 07:06:26 [INFO    ] [__main__] Tokenizer HPO complete | best_val_loss=0.045195 | best_params={'tokenizer_learning_rate': 0.0021627162139822797, 'adam_weight_decay': 0.18681142751959703, 'batch_size': 128, 'accumulation_steps': 1, 'tokenizer_pct_start': 0.10064227571054421, 'tokenizer_div_factor': 10.0, 'bsq_beta': 0.08215779086695435, 'bsq_gamma': 1.2363555228551528, 'bsq_zeta': 0.1837882061421475}
2026-05-17 07:06:26 [INFO    ] [__main__] Results atomically written to: finetuned/1d_t4_hpo/hpo_results/tokenizer_trials.json
2026-05-17 07:06:26 [DEBUG   ] [matplotlib] matplotlib data path: /usr/local/lib/python3.12/dist-packages/matplotlib/mpl-data
2026-05-17 07:06:26 [DEBUG   ] [matplotlib] CONFIGDIR=/root/.config/matplotlib
2026-05-17 07:06:26 [DEBUG   ] [matplotlib] interactive is False
2026-05-17 07:06:26 [DEBUG   ] [matplotlib] platform is linux
2026-05-17 07:06:26 [DEBUG   ] [matplotlib] CACHEDIR=/root/.cache/matplotlib
2026-05-17 07:06:26 [DEBUG   ] [matplotlib.font_manager] Using fontManager instance from /root/.cache/matplotlib/fontlist-v390.json
2026-05-17 07:06:28 [INFO    ] [__main__] Visualisation plots saved to: finetuned/1d_t4_hpo/hpo_results
2026-05-17 07:06:28 [INFO    ] [__main__] Training full tokenizer with best HPO params before basemodel phase.
2026-05-17 07:09:16 [INFO    ] [__main__] Best tokenizer training complete. Saved to: finetuned/1d_t4_hpo/tokenizer
2026-05-17 07:09:16 [INFO    ] [__main__] Starting basemodel HPO: 10 trials | device=cuda:0 | tokenizer=finetuned/1d_t4_hpo/tokenizer/best_model
2026-05-17 07:09:16 [INFO    ] [__main__] Creating/loading study 'nos_finetune_1d_hpo_basemodel' | storage=sqlite:///C:/Temp/nos_t40_hpo.db?timeout=60
2026-05-17 07:09:16 [INFO    ] [__main__] SQLite WAL mode, NullPool, and busy_timeout=300s applied via connection hook.
2026-05-17 07:09:16 [INFO    ] [__main__] Study ready: 0 existing trials loaded.
2026-05-17 07:09:17 [INFO    ] [__main__.BasemodelObjective.pid18131] Trial 0 | Sampled overrides: {'predictor_learning_rate': 2.8057582076672495e-05, 'adam_weight_decay': 0.2536999076681772, 'batch_size': 64, 'accumulation_steps': 1, 'basemodel_pct_start': 0.1799264218662403, 'basemodel_div_factor': 25.0, 'ffn_dropout_p': 0.39097295564859835, 'resid_dropout_p': 0.26648852816008434, 'basemodel_epochs': 5, 'log_interval': 999999, 'num_workers': 1}
2026-05-17 07:09:30 [INFO    ] [__main__.BasemodelObjective.pid18131] Trial 0 COMPLETE | val_loss=4.679457 | overrides={'predictor_learning_rate': 2.8057582076672495e-05, 'adam_weight_decay': 0.2536999076681772, 'batch_size': 64, 'accumulation_steps': 1, 'basemodel_pct_start': 0.1799264218662403, 'basemodel_div_factor': 25.0, 'ffn_dropout_p': 0.39097295564859835, 'resid_dropout_p': 0.26648852816008434, 'basemodel_epochs': 5, 'log_interval': 999999, 'num_workers': 1}
2026-05-17 07:09:31 [INFO    ] [__main__.BasemodelObjective.pid18131] Trial 1 | Sampled overrides: {'predictor_learning_rate': 1.3293771991636346e-05, 'adam_weight_decay': 0.01855998084649059, 'batch_size': 256, 'accumulation_steps': 1, 'basemodel_pct_start': 0.14177793420835694, 'basemodel_div_factor': 50.0, 'ffn_dropout_p': 0.2368209952651108, 'resid_dropout_p': 0.2570351922786027, 'basemodel_epochs': 5, 'log_interval': 999999, 'num_workers': 1}
2026-05-17 07:09:43 [INFO    ] [__main__.BasemodelObjective.pid18131] Trial 1 COMPLETE | val_loss=4.761098 | overrides={'predictor_learning_rate': 1.3293771991636346e-05, 'adam_weight_decay': 0.01855998084649059, 'batch_size': 256, 'accumulation_steps': 1, 'basemodel_pct_start': 0.14177793420835694, 'basemodel_div_factor': 50.0, 'ffn_dropout_p': 0.2368209952651108, 'resid_dropout_p': 0.2570351922786027, 'basemodel_epochs': 5, 'log_interval': 999999, 'num_workers': 1}
2026-05-17 07:09:43 [INFO    ] [__main__.BasemodelObjective.pid18131] Trial 2 | Sampled overrides: {'predictor_learning_rate': 1.2540578430226153e-05, 'adam_weight_decay': 0.05748924681991978, 'batch_size': 256, 'accumulation_steps': 1, 'basemodel_pct_start': 0.192332830588, 'basemodel_div_factor': 10.0, 'ffn_dropout_p': 0.12930163420191518, 'resid_dropout_p': 0.23684660530243137, 'basemodel_epochs': 5, 'log_interval': 999999, 'num_workers': 1}
2026-05-17 07:09:55 [INFO    ] [__main__.BasemodelObjective.pid18131] Trial 2 COMPLETE | val_loss=4.760827 | overrides={'predictor_learning_rate': 1.2540578430226153e-05, 'adam_weight_decay': 0.05748924681991978, 'batch_size': 256, 'accumulation_steps': 1, 'basemodel_pct_start': 0.192332830588, 'basemodel_div_factor': 10.0, 'ffn_dropout_p': 0.12930163420191518, 'resid_dropout_p': 0.23684660530243137, 'basemodel_epochs': 5, 'log_interval': 999999, 'num_workers': 1}
2026-05-17 07:09:56 [INFO    ] [__main__.BasemodelObjective.pid18131] Trial 3 | Sampled overrides: {'predictor_learning_rate': 3.7955524026413494e-05, 'adam_weight_decay': 0.015144860262751412, 'batch_size': 256, 'accumulation_steps': 2, 'basemodel_pct_start': 0.09675666141341166, 'basemodel_div_factor': 25.0, 'ffn_dropout_p': 0.39087538832936763, 'resid_dropout_p': 0.2550265646722229, 'basemodel_epochs': 5, 'log_interval': 999999, 'num_workers': 1}
2026-05-17 07:10:08 [INFO    ] [__main__.BasemodelObjective.pid18131] Trial 3 COMPLETE | val_loss=4.752116 | overrides={'predictor_learning_rate': 3.7955524026413494e-05, 'adam_weight_decay': 0.015144860262751412, 'batch_size': 256, 'accumulation_steps': 2, 'basemodel_pct_start': 0.09675666141341166, 'basemodel_div_factor': 25.0, 'ffn_dropout_p': 0.39087538832936763, 'resid_dropout_p': 0.2550265646722229, 'basemodel_epochs': 5, 'log_interval': 999999, 'num_workers': 1}
2026-05-17 07:10:08 [INFO    ] [__main__.BasemodelObjective.pid18131] Trial 4 | Sampled overrides: {'predictor_learning_rate': 0.0003784146030083808, 'adam_weight_decay': 0.20978213384576816, 'batch_size': 128, 'accumulation_steps': 1, 'basemodel_pct_start': 0.09879954961448967, 'basemodel_div_factor': 50.0, 'ffn_dropout_p': 0.2070259980080768, 'resid_dropout_p': 0.15618690193747614, 'basemodel_epochs': 5, 'log_interval': 999999, 'num_workers': 1}
2026-05-17 07:10:21 [INFO    ] [__main__.BasemodelObjective.pid18131] Trial 4 COMPLETE | val_loss=4.388323 | overrides={'predictor_learning_rate': 0.0003784146030083808, 'adam_weight_decay': 0.20978213384576816, 'batch_size': 128, 'accumulation_steps': 1, 'basemodel_pct_start': 0.09879954961448967, 'basemodel_div_factor': 50.0, 'ffn_dropout_p': 0.2070259980080768, 'resid_dropout_p': 0.15618690193747614, 'basemodel_epochs': 5, 'log_interval': 999999, 'num_workers': 1}
2026-05-17 07:10:21 [INFO    ] [__main__.BasemodelObjective.pid18131] Trial 5 | Sampled overrides: {'predictor_learning_rate': 6.086423540561219e-05, 'adam_weight_decay': 0.016149614799999188, 'batch_size': 256, 'accumulation_steps': 1, 'basemodel_pct_start': 0.050828317568540365, 'basemodel_div_factor': 10.0, 'ffn_dropout_p': 0.3313811040057838, 'resid_dropout_p': 0.11480893034681808, 'basemodel_epochs': 5, 'log_interval': 999999, 'num_workers': 1}
2026-05-17 07:10:33 [INFO    ] [__main__.BasemodelObjective.pid18131] Trial 5 COMPLETE | val_loss=4.666116 | overrides={'predictor_learning_rate': 6.086423540561219e-05, 'adam_weight_decay': 0.016149614799999188, 'batch_size': 256, 'accumulation_steps': 1, 'basemodel_pct_start': 0.050828317568540365, 'basemodel_div_factor': 10.0, 'ffn_dropout_p': 0.3313811040057838, 'resid_dropout_p': 0.11480893034681808, 'basemodel_epochs': 5, 'log_interval': 999999, 'num_workers': 1}
2026-05-17 07:10:34 [INFO    ] [__main__.BasemodelObjective.pid18131] Trial 6 | Sampled overrides: {'predictor_learning_rate': 2.6055622978941292e-05, 'adam_weight_decay': 0.014830392684568025, 'batch_size': 64, 'accumulation_steps': 2, 'basemodel_pct_start': 0.09877749830401206, 'basemodel_div_factor': 50.0, 'ffn_dropout_p': 0.2416644775485848, 'resid_dropout_p': 0.12391884918766034, 'basemodel_epochs': 5, 'log_interval': 999999, 'num_workers': 1}
2026-05-17 07:10:40 [INFO    ] [__main__.BasemodelObjective.pid18131] Trial 7 | Sampled overrides: {'predictor_learning_rate': 0.0001334933337137229, 'adam_weight_decay': 0.13297554090738672, 'batch_size': 128, 'accumulation_steps': 1, 'basemodel_pct_start': 0.05381286901161428, 'basemodel_div_factor': 50.0, 'ffn_dropout_p': 0.19430679432289802, 'resid_dropout_p': 0.20171413823294054, 'basemodel_epochs': 5, 'log_interval': 999999, 'num_workers': 1}
2026-05-17 07:10:56 [INFO    ] [__main__.BasemodelObjective.pid18131] Trial 8 | Sampled overrides: {'predictor_learning_rate': 0.00032666526101138684, 'adam_weight_decay': 0.02334720250903525, 'batch_size': 128, 'accumulation_steps': 2, 'basemodel_pct_start': 0.07418319308810067, 'basemodel_div_factor': 10.0, 'ffn_dropout_p': 0.3614381770563153, 'resid_dropout_p': 0.2607344153798229, 'basemodel_epochs': 5, 'log_interval': 999999, 'num_workers': 1}
2026-05-17 07:11:11 [INFO    ] [__main__.BasemodelObjective.pid18131] Trial 9 | Sampled overrides: {'predictor_learning_rate': 1.1806199622206298e-05, 'adam_weight_decay': 0.20816986844858934, 'batch_size': 256, 'accumulation_steps': 1, 'basemodel_pct_start': 0.08419027438129126, 'basemodel_div_factor': 50.0, 'ffn_dropout_p': 0.10208563915935721, 'resid_dropout_p': 0.20214946051551314, 'basemodel_epochs': 5, 'log_interval': 999999, 'num_workers': 1}
2026-05-17 07:11:31 [INFO    ] [__main__] Basemodel HPO complete | best_val_loss=4.388323 | best_params={'predictor_learning_rate': 0.0003784146030083808, 'adam_weight_decay': 0.20978213384576816, 'batch_size': 128, 'accumulation_steps': 1, 'basemodel_pct_start': 0.09879954961448967, 'basemodel_div_factor': 50.0, 'ffn_dropout_p': 0.2070259980080768, 'resid_dropout_p': 0.15618690193747614}
2026-05-17 07:11:31 [INFO    ] [__main__] Results atomically written to: finetuned/1d_t4_hpo/hpo_results/basemodel_trials.json
2026-05-17 07:11:32 [INFO    ] [__main__] Visualisation plots saved to: finetuned/1d_t4_hpo/hpo_results
2026-05-17 07:11:32 [INFO    ] [__main__] Best config written to: configs/config_t40_hpo_winner.yaml

```

---

### `finetuned/1d_t4_hpo/logs/basemodel_training_rank_0.log`

```text
2026-05-17 07:14:42 - basemodel_training_rank_0 - INFO - === Basemodel Training Started ===
2026-05-17 07:14:42 - basemodel_training_rank_0 - INFO - Experiment: 1d_t4_hpo
2026-05-17 07:14:42 - basemodel_training_rank_0 - INFO - Log dir:    finetuned/1d_t4_hpo/logs
2026-05-17 07:14:42 - basemodel_training_rank_0 - INFO - Rank:       0
2026-05-17 07:14:42 - basemodel_training_rank_0 - INFO - Timestamp:  2026-05-17 07:14:42
2026-05-17 07:14:42 - basemodel_training_rank_0 - INFO - Loading fine-tuned tokenizer...
2026-05-17 07:14:42 - basemodel_training_rank_0 - INFO - Loading pretrained predictor...
2026-05-17 07:14:42 - basemodel_training_rank_0 - INFO - Model parameters: 4,108,032
2026-05-17 07:14:42 - basemodel_training_rank_0 - INFO - === Training Configuration ===
2026-05-17 07:14:42 - basemodel_training_rank_0 - INFO - Data path: data/1d.csv
2026-05-17 07:14:42 - basemodel_training_rank_0 - INFO - Lookback window: 90
2026-05-17 07:14:42 - basemodel_training_rank_0 - INFO - Predict window: 14
2026-05-17 07:14:42 - basemodel_training_rank_0 - INFO - Batch size: 128
2026-05-17 07:14:42 - basemodel_training_rank_0 - INFO - Learning rate: 0.0003784146030083808
2026-05-17 07:14:42 - basemodel_training_rank_0 - INFO - Training epochs: 30
2026-05-17 07:14:42 - basemodel_training_rank_0 - INFO - Device: cuda:0
2026-05-17 07:14:42 - basemodel_training_rank_0 - INFO - Tokenizer path: finetuned/1d_t4_hpo/tokenizer/best_model
2026-05-17 07:14:42 - basemodel_training_rank_0 - INFO - Pretrained model path: models/nos_mini
2026-05-17 07:14:42 - basemodel_training_rank_0 - INFO - Starting fine-tuning training...
2026-05-17 07:14:42 - basemodel_training_rank_0 - INFO - Starting base-model training …
2026-05-17 07:14:44 - basemodel_training_rank_0 - INFO - [Epoch 1/30, Step 10/11] LR: 0.000092  Loss: 4.8909
2026-05-17 07:14:44 - basemodel_training_rank_0 - INFO - 
--- Epoch 1/30 Summary ---
  Train Loss : 4.963334
  Val   Loss : 4.645067
  Epoch Time : 0:00:02

2026-05-17 07:14:44 - basemodel_training_rank_0 - INFO - ✓ Best model saved → finetuned/1d_t4_hpo/basemodel/best_model  (val loss: 4.645067)
2026-05-17 07:14:46 - basemodel_training_rank_0 - INFO - [Epoch 2/30, Step 20/11] LR: 0.000268  Loss: 4.5775
2026-05-17 07:14:47 - basemodel_training_rank_0 - INFO - 
--- Epoch 2/30 Summary ---
  Train Loss : 4.693393
  Val   Loss : 4.520689
  Epoch Time : 0:00:02

2026-05-17 07:14:47 - basemodel_training_rank_0 - INFO - ✓ Best model saved → finetuned/1d_t4_hpo/basemodel/best_model  (val loss: 4.520689)
2026-05-17 07:14:49 - basemodel_training_rank_0 - INFO - [Epoch 3/30, Step 30/11] LR: 0.000376  Loss: 4.1889
2026-05-17 07:14:49 - basemodel_training_rank_0 - INFO - 
--- Epoch 3/30 Summary ---
  Train Loss : 4.252041
  Val   Loss : 4.329933
  Epoch Time : 0:00:02

2026-05-17 07:14:49 - basemodel_training_rank_0 - INFO - ✓ Best model saved → finetuned/1d_t4_hpo/basemodel/best_model  (val loss: 4.329933)
2026-05-17 07:14:51 - basemodel_training_rank_0 - INFO - [Epoch 4/30, Step 40/11] LR: 0.000378  Loss: 3.8931
2026-05-17 07:14:52 - basemodel_training_rank_0 - INFO - 
--- Epoch 4/30 Summary ---
  Train Loss : 3.947981
  Val   Loss : 4.284915
  Epoch Time : 0:00:02

2026-05-17 07:14:52 - basemodel_training_rank_0 - INFO - ✓ Best model saved → finetuned/1d_t4_hpo/basemodel/best_model  (val loss: 4.284915)
2026-05-17 07:14:53 - basemodel_training_rank_0 - INFO - [Epoch 5/30, Step 50/11] LR: 0.000375  Loss: 3.6784
2026-05-17 07:14:54 - basemodel_training_rank_0 - INFO - 
--- Epoch 5/30 Summary ---
  Train Loss : 3.662367
  Val   Loss : 4.433038
  Epoch Time : 0:00:02

2026-05-17 07:14:55 - basemodel_training_rank_0 - INFO - [Epoch 6/30, Step 60/11] LR: 0.000370  Loss: 3.3388
2026-05-17 07:14:57 - basemodel_training_rank_0 - INFO - 
--- Epoch 6/30 Summary ---
  Train Loss : 3.347033
  Val   Loss : 4.740929
  Epoch Time : 0:00:02

2026-05-17 07:14:58 - basemodel_training_rank_0 - INFO - [Epoch 7/30, Step 70/11] LR: 0.000363  Loss: 3.0353
2026-05-17 07:14:59 - basemodel_training_rank_0 - INFO - 
--- Epoch 7/30 Summary ---
  Train Loss : 3.022630
  Val   Loss : 5.068810
  Epoch Time : 0:00:02

2026-05-17 07:15:00 - basemodel_training_rank_0 - INFO - [Epoch 8/30, Step 80/11] LR: 0.000354  Loss: 2.7908
2026-05-17 07:15:02 - basemodel_training_rank_0 - INFO - 
--- Epoch 8/30 Summary ---
  Train Loss : 2.726871
  Val   Loss : 5.443906
  Epoch Time : 0:00:02

2026-05-17 07:15:02 - basemodel_training_rank_0 - INFO - [Epoch 9/30, Step 90/11] LR: 0.000344  Loss: 2.4924
2026-05-17 07:15:04 - basemodel_training_rank_0 - INFO - 
--- Epoch 9/30 Summary ---
  Train Loss : 2.504552
  Val   Loss : 5.675878
  Epoch Time : 0:00:02

2026-05-17 07:15:05 - basemodel_training_rank_0 - INFO - [Epoch 10/30, Step 100/11] LR: 0.000331  Loss: 2.5554
2026-05-17 07:15:07 - basemodel_training_rank_0 - INFO - [Epoch 10/30, Step 110/11] LR: 0.000317  Loss: 2.3115
2026-05-17 07:15:07 - basemodel_training_rank_0 - INFO - 
--- Epoch 10/30 Summary ---
  Train Loss : 2.348972
  Val   Loss : 5.722508
  Epoch Time : 0:00:02

2026-05-17 07:15:09 - basemodel_training_rank_0 - INFO - [Epoch 11/30, Step 120/11] LR: 0.000302  Loss: 2.0249
2026-05-17 07:15:10 - basemodel_training_rank_0 - INFO - 
--- Epoch 11/30 Summary ---
  Train Loss : 2.173679
  Val   Loss : 5.873799
  Epoch Time : 0:00:02

2026-05-17 07:15:11 - basemodel_training_rank_0 - INFO - [Epoch 12/30, Step 130/11] LR: 0.000285  Loss: 2.0338
2026-05-17 07:15:12 - basemodel_training_rank_0 - INFO - 
--- Epoch 12/30 Summary ---
  Train Loss : 2.040126
  Val   Loss : 5.966989
  Epoch Time : 0:00:02

2026-05-17 07:15:14 - basemodel_training_rank_0 - INFO - [Epoch 13/30, Step 140/11] LR: 0.000267  Loss: 1.8267
2026-05-17 07:15:15 - basemodel_training_rank_0 - INFO - 
--- Epoch 13/30 Summary ---
  Train Loss : 1.909443
  Val   Loss : 6.101976
  Epoch Time : 0:00:02

2026-05-17 07:15:16 - basemodel_training_rank_0 - INFO - [Epoch 14/30, Step 150/11] LR: 0.000249  Loss: 1.7927
2026-05-17 07:15:17 - basemodel_training_rank_0 - INFO - 
--- Epoch 14/30 Summary ---
  Train Loss : 1.813062
  Val   Loss : 6.195870
  Epoch Time : 0:00:02

2026-05-17 07:15:18 - basemodel_training_rank_0 - INFO - [Epoch 15/30, Step 160/11] LR: 0.000229  Loss: 1.6903
2026-05-17 07:15:20 - basemodel_training_rank_0 - INFO - 
--- Epoch 15/30 Summary ---
  Train Loss : 1.700253
  Val   Loss : 6.364306
  Epoch Time : 0:00:02

2026-05-17 07:15:21 - basemodel_training_rank_0 - INFO - [Epoch 16/30, Step 170/11] LR: 0.000210  Loss: 1.6306
2026-05-17 07:15:22 - basemodel_training_rank_0 - INFO - 
--- Epoch 16/30 Summary ---
  Train Loss : 1.632787
  Val   Loss : 6.429851
  Epoch Time : 0:00:02

2026-05-17 07:15:23 - basemodel_training_rank_0 - INFO - [Epoch 17/30, Step 180/11] LR: 0.000190  Loss: 1.5346
2026-05-17 07:15:25 - basemodel_training_rank_0 - INFO - 
--- Epoch 17/30 Summary ---
  Train Loss : 1.582093
  Val   Loss : 6.487051
  Epoch Time : 0:00:02

2026-05-17 07:15:25 - basemodel_training_rank_0 - INFO - [Epoch 18/30, Step 190/11] LR: 0.000170  Loss: 1.5352
2026-05-17 07:15:27 - basemodel_training_rank_0 - INFO - 
--- Epoch 18/30 Summary ---
  Train Loss : 1.519012
  Val   Loss : 6.514397
  Epoch Time : 0:00:02

2026-05-17 07:15:28 - basemodel_training_rank_0 - INFO - [Epoch 19/30, Step 200/11] LR: 0.000150  Loss: 1.5341
2026-05-17 07:15:30 - basemodel_training_rank_0 - INFO - 
--- Epoch 19/30 Summary ---
  Train Loss : 1.474047
  Val   Loss : 6.548087
  Epoch Time : 0:00:02

2026-05-17 07:15:30 - basemodel_training_rank_0 - INFO - [Epoch 20/30, Step 210/11] LR: 0.000131  Loss: 1.4659
2026-05-17 07:15:32 - basemodel_training_rank_0 - INFO - [Epoch 20/30, Step 220/11] LR: 0.000112  Loss: 1.4378
2026-05-17 07:15:32 - basemodel_training_rank_0 - INFO - 
--- Epoch 20/30 Summary ---
  Train Loss : 1.442221
  Val   Loss : 6.604577
  Epoch Time : 0:00:02

2026-05-17 07:15:34 - basemodel_training_rank_0 - INFO - [Epoch 21/30, Step 230/11] LR: 0.000094  Loss: 1.3581
2026-05-17 07:15:35 - basemodel_training_rank_0 - INFO - 
--- Epoch 21/30 Summary ---
  Train Loss : 1.399695
  Val   Loss : 6.626465
  Epoch Time : 0:00:02

2026-05-17 07:15:37 - basemodel_training_rank_0 - INFO - [Epoch 22/30, Step 240/11] LR: 0.000078  Loss: 1.3639
2026-05-17 07:15:37 - basemodel_training_rank_0 - INFO - 
--- Epoch 22/30 Summary ---
  Train Loss : 1.381617
  Val   Loss : 6.647118
  Epoch Time : 0:00:02

2026-05-17 07:15:39 - basemodel_training_rank_0 - INFO - [Epoch 23/30, Step 250/11] LR: 0.000062  Loss: 1.3635
2026-05-17 07:15:40 - basemodel_training_rank_0 - INFO - 
--- Epoch 23/30 Summary ---
  Train Loss : 1.358930
  Val   Loss : 6.681557
  Epoch Time : 0:00:02

2026-05-17 07:15:42 - basemodel_training_rank_0 - INFO - [Epoch 24/30, Step 260/11] LR: 0.000048  Loss: 1.3244
2026-05-17 07:15:43 - basemodel_training_rank_0 - INFO - 
--- Epoch 24/30 Summary ---
  Train Loss : 1.342545
  Val   Loss : 6.693789
  Epoch Time : 0:00:02

2026-05-17 07:15:44 - basemodel_training_rank_0 - INFO - [Epoch 25/30, Step 270/11] LR: 0.000036  Loss: 1.3191
2026-05-17 07:15:45 - basemodel_training_rank_0 - INFO - 
--- Epoch 25/30 Summary ---
  Train Loss : 1.329156
  Val   Loss : 6.711730
  Epoch Time : 0:00:02

2026-05-17 07:15:46 - basemodel_training_rank_0 - INFO - [Epoch 26/30, Step 280/11] LR: 0.000025  Loss: 1.3204
2026-05-17 07:15:48 - basemodel_training_rank_0 - INFO - 
--- Epoch 26/30 Summary ---
  Train Loss : 1.326763
  Val   Loss : 6.722200
  Epoch Time : 0:00:02

2026-05-17 07:15:49 - basemodel_training_rank_0 - INFO - [Epoch 27/30, Step 290/11] LR: 0.000016  Loss: 1.3297
2026-05-17 07:15:50 - basemodel_training_rank_0 - INFO - 
--- Epoch 27/30 Summary ---
  Train Loss : 1.316370
  Val   Loss : 6.721284
  Epoch Time : 0:00:02

2026-05-17 07:15:51 - basemodel_training_rank_0 - INFO - [Epoch 28/30, Step 300/11] LR: 0.000009  Loss: 1.3364
2026-05-17 07:15:53 - basemodel_training_rank_0 - INFO - 
--- Epoch 28/30 Summary ---
  Train Loss : 1.316487
  Val   Loss : 6.720575
  Epoch Time : 0:00:02

2026-05-17 07:15:53 - basemodel_training_rank_0 - INFO - [Epoch 29/30, Step 310/11] LR: 0.000004  Loss: 1.3150
2026-05-17 07:15:56 - basemodel_training_rank_0 - INFO - 
--- Epoch 29/30 Summary ---
  Train Loss : 1.315080
  Val   Loss : 6.722671
  Epoch Time : 0:00:02

2026-05-17 07:15:56 - basemodel_training_rank_0 - INFO - [Epoch 30/30, Step 320/11] LR: 0.000001  Loss: 1.3168
2026-05-17 07:15:58 - basemodel_training_rank_0 - INFO - [Epoch 30/30, Step 330/11] LR: 0.000000  Loss: 1.3774
2026-05-17 07:15:58 - basemodel_training_rank_0 - INFO - 
--- Epoch 30/30 Summary ---
  Train Loss : 1.314285
  Val   Loss : 6.719721
  Epoch Time : 0:00:02

2026-05-17 07:15:58 - basemodel_training_rank_0 - INFO - Basemodel training completed! Best validation loss: 4.2849
Training time: 1.27 minutes
Model saved to: finetuned/1d_t4_hpo/basemodel

```

---

### `finetuned/1d_t4_hpo/logs/tokenizer_training_rank_0.log`

```text
2026-05-17 07:11:49 - tokenizer_training_rank_0 - INFO - === Tokenizer Training Started ===
2026-05-17 07:11:49 - tokenizer_training_rank_0 - INFO - Experiment Name: 1d_t4_hpo
2026-05-17 07:11:49 - tokenizer_training_rank_0 - INFO - Log Directory: finetuned/1d_t4_hpo/logs
2026-05-17 07:11:49 - tokenizer_training_rank_0 - INFO - Rank: 0
2026-05-17 07:11:49 - tokenizer_training_rank_0 - INFO - Timestamp: 2026-05-17 07:11:49
2026-05-17 07:11:49 - tokenizer_training_rank_0 - INFO - Loading pretrained tokenizer...
2026-05-17 07:11:49 - tokenizer_training_rank_0 - INFO - Tokenizer parameters: 3,958,042
2026-05-17 07:11:49 - tokenizer_training_rank_0 - INFO - === Training Configuration ===
2026-05-17 07:11:49 - tokenizer_training_rank_0 - INFO - Data path: data/1d.csv
2026-05-17 07:11:49 - tokenizer_training_rank_0 - INFO - Lookback window: 90
2026-05-17 07:11:49 - tokenizer_training_rank_0 - INFO - Predict window: 14
2026-05-17 07:11:49 - tokenizer_training_rank_0 - INFO - Batch size: 128
2026-05-17 07:11:49 - tokenizer_training_rank_0 - INFO - Learning rate: 0.0021627162139822797
2026-05-17 07:11:49 - tokenizer_training_rank_0 - INFO - Training epochs: 50
2026-05-17 07:11:49 - tokenizer_training_rank_0 - INFO - Device: cuda:0
2026-05-17 07:11:49 - tokenizer_training_rank_0 - INFO - Distributed training: False
2026-05-17 07:11:49 - tokenizer_training_rank_0 - INFO - Starting tokenizer fine-tuning training...
2026-05-17 07:11:49 - tokenizer_training_rank_0 - INFO - Starting tokenizer training...
2026-05-17 07:11:54 - tokenizer_training_rank_0 - INFO - [Epoch 1/50, Step 10/11] LR: 0.000374, Loss: 0.0361
2026-05-17 07:11:55 - tokenizer_training_rank_0 - INFO - 
--- Epoch 1/50 Summary ---
Validation Loss: 0.119014
Epoch Time: 0:00:04

2026-05-17 07:11:55 - tokenizer_training_rank_0 - INFO - Best model saved: finetuned/1d_t4_hpo/tokenizer/best_model (val loss: 0.119014)
2026-05-17 07:11:57 - tokenizer_training_rank_0 - INFO - [Epoch 2/50, Step 20/11] LR: 0.000797, Loss: -0.0513
2026-05-17 07:11:58 - tokenizer_training_rank_0 - INFO - 
--- Epoch 2/50 Summary ---
Validation Loss: 0.091160
Epoch Time: 0:00:03

2026-05-17 07:11:58 - tokenizer_training_rank_0 - INFO - Best model saved: finetuned/1d_t4_hpo/tokenizer/best_model (val loss: 0.091160)
2026-05-17 07:12:01 - tokenizer_training_rank_0 - INFO - [Epoch 3/50, Step 30/11] LR: 0.001348, Loss: -0.1186
2026-05-17 07:12:02 - tokenizer_training_rank_0 - INFO - 
--- Epoch 3/50 Summary ---
Validation Loss: 0.079762
Epoch Time: 0:00:03

2026-05-17 07:12:02 - tokenizer_training_rank_0 - INFO - Best model saved: finetuned/1d_t4_hpo/tokenizer/best_model (val loss: 0.079762)
2026-05-17 07:12:04 - tokenizer_training_rank_0 - INFO - [Epoch 4/50, Step 40/11] LR: 0.001847, Loss: -0.1567
2026-05-17 07:12:05 - tokenizer_training_rank_0 - INFO - 
--- Epoch 4/50 Summary ---
Validation Loss: 0.068567
Epoch Time: 0:00:03

2026-05-17 07:12:05 - tokenizer_training_rank_0 - INFO - Best model saved: finetuned/1d_t4_hpo/tokenizer/best_model (val loss: 0.068567)
2026-05-17 07:12:07 - tokenizer_training_rank_0 - INFO - [Epoch 5/50, Step 50/11] LR: 0.002132, Loss: -0.1905
2026-05-17 07:12:09 - tokenizer_training_rank_0 - INFO - 
--- Epoch 5/50 Summary ---
Validation Loss: 0.064920
Epoch Time: 0:00:03

2026-05-17 07:12:09 - tokenizer_training_rank_0 - INFO - Best model saved: finetuned/1d_t4_hpo/tokenizer/best_model (val loss: 0.064920)
2026-05-17 07:12:11 - tokenizer_training_rank_0 - INFO - [Epoch 6/50, Step 60/11] LR: 0.002162, Loss: -0.2086
2026-05-17 07:12:13 - tokenizer_training_rank_0 - INFO - 
--- Epoch 6/50 Summary ---
Validation Loss: 0.059318
Epoch Time: 0:00:03

2026-05-17 07:12:13 - tokenizer_training_rank_0 - INFO - Best model saved: finetuned/1d_t4_hpo/tokenizer/best_model (val loss: 0.059318)
2026-05-17 07:12:14 - tokenizer_training_rank_0 - INFO - [Epoch 7/50, Step 70/11] LR: 0.002157, Loss: -0.2225
2026-05-17 07:12:16 - tokenizer_training_rank_0 - INFO - 
--- Epoch 7/50 Summary ---
Validation Loss: 0.056564
Epoch Time: 0:00:03

2026-05-17 07:12:16 - tokenizer_training_rank_0 - INFO - Best model saved: finetuned/1d_t4_hpo/tokenizer/best_model (val loss: 0.056564)
2026-05-17 07:12:17 - tokenizer_training_rank_0 - INFO - [Epoch 8/50, Step 80/11] LR: 0.002148, Loss: -0.2298
2026-05-17 07:12:20 - tokenizer_training_rank_0 - INFO - 
--- Epoch 8/50 Summary ---
Validation Loss: 0.058682
Epoch Time: 0:00:03

2026-05-17 07:12:20 - tokenizer_training_rank_0 - INFO - [Epoch 9/50, Step 90/11] LR: 0.002135, Loss: -0.2355
2026-05-17 07:12:23 - tokenizer_training_rank_0 - INFO - 
--- Epoch 9/50 Summary ---
Validation Loss: 0.052077
Epoch Time: 0:00:03

2026-05-17 07:12:23 - tokenizer_training_rank_0 - INFO - Best model saved: finetuned/1d_t4_hpo/tokenizer/best_model (val loss: 0.052077)
2026-05-17 07:12:24 - tokenizer_training_rank_0 - INFO - [Epoch 10/50, Step 100/11] LR: 0.002118, Loss: -0.2412
2026-05-17 07:12:26 - tokenizer_training_rank_0 - INFO - [Epoch 10/50, Step 110/11] LR: 0.002096, Loss: -0.2398
2026-05-17 07:12:27 - tokenizer_training_rank_0 - INFO - 
--- Epoch 10/50 Summary ---
Validation Loss: 0.054823
Epoch Time: 0:00:03

2026-05-17 07:12:30 - tokenizer_training_rank_0 - INFO - [Epoch 11/50, Step 120/11] LR: 0.002070, Loss: -0.2474
2026-05-17 07:12:30 - tokenizer_training_rank_0 - INFO - 
--- Epoch 11/50 Summary ---
Validation Loss: 0.054190
Epoch Time: 0:00:03

2026-05-17 07:12:33 - tokenizer_training_rank_0 - INFO - [Epoch 12/50, Step 130/11] LR: 0.002040, Loss: -0.2505
2026-05-17 07:12:33 - tokenizer_training_rank_0 - INFO - 
--- Epoch 12/50 Summary ---
Validation Loss: 0.051983
Epoch Time: 0:00:03

2026-05-17 07:12:33 - tokenizer_training_rank_0 - INFO - Best model saved: finetuned/1d_t4_hpo/tokenizer/best_model (val loss: 0.051983)
2026-05-17 07:12:36 - tokenizer_training_rank_0 - INFO - [Epoch 13/50, Step 140/11] LR: 0.002007, Loss: -0.2478
2026-05-17 07:12:37 - tokenizer_training_rank_0 - INFO - 
--- Epoch 13/50 Summary ---
Validation Loss: 0.052626
Epoch Time: 0:00:03

2026-05-17 07:12:39 - tokenizer_training_rank_0 - INFO - [Epoch 14/50, Step 150/11] LR: 0.001969, Loss: -0.2547
2026-05-17 07:12:40 - tokenizer_training_rank_0 - INFO - 
--- Epoch 14/50 Summary ---
Validation Loss: 0.053510
Epoch Time: 0:00:03

2026-05-17 07:12:42 - tokenizer_training_rank_0 - INFO - [Epoch 15/50, Step 160/11] LR: 0.001928, Loss: -0.2564
2026-05-17 07:12:43 - tokenizer_training_rank_0 - INFO - 
--- Epoch 15/50 Summary ---
Validation Loss: 0.051254
Epoch Time: 0:00:03

2026-05-17 07:12:43 - tokenizer_training_rank_0 - INFO - Best model saved: finetuned/1d_t4_hpo/tokenizer/best_model (val loss: 0.051254)
2026-05-17 07:12:45 - tokenizer_training_rank_0 - INFO - [Epoch 16/50, Step 170/11] LR: 0.001884, Loss: -0.2576
2026-05-17 07:12:47 - tokenizer_training_rank_0 - INFO - 
--- Epoch 16/50 Summary ---
Validation Loss: 0.051031
Epoch Time: 0:00:03

2026-05-17 07:12:47 - tokenizer_training_rank_0 - INFO - Best model saved: finetuned/1d_t4_hpo/tokenizer/best_model (val loss: 0.051031)
2026-05-17 07:12:48 - tokenizer_training_rank_0 - INFO - [Epoch 17/50, Step 180/11] LR: 0.001836, Loss: -0.2612
2026-05-17 07:12:50 - tokenizer_training_rank_0 - INFO - 
--- Epoch 17/50 Summary ---
Validation Loss: 0.057854
Epoch Time: 0:00:03

2026-05-17 07:12:51 - tokenizer_training_rank_0 - INFO - [Epoch 18/50, Step 190/11] LR: 0.001786, Loss: -0.2596
2026-05-17 07:12:53 - tokenizer_training_rank_0 - INFO - 
--- Epoch 18/50 Summary ---
Validation Loss: 0.054282
Epoch Time: 0:00:03

2026-05-17 07:12:54 - tokenizer_training_rank_0 - INFO - [Epoch 19/50, Step 200/11] LR: 0.001732, Loss: -0.2637
2026-05-17 07:12:57 - tokenizer_training_rank_0 - INFO - 
--- Epoch 19/50 Summary ---
Validation Loss: 0.054062
Epoch Time: 0:00:03

2026-05-17 07:12:57 - tokenizer_training_rank_0 - INFO - [Epoch 20/50, Step 210/11] LR: 0.001676, Loss: -0.2641
2026-05-17 07:13:00 - tokenizer_training_rank_0 - INFO - [Epoch 20/50, Step 220/11] LR: 0.001617, Loss: -0.2632
2026-05-17 07:13:00 - tokenizer_training_rank_0 - INFO - 
--- Epoch 20/50 Summary ---
Validation Loss: 0.052563
Epoch Time: 0:00:03

2026-05-17 07:13:03 - tokenizer_training_rank_0 - INFO - [Epoch 21/50, Step 230/11] LR: 0.001557, Loss: -0.2661
2026-05-17 07:13:03 - tokenizer_training_rank_0 - INFO - 
--- Epoch 21/50 Summary ---
Validation Loss: 0.050338
Epoch Time: 0:00:03

2026-05-17 07:13:03 - tokenizer_training_rank_0 - INFO - Best model saved: finetuned/1d_t4_hpo/tokenizer/best_model (val loss: 0.050338)
2026-05-17 07:13:06 - tokenizer_training_rank_0 - INFO - [Epoch 22/50, Step 240/11] LR: 0.001494, Loss: -0.2659
2026-05-17 07:13:07 - tokenizer_training_rank_0 - INFO - 
--- Epoch 22/50 Summary ---
Validation Loss: 0.053006
Epoch Time: 0:00:03

2026-05-17 07:13:09 - tokenizer_training_rank_0 - INFO - [Epoch 23/50, Step 250/11] LR: 0.001430, Loss: -0.2692
2026-05-17 07:13:10 - tokenizer_training_rank_0 - INFO - 
--- Epoch 23/50 Summary ---
Validation Loss: 0.053697
Epoch Time: 0:00:03

2026-05-17 07:13:12 - tokenizer_training_rank_0 - INFO - [Epoch 24/50, Step 260/11] LR: 0.001364, Loss: -0.2701
2026-05-17 07:13:14 - tokenizer_training_rank_0 - INFO - 
--- Epoch 24/50 Summary ---
Validation Loss: 0.052960
Epoch Time: 0:00:03

2026-05-17 07:13:15 - tokenizer_training_rank_0 - INFO - [Epoch 25/50, Step 270/11] LR: 0.001297, Loss: -0.2689
2026-05-17 07:13:17 - tokenizer_training_rank_0 - INFO - 
--- Epoch 25/50 Summary ---
Validation Loss: 0.053371
Epoch Time: 0:00:03

2026-05-17 07:13:18 - tokenizer_training_rank_0 - INFO - [Epoch 26/50, Step 280/11] LR: 0.001230, Loss: -0.2707
2026-05-17 07:13:20 - tokenizer_training_rank_0 - INFO - 
--- Epoch 26/50 Summary ---
Validation Loss: 0.052022
Epoch Time: 0:00:03

2026-05-17 07:13:22 - tokenizer_training_rank_0 - INFO - [Epoch 27/50, Step 290/11] LR: 0.001161, Loss: -0.2716
2026-05-17 07:13:24 - tokenizer_training_rank_0 - INFO - 
--- Epoch 27/50 Summary ---
Validation Loss: 0.054940
Epoch Time: 0:00:03

2026-05-17 07:13:25 - tokenizer_training_rank_0 - INFO - [Epoch 28/50, Step 300/11] LR: 0.001093, Loss: -0.2725
2026-05-17 07:13:27 - tokenizer_training_rank_0 - INFO - 
--- Epoch 28/50 Summary ---
Validation Loss: 0.054098
Epoch Time: 0:00:03

2026-05-17 07:13:28 - tokenizer_training_rank_0 - INFO - [Epoch 29/50, Step 310/11] LR: 0.001024, Loss: -0.2716
2026-05-17 07:13:31 - tokenizer_training_rank_0 - INFO - 
--- Epoch 29/50 Summary ---
Validation Loss: 0.053061
Epoch Time: 0:00:03

2026-05-17 07:13:31 - tokenizer_training_rank_0 - INFO - [Epoch 30/50, Step 320/11] LR: 0.000956, Loss: -0.2726
2026-05-17 07:13:34 - tokenizer_training_rank_0 - INFO - [Epoch 30/50, Step 330/11] LR: 0.000888, Loss: -0.2745
2026-05-17 07:13:34 - tokenizer_training_rank_0 - INFO - 
--- Epoch 30/50 Summary ---
Validation Loss: 0.054280
Epoch Time: 0:00:03

2026-05-17 07:13:37 - tokenizer_training_rank_0 - INFO - [Epoch 31/50, Step 340/11] LR: 0.000821, Loss: -0.2737
2026-05-17 07:13:38 - tokenizer_training_rank_0 - INFO - 
--- Epoch 31/50 Summary ---
Validation Loss: 0.055258
Epoch Time: 0:00:03

2026-05-17 07:13:40 - tokenizer_training_rank_0 - INFO - [Epoch 32/50, Step 350/11] LR: 0.000755, Loss: -0.2746
2026-05-17 07:13:41 - tokenizer_training_rank_0 - INFO - 
--- Epoch 32/50 Summary ---
Validation Loss: 0.053723
Epoch Time: 0:00:03

2026-05-17 07:13:43 - tokenizer_training_rank_0 - INFO - [Epoch 33/50, Step 360/11] LR: 0.000690, Loss: -0.2764
2026-05-17 07:13:45 - tokenizer_training_rank_0 - INFO - 
--- Epoch 33/50 Summary ---
Validation Loss: 0.052671
Epoch Time: 0:00:03

2026-05-17 07:13:47 - tokenizer_training_rank_0 - INFO - [Epoch 34/50, Step 370/11] LR: 0.000627, Loss: -0.2761
2026-05-17 07:13:48 - tokenizer_training_rank_0 - INFO - 
--- Epoch 34/50 Summary ---
Validation Loss: 0.054220
Epoch Time: 0:00:03

2026-05-17 07:13:50 - tokenizer_training_rank_0 - INFO - [Epoch 35/50, Step 380/11] LR: 0.000565, Loss: -0.2755
2026-05-17 07:13:51 - tokenizer_training_rank_0 - INFO - 
--- Epoch 35/50 Summary ---
Validation Loss: 0.054730
Epoch Time: 0:00:03

2026-05-17 07:13:53 - tokenizer_training_rank_0 - INFO - [Epoch 36/50, Step 390/11] LR: 0.000506, Loss: -0.2770
2026-05-17 07:13:55 - tokenizer_training_rank_0 - INFO - 
--- Epoch 36/50 Summary ---
Validation Loss: 0.054964
Epoch Time: 0:00:03

2026-05-17 07:13:56 - tokenizer_training_rank_0 - INFO - [Epoch 37/50, Step 400/11] LR: 0.000449, Loss: -0.2783
2026-05-17 07:13:58 - tokenizer_training_rank_0 - INFO - 
--- Epoch 37/50 Summary ---
Validation Loss: 0.055089
Epoch Time: 0:00:03

2026-05-17 07:13:59 - tokenizer_training_rank_0 - INFO - [Epoch 38/50, Step 410/11] LR: 0.000395, Loss: -0.2771
2026-05-17 07:14:01 - tokenizer_training_rank_0 - INFO - 
--- Epoch 38/50 Summary ---
Validation Loss: 0.053634
Epoch Time: 0:00:03

2026-05-17 07:14:02 - tokenizer_training_rank_0 - INFO - [Epoch 39/50, Step 420/11] LR: 0.000343, Loss: -0.2769
2026-05-17 07:14:05 - tokenizer_training_rank_0 - INFO - 
--- Epoch 39/50 Summary ---
Validation Loss: 0.054066
Epoch Time: 0:00:03

2026-05-17 07:14:05 - tokenizer_training_rank_0 - INFO - [Epoch 40/50, Step 430/11] LR: 0.000294, Loss: -0.2789
2026-05-17 07:14:08 - tokenizer_training_rank_0 - INFO - [Epoch 40/50, Step 440/11] LR: 0.000249, Loss: -0.2784
2026-05-17 07:14:08 - tokenizer_training_rank_0 - INFO - 
--- Epoch 40/50 Summary ---
Validation Loss: 0.054495
Epoch Time: 0:00:03

2026-05-17 07:14:11 - tokenizer_training_rank_0 - INFO - [Epoch 41/50, Step 450/11] LR: 0.000207, Loss: -0.2784
2026-05-17 07:14:11 - tokenizer_training_rank_0 - INFO - 
--- Epoch 41/50 Summary ---
Validation Loss: 0.054262
Epoch Time: 0:00:03

2026-05-17 07:14:14 - tokenizer_training_rank_0 - INFO - [Epoch 42/50, Step 460/11] LR: 0.000168, Loss: -0.2787
2026-05-17 07:14:15 - tokenizer_training_rank_0 - INFO - 
--- Epoch 42/50 Summary ---
Validation Loss: 0.054672
Epoch Time: 0:00:03

2026-05-17 07:14:17 - tokenizer_training_rank_0 - INFO - [Epoch 43/50, Step 470/11] LR: 0.000133, Loss: -0.2784
2026-05-17 07:14:18 - tokenizer_training_rank_0 - INFO - 
--- Epoch 43/50 Summary ---
Validation Loss: 0.054912
Epoch Time: 0:00:03

2026-05-17 07:14:20 - tokenizer_training_rank_0 - INFO - [Epoch 44/50, Step 480/11] LR: 0.000102, Loss: -0.2792
2026-05-17 07:14:21 - tokenizer_training_rank_0 - INFO - 
--- Epoch 44/50 Summary ---
Validation Loss: 0.055603
Epoch Time: 0:00:03

2026-05-17 07:14:23 - tokenizer_training_rank_0 - INFO - [Epoch 45/50, Step 490/11] LR: 0.000075, Loss: -0.2781
2026-05-17 07:14:25 - tokenizer_training_rank_0 - INFO - 
--- Epoch 45/50 Summary ---
Validation Loss: 0.055655
Epoch Time: 0:00:03

2026-05-17 07:14:26 - tokenizer_training_rank_0 - INFO - [Epoch 46/50, Step 500/11] LR: 0.000052, Loss: -0.2779
2026-05-17 07:14:28 - tokenizer_training_rank_0 - INFO - 
--- Epoch 46/50 Summary ---
Validation Loss: 0.055599
Epoch Time: 0:00:03

2026-05-17 07:14:29 - tokenizer_training_rank_0 - INFO - [Epoch 47/50, Step 510/11] LR: 0.000033, Loss: -0.2782
2026-05-17 07:14:31 - tokenizer_training_rank_0 - INFO - 
--- Epoch 47/50 Summary ---
Validation Loss: 0.055318
Epoch Time: 0:00:03

2026-05-17 07:14:32 - tokenizer_training_rank_0 - INFO - [Epoch 48/50, Step 520/11] LR: 0.000018, Loss: -0.2794
2026-05-17 07:14:35 - tokenizer_training_rank_0 - INFO - 
--- Epoch 48/50 Summary ---
Validation Loss: 0.055422
Epoch Time: 0:00:03

2026-05-17 07:14:35 - tokenizer_training_rank_0 - INFO - [Epoch 49/50, Step 530/11] LR: 0.000008, Loss: -0.2788
2026-05-17 07:14:38 - tokenizer_training_rank_0 - INFO - 
--- Epoch 49/50 Summary ---
Validation Loss: 0.055499
Epoch Time: 0:00:03

2026-05-17 07:14:38 - tokenizer_training_rank_0 - INFO - [Epoch 50/50, Step 540/11] LR: 0.000002, Loss: -0.2784
2026-05-17 07:14:41 - tokenizer_training_rank_0 - INFO - [Epoch 50/50, Step 550/11] LR: 0.000000, Loss: -0.2791
2026-05-17 07:14:41 - tokenizer_training_rank_0 - INFO - 
--- Epoch 50/50 Summary ---
Validation Loss: 0.055485
Epoch Time: 0:00:03

2026-05-17 07:14:42 - tokenizer_training_rank_0 - INFO - Tokenizer training completed! Best validation loss: 0.0503
Training time: 2.88 minutes
Model saved to: finetuned/1d_t4_hpo/tokenizer

```

---

### `finetuned/1d_t4_hpo/tokenizer/trial.log`

```text
2026-05-17 07:06:29 [INFO] Starting tokenizer training...
2026-05-17 07:06:32 [INFO] [Epoch 1/50, Step 10/11] LR: 0.000374, Loss: -0.1784
2026-05-17 07:06:32 [INFO] 
--- Epoch 1/50 Summary ---
Validation Loss: 0.087068
Epoch Time: 0:00:03

2026-05-17 07:06:32 [INFO] Best model saved: finetuned/1d_t4_hpo/tokenizer/best_model (val loss: 0.087068)
2026-05-17 07:06:35 [INFO] [Epoch 2/50, Step 20/11] LR: 0.000797, Loss: -0.2155
2026-05-17 07:06:36 [INFO] 
--- Epoch 2/50 Summary ---
Validation Loss: 0.066257
Epoch Time: 0:00:03

2026-05-17 07:06:36 [INFO] Best model saved: finetuned/1d_t4_hpo/tokenizer/best_model (val loss: 0.066257)
2026-05-17 07:06:38 [INFO] [Epoch 3/50, Step 30/11] LR: 0.001348, Loss: -0.2384
2026-05-17 07:06:39 [INFO] 
--- Epoch 3/50 Summary ---
Validation Loss: 0.054230
Epoch Time: 0:00:03

2026-05-17 07:06:39 [INFO] Best model saved: finetuned/1d_t4_hpo/tokenizer/best_model (val loss: 0.054230)
2026-05-17 07:06:41 [INFO] [Epoch 4/50, Step 40/11] LR: 0.001847, Loss: -0.2449
2026-05-17 07:06:43 [INFO] 
--- Epoch 4/50 Summary ---
Validation Loss: 0.047967
Epoch Time: 0:00:03

2026-05-17 07:06:43 [INFO] Best model saved: finetuned/1d_t4_hpo/tokenizer/best_model (val loss: 0.047967)
2026-05-17 07:06:44 [INFO] [Epoch 5/50, Step 50/11] LR: 0.002132, Loss: -0.2566
2026-05-17 07:06:46 [INFO] 
--- Epoch 5/50 Summary ---
Validation Loss: 0.045639
Epoch Time: 0:00:03

2026-05-17 07:06:46 [INFO] Best model saved: finetuned/1d_t4_hpo/tokenizer/best_model (val loss: 0.045639)
2026-05-17 07:06:48 [INFO] [Epoch 6/50, Step 60/11] LR: 0.002162, Loss: -0.2640
2026-05-17 07:06:50 [INFO] 
--- Epoch 6/50 Summary ---
Validation Loss: 0.045230
Epoch Time: 0:00:03

2026-05-17 07:06:50 [INFO] Best model saved: finetuned/1d_t4_hpo/tokenizer/best_model (val loss: 0.045230)
2026-05-17 07:06:51 [INFO] [Epoch 7/50, Step 70/11] LR: 0.002157, Loss: -0.2656
2026-05-17 07:06:53 [INFO] 
--- Epoch 7/50 Summary ---
Validation Loss: 0.047613
Epoch Time: 0:00:03

2026-05-17 07:06:54 [INFO] [Epoch 8/50, Step 80/11] LR: 0.002148, Loss: -0.2728
2026-05-17 07:06:57 [INFO] 
--- Epoch 8/50 Summary ---
Validation Loss: 0.045344
Epoch Time: 0:00:03

2026-05-17 07:06:57 [INFO] [Epoch 9/50, Step 90/11] LR: 0.002135, Loss: -0.2738
2026-05-17 07:07:00 [INFO] 
--- Epoch 9/50 Summary ---
Validation Loss: 0.046707
Epoch Time: 0:00:03

2026-05-17 07:07:00 [INFO] [Epoch 10/50, Step 100/11] LR: 0.002118, Loss: -0.2750
2026-05-17 07:07:03 [INFO] [Epoch 10/50, Step 110/11] LR: 0.002096, Loss: -0.2779
2026-05-17 07:07:03 [INFO] 
--- Epoch 10/50 Summary ---
Validation Loss: 0.049689
Epoch Time: 0:00:03

2026-05-17 07:07:06 [INFO] [Epoch 11/50, Step 120/11] LR: 0.002070, Loss: -0.2773
2026-05-17 07:07:07 [INFO] 
--- Epoch 11/50 Summary ---
Validation Loss: 0.047271
Epoch Time: 0:00:03

2026-05-17 07:07:09 [INFO] [Epoch 12/50, Step 130/11] LR: 0.002040, Loss: -0.2813
2026-05-17 07:07:10 [INFO] 
--- Epoch 12/50 Summary ---
Validation Loss: 0.046672
Epoch Time: 0:00:03

2026-05-17 07:07:12 [INFO] [Epoch 13/50, Step 140/11] LR: 0.002007, Loss: -0.2795
2026-05-17 07:07:13 [INFO] 
--- Epoch 13/50 Summary ---
Validation Loss: 0.049256
Epoch Time: 0:00:03

2026-05-17 07:07:15 [INFO] [Epoch 14/50, Step 150/11] LR: 0.001969, Loss: -0.2835
2026-05-17 07:07:17 [INFO] 
--- Epoch 14/50 Summary ---
Validation Loss: 0.048708
Epoch Time: 0:00:03

2026-05-17 07:07:18 [INFO] [Epoch 15/50, Step 160/11] LR: 0.001928, Loss: -0.2839
2026-05-17 07:07:20 [INFO] 
--- Epoch 15/50 Summary ---
Validation Loss: 0.048479
Epoch Time: 0:00:03

2026-05-17 07:07:21 [INFO] [Epoch 16/50, Step 170/11] LR: 0.001884, Loss: -0.2855
2026-05-17 07:07:23 [INFO] 
--- Epoch 16/50 Summary ---
Validation Loss: 0.049386
Epoch Time: 0:00:03

2026-05-17 07:07:24 [INFO] [Epoch 17/50, Step 180/11] LR: 0.001836, Loss: -0.2849
2026-05-17 07:07:26 [INFO] 
--- Epoch 17/50 Summary ---
Validation Loss: 0.052336
Epoch Time: 0:00:03

2026-05-17 07:07:27 [INFO] [Epoch 18/50, Step 190/11] LR: 0.001786, Loss: -0.2851
2026-05-17 07:07:30 [INFO] 
--- Epoch 18/50 Summary ---
Validation Loss: 0.051633
Epoch Time: 0:00:03

2026-05-17 07:07:30 [INFO] [Epoch 19/50, Step 200/11] LR: 0.001732, Loss: -0.2881
2026-05-17 07:07:33 [INFO] 
--- Epoch 19/50 Summary ---
Validation Loss: 0.052634
Epoch Time: 0:00:03

2026-05-17 07:07:33 [INFO] [Epoch 20/50, Step 210/11] LR: 0.001676, Loss: -0.2883
2026-05-17 07:07:36 [INFO] [Epoch 20/50, Step 220/11] LR: 0.001617, Loss: -0.2883
2026-05-17 07:07:36 [INFO] 
--- Epoch 20/50 Summary ---
Validation Loss: 0.054674
Epoch Time: 0:00:03

2026-05-17 07:07:39 [INFO] [Epoch 21/50, Step 230/11] LR: 0.001557, Loss: -0.2894
2026-05-17 07:07:39 [INFO] 
--- Epoch 21/50 Summary ---
Validation Loss: 0.053140
Epoch Time: 0:00:03

2026-05-17 07:07:42 [INFO] [Epoch 22/50, Step 240/11] LR: 0.001494, Loss: -0.2901
2026-05-17 07:07:42 [INFO] 
--- Epoch 22/50 Summary ---
Validation Loss: 0.055888
Epoch Time: 0:00:03

2026-05-17 07:07:45 [INFO] [Epoch 23/50, Step 250/11] LR: 0.001430, Loss: -0.2905
2026-05-17 07:07:46 [INFO] 
--- Epoch 23/50 Summary ---
Validation Loss: 0.056180
Epoch Time: 0:00:03

2026-05-17 07:07:48 [INFO] [Epoch 24/50, Step 260/11] LR: 0.001364, Loss: -0.2929
2026-05-17 07:07:49 [INFO] 
--- Epoch 24/50 Summary ---
Validation Loss: 0.057684
Epoch Time: 0:00:03

2026-05-17 07:07:51 [INFO] [Epoch 25/50, Step 270/11] LR: 0.001297, Loss: -0.2931
2026-05-17 07:07:52 [INFO] 
--- Epoch 25/50 Summary ---
Validation Loss: 0.058917
Epoch Time: 0:00:03

2026-05-17 07:07:54 [INFO] [Epoch 26/50, Step 280/11] LR: 0.001230, Loss: -0.2932
2026-05-17 07:07:56 [INFO] 
--- Epoch 26/50 Summary ---
Validation Loss: 0.057438
Epoch Time: 0:00:03

2026-05-17 07:07:57 [INFO] [Epoch 27/50, Step 290/11] LR: 0.001161, Loss: -0.2944
2026-05-17 07:07:59 [INFO] 
--- Epoch 27/50 Summary ---
Validation Loss: 0.059616
Epoch Time: 0:00:03

2026-05-17 07:08:00 [INFO] [Epoch 28/50, Step 300/11] LR: 0.001093, Loss: -0.2949
2026-05-17 07:08:02 [INFO] 
--- Epoch 28/50 Summary ---
Validation Loss: 0.059907
Epoch Time: 0:00:03

2026-05-17 07:08:03 [INFO] [Epoch 29/50, Step 310/11] LR: 0.001024, Loss: -0.2952
2026-05-17 07:08:06 [INFO] 
--- Epoch 29/50 Summary ---
Validation Loss: 0.060406
Epoch Time: 0:00:03

2026-05-17 07:08:06 [INFO] [Epoch 30/50, Step 320/11] LR: 0.000956, Loss: -0.2948
2026-05-17 07:08:09 [INFO] [Epoch 30/50, Step 330/11] LR: 0.000888, Loss: -0.2955
2026-05-17 07:08:09 [INFO] 
--- Epoch 30/50 Summary ---
Validation Loss: 0.060676
Epoch Time: 0:00:03

2026-05-17 07:08:12 [INFO] [Epoch 31/50, Step 340/11] LR: 0.000821, Loss: -0.2968
2026-05-17 07:08:12 [INFO] 
--- Epoch 31/50 Summary ---
Validation Loss: 0.061203
Epoch Time: 0:00:03

2026-05-17 07:08:15 [INFO] [Epoch 32/50, Step 350/11] LR: 0.000755, Loss: -0.2964
2026-05-17 07:08:16 [INFO] 
--- Epoch 32/50 Summary ---
Validation Loss: 0.061604
Epoch Time: 0:00:03

2026-05-17 07:08:18 [INFO] [Epoch 33/50, Step 360/11] LR: 0.000690, Loss: -0.2970
2026-05-17 07:08:19 [INFO] 
--- Epoch 33/50 Summary ---
Validation Loss: 0.062301
Epoch Time: 0:00:03

2026-05-17 07:08:21 [INFO] [Epoch 34/50, Step 370/11] LR: 0.000627, Loss: -0.2978
2026-05-17 07:08:23 [INFO] 
--- Epoch 34/50 Summary ---
Validation Loss: 0.062812
Epoch Time: 0:00:03

2026-05-17 07:08:24 [INFO] [Epoch 35/50, Step 380/11] LR: 0.000565, Loss: -0.2979
2026-05-17 07:08:26 [INFO] 
--- Epoch 35/50 Summary ---
Validation Loss: 0.064499
Epoch Time: 0:00:03

2026-05-17 07:08:27 [INFO] [Epoch 36/50, Step 390/11] LR: 0.000506, Loss: -0.2981
2026-05-17 07:08:29 [INFO] 
--- Epoch 36/50 Summary ---
Validation Loss: 0.064907
Epoch Time: 0:00:03

2026-05-17 07:08:31 [INFO] [Epoch 37/50, Step 400/11] LR: 0.000449, Loss: -0.2983
2026-05-17 07:08:33 [INFO] 
--- Epoch 37/50 Summary ---
Validation Loss: 0.066471
Epoch Time: 0:00:03

2026-05-17 07:08:34 [INFO] [Epoch 38/50, Step 410/11] LR: 0.000395, Loss: -0.2983
2026-05-17 07:08:36 [INFO] 
--- Epoch 38/50 Summary ---
Validation Loss: 0.066884
Epoch Time: 0:00:03

2026-05-17 07:08:37 [INFO] [Epoch 39/50, Step 420/11] LR: 0.000343, Loss: -0.2990
2026-05-17 07:08:39 [INFO] 
--- Epoch 39/50 Summary ---
Validation Loss: 0.066256
Epoch Time: 0:00:03

2026-05-17 07:08:40 [INFO] [Epoch 40/50, Step 430/11] LR: 0.000294, Loss: -0.2994
2026-05-17 07:08:42 [INFO] [Epoch 40/50, Step 440/11] LR: 0.000249, Loss: -0.2996
2026-05-17 07:08:43 [INFO] 
--- Epoch 40/50 Summary ---
Validation Loss: 0.066787
Epoch Time: 0:00:03

2026-05-17 07:08:45 [INFO] [Epoch 41/50, Step 450/11] LR: 0.000207, Loss: -0.2999
2026-05-17 07:08:46 [INFO] 
--- Epoch 41/50 Summary ---
Validation Loss: 0.066963
Epoch Time: 0:00:03

2026-05-17 07:08:49 [INFO] [Epoch 42/50, Step 460/11] LR: 0.000168, Loss: -0.2997
2026-05-17 07:08:49 [INFO] 
--- Epoch 42/50 Summary ---
Validation Loss: 0.067538
Epoch Time: 0:00:03

2026-05-17 07:08:52 [INFO] [Epoch 43/50, Step 470/11] LR: 0.000133, Loss: -0.2995
2026-05-17 07:08:53 [INFO] 
--- Epoch 43/50 Summary ---
Validation Loss: 0.068086
Epoch Time: 0:00:03

2026-05-17 07:08:55 [INFO] [Epoch 44/50, Step 480/11] LR: 0.000102, Loss: -0.3004
2026-05-17 07:08:56 [INFO] 
--- Epoch 44/50 Summary ---
Validation Loss: 0.068216
Epoch Time: 0:00:03

2026-05-17 07:08:58 [INFO] [Epoch 45/50, Step 490/11] LR: 0.000075, Loss: -0.3003
2026-05-17 07:08:59 [INFO] 
--- Epoch 45/50 Summary ---
Validation Loss: 0.067886
Epoch Time: 0:00:03

2026-05-17 07:09:01 [INFO] [Epoch 46/50, Step 500/11] LR: 0.000052, Loss: -0.3007
2026-05-17 07:09:03 [INFO] 
--- Epoch 46/50 Summary ---
Validation Loss: 0.068284
Epoch Time: 0:00:03

2026-05-17 07:09:04 [INFO] [Epoch 47/50, Step 510/11] LR: 0.000033, Loss: -0.3003
2026-05-17 07:09:06 [INFO] 
--- Epoch 47/50 Summary ---
Validation Loss: 0.068507
Epoch Time: 0:00:03

2026-05-17 07:09:07 [INFO] [Epoch 48/50, Step 520/11] LR: 0.000018, Loss: -0.3002
2026-05-17 07:09:09 [INFO] 
--- Epoch 48/50 Summary ---
Validation Loss: 0.068597
Epoch Time: 0:00:03

2026-05-17 07:09:10 [INFO] [Epoch 49/50, Step 530/11] LR: 0.000008, Loss: -0.3005
2026-05-17 07:09:13 [INFO] 
--- Epoch 49/50 Summary ---
Validation Loss: 0.068614
Epoch Time: 0:00:03

2026-05-17 07:09:13 [INFO] [Epoch 50/50, Step 540/11] LR: 0.000002, Loss: -0.3008
2026-05-17 07:09:16 [INFO] [Epoch 50/50, Step 550/11] LR: 0.000000, Loss: -0.3003
2026-05-17 07:09:16 [INFO] 
--- Epoch 50/50 Summary ---
Validation Loss: 0.068614
Epoch Time: 0:00:03


```

---

### `finetuned/1d_t4_hpo/tokenizer/best_model/config.json`

```json
{
  "attn_dropout_p": 0.0,
  "beta": 0.05,
  "d_in": 6,
  "d_model": 256,
  "ff_dim": 512,
  "ffn_dropout_p": 0.0,
  "gamma": 1.1,
  "gamma0": 1.0,
  "group_size": 5,
  "n_dec_layers": 4,
  "n_enc_layers": 4,
  "n_heads": 4,
  "resid_dropout_p": 0.0,
  "s1_bits": 10,
  "s2_bits": 10,
  "zeta": 0.05
}
```

---

### `finetuned/1d_t4_hpo/tokenizer/best_model/model.safetensors`

```text
[File too large to display: 15.1 MB]
```

---

### `finetuned/1d_t4_hpo/tokenizer/best_model/README.md`

```markdown
---
tags:
- model_hub_mixin
- pytorch_model_hub_mixin
---

This model has been pushed to the Hub using the [PytorchModelHubMixin](https://huggingface.co/docs/huggingface_hub/package_reference/mixins#huggingface_hub.PyTorchModelHubMixin) integration:
- Code: [More Information Needed]
- Paper: [More Information Needed]
- Docs: [More Information Needed]
```

---

### `finetuned/tests/model.zip`

```text
[Binary file — content not displayed]
```

---

### `model/__init__.py`

```python
from .nos import NosTokenizer, Nos, NosPredictor

model_dict = {
    'nos_tokenizer': NosTokenizer,
    'nos': Nos,
    'nos_predictor': NosPredictor
}


def get_model_class(model_name):
    if model_name in model_dict:
        return model_dict[model_name]
    else:
        print(f"Model {model_name} not found in model_dict")
        raise NotImplementedError



```

---

### `model/module.py`

```python
import math

from einops import rearrange, reduce
from typing import Tuple
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F

class DifferentiableEntropyFunction(Function):
    @staticmethod
    def forward(ctx, zq, basis, K, eps):
        # FORCE FP32: Prevents eps=1e-8 from rounding to 0.0 in autocast (FP16),
        # which would otherwise cause torch.log(0.0) -> NaNs.
        zq_32 = zq.float()
        zb = (zq_32 + 1) / 2
        zi = ((zb * basis).sum(-1)).to(torch.int64)

        # Initialize counts and compute probabilities entirely in float32
        cnt = torch.scatter_reduce(
            torch.zeros(2 ** K, device=zq.device, dtype=torch.float32),
            0,
            zi.flatten(),
            torch.ones_like(zi.flatten(), dtype=torch.float32),
            'sum'
        )
        
        prob = (cnt + eps) / (cnt + eps).sum()
        H = -(prob * torch.log(prob)).sum()
        
        # Save tensors needed for backward. prob is already float32.
        ctx.save_for_backward(zq, zi, prob)
        ctx.K = K
        
        # Return loss cast back to the original mixed-precision dtype
        return H.to(zq.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        zq, zi, prob = ctx.saved_tensors
        
        # Force incoming gradients to FP32 for numerical stability
        grad_output_32 = grad_output.float()
        
        grad_array = -grad_output_32 * (torch.log(prob) + 1) / zi.numel() / ctx.K
        reord_grad = grad_array[zi.flatten()].reshape(zi.shape)
        
        # Compute final input gradient and cast back to original dtype
        grad_input = (reord_grad.unsqueeze(-1) * zq.float()).to(zq.dtype)
        
        # Return gradients for (zq, basis, K, eps)
        return grad_input, None, None, None


def codebook_entropy(zq, basis, K, eps=1e-4):
    return DifferentiableEntropyFunction.apply(zq, basis, K, eps)


class BinarySphericalQuantizer(nn.Module):
    def __init__(self, embed_dim, beta, gamma0, gamma, zeta,
                 input_format='bchw',
                 soft_entropy=True, group_size=9,
                 persample_entropy_compute='analytical',
                 cb_entropy_compute='group',
                 l2_norm=True,
                 inv_temperature=1):
        """
        Paper link: https://arxiv.org/pdf/2406.07548.pdf
        Here we use the official implementation of the BinarySphericalQuantizer.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.beta = beta  # loss weight for commit loss
        self.gamma0 = gamma0  # loss weight for entropy penalty
        self.gamma = gamma  # loss weight for entropy penalty
        self.zeta = zeta  # loss weight for entire entropy penalty
        self.input_format = input_format
        assert self.embed_dim % group_size == 0, "embed_dim must be divisible by group_size"
        self.num_groups = self.embed_dim // group_size
        self.group_size = group_size
        assert persample_entropy_compute in ['group', 'analytical'], "persample_entropy_compute must be either 'group' or 'analytical'"
        assert cb_entropy_compute in ['group', 'nce'], "cb_entropy_compute must be either 'group' or 'nce'"
        self.persample_entropy_compute = persample_entropy_compute
        self.cb_entropy_compute = cb_entropy_compute
        self.l2_norm = l2_norm
        self.inv_temperature = inv_temperature

        self.register_buffer('basis', 2 ** torch.arange(embed_dim - 1, -1, -1))
        self.register_buffer('group_basis', 2 ** torch.arange(group_size - 1, -1, -1))

        self.num_dimensions = 2 ** embed_dim
        self.bits_per_index = embed_dim

        # we only need to keep the codebook portion up to the group size
        # because we approximate the H loss with this subcode
        group_codes = torch.arange(2 ** self.group_size)
        group_codebook = self.indexes_to_codes(group_codes).float()[:, -group_size:]
        self.register_buffer('group_codebook', group_codebook, persistent=False)

        self.soft_entropy = soft_entropy  # soft_entropy: Sec 3.2 of https://arxiv.org/pdf/1911.05894.pdf

    def quantize(self, z):
        assert z.shape[-1] == self.embed_dim, f"Expected {self.embed_dim} dimensions, got {z.shape[-1]}"

        zhat = torch.where(z > 0,
                           torch.tensor(1, dtype=z.dtype, device=z.device),
                           torch.tensor(-1, dtype=z.dtype, device=z.device))
        return z + (zhat - z).detach()

    def forward(self, z):
        zq = self.quantize(z)

        indices = self.codes_to_indexes(zq.detach())
        group_indices = self.codes_to_group_indexes(zq.detach())
        if not self.training:
            used_codes = torch.unique(indices, return_counts=False)
        else:
            used_codes = None

        q_scale = 1. / (self.embed_dim ** 0.5) if self.l2_norm else 1.

        if self.soft_entropy:
            persample_entropy, cb_entropy, avg_prob = self.soft_entropy_loss(z)
            entropy_penalty = self.gamma0 * persample_entropy - self.gamma * cb_entropy
        else:
            zb_by_sample = ((zq + 1) / 2).reshape(z.shape[0], -1, z.shape[-1]).to(torch.float32)
            persample_entropy = self.get_hard_per_sample_entropy(zb_by_sample)
            cb_entropy = codebook_entropy(zq, self.basis, self.embed_dim)
            entropy_penalty = self.gamma0 * persample_entropy - self.gamma * cb_entropy

        zq = zq * q_scale

        # commit loss
        commit_loss = self.beta * torch.mean(((zq.detach() - z) ** 2).sum(dim=-1))

        return (
            zq,
            commit_loss + self.zeta * entropy_penalty / self.inv_temperature,
            {"H": cb_entropy, "used_codes": used_codes, "indices": indices, "group_indices": group_indices,
             "avg_prob": avg_prob}
        )

    def soft_entropy_loss(self, z):
        group_code_book = self.group_codebook / (self.embed_dim ** 0.5 if self.l2_norm else 1)
        divided_z = rearrange(z, '... (g c) -> ... g c', c=self.group_size)

        distance = - 2 * torch.einsum('... g c, d c ->... g d', divided_z, group_code_book)
        prob = (-distance * self.inv_temperature).softmax(dim=-1)
        if self.persample_entropy_compute == 'analytical':
            if self.l2_norm:
                p = torch.sigmoid(-4 * z / (self.embed_dim ** 0.5) * self.inv_temperature)
            else:
                p = torch.sigmoid(-4 * z * self.inv_temperature)
            prob = torch.stack([p, 1 - p], dim=-1)
            per_sample_entropy = self.get_entropy(prob, dim=-1, normalize=False).sum(dim=-1).mean()
        else:
            per_sample_entropy = self.get_entropy(prob, dim=-1, normalize=False).sum(dim=-1).mean()

        avg_prob = reduce(prob, '... g d ->g d', 'mean')
        codebook_entropy = self.get_entropy(avg_prob, dim=-1, normalize=False)

        return per_sample_entropy, codebook_entropy.sum(), avg_prob

    def get_hard_per_sample_entropy(self, zb_by_sample):
        probs_per_dim = zb_by_sample.sum(1) / zb_by_sample.shape[1]
        persample_entropy = - probs_per_dim * torch.log(probs_per_dim + 1e-8) - (1 - probs_per_dim) * torch.log(1 - probs_per_dim + 1e-8)
        persample_entropy = persample_entropy.sum(-1)
        return persample_entropy.mean()

    def codes_to_indexes(self, zhat):
        assert zhat.shape[-1] == self.embed_dim, f"Expected {self.embed_dim} dimensions, got {zhat.shape[-1]}"
        return ((zhat + 1) / 2 * self.basis).sum(axis=-1).to(torch.int64)

    def codes_to_group_indexes(self, zhat):
        zhat_in_group = rearrange(zhat, 'b ... (g c) -> b ... g c', c=self.group_size)
        return ((zhat_in_group + 1) / 2 * self.group_basis).sum(axis=-1).to(torch.int64)

    def indexes_to_codes(self, indices):
        indices = indices.unsqueeze(-1)
        codes_non_centered = torch.remainder(
            torch.floor_divide(indices, self.basis), 2
        )
        return codes_non_centered * 2 - 1

    def group_indexes_to_codes(self, group_indices):
        group_indices = group_indices.unsqueeze(-1)
        codes_non_centered = torch.remainder(
            torch.floor_divide(group_indices, self.group_basis), 2
        )
        codes_non_centered = rearrange(codes_non_centered, 'b ... g c -> b ... (g c)')
        return codes_non_centered * 2 - 1

    def get_entropy(self, count, dim=-1, eps=1e-4, normalize=True):
        if normalize:
            probs = (count + eps) / (count + eps).sum(dim=dim, keepdim=True)
        else:
            probs = count
        H = -(probs * torch.log(probs + 1e-8)).sum(dim=dim)
        return H

    def get_group_codebook_entry(self, group_indices):
        z_q = self.group_indexes_to_codes(group_indices)
        q_scale = 1. / (self.embed_dim ** 0.5) if self.l2_norm else 1.
        z_q = z_q * q_scale
        if self.input_format == 'bchw':
            h, w = int(z_q.shape[1] ** 0.5)
            assert h * w == z_q.shape[1], 'Invalid sequence length'
            z_q = rearrange(z_q, 'b (h w) c -> b c h w', h=h)
        return z_q

    def get_codebook_entry(self, indices):
        z_q = self.indexes_to_codes(indices)
        q_scale = 1. / (self.embed_dim ** 0.5) if self.l2_norm else 1.
        z_q = z_q * q_scale
        if self.input_format == 'bchw':
            h, w = int(z_q.shape[1] ** 0.5)
            assert h * w == z_q.shape[1], 'Invalid sequence length'
            z_q = rearrange(z_q, 'b (h w) c -> b c h w', h=h)
        return z_q


class BSQuantizer(nn.Module):
    def __init__(self, s1_bits, s2_bits, beta, gamma0, gamma, zeta, group_size):
        super().__init__()
        self.codebook_dim = s1_bits + s2_bits
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.bsq = BinarySphericalQuantizer(self.codebook_dim, beta, gamma0, gamma, zeta, group_size=group_size)

    def bits_to_indices(self, bits):
        bits = (bits >= 0).to(torch.long)
        indices = 2 ** torch.arange(
            0,
            bits.shape[-1],
            1,
            dtype=torch.long,
            device=bits.device,
        )
        return (bits * indices).sum(-1)

    def forward(self, z, half=False):
        z = F.normalize(z, dim=-1)
        quantized, bsq_loss, metrics = self.bsq(z)
        if half:
            q_pre = quantized[:, :, :self.s1_bits]
            q_post = quantized[:, :, self.s1_bits:]
            z_indices = [self.bits_to_indices(q_pre), self.bits_to_indices(q_post)]
        else:
            z_indices = self.bits_to_indices(quantized)
        return bsq_loss, quantized, z_indices


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(self, d_model, ff_dim, ffn_dropout_p=0.0):
        super().__init__()

        self.w1 = nn.Linear(d_model, ff_dim, bias=False)
        self.w3 = nn.Linear(d_model, ff_dim, bias=False)
        self.w2 = nn.Linear(ff_dim, d_model, bias=False)
        self.ffn_dropout = nn.Dropout(ffn_dropout_p)

    def forward(self, x):
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(
                f"RotaryPositionalEmbedding requires an even head dimension. "
                f"Got dim={dim}."
            )
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=True)

        self.register_buffer(
            "cos_cached", torch.empty(0, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "sin_cached", torch.empty(0, dtype=torch.float32), persistent=False
        )
        self._seq_len_cached: int = 0

    def _update_cos_sin_cache(
        self, x: torch.Tensor, seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len != self._seq_len_cached:
            self._seq_len_cached = seq_len

            t = torch.arange(
                seq_len,
                device=self.inv_freq.device,
                dtype=self.inv_freq.dtype,
            )
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)

            self.cos_cached = emb.cos()[None, None, :, :].to(dtype=x.dtype)
            self.sin_cached = emb.sin()[None, None, :, :].to(dtype=x.dtype)

        return self.cos_cached, self.sin_cached

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        q_offset: int = 0  # <--- NEW: Offset parameter for AR generation
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_len = q.shape[-2]
        k_len = k.shape[-2]
        
        # Cache must be large enough to cover the absolute maximum position needed
        max_len = max(q_len + q_offset, k_len)
        cos, sin = self._update_cos_sin_cache(q, max_len)

        # Slice cache for Query based on its precise temporal offset
        q_cos = cos[:, :, q_offset : q_offset + q_len, :]
        q_sin = sin[:, :, q_offset : q_offset + q_len, :]
        
        # Slice cache for Key based on its full length (always starts at t=0)
        k_cos = cos[:, :, :k_len, :]
        k_sin = sin[:, :, :k_len, :]

        rotated_q = (q * q_cos) + (self._rotate_half(q) * q_sin)
        rotated_k = (k * k_cos) + (self._rotate_half(k) * k_sin)
        return rotated_q, rotated_k


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float = None,
    training: bool = True
) -> torch.Tensor:
    """
    Delegates directly to PyTorch 2.0+ optimized C++ kernels.
    Resolves O(N^2) memory scaling by avoiding explicit attention matrix materialization.
    """
    effective_dropout = dropout_p if training else 0.0

    # Fast-path for causal attention without custom masks
    if is_causal and attn_mask is None:
        return F.scaled_dot_product_attention(
            query, key, value,
            dropout_p=effective_dropout,
            is_causal=True,
            scale=scale
        )

    # Handle custom masks
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            # F.sdpa expects True for elements that *are* allowed to attend.
            # Standard Transformer convention usually uses True to mask *out*.
            # Invert the boolean mask to match F.sdpa's expectation.
            attn_mask = ~attn_mask

        # FIXED: Prevent Causal Annihilation
        # Combine the padding mask with a causal mask if the layer is autoregressive.
        if is_causal:
            q_len, k_len = query.size(-2), key.size(-2)
            # Create a causal mask (True where attention is allowed)
            causal_mask = torch.tril(
                torch.ones((q_len, k_len), dtype=torch.bool, device=query.device)
            )

            if attn_mask.dtype == torch.bool:
                # Logical AND: A token can be attended to only if BOTH masks allow it
                attn_mask = attn_mask & causal_mask
            else:
                # Mixed-precision safe: use lowest finite value for active dtype
                # instead of hard -inf to prevent NaN in FP16/BF16 softmax backward
                mask_value = torch.finfo(query.dtype).min
                attn_mask = attn_mask.to(query.dtype).masked_fill(~causal_mask, mask_value)

    # Determine final is_causal flag:
    # - If attn_mask is None and is_causal=True, use PyTorch's built-in causal handling (fast path at line 368 already handled this)
    # - If attn_mask was provided, we combined it with causal mask above, so use is_causal=False since causal is baked into attn_mask
    # - If is_causal=False, ensure causal is disabled
    final_is_causal = is_causal and attn_mask is None

    return F.scaled_dot_product_attention(
        query, key, value,
        attn_mask=attn_mask,
        dropout_p=effective_dropout,
        is_causal=final_is_causal,
        scale=scale
    )

class MultiHeadAttentionWithRoPE(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout_p=0.0, resid_dropout_p=0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rotary = RotaryPositionalEmbedding(self.head_dim)
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout = nn.Dropout(resid_dropout_p)

    def forward(self, x, key_padding_mask=None):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Self-attention always processes aligned sequences, so q_offset remains 0
        q, k = self.rotary(q, k)

        if key_padding_mask is not None:
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            attn_mask = attn_mask.expand(-1, self.n_heads, seq_len, -1)  # [batch, n_heads, q_len, k_len]
        else:
            attn_mask = None

        attn_output = scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout_p,
            is_causal=True,
            training=self.training
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.resid_dropout(self.out_proj(attn_output))


class MultiHeadCrossAttentionWithRoPE(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout_p=0.0, resid_dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rotary = RotaryPositionalEmbedding(self.head_dim)
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout = nn.Dropout(resid_dropout)

    def forward(self, query, key, value, key_padding_mask=None):
        batch_size, q_len, _ = query.shape
        _, seq_len, _ = key.shape

        q = self.q_proj(query).view(batch_size, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # <--- NEW: Dynamically infer the offset based on sequence lengths
        q_offset = seq_len - q_len if q_len < seq_len else 0
        q, k = self.rotary(q, k, q_offset=q_offset)

        if key_padding_mask is not None:
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.expand(-1, self.n_heads, q_len, -1)
        else:
            attn_mask = None

        is_causal_flag = q_len > 1

        attn_output = scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout_p,
            is_causal=is_causal_flag,
            training=self.training
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        return self.resid_dropout(self.out_proj(attn_output))

class HierarchicalEmbedding(nn.Module):
    def __init__(self, s1_bits, s2_bits, d_model=256):
        super().__init__()
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits

        vocab_s1 = 2 ** s1_bits
        vocab_s2 = 2 ** s2_bits

        self.emb_s1 = nn.Embedding(vocab_s1, d_model)
        self.emb_s2 = nn.Embedding(vocab_s2, d_model)
        self.d_model = d_model
        self.fusion_proj = nn.Linear(d_model * 2, d_model)

        nn.init.normal_(self.emb_s1.weight, mean=0, std=d_model ** -0.5)
        nn.init.normal_(self.emb_s2.weight, mean=0, std=d_model ** -0.5)

    def forward(self, token_ids):
        if isinstance(token_ids, tuple) or isinstance(token_ids, list):
            s1_ids, s2_ids = token_ids
        else:
            s1_ids, s2_ids = self.split_token(token_ids, self.s2_bits)
        s1_emb = self.emb_s1(s1_ids) * math.sqrt(self.d_model)
        s2_emb = self.emb_s2(s2_ids) * math.sqrt(self.d_model)
        return self.fusion_proj(torch.cat([s1_emb, s2_emb], dim=-1))


class DependencyAwareLayer(nn.Module):
    def __init__(self, d_model, n_heads=4, attn_dropout_p=0.0, resid_dropout=0.0):
        super().__init__()
        self.cross_attn = MultiHeadCrossAttentionWithRoPE(d_model, n_heads, attn_dropout_p, resid_dropout)
        self.norm = RMSNorm(d_model)

    def forward(self, hidden_states, sibling_embed, key_padding_mask=None):
        attn_out = self.cross_attn(
            query=sibling_embed,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=key_padding_mask
        )
        
        # <--- FIXED: Slice residual to match query length dynamically
        q_len = sibling_embed.shape[1]
        residual = hidden_states[:, -q_len:, :]
        
        return self.norm(residual + attn_out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim=1024, ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.self_attn = MultiHeadAttentionWithRoPE(d_model, n_heads, attn_dropout_p, resid_dropout_p)
        self.norm2 = RMSNorm(d_model)
        self.ffn = FeedForward(d_model, ff_dim, ffn_dropout_p)

    def forward(self, x, key_padding_mask=None):
        residual = x
        x = self.norm1(x)
        attn_out = self.self_attn(x, key_padding_mask=key_padding_mask)
        x = residual + attn_out

        residual = x
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        x = residual + ffn_out
        return x


class DualHead(nn.Module):
    def __init__(self, s1_bits, s2_bits, d_model):
        super().__init__()
        self.vocab_s1 = 2 ** s1_bits
        self.vocab_s2 = 2 ** s2_bits
        self.proj_s1 = nn.Linear(d_model, self.vocab_s1)
        self.proj_s2 = nn.Linear(d_model, self.vocab_s2)

    def compute_loss(self, s1_logits, s2_logits, s1_targets, s2_targets, padding_mask=None):
        if padding_mask is not None:
            valid_mask = (padding_mask == 0)
            s1_logits = s1_logits[valid_mask]
            s2_logits = s2_logits[valid_mask]
            s1_targets = s1_targets[valid_mask]
            s2_targets = s2_targets[valid_mask]
            ce_s1 = F.cross_entropy(s1_logits, s1_targets)
            ce_s2 = F.cross_entropy(s2_logits, s2_targets)
        else:
            ce_s1 = F.cross_entropy(s1_logits.reshape(-1, self.vocab_s1), s1_targets.reshape(-1))
            ce_s2 = F.cross_entropy(s2_logits.reshape(-1, self.vocab_s2), s2_targets.reshape(-1))
        ce_loss = (ce_s1 + ce_s2) / 2
        return ce_loss, ce_s1, ce_s2

    def forward(self, x):
        return self.proj_s1(x)

    def cond_forward(self, x2):
        return self.proj_s2(x2)


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, learn_pe):
        super(TemporalEmbedding, self).__init__()

        minute_size = 60
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if not learn_pe else nn.Embedding
        self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 0])
        hour_x = self.hour_embed(x[:, :, 1])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 3])
        month_x = self.month_embed(x[:, :, 4])

        return hour_x + weekday_x + day_x + month_x + minute_x
```

---

### `model/nos.py`

```python
import numpy as np
import pandas as pd
import torch
from huggingface_hub import PyTorchModelHubMixin
import sys

from tqdm import trange

sys.path.append("../")
from model.module import *


class NosTokenizer(nn.Module, PyTorchModelHubMixin):
    """
    NosTokenizer module for tokenizing input data using a hybrid quantization approach.

    This tokenizer utilizes a combination of encoder and decoder Transformer blocks
    along with the Binary Spherical Quantization (BSQuantizer) to compress and decompress input data.

    Args:
           d_in (int): Input dimension.
           d_model (int): Model dimension.
           n_heads (int): Number of attention heads.
           ff_dim (int): Feed-forward dimension.
           n_enc_layers (int): Number of encoder layers.
           n_dec_layers (int): Number of decoder layers.
           ffn_dropout_p (float): Dropout probability for feed-forward networks.
           attn_dropout_p (float): Dropout probability for attention mechanisms.
           resid_dropout_p (float): Dropout probability for residual connections.
           s1_bits (int): Number of bits for the pre token in BSQuantizer.
           s2_bits (int): Number of bits for the post token in BSQuantizer.
           beta (float): Beta parameter for BSQuantizer.
           gamma0 (float): Gamma0 parameter for BSQuantizer.
           gamma (float): Gamma parameter for BSQuantizer.
           zeta (float): Zeta parameter for BSQuantizer.
           group_size (int): Group size parameter for BSQuantizer.

    """

    def __init__(self, d_in, d_model, n_heads, ff_dim, n_enc_layers, n_dec_layers, ffn_dropout_p, attn_dropout_p, resid_dropout_p, s1_bits, s2_bits, beta, gamma0, gamma, zeta, group_size):

        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.enc_layers = n_enc_layers
        self.dec_layers = n_dec_layers
        self.ffn_dropout_p = ffn_dropout_p
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout_p = resid_dropout_p

        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.codebook_dim = s1_bits + s2_bits # Total dimension of the codebook after quantization
        self.embed = nn.Linear(self.d_in, self.d_model)
        self.head = nn.Linear(self.d_model, self.d_in)

        # Encoder Transformer Blocks
        self.encoder = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, self.ff_dim, self.ffn_dropout_p, self.attn_dropout_p, self.resid_dropout_p)
            for _ in range(self.enc_layers - 1)
        ])
        # Decoder Transformer Blocks
        self.decoder = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, self.ff_dim, self.ffn_dropout_p, self.attn_dropout_p, self.resid_dropout_p)
            for _ in range(self.dec_layers - 1)
        ])
        self.quant_embed = nn.Linear(in_features=self.d_model, out_features=self.codebook_dim) # Linear layer before quantization
        self.post_quant_embed_pre = nn.Linear(in_features=self.s1_bits, out_features=self.d_model) # Linear layer after quantization (pre part - s1 bits)
        self.post_quant_embed = nn.Linear(in_features=self.codebook_dim, out_features=self.d_model) # Linear layer after quantization (full codebook)
        self.tokenizer = BSQuantizer(self.s1_bits, self.s2_bits, beta, gamma0, gamma, zeta, group_size) # BSQuantizer module

    def forward(self, x):
        """
        Forward pass of the NosTokenizer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_in).

        Returns:
            tuple: A tuple containing:
                - tuple: (z_pre, z) - Reconstructed outputs from decoder with s1_bits and full codebook respectively,
                         both of shape (batch_size, seq_len, d_in).
                - torch.Tensor: bsq_loss - Loss from the BSQuantizer.
                - torch.Tensor: quantized - Quantized representation from BSQuantizer.
                - torch.Tensor: z_indices - Indices from the BSQuantizer.
        """
        z = self.embed(x)

        for layer in self.encoder:
            z = layer(z)

        z = self.quant_embed(z) # (B, T, codebook)

        bsq_loss, quantized, z_indices = self.tokenizer(z)

        quantized_pre = quantized[:, :, :self.s1_bits] # Extract the first part of quantized representation (s1_bits)
        z_pre = self.post_quant_embed_pre(quantized_pre)

        z = self.post_quant_embed(quantized)

        # Decoder layers (for pre part - s1 bits)
        for layer in self.decoder:
            z_pre = layer(z_pre)
        z_pre = self.head(z_pre)

        # Decoder layers (for full codebook)
        for layer in self.decoder:
            z = layer(z)
        z = self.head(z)

        return (z_pre, z), bsq_loss, quantized, z_indices

    def indices_to_bits(self, x, half=False):
        """
        Converts indices to bit representations and scales them.

        Args:
            x (torch.Tensor): Indices tensor.
            half (bool, optional): Whether to process only half of the codebook dimension. Defaults to False.

        Returns:
            torch.Tensor: Bit representation tensor.
        """
        if half:
            x1 = x[0] # Assuming x is a tuple of indices if half is True
            x2 = x[1]
            mask1 = 2 ** torch.arange(self.s1_bits, device=x1.device, dtype=torch.long)
            mask2 = 2 ** torch.arange(self.s2_bits, device=x2.device, dtype=torch.long)
            x1 = (x1.unsqueeze(-1) & mask1) != 0 
            x2 = (x2.unsqueeze(-1) & mask2) != 0 
            x = torch.cat([x1, x2], dim=-1) # Concatenate the bit representations
        else:
            mask = 2 ** torch.arange(self.codebook_dim, device=x.device, dtype=torch.long) # Create a mask for bit extraction
            x = (x.unsqueeze(-1) & mask) != 0 # Extract bits

        x = x.float() * 2 - 1 # Convert boolean to bipolar (-1, 1)
        q_scale = 1. / (self.codebook_dim ** 0.5) # Scaling factor
        x = x * q_scale
        return x

    def encode(self, x, half=False):
        """
        Encodes the input data into quantized indices.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_in).
            half (bool, optional): Whether to use half quantization in BSQuantizer. Defaults to False.

        Returns:
            torch.Tensor: Quantized indices from BSQuantizer.
        """
        z = self.embed(x)
        for layer in self.encoder:
            z = layer(z)
        z = self.quant_embed(z)

        bsq_loss, quantized, z_indices = self.tokenizer(z, half)
        return z_indices

    def decode(self, x, half=False):
        """
        Decodes quantized indices back to the input data space.

        Args:
            x (torch.Tensor): Quantized indices tensor.
            half (bool, optional): Whether the indices were generated with half quantization. Defaults to False.

        Returns:
            torch.Tensor: Reconstructed output tensor of shape (batch_size, seq_len, d_in).
        """
        quantized = self.indices_to_bits(x, half)
        z = self.post_quant_embed(quantized)
        for layer in self.decoder:
            z = layer(z)
        z = self.head(z)
        return z


class Nos(nn.Module, PyTorchModelHubMixin):
    """
    Nos Model.

    Args:
        s1_bits (int): Number of bits for pre tokens.
        s2_bits (int): Number of bits for post tokens.
        n_layers (int): Number of Transformer blocks.
        d_model (int): Dimension of the model's embeddings and hidden states.
        n_heads (int): Number of attention heads in the MultiheadAttention layers.
        ff_dim (int): Dimension of the feedforward network in the Transformer blocks.
        ffn_dropout_p (float): Dropout probability for the feedforward network.
        attn_dropout_p (float): Dropout probability for the attention layers.
        resid_dropout_p (float): Dropout probability for residual connections.
        token_dropout_p (float): Dropout probability for token embeddings.
        learn_te (bool): Whether to use learnable temporal embeddings.
    """

    def __init__(self, s1_bits, s2_bits, n_layers, d_model, n_heads, ff_dim, ffn_dropout_p, attn_dropout_p, resid_dropout_p, token_dropout_p, learn_te):
        super().__init__()
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.learn_te = learn_te
        self.ff_dim = ff_dim
        self.ffn_dropout_p = ffn_dropout_p
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout_p = resid_dropout_p
        self.token_dropout_p = token_dropout_p

        self.s1_vocab_size = 2 ** self.s1_bits
        self.token_drop = nn.Dropout(self.token_dropout_p)
        self.embedding = HierarchicalEmbedding(self.s1_bits, self.s2_bits, self.d_model)
        self.time_emb = TemporalEmbedding(self.d_model, self.learn_te)
        self.transformer = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, self.ff_dim, self.ffn_dropout_p, self.attn_dropout_p, self.resid_dropout_p)
            for _ in range(self.n_layers)
        ])
        self.norm = RMSNorm(self.d_model)
        self.dep_layer = DependencyAwareLayer(self.d_model, n_heads=self.n_heads) # <--- FIXED
        self.head = DualHead(self.s1_bits, self.s2_bits, self.d_model)

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=self.embedding.d_model ** -0.5)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(self, s1_ids, s2_ids, stamp=None, padding_mask=None, use_teacher_forcing=False, s1_targets=None, gumbel_tau=1.0):
        """
        Args:
            s1_ids (torch.Tensor): Input tensor of s1 token IDs. Shape: [batch_size, seq_len]
            s2_ids (torch.Tensor): Input tensor of s2 token IDs. Shape: [batch_size, seq_len]
            stamp (torch.Tensor, optional): Temporal stamp tensor. Shape: [batch_size, seq_len]. Defaults to None.
            padding_mask (torch.Tensor, optional): Mask for padding tokens. Shape: [batch_size, seq_len]. Defaults to None.
            use_teacher_forcing (bool, optional): Whether to use teacher forcing for s1 decoding. Defaults to False.
            s1_targets (torch.Tensor, optional): Target s1 token IDs for teacher forcing. Shape: [batch_size, seq_len]. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - s1 logits: Logits for s1 token predictions. Shape: [batch_size, seq_len, s1_vocab_size]
                - s2_logits: Logits for s2 token predictions, conditioned on s1. Shape: [batch_size, seq_len, s2_vocab_size]
        """
        x = self.embedding([s1_ids, s2_ids])
        if stamp is not None:
            time_embedding = self.time_emb(stamp)
            x = x + time_embedding
        x = self.token_drop(x)

        for layer in self.transformer:
            x = layer(x, key_padding_mask=padding_mask)

        x = self.norm(x)

        s1_logits = self.head(x)

        if use_teacher_forcing and s1_targets is not None:
            # State 1: Explicit Teacher Forcing (Standard for early training/finetuning)
            sibling_embed = self.embedding.emb_s1(s1_targets)
        elif self.training:
            # State 2: Autoregressive Training (Maintains gradient graph)
            # Uses Straight-Through Gumbel-Softmax so S2 can backpropagate into S1
            s1_probs = F.gumbel_softmax(s1_logits, tau=gumbel_tau, hard=True)
            sibling_embed = torch.matmul(s1_probs, self.embedding.emb_s1.weight)
        else:
            # State 3: Standard Inference (Detached, memory-efficient)
            s1_probs = F.softmax(s1_logits.detach() / 1.0, dim=-1) # Can inject temp here later
            sample_s1_ids = torch.multinomial(s1_probs.view(-1, self.s1_vocab_size), 1).view(s1_ids.shape)
            sibling_embed = self.embedding.emb_s1(sample_s1_ids)

        x2 = self.dep_layer(x, sibling_embed, key_padding_mask=padding_mask) # Dependency Aware Layer: Condition on s1 embeddings
        s2_logits = self.head.cond_forward(x2)
        return s1_logits, s2_logits

    def decode_s1(self, s1_ids, s2_ids, stamp=None, padding_mask=None):
        """
        Decodes only the s1 tokens.

        This method performs a forward pass to predict only s1 tokens. It returns the s1 logits
        and the context representation from the Transformer, which can be used for subsequent s2 decoding.

        Args:
            s1_ids (torch.Tensor): Input tensor of s1 token IDs. Shape: [batch_size, seq_len]
            s2_ids (torch.Tensor): Input tensor of s2 token IDs. Shape: [batch_size, seq_len]
            stamp (torch.Tensor, optional): Temporal stamp tensor. Shape: [batch_size, seq_len]. Defaults to None.
            padding_mask (torch.Tensor, optional): Mask for padding tokens. Shape: [batch_size, seq_len]. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - s1 logits: Logits for s1 token predictions. Shape: [batch_size, seq_len, s1_vocab_size]
                - context: Context representation from the Transformer. Shape: [batch_size, seq_len, d_model]
        """
        x = self.embedding([s1_ids, s2_ids])
        if stamp is not None:
            time_embedding = self.time_emb(stamp)
            x = x + time_embedding
        x = self.token_drop(x)

        for layer in self.transformer:
            x = layer(x, key_padding_mask=padding_mask)

        x = self.norm(x)

        s1_logits = self.head(x)
        return s1_logits, x

    def decode_s2(self, context, s1_ids, padding_mask=None):
        """
        Decodes the s2 tokens, conditioned on the context and s1 tokens.

        This method decodes s2 tokens based on a pre-computed context representation (typically from `decode_s1`)
        and the s1 token IDs. It uses the dependency-aware layer and the conditional s2 head to predict s2 tokens.

        Args:
            context (torch.Tensor): Context representation from the transformer (output of decode_s1).
                                     Shape: [batch_size, seq_len, d_model]
            s1_ids (torch.torch.Tensor): Input tensor of s1 token IDs. Shape: [batch_size, seq_len]
            padding_mask (torch.Tensor, optional): Mask for padding tokens. Shape: [batch_size, seq_len]. Defaults to None.

        Returns:
            torch.Tensor: s2 logits. Shape: [batch_size, seq_len, s2_vocab_size]
        """
        sibling_embed = self.embedding.emb_s1(s1_ids)
        x2 = self.dep_layer(context, sibling_embed, key_padding_mask=padding_mask)
        return self.head.cond_forward(x2)


def top_k_top_p_filtering(
        logits,
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
        return logits

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
        return logits


def sample_from_logits(logits, temperature=1.0, top_k=None, top_p=None, sample_logits=True):
    logits = logits / temperature
    if top_k is not None or top_p is not None:
        if top_k > 0 or top_p < 1.0:
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

    probs = F.softmax(logits, dim=-1)

    if not sample_logits:
        _, x = top_k(probs, k=1, dim=-1)
    else:
        x = torch.multinomial(probs, num_samples=1)

    return x


def auto_regressive_inference(tokenizer, model, x, x_stamp, y_stamp, max_context, pred_len, clip=5, T=1.0, top_k=0, top_p=0.99, sample_count=5, verbose=False):
    with torch.no_grad():
        batch_size = x.size(0)
        initial_seq_len = x.size(1)
        x = torch.clip(x, -clip, clip)

        device = x.device
        x = x.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x.size(1), x.size(2)).to(device)
        x_stamp = x_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x_stamp.size(1), x_stamp.size(2)).to(device)
        y_stamp = y_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, y_stamp.size(1), y_stamp.size(2)).to(device)

        x_token = tokenizer.encode(x, half=True)

        def get_dynamic_stamp(x_stamp, y_stamp, current_seq_len, pred_step):

            if current_seq_len <= max_context - pred_step:
                return torch.cat([x_stamp, y_stamp[:, :pred_step, :]], dim=1)
            else:
                start_idx = max_context - pred_step
                return torch.cat([x_stamp[:, -start_idx:, :], y_stamp[:, :pred_step, :]], dim=1)

        if verbose:
            ran = trange
        else:
            ran = range
        for i in ran(pred_len):
            current_seq_len = initial_seq_len + i

            if current_seq_len <= max_context:
                input_tokens = x_token
            else:
                input_tokens = [t[:, -max_context:].contiguous() for t in x_token]

            current_stamp = get_dynamic_stamp(x_stamp, y_stamp, current_seq_len, i)

            s1_logits, context = model.decode_s1(input_tokens[0], input_tokens[1], current_stamp)
            s1_logits = s1_logits[:, -1, :]
            sample_pre = sample_from_logits(s1_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)

            s2_logits = model.decode_s2(context, sample_pre)
            s2_logits = s2_logits[:, -1, :]
            sample_post = sample_from_logits(s2_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)

            x_token[0] = torch.cat([x_token[0], sample_pre], dim=1)
            x_token[1] = torch.cat([x_token[1], sample_post], dim=1)

            torch.cuda.empty_cache()

        input_tokens = [t[:, -max_context:].contiguous() for t in x_token]
        z = tokenizer.decode(input_tokens, half=True)
        z = z.reshape(batch_size, sample_count, z.size(1), z.size(2))
        preds = z.cpu().numpy()
        preds = np.mean(preds, axis=1)

        return preds


def calc_time_stamps(x_timestamp):
    time_df = pd.DataFrame()
    time_df['minute'] = x_timestamp.dt.minute
    time_df['hour'] = x_timestamp.dt.hour
    time_df['weekday'] = x_timestamp.dt.weekday
    time_df['day'] = x_timestamp.dt.day
    time_df['month'] = x_timestamp.dt.month
    return time_df


class NosPredictor:

    def __init__(self, model, tokenizer, device="cuda:0", max_context=512, clip=5):
        self.tokenizer = tokenizer
        self.model = model
        self.max_context = max_context
        self.clip = clip
        self.price_cols = ['open', 'high', 'low', 'close']
        self.vol_col = 'volume'
        self.amt_vol = 'amount'
        self.time_cols = ['minute', 'hour', 'weekday', 'day', 'month']
        self.device = device

        self.tokenizer = self.tokenizer.to(self.device)
        self.model = self.model.to(self.device)

    def generate(self, x, x_stamp, y_stamp, pred_len, T, top_k, top_p, sample_count, verbose):
        x_tensor = torch.from_numpy(np.array(x).astype(np.float32)).to(self.device)
        x_stamp_tensor = torch.from_numpy(np.array(x_stamp).astype(np.float32)).to(self.device)
        y_stamp_tensor = torch.from_numpy(np.array(y_stamp).astype(np.float32)).to(self.device)

        preds = auto_regressive_inference(self.tokenizer, self.model, x_tensor, x_stamp_tensor, y_stamp_tensor, self.max_context, pred_len,
                                          self.clip, T, top_k, top_p, sample_count, verbose)
        preds = preds[:, -pred_len:, :]
        return preds

    def predict(self, df, x_timestamp, y_timestamp, pred_len, T=1.0, top_k=0, top_p=0.9, sample_count=1, verbose=True):

        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        if not all(col in df.columns for col in self.price_cols):
            raise ValueError(f"Price columns {self.price_cols} not found in DataFrame.")

        df = df.copy()
        if self.vol_col not in df.columns:
            df[self.vol_col] = 0.0
            df[self.amt_vol] = 0.0
        if self.amt_vol not in df.columns and self.vol_col in df.columns:
            df[self.amt_vol] = df[self.vol_col] * df[self.price_cols].mean(axis=1)

        if df[self.price_cols + [self.vol_col, self.amt_vol]].isnull().values.any():
            raise ValueError("Input DataFrame contains NaN values in price or volume columns.")

        x_time_df = calc_time_stamps(x_timestamp)
        y_time_df = calc_time_stamps(y_timestamp)

        # ── PROFESSIONAL UPGRADE: Store last absolute price & convert to Log Returns ──
        last_abs_prices = df[self.price_cols].iloc[-1].values

        df[self.price_cols] = np.log(df[self.price_cols] / df[self.price_cols].shift(1))
        df[self.price_cols] = df[self.price_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        x = df[self.price_cols + [self.vol_col, self.amt_vol]].values.astype(np.float32)
        x_stamp = x_time_df.values.astype(np.float32)
        y_stamp = y_time_df.values.astype(np.float32)

        recent_x = x[-self.max_context:] if x.shape[0] > self.max_context else x
        x_mean, x_std = np.mean(recent_x, axis=0), np.std(recent_x, axis=0)

        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.clip, self.clip)

        x = x[np.newaxis, :]
        x_stamp = x_stamp[np.newaxis, :]
        y_stamp = y_stamp[np.newaxis, :]

        preds = self.generate(x, x_stamp, y_stamp, pred_len, T, top_k, top_p, sample_count, verbose)

        preds = preds.squeeze(0)

        # Un-normalize Z-score (these are now predicted Log Returns)
        preds = preds * (x_std + 1e-5) + x_mean

        # ── PROFESSIONAL UPGRADE: Reconstruct absolute prices ──
        num_price_cols = len(self.price_cols)
        pred_log_returns = preds[:, :num_price_cols]

        # Formula: P_t = P_0 * exp(cumsum(R_log))
        cum_log_returns = np.cumsum(pred_log_returns, axis=0)
        pred_abs_prices = last_abs_prices * np.exp(cum_log_returns)

        # Overwrite the log return columns with the newly reconstructed absolute prices
        preds[:, :num_price_cols] = pred_abs_prices

        pred_df = pd.DataFrame(preds, columns=self.price_cols + [self.vol_col, self.amt_vol], index=y_timestamp)
        return pred_df

    def predict_batch(self, df_list, x_timestamp_list, y_timestamp_list, pred_len, T=1.0, top_k=0, top_p=0.9, sample_count=1, verbose=True):
        if not isinstance(df_list, (list, tuple)) or not isinstance(x_timestamp_list, (list, tuple)) or not isinstance(y_timestamp_list, (list, tuple)):
            raise ValueError("df_list, x_timestamp_list, y_timestamp_list must be list or tuple types.")
        if not (len(df_list) == len(x_timestamp_list) == len(y_timestamp_list)):
            raise ValueError("df_list, x_timestamp_list, y_timestamp_list must have consistent lengths.")

        num_series = len(df_list)

        x_list = []
        x_stamp_list = []
        y_stamp_list = []
        means = []
        stds = []
        seq_lens = []
        y_lens = []

        # ── Store last absolute prices for the entire batch ──
        last_abs_prices_list = []

        for i in range(num_series):
            df = df_list[i]
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"Input at index {i} is not a pandas DataFrame.")
            if not all(col in df.columns for col in self.price_cols):
                raise ValueError(f"DataFrame at index {i} is missing price columns {self.price_cols}.")

            df = df.copy()
            if self.vol_col not in df.columns:
                df[self.vol_col] = 0.0
                df[self.amt_vol] = 0.0
            if self.amt_vol not in df.columns and self.vol_col in df.columns:
                df[self.amt_vol] = df[self.vol_col] * df[self.price_cols].mean(axis=1)

            if df[self.price_cols + [self.vol_col, self.amt_vol]].isnull().values.any():
                raise ValueError(f"DataFrame at index {i} contains NaN values in price or volume columns.")

            x_timestamp = x_timestamp_list[i]
            y_timestamp = y_timestamp_list[i]

            x_time_df = calc_time_stamps(x_timestamp)
            y_time_df = calc_time_stamps(y_timestamp)

            # ── Capture absolute price, then convert dataframe to Log Returns ──
            last_abs_prices_list.append(df[self.price_cols].iloc[-1].values)

            df[self.price_cols] = np.log(df[self.price_cols] / df[self.price_cols].shift(1))
            df[self.price_cols] = df[self.price_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

            x = df[self.price_cols + [self.vol_col, self.amt_vol]].values.astype(np.float32)
            x_stamp = x_time_df.values.astype(np.float32)
            y_stamp = y_time_df.values.astype(np.float32)

            if x.shape[0] != x_stamp.shape[0]:
                raise ValueError(f"Inconsistent lengths at index {i}: x has {x.shape[0]} vs x_stamp has {x_stamp.shape[0]}.")
            if y_stamp.shape[0] != pred_len:
                raise ValueError(f"y_timestamp length at index {i} should equal pred_len={pred_len}, got {y_stamp.shape[0]}.")

            recent_x = x[-self.max_context:] if x.shape[0] > self.max_context else x
            x_mean, x_std = np.mean(recent_x, axis=0), np.std(recent_x, axis=0)

            x_norm = (x - x_mean) / (x_std + 1e-5)
            x_norm = np.clip(x_norm, -self.clip, self.clip)

            x_list.append(x_norm)
            x_stamp_list.append(x_stamp)
            y_stamp_list.append(y_stamp)
            means.append(x_mean)
            stds.append(x_std)

            seq_lens.append(x_norm.shape[0])
            y_lens.append(y_stamp.shape[0])

        if len(set(seq_lens)) != 1:
            raise ValueError(f"Parallel prediction requires all series to have consistent historical lengths, got: {seq_lens}")
        if len(set(y_lens)) != 1:
            raise ValueError(f"Parallel prediction requires all series to have consistent prediction lengths, got: {y_lens}")

        x_batch = np.stack(x_list, axis=0).astype(np.float32)
        x_stamp_batch = np.stack(x_stamp_list, axis=0).astype(np.float32)
        y_stamp_batch = np.stack(y_stamp_list, axis=0).astype(np.float32)

        preds = self.generate(x_batch, x_stamp_batch, y_stamp_batch, pred_len, T, top_k, top_p, sample_count, verbose)

        pred_dfs = []
        num_price_cols = len(self.price_cols)

        for i in range(num_series):
            # Un-normalize Z-score (Log Returns)
            preds_i = preds[i] * (stds[i] + 1e-5) + means[i]

            # ── Reconstruct absolute prices for this specific batch item ──
            pred_log_returns = preds_i[:, :num_price_cols]
            cum_log_returns = np.cumsum(pred_log_returns, axis=0)
            pred_abs_prices = last_abs_prices_list[i] * np.exp(cum_log_returns)

            preds_i[:, :num_price_cols] = pred_abs_prices

            pred_df = pd.DataFrame(preds_i, columns=self.price_cols + [self.vol_col, self.amt_vol], index=y_timestamp_list[i])
            pred_dfs.append(pred_df)

        return pred_dfs


```

---

### `models/nos_base/config.json`

```json
{
  "attn_dropout_p": 0.0,
  "d_model": 832,
  "ff_dim": 2048,
  "ffn_dropout_p": 0.2,
  "learn_te": true,
  "n_heads": 16,
  "n_layers": 12,
  "resid_dropout_p": 0.2,
  "s1_bits": 10,
  "s2_bits": 10,
  "token_dropout_p": 0.0
}
```

---

### `models/nos_base/model.safetensors`

```text
[File too large to display: 390.3 MB]
```

---

### `models/nos_mini/config.json`

```json
{
  "attn_dropout_p": 0.0,
  "d_model": 256,
  "ff_dim": 512,
  "ffn_dropout_p": 0.2,
  "learn_te": true,
  "n_heads": 4,
  "n_layers": 4,
  "resid_dropout_p": 0.2,
  "s1_bits": 10,
  "s2_bits": 10,
  "token_dropout_p": 0.0
}
```

---

### `models/nos_mini/model.safetensors`

```text
[File too large to display: 15.7 MB]
```

---

### `models/nos_small/config.json`

```json
{
  "attn_dropout_p": 0.1,
  "d_model": 512,
  "ff_dim": 1024,
  "ffn_dropout_p": 0.25,
  "learn_te": true,
  "n_heads": 8,
  "n_layers": 8,
  "resid_dropout_p": 0.25,
  "s1_bits": 10,
  "s2_bits": 10,
  "token_dropout_p": 0.1
}
```

---

### `models/nos_small/model.safetensors`

```text
[File too large to display: 94.4 MB]
```

---

### `models/nos_tokenizer_2k/config.json`

```json
{
  "attn_dropout_p": 0.0,
  "beta": 0.05,
  "d_in": 6,
  "d_model": 256,
  "ff_dim": 512,
  "ffn_dropout_p": 0.0,
  "gamma": 1.1,
  "gamma0": 1.0,
  "group_size": 5,
  "n_dec_layers": 4,
  "n_enc_layers": 4,
  "n_heads": 4,
  "resid_dropout_p": 0.0,
  "s1_bits": 10,
  "s2_bits": 10,
  "zeta": 0.05
}
```

---

### `models/nos_tokenizer_2k/model.safetensors`

```text
[File too large to display: 15.1 MB]
```

---

### `models/nos_tokenizer_base/config.json`

```json
{
  "attn_dropout_p": 0.0,
  "beta": 0.05,
  "d_in": 6,
  "d_model": 256,
  "ff_dim": 512,
  "ffn_dropout_p": 0.0,
  "gamma": 1.1,
  "gamma0": 1.0,
  "group_size": 4,
  "n_dec_layers": 4,
  "n_enc_layers": 4,
  "n_heads": 4,
  "resid_dropout_p": 0.0,
  "s1_bits": 10,
  "s2_bits": 10,
  "zeta": 0.05
}
```

---

### `models/nos_tokenizer_base/model.safetensors`

```text
[File too large to display: 15.1 MB]
```

---

### `config_loader.py`

```python
import os
import yaml
from typing import Dict, Any


class ConfigLoader:

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"config file not found: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        config = self._resolve_dynamic_paths(config)
        return config

    def _resolve_dynamic_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        exp_name = config.get('model_paths', {}).get('exp_name', '')
        if not exp_name:
            return config

        base_path = config.get('model_paths', {}).get('base_path', '')
        path_templates = {
            'base_save_path': f"{base_path}/{exp_name}",
            'finetuned_tokenizer': f"{base_path}/{exp_name}/tokenizer/best_model"
        }

        if 'model_paths' in config:
            for key, template in path_templates.items():
                if key in config['model_paths']:
                    current_value = config['model_paths'][key]
                    if current_value == "" or current_value is None:
                        config['model_paths'][key] = template
                    else:
                        if isinstance(current_value, str) and '{exp_name}' in current_value:
                            config['model_paths'][key] = current_value.format(exp_name=exp_name)

        return config

    def get(self, key: str, default=None):
        """
        Retrieves a (possibly nested) config value using dot-notation keys.

        Examples:
            loader.get('training.batch_size')
            loader.get('hpo.search_space.tokenizer_learning_rate.low')

        Args:
            key:     Dot-separated key path into the config dict.
            default: Value to return if any key in the path is missing.

        Returns:
            Config value at the given path, or default if not found.
        """
        keys  = key.split('.')
        value = self.config
        try:
            for k in keys:
                if not isinstance(value, dict):
                    return default
                value = value[k]
            # Treat explicit YAML null (None) as absent — return default
            return value if value is not None else default
        except (KeyError, TypeError):
            return default

    def get_data_config(self) -> Dict[str, Any]:
        return self.config.get('data', {})

    def get_training_config(self) -> Dict[str, Any]:
        return self.config.get('training', {})

    def get_model_paths(self) -> Dict[str, str]:
        return self.config.get('model_paths', {})

    def get_experiment_config(self) -> Dict[str, Any]:
        return self.config.get('experiment', {})

    def get_device_config(self) -> Dict[str, Any]:
        return self.config.get('device', {})

    def get_distributed_config(self) -> Dict[str, Any]:
        return self.config.get('distributed', {})

    def get_dynamic_tuning_config(self) -> Dict[str, Any]:
        return self.config.get('dynamic_tuning', {})
    
    def get_hpo_config(self) -> Dict[str, Any]:
        """
        Returns the full HPO configuration block.

        Returns an empty dict (not None) if the block is absent, so callers
        can safely use .get() without None-checks.
        """
        return self.config.get('hpo') or {}

    def update_config(self, updates: Dict[str, Any]):
        def update_nested_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_nested_dict(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        self.config = update_nested_dict(self.config, updates)

    def save_config(self, save_path: str = None):
        if save_path is None:
            save_path = self.config_path
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False,
                      allow_unicode=True, indent=2)

    def print_config(self):
        print("=" * 50)
        print("Current configuration:")
        print("=" * 50)
        yaml.dump(self.config, default_flow_style=False,
                  allow_unicode=True, indent=2)
        print("=" * 50)


class CustomFinetuneConfig:

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')

        self.loader = ConfigLoader(config_path)
        self._load_all_configs()

    def _load_all_configs(self):
        # ── Data ──────────────────────────────────────────
        data_config = self.loader.get_data_config()
        self.data_path = data_config.get('data_path')
        self.lookback_window = data_config.get('lookback_window', 512)
        self.predict_window = data_config.get('predict_window', 48)
        self.max_context = data_config.get('max_context', 512)
        self.clip = data_config.get('clip', 5.0)
        self.train_ratio = data_config.get('train_ratio', 0.9)
        self.val_ratio = data_config.get('val_ratio', 0.1)
        self.test_ratio = data_config.get('test_ratio', 0.0)

        # ── Training ──────────────────────────────────────────────
        training_config = self.loader.get_training_config()

        self.tokenizer_epochs = training_config.get('tokenizer_epochs', 30)
        self.basemodel_epochs = training_config.get('basemodel_epochs', 30)
        if 'epochs' in training_config and 'tokenizer_epochs' not in training_config:
            self.tokenizer_epochs = training_config.get('epochs', 30)
        if 'epochs' in training_config and 'basemodel_epochs' not in training_config:
            self.basemodel_epochs = training_config.get('epochs', 30)

        self.batch_size = training_config.get('batch_size', 160)
        self.log_interval = training_config.get('log_interval', 50)
        self.num_workers = training_config.get('num_workers', 6)
        self.persistent_workers = training_config.get('persistent_workers', False)
        self.seed = training_config.get('seed', 100)

        self.adam_beta1 = training_config.get('adam_beta1', 0.9)
        self.adam_beta2 = training_config.get('adam_beta2', 0.95)
        self.adam_weight_decay = training_config.get('adam_weight_decay', 0.1)
        self.adam_eps = training_config.get('adam_eps', 1e-6)

        self.tokenizer_learning_rate = training_config.get('tokenizer_learning_rate', 2e-4)
        self.predictor_learning_rate = training_config.get('predictor_learning_rate', 4e-5)

        self.tokenizer_grad_clip = float(
            training_config.get('tokenizer_grad_clip')
            or training_config.get('tokenizer_max_grad_norm')
            or 2.0
        )
        self.basemodel_grad_clip = float(
            training_config.get('basemodel_grad_clip')
            or training_config.get('basemodel_max_grad_norm')
            or 3.0
        )
        self.grad_clip_norm_type = float(
            training_config.get('grad_clip_norm_type', 2.0)
        )

        self.scheduler_type = training_config.get('scheduler_type', 'cosine_warmup')
        self.scheduler_pct_start = training_config.get('scheduler_pct_start', 0.05)
        self.scheduler_div_factor = training_config.get('scheduler_div_factor', 25.0)
        self.scheduler_final_div_factor = training_config.get('scheduler_final_div_factor', 1000.0)

        self.tokenizer_recon_pre_weight = training_config.get('tokenizer_recon_pre_weight', 1.0)
        self.tokenizer_recon_all_weight = training_config.get('tokenizer_recon_all_weight', 1.0)
        self.tokenizer_bsq_weight = training_config.get('tokenizer_bsq_weight', 0.5)
        self.tokenizer_recon_weight = training_config.get('tokenizer_recon_weight', 0.5)
        self.basemodel_s1_weight = training_config.get('basemodel_s1_weight', 0.5)
        self.basemodel_s2_weight = training_config.get('basemodel_s2_weight', 0.5)

        self.label_smoothing = training_config.get('label_smoothing', 0.0)
        self.bsq_inv_temperature = training_config.get('bsq_inv_temperature', 1.0)
        self.rope_base = training_config.get('rope_base', 10000)
        self.accumulation_steps = training_config.get('accumulation_steps', 1)

        # ── NEW: HPO search space params that must exist as real attributes ──
        # Without these, getattr() fallbacks in training code ignore YAML values
        # whenever HPO doesn't sample a given parameter in a trial.
        self.tokenizer_pct_start = training_config.get(
            'tokenizer_pct_start', self.scheduler_pct_start
        )
        self.tokenizer_div_factor = training_config.get(
            'tokenizer_div_factor', self.scheduler_div_factor
        )
        self.tokenizer_max_grad_norm = training_config.get(
            'tokenizer_max_grad_norm', self.tokenizer_grad_clip
        )
        self.basemodel_pct_start = training_config.get(
            'basemodel_pct_start', self.scheduler_pct_start
        )
        self.basemodel_div_factor = training_config.get(
            'basemodel_div_factor', self.scheduler_div_factor
        )
        self.basemodel_max_grad_norm = training_config.get(
            'basemodel_max_grad_norm', self.basemodel_grad_clip
        )
        self.bsq_beta   = training_config.get('bsq_beta',   None)
        self.bsq_gamma0 = training_config.get('bsq_gamma0', None)
        self.bsq_gamma  = training_config.get('bsq_gamma',  None)
        self.bsq_zeta   = training_config.get('bsq_zeta',   None)
        self.ffn_dropout_p   = training_config.get('ffn_dropout_p',   None)
        self.attn_dropout_p  = training_config.get('attn_dropout_p',  None)
        self.resid_dropout_p = training_config.get('resid_dropout_p', None)
        self.token_dropout_p = training_config.get('token_dropout_p', None)
        # ── END NEW ──────────────────────────────────────────────────────────

        # ── Model paths ───────────────────────────────────
        model_paths = self.loader.get_model_paths()
        self.exp_name = model_paths.get('exp_name', 'default_experiment')
        self.pretrained_tokenizer_path = model_paths.get('pretrained_tokenizer')
        self.pretrained_predictor_path = model_paths.get('pretrained_predictor')
        self.base_save_path = model_paths.get('base_save_path') or ''
        self.tokenizer_save_name = model_paths.get('tokenizer_save_name', 'tokenizer')
        self.basemodel_save_name = model_paths.get('basemodel_save_name', 'basemodel')
        self.finetuned_tokenizer_path = model_paths.get('finetuned_tokenizer')

        # ── Experiment ────────────────────────────────────
        experiment_config = self.loader.get_experiment_config()
        self.experiment_name = experiment_config.get('name', 'Nos_custom_finetune')
        self.experiment_description = experiment_config.get('description', '')
        self.use_comet = experiment_config.get('use_comet', False)
        self.train_tokenizer = experiment_config.get('train_tokenizer', True)
        self.train_basemodel = experiment_config.get('train_basemodel', True)
        self.skip_existing = experiment_config.get('skip_existing', False)

        unified_pretrained = experiment_config.get('pre_trained', None)
        self.pre_trained_tokenizer = experiment_config.get(
            'pre_trained_tokenizer',
            unified_pretrained if unified_pretrained is not None else True)
        self.pre_trained_predictor = experiment_config.get(
            'pre_trained_predictor',
            unified_pretrained if unified_pretrained is not None else True)

        # ── Device ────────────────────────────────────────
        device_config = self.loader.get_device_config()
        self.use_cuda = device_config.get('use_cuda', True)
        self.device_id = device_config.get('device_id', 0)

        # ── Distributed ───────────────────────────────────
        distributed_config = self.loader.get_distributed_config()
        self.use_ddp = distributed_config.get('use_ddp', False)
        self.ddp_backend = distributed_config.get('backend', 'nccl')

        # ── Dynamic tuning ────────────────────────────────
        dt = self.loader.get_dynamic_tuning_config()
        self.dynamic_tuning_enabled = dt.get('enabled', False)
        self.dynamic_tuning_mode = dt.get('mode', 'moderate')

        self.tune_learning_rate = dt.get('tune_learning_rate', True)
        self.tune_grad_clip = dt.get('tune_grad_clip', True)
        self.tune_betas = dt.get('tune_betas', True)
        self.tune_loss_weights = dt.get('tune_loss_weights', True)
        self.tune_label_smoothing = dt.get('tune_label_smoothing', True)
        self.tune_bsq_temperature = dt.get('tune_bsq_temperature', True)
        self.tune_scheduler_params = dt.get('tune_scheduler_params', True)
        self.tune_weight_decay = dt.get('tune_weight_decay', True)
        self.tune_dropout = dt.get('tune_dropout', True)
        self.tune_adam_eps = dt.get('tune_adam_eps', True)

        # Bounds
        self.dt_lr_min = dt.get('lr_min', 1e-8)
        self.dt_lr_max = dt.get('lr_max', 1e-3)
        self.dt_beta1_min = dt.get('beta1_min', 0.85)
        self.dt_beta1_max = dt.get('beta1_max', 0.95)
        self.dt_beta2_min = dt.get('beta2_min', 0.90)
        self.dt_beta2_max = dt.get('beta2_max', 0.999)
        self.dt_grad_clip_min = dt.get('grad_clip_min', 0.5)
        self.dt_grad_clip_max = dt.get('grad_clip_max', 10.0)
        self.dt_loss_weight_min = dt.get('loss_weight_min', 0.1)
        self.dt_loss_weight_max = dt.get('loss_weight_max', 2.0)
        self.dt_label_smoothing_min = dt.get('label_smoothing_min', 0.0)
        self.dt_label_smoothing_max = dt.get('label_smoothing_max', 0.15)
        self.dt_bsq_inv_temp_min = dt.get('bsq_inv_temp_min', 0.5)
        self.dt_bsq_inv_temp_max = dt.get('bsq_inv_temp_max', 5.0)
        self.dt_weight_decay_min = dt.get('weight_decay_min', 0.0)
        self.dt_weight_decay_max = dt.get('weight_decay_max', 0.5)

        # Detection thresholds
        self.dt_plateau_window = dt.get('plateau_window_epochs', 4)
        self.dt_plateau_threshold = dt.get('plateau_threshold', 5e-4)
        self.dt_overfit_gap_ratio = dt.get('overfit_gap_ratio', 0.10)
        self.dt_severe_overfit_gap_ratio = dt.get('severe_overfit_gap_ratio', 0.30)
        self.dt_grad_explosion_threshold = dt.get('grad_explosion_threshold', 8.0)

        # Cooldowns
        self.dt_lr_cooldown = dt.get('lr_cooldown_epochs', 2)
        self.dt_beta_cooldown = dt.get('beta_cooldown_epochs', 4)
        self.dt_loss_weight_cooldown = dt.get('loss_weight_cooldown_epochs', 3)
        self.dt_label_smooth_cooldown = dt.get('label_smooth_cooldown_epochs', 4)
        self.dt_bsq_temp_cooldown = dt.get('bsq_temp_cooldown_epochs', 5)
        self.dt_grad_clip_cooldown = dt.get('grad_clip_cooldown_epochs', 1)
        self.dt_scheduler_swap_cooldown = dt.get('scheduler_swap_cooldown_epochs', 5)
        self.dt_tuning_warmup = dt.get('tuning_warmup_epochs', 2)

        self._compute_full_paths()

        # ── HPO ───────────────────────────────────────────
        hpo_config = self.loader.get('hpo') or {}

        self.hpo_enabled            = bool(hpo_config.get('enabled', False))
        self.hpo_n_trials           = int(hpo_config.get('n_trials', 10))
        self.hpo_direction          = str(hpo_config.get('direction', 'minimize'))
        self.hpo_sampler            = str(hpo_config.get('sampler', 'tpe'))
        self.hpo_pruner             = str(hpo_config.get('pruner', 'median'))
        self.hpo_storage            = hpo_config.get('storage', None)
        self.hpo_study_name         = str(hpo_config.get('study_name', 'nos_hpo'))

        self.hpo_optimize_tokenizer = bool(hpo_config.get('optimize_tokenizer', True))
        self.hpo_optimize_basemodel = bool(hpo_config.get('optimize_basemodel', True))
        self.hpo_tokenizer_epochs   = int(hpo_config.get('hpo_tokenizer_epochs', 5))
        self.hpo_basemodel_epochs   = int(hpo_config.get('hpo_basemodel_epochs', 3))

        # Deep-copy the search space so that trial clones can never mutate
        # the base config's search space dict via shared reference.
        import copy as _copy
        raw_search_space = hpo_config.get('search_space', {}) or {}
        self.hpo_search_space: Dict[str, Any] = _copy.deepcopy(raw_search_space)

        # ── Search space key aliasing ─────────────────────────────────────────
        # The YAML uses tokenizer_max_grad_norm / basemodel_max_grad_norm but
        # the training code reads tokenizer_grad_clip / basemodel_grad_clip.
        # Normalise here so both YAML spellings work transparently.
        _alias_map = {
            'tokenizer_max_grad_norm': 'tokenizer_grad_clip',
            'basemodel_max_grad_norm': 'basemodel_grad_clip',
        }
        for yaml_key, code_key in _alias_map.items():
            if yaml_key in self.hpo_search_space and code_key not in self.hpo_search_space:
                self.hpo_search_space[code_key] = _copy.deepcopy(
                    self.hpo_search_space[yaml_key]
                )

        # ── Validate storage URI ──────────────────────────────────────────────
        # Warn loudly if SQLite is configured without a timeout parameter.
        # The timeout prevents "database is locked" errors under 8 concurrent
        # workers. See configs/config.yaml for the recommended URI format.
        if (
            self.hpo_storage is not None
            and self.hpo_storage.startswith('sqlite')
            and 'timeout=' not in self.hpo_storage
        ):
            import logging
            logging.getLogger(__name__).warning(
                f"HPO storage URI '{self.hpo_storage}' is SQLite but does not "
                f"include a timeout parameter. Under concurrent multi-GPU HPO "
                f"this will cause 'database is locked' errors. "
                f"Recommended: append '?timeout=60' to the URI."
            )

        # ── Validate direction ────────────────────────────────────────────────
        if self.hpo_direction not in ('minimize', 'maximize'):
            raise ValueError(
                f"hpo.direction must be 'minimize' or 'maximize'. "
                f"Got: '{self.hpo_direction}'"
            )

    def _compute_full_paths(self):
        self.tokenizer_save_path = os.path.join(
            self.base_save_path, self.tokenizer_save_name)
        self.tokenizer_best_model_path = os.path.join(
            self.tokenizer_save_path, 'best_model')
        self.basemodel_save_path = os.path.join(
            self.base_save_path, self.basemodel_save_name)
        self.basemodel_best_model_path = os.path.join(
            self.basemodel_save_path, 'best_model')

    def clone_with_overrides(self, overrides: Dict[str, Any]) -> 'CustomFinetuneConfig':
        
        import copy
        import logging

        new_config = copy.deepcopy(self)

        for k, v in overrides.items():
            if not hasattr(new_config, k):
                logging.getLogger(__name__).warning(
                    f"clone_with_overrides: attribute '{k}' does not exist "
                    f"on CustomFinetuneConfig. Adding dynamically. "
                    f"Verify this parameter name matches the training code."
                )
            setattr(new_config, k, v)

        new_config._compute_full_paths()

        return new_config   

    def get_tokenizer_config(self):
        return {
            'data_path': self.data_path,
            'lookback_window': self.lookback_window,
            'predict_window': self.predict_window,
            'max_context': self.max_context,
            'clip': self.clip,
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'epochs': self.tokenizer_epochs,
            'batch_size': self.batch_size,
            'log_interval': self.log_interval,
            'num_workers': self.num_workers,
            'seed': self.seed,
            'learning_rate': self.tokenizer_learning_rate,
            'adam_beta1': self.adam_beta1,
            'adam_beta2': self.adam_beta2,
            'adam_weight_decay': self.adam_weight_decay,
            'adam_eps': self.adam_eps,
            'accumulation_steps': self.accumulation_steps,
            'pretrained_model_path': self.pretrained_tokenizer_path,
            'save_path': self.tokenizer_save_path,
            'use_comet': self.use_comet,
            'tokenizer_grad_clip': self.tokenizer_grad_clip,
            'scheduler_type': self.scheduler_type,
            'scheduler_pct_start': self.scheduler_pct_start,
            'scheduler_div_factor': self.scheduler_div_factor,
            'scheduler_final_div_factor': self.scheduler_final_div_factor,
            'tokenizer_recon_pre_weight': self.tokenizer_recon_pre_weight,
            'tokenizer_recon_all_weight': self.tokenizer_recon_all_weight,
            'tokenizer_bsq_weight': self.tokenizer_bsq_weight,
            'tokenizer_recon_weight': self.tokenizer_recon_weight,
            'label_smoothing': self.label_smoothing,
            'bsq_inv_temperature': self.bsq_inv_temperature,
        }

    def get_basemodel_config(self):
        return {
            'data_path': self.data_path,
            'lookback_window': self.lookback_window,
            'predict_window': self.predict_window,
            'max_context': self.max_context,
            'clip': self.clip,
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'epochs': self.basemodel_epochs,
            'batch_size': self.batch_size,
            'log_interval': self.log_interval,
            'num_workers': self.num_workers,
            'seed': self.seed,
            'predictor_learning_rate': self.predictor_learning_rate,
            'tokenizer_learning_rate': self.tokenizer_learning_rate,
            'adam_beta1': self.adam_beta1,
            'adam_beta2': self.adam_beta2,
            'adam_weight_decay': self.adam_weight_decay,
            'adam_eps': self.adam_eps,
            'pretrained_tokenizer_path': self.finetuned_tokenizer_path,
            'pretrained_predictor_path': self.pretrained_predictor_path,
            'save_path': self.basemodel_save_path,
            'use_comet': self.use_comet,
            'basemodel_grad_clip': self.basemodel_grad_clip,
            'scheduler_type': self.scheduler_type,
            'scheduler_pct_start': self.scheduler_pct_start,
            'scheduler_div_factor': self.scheduler_div_factor,
            'scheduler_final_div_factor': self.scheduler_final_div_factor,
            'basemodel_s1_weight': self.basemodel_s1_weight,
            'basemodel_s2_weight': self.basemodel_s2_weight,
            'label_smoothing': self.label_smoothing,
        }

    def print_config_summary(self):
        print("=" * 60)
        print("Nos finetuning configuration summary")
        print("=" * 60)
        print(f"Experiment name:          {self.exp_name}")
        print(f"Data path:                {self.data_path}")
        print(f"Lookback window:          {self.lookback_window}")
        print(f"Predict window:           {self.predict_window}")
        print(f"Tokenizer epochs:         {self.tokenizer_epochs}")
        print(f"Basemodel epochs:         {self.basemodel_epochs}")
        print(f"Batch size:               {self.batch_size}")
        print(f"Tokenizer LR:             {self.tokenizer_learning_rate}")
        print(f"Predictor LR:             {self.predictor_learning_rate}")
        print(f"Adam betas:               ({self.adam_beta1}, {self.adam_beta2})")
        print(f"Adam eps:                 {self.adam_eps}")
        print(f"Scheduler type:           {self.scheduler_type}")
        print(f"Scheduler pct_start:      {self.scheduler_pct_start}")
        print(f"Tokenizer grad clip:      {self.tokenizer_grad_clip}")
        print(f"Basemodel grad clip:      {self.basemodel_grad_clip}")
        print(f"Label smoothing:          {self.label_smoothing}")
        print(f"BSQ inv temperature:      {self.bsq_inv_temperature}")
        print(f"RoPE base:                {self.rope_base}")
        print(f"Dynamic tuning enabled:   {self.dynamic_tuning_enabled}")
        print(f"Dynamic tuning mode:      {self.dynamic_tuning_mode}")
        print(f"Base save path:           {self.base_save_path}")
        print("=" * 60)
```

---

### `hpo_tuner.py`

```python
"""
Automatic Hyperparameter Tuning for Nos Model Finetuning.

Architecture: Asynchronous multi-process HPO where each GPU process runs
independently, sharing only an Optuna SQLite/PostgreSQL study database.
No DDP is used — process isolation is enforced at the launcher level.

Launch (single GPU, development):
    python hpo_tuner.py --config configs/config.yaml --phase both

Launch (8-GPU cluster, production):
    ./launch_hpo.sh --config configs/config.yaml --phase both --n-trials 30

Launch (specific phase):
    python hpo_tuner.py --config configs/config.yaml --phase tokenizer
    python hpo_tuner.py --config configs/config.yaml --phase basemodel --tokenizer-path path/to/tokenizer
    python hpo_tuner.py --config configs/config.yaml --apply-best
"""

from __future__ import annotations

import copy
import datetime
import gc
import json
import logging
import argparse
import multiprocessing
import os
import shutil
import sys
import tempfile
import time
import warnings  # <-- Correct standard library import
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy.pool import NullPool

import torch
import numpy as np
import yaml

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ── Optional Optuna import ─────────────────────────────────────────────────
try:
    import optuna
    from optuna.exceptions import ExperimentalWarning  # <-- Moved inside the safe block
    from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
    from optuna.pruners import MedianPruner, HyperbandPruner, NopPruner
    from optuna.storages import RDBStorage
    
    # Suppress Optuna's experimental feature warnings safely
    warnings.filterwarnings("ignore", category=ExperimentalWarning)
    
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from config_loader import CustomFinetuneConfig
from model import Nos, NosTokenizer


# ══════════════════════════════════════════════════════════════════════════════
# Module-level logger (console output for the orchestrator process)
# ══════════════════════════════════════════════════════════════════════════════

def _configure_root_logger(log_dir: str, worker_tag: str) -> logging.Logger:
    """
    Configures the root logger for this HPO worker process.

    Writes INFO+ to both console (WARNING+ only to reduce noise) and a
    dedicated per-worker log file with full DEBUG output.

    Args:
        log_dir:    Directory where the worker log file will be written.
        worker_tag: Unique identifier string (e.g., "pid12345_gpu0").

    Returns:
        Configured root logger for this process.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"worker_{worker_tag}.log")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Remove any existing handlers to prevent duplication on re-import
    root_logger.handlers.clear()

    # ── File handler: full DEBUG output per worker ─────────────────────────
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-8s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # ── Console handler: WARNING+ only to keep stdout clean ───────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    return root_logger


def _get_trial_logger(name: str, trial_dir: str) -> logging.Logger:
    """
    Creates an isolated, non-propagating file logger for a single trial.

    Using a unique logger name (including PID) prevents Python's logging
    singleton from returning a cached logger from a previous trial, which
    would cause handler accumulation (each call adding another FileHandler).

    Args:
        name:      Base name for the logger (e.g., "hpo_tok_trial_7").
        trial_dir: Path to the trial's isolated working directory.

    Returns:
        Configured logger that writes only to trial_dir/trial.log.
    """
    unique_name = f"{name}_pid{os.getpid()}"
    logger = logging.getLogger(unique_name)

    # Always clear handlers — prevents accumulation if function is called
    # multiple times with the same effective logger name (e.g., trial restarts)
    logger.handlers.clear()
    logger.propagate = False  # Do not bubble up to root logger
    logger.setLevel(logging.INFO)

    log_path = os.path.join(trial_dir, "trial.log")
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    return logger


# ══════════════════════════════════════════════════════════════════════════════
# Model Override Helpers
# ══════════════════════════════════════════════════════════════════════════════

def apply_dropout_overrides(
    model: torch.nn.Module,
    config: CustomFinetuneConfig,
) -> torch.nn.Module:
    """
    Walks all named submodules and overrides dropout probabilities in-place
    when the corresponding config attribute is set to a non-None value.

    This modifies only scalar dropout probability attributes and the `p`
    attribute of nn.Dropout instances. Model weights are never touched.

    Args:
        model:  Any PyTorch module (NosTokenizer or Nos predictor).
        config: Trial config carrying dropout override values.

    Returns:
        The same model instance with dropout values updated.
    """
    ffn_dp    = getattr(config, "ffn_dropout_p",   None)
    attn_dp   = getattr(config, "attn_dropout_p",  None)
    resid_dp  = getattr(config, "resid_dropout_p", None)
    token_dp  = getattr(config, "token_dropout_p", None)

    for _name, module in model.named_modules():
        # FeedForward ffn_dropout (nn.Dropout instance on FeedForward blocks)
        if ffn_dp is not None and hasattr(module, "ffn_dropout"):
            if isinstance(module.ffn_dropout, torch.nn.Dropout):
                module.ffn_dropout.p = float(ffn_dp)

        # Residual dropout in attention blocks (nn.Dropout instance)
        if resid_dp is not None and hasattr(module, "resid_dropout"):
            if isinstance(module.resid_dropout, torch.nn.Dropout):
                module.resid_dropout.p = float(resid_dp)

        # Token dropout on Nos predictor (nn.Dropout instance)
        if token_dp is not None and hasattr(module, "token_drop"):
            if isinstance(module.token_drop, torch.nn.Dropout):
                module.token_drop.p = float(token_dp)

        # Attention dropout stored as a plain float scalar (not nn.Dropout)
        if attn_dp is not None and hasattr(module, "attn_dropout_p"):
            if isinstance(module.attn_dropout_p, float):
                module.attn_dropout_p = float(attn_dp)

    return model


def apply_bsq_overrides(
    tokenizer: NosTokenizer,
    config: CustomFinetuneConfig,
) -> NosTokenizer:
    """
    Overrides BSQ (Binary Spherical Quantizer) loss weight scalars on a
    loaded tokenizer. Only modifies scalar hyperparameter attributes;
    quantizer weight tensors are never modified.

    Args:
        tokenizer: Loaded NosTokenizer instance.
        config:    Trial config carrying BSQ override values.

    Returns:
        The same tokenizer instance with BSQ weights updated.

    Raises:
        AttributeError: If tokenizer does not have the expected BSQ structure.
    """
    try:
        bsq = tokenizer.tokenizer.bsq  # BinarySphericalQuantizer instance
    except AttributeError as exc:
        raise AttributeError(
            "Could not access tokenizer.tokenizer.bsq. "
            "Verify NosTokenizer architecture has not changed."
        ) from exc

    override_map = {
        "bsq_beta":   "beta",
        "bsq_gamma0": "gamma0",
        "bsq_gamma":  "gamma",
        "bsq_zeta":   "zeta",
    }
    for config_attr, bsq_attr in override_map.items():
        value = getattr(config, config_attr, None)
        if value is not None:
            setattr(bsq, bsq_attr, float(value))

    return tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# GPU Memory Management
# ══════════════════════════════════════════════════════════════════════════════

def release_gpu_memory(*model_refs) -> None:
    """
    Performs deterministic GPU memory cleanup after a trial completes.

    Executes the full cleanup chain required to return memory to the CUDA
    driver (not just to PyTorch's internal allocator cache):
      1. Delete all passed model references from Python's reference count.
      2. Force Python's cyclic garbage collector to collect any cycles.
      3. Release PyTorch's allocator cache back to the CUDA driver.
      4. Synchronize the CUDA stream to ensure all ops have completed.

    Without step 3 (`empty_cache`), PyTorch holds memory in its internal
    pool. Across many HPO trials this accumulates until the next trial's
    allocation fails with OutOfMemoryError even though logically the memory
    was "freed" after the previous trial.

    Args:
        *model_refs: Any number of PyTorch module references to delete.
    """
    for ref in model_refs:
        if ref is not None:
            del ref

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ══════════════════════════════════════════════════════════════════════════════
# Search Space Builder
# ══════════════════════════════════════════════════════════════════════════════

class SearchSpaceBuilder:
    """
    Translates the `hpo.search_space` YAML block into Optuna trial suggestions.

    The search space config block uses a typed specification format:
        param_name:
            type: float | int | categorical
            low:  <number>          # float and int only
            high: <number>          # float and int only
            log:  true | false      # float and int only (log-scale sampling)
            choices: [a, b, c]      # categorical only

    Parameters are grouped by training phase so that tokenizer-only or
    basemodel-only HPO runs sample only the relevant hyperparameters.
    """

    # Maps each parameter to its owning training phase.
    # Parameters listed under 'shared' are always included regardless of phase.
    PARAM_GROUPS: Dict[str, List[str]] = {
        "tokenizer": [
            "tokenizer_learning_rate",
            "tokenizer_pct_start",
            "tokenizer_div_factor",
            "tokenizer_max_grad_norm",
            "bsq_beta",
            "bsq_gamma0",
            "bsq_gamma",
            "bsq_zeta",
        ],
        "basemodel": [
            "predictor_learning_rate",
            "basemodel_pct_start",
            "basemodel_div_factor",
            "basemodel_max_grad_norm",
            "ffn_dropout_p",
            "attn_dropout_p",
            "resid_dropout_p",
            "token_dropout_p",
        ],
        "shared": [
            "adam_weight_decay",
            "adam_beta1",
            "adam_beta2",
            "batch_size",
            "accumulation_steps",
            "clip",
        ],
    }

    def __init__(self, search_space: Dict[str, Any], phase: str = "both") -> None:
        """
        Args:
            search_space: The parsed `hpo.search_space` dict from config YAML.
            phase:        Which phase to sample for: 'tokenizer', 'basemodel', or 'both'.
        """
        if phase not in ("tokenizer", "basemodel", "both"):
            raise ValueError(
                f"phase must be 'tokenizer', 'basemodel', or 'both'. Got: '{phase}'"
            )
        self.search_space = search_space
        self.phase = phase

    def _should_include(self, param_name: str) -> bool:
        """Returns True if param_name is relevant to the current HPO phase."""
        if self.phase == "both":
            return True
        phase_params = self.PARAM_GROUPS.get(self.phase, [])
        shared_params = self.PARAM_GROUPS["shared"]
        return param_name in phase_params or param_name in shared_params

    def suggest(self, trial: "optuna.Trial", param_name: str) -> Optional[Any]:
        """
        Suggests a single hyperparameter value using the Optuna trial API.

        Args:
            trial:      The current Optuna trial object.
            param_name: Name of the parameter to suggest.

        Returns:
            Suggested value, or None if param_name is not in the search space.

        Raises:
            ValueError: If the spec type is unrecognised.
        """
        if param_name not in self.search_space:
            return None

        spec = self.search_space[param_name]
        ptype = spec.get("type", "").lower()

        if ptype == "float":
            return trial.suggest_float(
                param_name,
                float(spec["low"]),
                float(spec["high"]),
                log=bool(spec.get("log", False)),
            )
        elif ptype == "int":
            return trial.suggest_int(
                param_name,
                int(spec["low"]),
                int(spec["high"]),
                log=bool(spec.get("log", False)),
            )
        elif ptype == "categorical":
            return trial.suggest_categorical(param_name, spec["choices"])
        else:
            raise ValueError(
                f"Unknown search space type '{ptype}' for parameter '{param_name}'. "
                f"Valid types: float, int, categorical."
            )

    def build_overrides(self, trial: "optuna.Trial") -> Dict[str, Any]:
        """
        Samples all relevant hyperparameters for this phase and returns
        them as a flat dict suitable for `config.clone_with_overrides()`.

        Args:
            trial: The current Optuna trial object.

        Returns:
            Dict mapping parameter names to their suggested values.
        """
        overrides: Dict[str, Any] = {}
        for param_name in self.search_space:
            if self._should_include(param_name):
                value = self.suggest(trial, param_name)
                if value is not None:
                    overrides[param_name] = value
        return overrides


# ══════════════════════════════════════════════════════════════════════════════
# Optuna Trial Failure Callback
# ══════════════════════════════════════════════════════════════════════════════

def _build_failure_rate_callback(
    max_failure_rate: float = 0.50,
    min_trials_before_check: int = 4,
) -> "Callable":
    """
    Builds an Optuna callback that aborts the study if the trial failure
    rate exceeds a threshold. This prevents silent budget exhaustion when
    a systematic bug causes every trial to crash.

    Without this guard, `catch=(RuntimeError, ...)` in study.optimize will
    silently mark all trials as FAIL and consume the full n_trials budget
    before the user notices nothing is working.

    Args:
        max_failure_rate:          Fraction of failed trials that triggers abort [0.0, 1.0].
        min_trials_before_check:   Minimum number of trials before rate is evaluated.

    Returns:
        Callable matching Optuna's callback signature: f(study, trial) -> None.
    """
    logger = logging.getLogger(__name__)

    def callback(study: "optuna.Study", trial: "optuna.Trial") -> None:
        if trial.state != optuna.trial.TrialState.FAIL:
            return

        completed_and_failed = [
            t for t in study.trials
            if t.state in (
                optuna.trial.TrialState.COMPLETE,
                optuna.trial.TrialState.FAIL,
            )
        ]
        total = len(completed_and_failed)
        if total < min_trials_before_check:
            return

        failed_count = sum(
            1 for t in completed_and_failed
            if t.state == optuna.trial.TrialState.FAIL
        )
        fail_rate = failed_count / total

        logger.warning(
            f"Trial {trial.number} FAILED. "
            f"Cumulative failure rate: {failed_count}/{total} ({fail_rate:.0%})"
        )

        if fail_rate > max_failure_rate:
            logger.error(
                f"Failure rate {fail_rate:.0%} exceeds {max_failure_rate:.0%} "
                f"threshold after {total} trials. Aborting study to prevent "
                f"wasting the remaining trial budget. "
                f"Inspect per-trial logs for root cause."
            )
            study.stop()

    return callback


# ══════════════════════════════════════════════════════════════════════════════
# Objective Functions
# ══════════════════════════════════════════════════════════════════════════════

class TokenizerObjective:
    """
    Optuna objective function for tokenizer HPO.

    Each call to __call__ corresponds to one Optuna trial. The lifecycle is:
      1. Sample hyperparameters from the search space.
      2. Build a fully isolated trial config (unique save directory, PID-stamped).
      3. Load the pretrained tokenizer and apply config overrides.
      4. Run abbreviated training via the external train_tokenizer function.
      5. Return the validation loss as the optimization target.
      6. Deterministically release all GPU memory regardless of outcome.

    The trial directory is always deleted in the finally block. Trial artifacts
    are never shared between workers because the directory name includes both
    the trial number (assigned by Optuna DB) and the worker PID.
    """

    def __init__(
        self,
        base_config: CustomFinetuneConfig,
        search_space_builder: SearchSpaceBuilder,
        device: torch.device,
        trial_base_dir: str,
        safe_num_workers: int,
    ) -> None:
        """
        Args:
            base_config:           Original config (never mutated).
            search_space_builder:  Configured SearchSpaceBuilder instance.
            device:                CUDA device for this worker process.
            trial_base_dir:        Root directory under which per-trial dirs are created.
            safe_num_workers:      Pre-computed safe DataLoader worker count.
        """
        self.base_config = base_config
        self.ssb = search_space_builder
        self.device = device
        self.trial_base_dir = trial_base_dir
        self.safe_num_workers = safe_num_workers
        self._logger = logging.getLogger(
            f"{__name__}.TokenizerObjective.pid{os.getpid()}"
        )

    def __call__(self, trial: "optuna.Trial") -> float:
        """
        Executes one HPO trial for the tokenizer.

        Returns:
            Validation loss (float) to be minimized by Optuna.

        Raises:
            optuna.exceptions.TrialPruned: On model load failure or training crash.
        """
        tokenizer: Optional[NosTokenizer] = None
        trial_dir: Optional[str] = None

        try:
            # ── Step 1: Sample hyperparameters ────────────────────────────
            overrides = self.ssb.build_overrides(trial)

            # Use reduced epochs for HPO speed; suppress per-step log spam
            overrides["tokenizer_epochs"] = self.base_config.hpo_tokenizer_epochs
            overrides["log_interval"]     = 999_999
            overrides["num_workers"]      = self.safe_num_workers

            self._logger.info(
                f"Trial {trial.number} | Sampled overrides: {overrides}"
            )

            # ── Step 2: Build fully isolated trial config ──────────────────
            # PID inclusion guarantees uniqueness even if the Optuna DB
            # assigns the same trial number to two workers simultaneously
            # (which can happen when a worker restarts mid-study).
            worker_tag = f"pid{os.getpid()}_trial{trial.number}"
            trial_dir  = os.path.join(
                self.trial_base_dir, f"tokenizer_{worker_tag}"
            )
            os.makedirs(trial_dir, exist_ok=True)

            trial_config = self.base_config.clone_with_overrides(overrides)

            # Force ALL save paths into the isolated trial directory.
            # This prevents any overlap with the canonical save paths
            # that other workers or the main training pipeline might use.
            trial_config.tokenizer_save_path       = trial_dir
            trial_config.tokenizer_best_model_path = os.path.join(trial_dir, "best_model")
            trial_config.base_save_path            = trial_dir

            # ── Step 3: Load model ────────────────────────────────────────
            try:
                if getattr(trial_config, "pre_trained_tokenizer", True):
                    tokenizer = NosTokenizer.from_pretrained(
                        self.base_config.pretrained_tokenizer_path
                    )
                else:
                    arch_path = os.path.join(
                        self.base_config.pretrained_tokenizer_path, "config.json"
                    )
                    with open(arch_path, "r", encoding="utf-8") as f:
                        arch = json.load(f)
                    valid_keys = [
                        "d_in", "d_model", "n_heads", "ff_dim",
                        "n_enc_layers", "n_dec_layers", "ffn_dropout_p",
                        "attn_dropout_p", "resid_dropout_p", "s1_bits",
                        "s2_bits", "beta", "gamma0", "gamma", "zeta", "group_size",
                    ]
                    tokenizer = NosTokenizer(
                        **{k: arch[k] for k in valid_keys if k in arch}
                    )

                tokenizer = apply_bsq_overrides(tokenizer, trial_config)
                tokenizer = apply_dropout_overrides(tokenizer, trial_config)
                tokenizer = tokenizer.to(self.device)

            except FileNotFoundError as exc:
                self._logger.error(
                    f"Trial {trial.number}: Pretrained tokenizer not found "
                    f"at '{self.base_config.pretrained_tokenizer_path}': {exc}"
                )
                raise optuna.exceptions.TrialPruned() from exc
            except Exception as exc:
                self._logger.error(
                    f"Trial {trial.number}: Model loading failed: {exc}",
                    exc_info=True,
                )
                raise optuna.exceptions.TrialPruned() from exc

            # ── Step 4: Run training ──────────────────────────────────────
            trial_logger = _get_trial_logger(
                f"hpo_tok_trial_{trial.number}", trial_dir
            )

            try:
                from finetune_tokenizer import train_tokenizer  # type: ignore
                val_loss: float = train_tokenizer(
                    tokenizer, self.device, trial_config, trial_dir, trial_logger, trial=trial
                )
            except optuna.exceptions.TrialPruned:
                raise  # Pruning signals must propagate — never catch them
            except (RuntimeError, ValueError, torch.cuda.OutOfMemoryError) as exc:
                self._logger.warning(
                    f"Trial {trial.number}: Training failed with expected "
                    f"transient error ({type(exc).__name__}): {exc}"
                )
                raise optuna.exceptions.TrialPruned() from exc
            except Exception as exc:
                self._logger.error(
                    f"Trial {trial.number}: Training failed with unexpected "
                    f"error: {exc}",
                    exc_info=True,
                )
                raise optuna.exceptions.TrialPruned() from exc

            self._logger.info(
                f"Trial {trial.number} COMPLETE | "
                f"val_loss={val_loss:.6f} | overrides={overrides}"
            )
            print(
                f"  [GPU {self.device}] Trial {trial.number}: "
                f"val_loss={val_loss:.6f}"
            )
            return val_loss

        finally:
            # ── Step 5: Deterministic GPU memory release ───────────────────
            # This executes regardless of success, pruning, or exception.
            # Skipping this causes PyTorch allocator cache accumulation,
            # which leads to OOM on trial N+K even though trial N "freed" its memory.
            # Explicitly delete local references so GC can collect them.
            if 'tokenizer' in locals():
                del tokenizer

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # ── Step 6: Delete trial artifacts ────────────────────────────
            if trial_dir and os.path.exists(trial_dir):
                shutil.rmtree(trial_dir, ignore_errors=True)


class BasemodelObjective:
    """
    Optuna objective function for predictor/basemodel HPO.

    Assumes a finetuned (or pretrained) tokenizer is available at a fixed
    path. The tokenizer is frozen (eval mode, no gradients) during basemodel
    HPO to isolate the predictor hyperparameter search from tokenizer variance.

    The lifecycle mirrors TokenizerObjective with the addition of loading
    and freezing the tokenizer before loading the predictor.
    """

    def __init__(
        self,
        base_config: CustomFinetuneConfig,
        search_space_builder: SearchSpaceBuilder,
        device: torch.device,
        trial_base_dir: str,
        tokenizer_path: str,
        safe_num_workers: int,
    ) -> None:
        """
        Args:
            base_config:           Original config (never mutated).
            search_space_builder:  Configured SearchSpaceBuilder instance.
            device:                CUDA device for this worker process.
            trial_base_dir:        Root directory for per-trial isolation.
            tokenizer_path:        Path to the finetuned tokenizer checkpoint.
            safe_num_workers:      Pre-computed safe DataLoader worker count.
        """
        self.base_config = base_config
        self.ssb = search_space_builder
        self.device = device
        self.trial_base_dir = trial_base_dir
        self.tokenizer_path = tokenizer_path
        self.safe_num_workers = safe_num_workers
        self._logger = logging.getLogger(
            f"{__name__}.BasemodelObjective.pid{os.getpid()}"
        )

    def __call__(self, trial: "optuna.Trial") -> float:
        """
        Executes one HPO trial for the basemodel predictor.

        Returns:
            Validation loss (float) to be minimized by Optuna.

        Raises:
            optuna.exceptions.TrialPruned: On model load failure or training crash.
        """
        tokenizer: Optional[NosTokenizer] = None
        model: Optional[Nos] = None
        trial_dir: Optional[str] = None

        try:
            # ── Step 1: Sample hyperparameters ────────────────────────────
            overrides = self.ssb.build_overrides(trial)
            overrides["basemodel_epochs"] = self.base_config.hpo_basemodel_epochs
            overrides["log_interval"]     = 999_999
            overrides["num_workers"]      = self.safe_num_workers

            self._logger.info(
                f"Trial {trial.number} | Sampled overrides: {overrides}"
            )

            # ── Step 2: Build isolated trial config ────────────────────────
            worker_tag = f"pid{os.getpid()}_trial{trial.number}"
            trial_dir  = os.path.join(
                self.trial_base_dir, f"basemodel_{worker_tag}"
            )
            os.makedirs(trial_dir, exist_ok=True)

            trial_config = self.base_config.clone_with_overrides(overrides)
            trial_config.finetuned_tokenizer_path  = self.tokenizer_path
            trial_config.basemodel_save_path       = trial_dir
            trial_config.basemodel_best_model_path = os.path.join(trial_dir, "best_model")
            trial_config.base_save_path            = trial_dir

            # ── Step 3: Load and freeze tokenizer ─────────────────────────
            try:
                tokenizer = NosTokenizer.from_pretrained(self.tokenizer_path)
                tokenizer = tokenizer.to(self.device)
                tokenizer.eval()
                for param in tokenizer.parameters():
                    param.requires_grad_(False)
            except FileNotFoundError as exc:
                self._logger.error(
                    f"Trial {trial.number}: Tokenizer not found "
                    f"at '{self.tokenizer_path}': {exc}"
                )
                raise optuna.exceptions.TrialPruned() from exc
            except Exception as exc:
                self._logger.error(
                    f"Trial {trial.number}: Tokenizer loading failed: {exc}",
                    exc_info=True,
                )
                raise optuna.exceptions.TrialPruned() from exc

            # ── Step 4: Load predictor model ──────────────────────────────
            try:
                if getattr(trial_config, "pre_trained_predictor", True):
                    model = Nos.from_pretrained(
                        self.base_config.pretrained_predictor_path
                    )
                else:
                    arch_path = os.path.join(
                        self.base_config.pretrained_predictor_path, "config.json"
                    )
                    with open(arch_path, "r", encoding="utf-8") as f:
                        arch = json.load(f)
                    valid_keys = [
                        "s1_bits", "s2_bits", "n_layers", "d_model",
                        "n_heads", "ff_dim", "ffn_dropout_p", "attn_dropout_p",
                        "resid_dropout_p", "token_dropout_p", "learn_te",
                    ]
                    model = Nos(**{k: arch[k] for k in valid_keys if k in arch})

                model = apply_dropout_overrides(model, trial_config)
                model = model.to(self.device)

            except FileNotFoundError as exc:
                self._logger.error(
                    f"Trial {trial.number}: Predictor model not found "
                    f"at '{self.base_config.pretrained_predictor_path}': {exc}"
                )
                raise optuna.exceptions.TrialPruned() from exc
            except Exception as exc:
                self._logger.error(
                    f"Trial {trial.number}: Predictor loading failed: {exc}",
                    exc_info=True,
                )
                raise optuna.exceptions.TrialPruned() from exc

            # ── Step 5: Run training ──────────────────────────────────────
            trial_logger = _get_trial_logger(
                f"hpo_base_trial_{trial.number}", trial_dir
            )

            try:
                from finetune_base_model import train_model  # type: ignore
                val_loss: float = train_model(
                    model, tokenizer, self.device,
                    trial_config, trial_dir, trial_logger, trial=trial
                )
            except optuna.exceptions.TrialPruned:
                raise
            except (RuntimeError, ValueError, torch.cuda.OutOfMemoryError) as exc:
                self._logger.warning(
                    f"Trial {trial.number}: Training failed with expected "
                    f"transient error ({type(exc).__name__}): {exc}"
                )
                raise optuna.exceptions.TrialPruned() from exc
            except Exception as exc:
                self._logger.error(
                    f"Trial {trial.number}: Training failed with unexpected "
                    f"error: {exc}",
                    exc_info=True,
                )
                raise optuna.exceptions.TrialPruned() from exc

            self._logger.info(
                f"Trial {trial.number} COMPLETE | "
                f"val_loss={val_loss:.6f} | overrides={overrides}"
            )
            print(
                f"  [GPU {self.device}] Trial {trial.number}: "
                f"val_loss={val_loss:.6f}"
            )
            return val_loss

        finally:
            # Explicitly delete local references so GC can collect them.
            if 'model' in locals():
                del model
            if 'tokenizer' in locals():
                del tokenizer

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            if trial_dir and os.path.exists(trial_dir):
                shutil.rmtree(trial_dir, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════════════
# HPO Runner
# ══════════════════════════════════════════════════════════════════════════════

class NosHPOTuner:
    """
    Main HPO orchestrator for the Nos two-phase training pipeline.

    Responsibilities:
    - Device setup with CUDA_VISIBLE_DEVICES awareness for multi-GPU isolation.
    - Optuna study creation with SQLite WAL mode and concurrency hardening.
    - Tokenizer and basemodel HPO phases with correct sampler configuration.
    - Atomic result persistence (race-condition-safe JSON writes).
    - Best parameter application back to YAML config.

    Multi-GPU Usage:
        Do NOT use torchrun. Instead, launch independent processes:

            CUDA_VISIBLE_DEVICES=0 python hpo_tuner.py --config cfg.yaml &
            CUDA_VISIBLE_DEVICES=1 python hpo_tuner.py --config cfg.yaml &
            ...
            # Or use launch_hpo.sh which automates this pattern.

        All workers share the same Optuna storage (SQLite or PostgreSQL) and
        independently pick up and execute trials. No synchronisation barriers.
    """

    def __init__(self, config_path: str) -> None:
        """
        Initialises the HPO tuner, sets up device, logging, and directories.

        Args:
            config_path: Path to the YAML configuration file.

        Raises:
            RuntimeError: If Optuna is not installed.
            FileNotFoundError: If config_path does not exist.
        """
        if not OPTUNA_AVAILABLE:
            raise RuntimeError(
                "Optuna is not installed. "
                "Run: pip install optuna plotly"
            )

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")

        self.config_path = config_path
        self.config = CustomFinetuneConfig(config_path)

        # ── Device setup (must precede logging setup for log annotations) ─
        self.device = self._setup_device()
        self._worker_tag = f"pid{os.getpid()}_{str(self.device).replace(':', '')}"

        # ── Directory layout ───────────────────────────────────────────────
        self.trial_base_dir = os.path.join(
            self.config.base_save_path, "hpo_trials"
        )
        self.results_dir = os.path.join(
            self.config.base_save_path, "hpo_results"
        )
        self.log_dir = os.path.join(
            self.config.base_save_path, "hpo_logs"
        )
        for directory in (self.trial_base_dir, self.results_dir, self.log_dir):
            os.makedirs(directory, exist_ok=True)

        # ── Logging ───────────────────────────────────────────────────────
        _configure_root_logger(self.log_dir, self._worker_tag)
        self._logger = logging.getLogger(__name__)
        self._logger.info(
            f"NosHPOTuner initialised | config={config_path} | "
            f"device={self.device} | worker_tag={self._worker_tag}"
        )

        # ── Pre-compute safe DataLoader concurrency ────────────────────────
        self._safe_num_workers = _compute_safe_num_workers(
            n_gpu_workers=self._detect_n_concurrent_workers()
        )

        # Results accumulator for this worker's session
        self.results: Dict[str, Any] = {}

    # ── Device Setup ──────────────────────────────────────────────────────────

    def _setup_device(self) -> torch.device:
        """
        Selects the correct GPU device for this worker process.

        Priority resolution order:
          1. CUDA_VISIBLE_DEVICES environment variable (set by launch_hpo.sh).
             When set to a single GPU ID (e.g., "3"), CUDA remaps it to
             logical index 0 within this process. We use cuda:0 in this case.
          2. device_id from YAML config (for single-machine manual override).
          3. CPU fallback if CUDA is unavailable.

        Returns:
            Resolved torch.device for this worker.
        """
        if not (self.config.use_cuda and torch.cuda.is_available()):
            logging.getLogger(__name__).info(
                "CUDA not available or disabled in config. Using CPU."
            )
            return torch.device("cpu")

        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)

        if cuda_visible is not None:
            # Process-isolation mode: the bash launcher set this env var.
            # CUDA remaps the assigned physical GPU to logical index 0.
            visible_list = [
                v.strip() for v in cuda_visible.split(",")
                if v.strip().lstrip("-").isdigit()
            ]

            if len(visible_list) == 0:
                # Malformed or empty CUDA_VISIBLE_DEVICES — fall back to config
                logical_id  = int(self.config.device_id)
                physical_id = logical_id
                logging.getLogger(__name__).warning(
                    f"CUDA_VISIBLE_DEVICES='{cuda_visible}' is not a valid "
                    f"GPU index list. Falling back to config device_id={logical_id}."
                )
            elif len(visible_list) == 1:
                # Exactly one GPU visible — correct process isolation
                logical_id  = 0
                physical_id = int(visible_list[0])
            else:
                # Multiple GPUs visible — warn and use first
                logical_id  = 0
                physical_id = int(visible_list[0])
                logging.getLogger(__name__).warning(
                    f"CUDA_VISIBLE_DEVICES='{cuda_visible}' exposes multiple GPUs. "
                    f"For true process isolation, assign one GPU per worker. "
                    f"Using logical cuda:0 (physical GPU {physical_id})."
                )
        else:
            # Single-machine mode — use config device_id directly
            logical_id  = int(self.config.device_id)
            physical_id = logical_id

        torch.cuda.set_device(logical_id)
        device = torch.device(f"cuda:{logical_id}")

        gpu_props = torch.cuda.get_device_properties(device)
        logging.getLogger(__name__).info(
            f"[PID {os.getpid()}] Using cuda:{logical_id} "
            f"(physical GPU {physical_id}: {gpu_props.name}, "
            f"{gpu_props.total_memory / 1e9:.1f} GB VRAM)"
        )
        return device

    # ── Worker Concurrency Detection ──────────────────────────────────────────

    def _detect_n_concurrent_workers(self) -> int:
        """
        Estimates the number of concurrent GPU workers in this HPO run.

        Used to compute a safe per-process DataLoader num_workers count.
        Reads CUDA_VISIBLE_DEVICES if set by the launcher; otherwise assumes
        single-worker mode.

        Returns:
            Estimated number of concurrent GPU workers (>=1).
        """
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible and cuda_visible not in ("NoDevFiles", "-1"):
            n = len([v for v in cuda_visible.split(",") if v.strip().isdigit()])
            if n > 0:
                return n
        # If CUDA_VISIBLE_DEVICES is not set, count all available GPUs
        if torch.cuda.is_available():
            return max(1, torch.cuda.device_count())
        return 1

    # ── Optuna Internals ──────────────────────────────────────────────────────

    def _build_sampler(self) -> "optuna.samplers.BaseSampler":
        """
        Constructs the Optuna sampler.

        For async multi-worker HPO, TPESampler with constant_liar=True is
        strongly recommended. Without constant_liar, all workers sample
        simultaneously before any trial completes, so the TPE model has
        zero completed trials to learn from and every worker independently
        suggests the same near-identical parameters (correlated exploration).

        constant_liar=True tells TPE to treat in-progress trials as if they
        returned the current worst observed value, encouraging each worker
        to explore a different region of the search space.

        Returns:
            Configured sampler instance.
        """
        name = self.config.hpo_sampler.lower()
        seed = getattr(self.config, "seed", 42)
        # Need enough startup trials before TPE's probabilistic model is reliable.
        # Require at least 2 full rounds of workers to complete before TPE kicks in.
        n_workers  = self._detect_n_concurrent_workers()
        n_startup  = max(10, 2 * n_workers)

        if name == "tpe":
            return TPESampler(
                seed=seed,
                multivariate=True,      # Model parameter correlations
                constant_liar=True,     # Critical for async multi-worker HPO
                n_startup_trials=n_startup,
            )
        elif name == "random":
            return RandomSampler(seed=seed)
        elif name == "cmaes":
            self._logger.warning(
                "CMA-ES sampler does not support constant_liar. "
                "Workers will suggest correlated parameters in async mode. "
                "TPE is recommended for distributed HPO."
            )
            return CmaEsSampler(seed=seed)
        else:
            self._logger.warning(
                f"Unknown sampler '{name}'. Defaulting to TPE with constant_liar."
            )
            return TPESampler(
                seed=seed,
                constant_liar=True,
                n_startup_trials=n_startup,
            )

    def _build_pruner(self) -> "optuna.pruners.BasePruner":
        """
        Constructs the Optuna pruner.

        MedianPruner is conservative and works well for training workloads
        where intermediate values are reported via trial.report().
        HyperbandPruner is more aggressive and better for very large search spaces.

        Returns:
            Configured pruner instance.
        """
        name = self.config.hpo_pruner.lower()
        if name == "median":
            return MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=1,
                interval_steps=1,
            )
        elif name == "hyperband":
            return HyperbandPruner()
        else:
            return NopPruner()

    def _create_study(self, study_name_suffix: str = "") -> "optuna.Study":
        """
        Creates or loads an Optuna study from the configured storage backend.

        For SQLite storage, applies WAL (Write-Ahead Logging) mode via a
        connection event hook. WAL allows concurrent readers while one
        writer is active, dramatically reducing lock contention when 8
        workers simultaneously commit trial results.

        For non-SQLite storage (PostgreSQL, MySQL), the URI is passed
        directly without modification.

        load_if_exists=True is mandatory for multi-worker mode: all workers
        must join the same existing study rather than each creating a new one.

        Args:
            study_name_suffix: Appended to base study name (e.g., "_tokenizer").

        Returns:
            Configured optuna.Study instance.
        """
        study_name = f"{self.config.hpo_study_name}{study_name_suffix}"
        storage_uri = self.config.hpo_storage

        self._logger.info(
            f"Creating/loading study '{study_name}' | storage={storage_uri}"
        )

        if storage_uri is not None and storage_uri.startswith("sqlite"):
            storage = self._build_sqlite_storage(storage_uri)
        else:
            # None → in-memory (single process only)
            # postgresql:// or mysql:// → passed through directly
            storage = storage_uri

        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction=self.config.hpo_direction,
            sampler=self._build_sampler(),
            pruner=self._build_pruner(),
            load_if_exists=True,  # Mandatory for multi-worker mode
        )
        self._logger.info(
            f"Study ready: {len(study.trials)} existing trials loaded."
        )
        return study

    def _build_sqlite_storage(self, uri: str) -> "RDBStorage":
        db_path = uri.split("?")[0].replace("sqlite:///", "")
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

        storage = RDBStorage(
            url=uri,
            engine_kwargs={
                "connect_args": {
                     "timeout": 60,
                     "check_same_thread": False,
                },
                "poolclass": NullPool,  # CRITICAL FIX: Disable pooling to prevent lock storms
           },
           skip_compatibility_check=False,
       )

        try:
             from sqlalchemy import event as sa_event

             @sa_event.listens_for(storage.engine, "connect")
             def _apply_wal_pragmas(dbapi_connection, connection_record):
                 cursor = dbapi_connection.cursor()
                 cursor.execute("PRAGMA journal_mode=WAL")
                 cursor.execute("PRAGMA synchronous=NORMAL")
                 cursor.execute("PRAGMA busy_timeout=300000")
                 cursor.close()

             self._logger.info(
                 "SQLite WAL mode, NullPool, and busy_timeout=300s applied via connection hook."
             )
        except Exception as exc:
             self._logger.warning(
                 f"Could not apply WAL mode pragmas: {exc}. "
                 f"Falling back to URI timeout parameter only."
            )

        return storage

    # ── Phase: Tokenizer ──────────────────────────────────────────────────────

    def tune_tokenizer(
        self, n_trials: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Runs Optuna HPO for the tokenizer training phase.

        Args:
            n_trials: Number of trials to run. Overrides config if provided.

        Returns:
            Dict with keys 'value' (best val_loss) and 'params' (best hyperparams).
        """
        n = n_trials or self.config.hpo_n_trials
        self._logger.info(
            f"Starting tokenizer HPO: {n} trials | device={self.device}"
        )
        print(f"\n{'='*60}")
        print(f"HPO Phase 1: Tokenizer | {n} trials | {self.device}")
        print(f"{'='*60}")

        ssb = SearchSpaceBuilder(
            self.config.hpo_search_space, phase="tokenizer"
        )
        objective = TokenizerObjective(
            base_config=self.config,
            search_space_builder=ssb,
            device=self.device,
            trial_base_dir=self.trial_base_dir,
            safe_num_workers=self._safe_num_workers,
        )
        study = self._create_study("_tokenizer")
        failure_callback = _build_failure_rate_callback(
            max_failure_rate=0.5,
            min_trials_before_check=4,
        )

        study.optimize(
            objective,
            n_trials=n,
            show_progress_bar=True,
            # Only catch expected transient failures caused by extreme
            # hyperparameter values (OOM from huge batch, NaN from bad LR).
            # Systematic bugs (wrong data path, import error) will NOT be
            # caught here — they will raise and crash the worker intentionally.
            catch=(
                RuntimeError,
                ValueError,
                torch.cuda.OutOfMemoryError,
            ),
            callbacks=[failure_callback],
        )

        best = self._extract_best(study)
        self.results["tokenizer"] = best

        self._logger.info(
            f"Tokenizer HPO complete | best_val_loss={best['value']:.6f} | "
            f"best_params={best['params']}"
        )
        print(f"\n✅ Best tokenizer val_loss: {best['value']:.6f}")
        print(f"   Best params: {best['params']}")

        self._save_study_results(study, "tokenizer")
        return best

    # ── Phase: Basemodel ──────────────────────────────────────────────────────

    def tune_basemodel(
        self,
        tokenizer_path: Optional[str] = None,
        n_trials: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Runs Optuna HPO for the basemodel predictor training phase.

        Requires a finetuned (or original pretrained) tokenizer checkpoint.
        The tokenizer is frozen throughout all basemodel trials.

        Args:
            tokenizer_path: Path to tokenizer checkpoint. Defaults to
                            config.finetuned_tokenizer_path.
            n_trials:       Number of trials. Overrides config if provided.

        Returns:
            Dict with keys 'value' (best val_loss) and 'params' (best hyperparams).

        Raises:
            FileNotFoundError: If tokenizer_path does not exist.
        """
        n = n_trials or self.config.hpo_n_trials

        if tokenizer_path is None:
            tokenizer_path = self.config.finetuned_tokenizer_path
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(
                f"Tokenizer checkpoint not found at: '{tokenizer_path}'\n"
                f"Run tokenizer finetuning or tokenizer HPO first, or "
                f"provide --tokenizer-path explicitly."
            )

        self._logger.info(
            f"Starting basemodel HPO: {n} trials | "
            f"device={self.device} | tokenizer={tokenizer_path}"
        )
        print(f"\n{'='*60}")
        print(f"HPO Phase 2: Basemodel | {n} trials | {self.device}")
        print(f"Tokenizer: {tokenizer_path}")
        print(f"{'='*60}")

        ssb = SearchSpaceBuilder(
            self.config.hpo_search_space, phase="basemodel"
        )
        objective = BasemodelObjective(
            base_config=self.config,
            search_space_builder=ssb,
            device=self.device,
            trial_base_dir=self.trial_base_dir,
            tokenizer_path=tokenizer_path,
            safe_num_workers=self._safe_num_workers,
        )
        study = self._create_study("_basemodel")
        failure_callback = _build_failure_rate_callback(
            max_failure_rate=0.5,
            min_trials_before_check=4,
        )

        study.optimize(
            objective,
            n_trials=n,
            show_progress_bar=True,
            catch=(
                RuntimeError,
                ValueError,
                torch.cuda.OutOfMemoryError,
            ),
            callbacks=[failure_callback],
        )

        best = self._extract_best(study)
        self.results["basemodel"] = best

        self._logger.info(
            f"Basemodel HPO complete | best_val_loss={best['value']:.6f} | "
            f"best_params={best['params']}"
        )
        print(f"\n✅ Best basemodel val_loss: {best['value']:.6f}")
        print(f"   Best params: {best['params']}")

        self._save_study_results(study, "basemodel")
        return best

    # ── Full Pipeline ─────────────────────────────────────────────────────────

    def tune_full_pipeline(self) -> Dict[str, Any]:
        """
        Runs the complete two-phase HPO pipeline:
          1. Tokenizer HPO → find best tokenizer hyperparameters.
          2. Full tokenizer training with best params (to produce a stable checkpoint).
          3. Basemodel HPO using the stable tokenizer checkpoint.

        Returns:
            Dict containing results from both phases.
        """
        all_results: Dict[str, Any] = {}

        if self.config.hpo_optimize_tokenizer:
            tok_best = self.tune_tokenizer()
            all_results["tokenizer"] = tok_best

            # Train the tokenizer with best params at full epochs so the
            # basemodel HPO uses a well-converged tokenizer checkpoint.
            print("\n📦 Training tokenizer with best params for basemodel HPO...")
            self._logger.info(
                "Training full tokenizer with best HPO params before basemodel phase."
            )
            best_tok_config = self.config.clone_with_overrides(tok_best["params"])
            self._train_best_tokenizer(best_tok_config)

        if self.config.hpo_optimize_basemodel:
            tok_path = self.config.tokenizer_best_model_path
            base_best = self.tune_basemodel(tokenizer_path=tok_path)
            all_results["basemodel"] = base_best

        self._print_final_report(all_results)
        return all_results

    def _train_best_tokenizer(self, config: CustomFinetuneConfig) -> None:
        """
        Runs one full (non-HPO) tokenizer training pass with the given config.
        Used to produce a stable tokenizer checkpoint before basemodel HPO.

        Args:
            config: Config instance with best HPO hyperparameters applied.
        """
        tokenizer: Optional[NosTokenizer] = None
        try:
            if getattr(config, "pre_trained_tokenizer", True):
                tokenizer = NosTokenizer.from_pretrained(
                    config.pretrained_tokenizer_path
                )
            else:
                raise ValueError(
                    "pre_trained_tokenizer=False requires manual model "
                    "instantiation before calling _train_best_tokenizer."
                )
            tokenizer = apply_bsq_overrides(tokenizer, config)
            tokenizer = apply_dropout_overrides(tokenizer, config)
            tokenizer = tokenizer.to(self.device)

            os.makedirs(config.tokenizer_save_path, exist_ok=True)
            logger = _get_trial_logger("hpo_best_tokenizer", config.tokenizer_save_path)

            from finetune_tokenizer import train_tokenizer  # type: ignore
            train_tokenizer(
                tokenizer, self.device, config,
                config.tokenizer_save_path, logger,
            )
            self._logger.info(
                f"Best tokenizer training complete. "
                f"Saved to: {config.tokenizer_save_path}"
            )
        finally:
            release_gpu_memory(tokenizer)

    # ── Results Persistence ───────────────────────────────────────────────────

    @staticmethod
    def _extract_best(study: "optuna.Study") -> Dict[str, Any]:
        """
        Safely extracts best trial information from a study.

        Handles the edge case where no trials completed successfully
        (all pruned or failed), which would cause study.best_value to raise.

        Args:
            study: Completed Optuna study.

        Returns:
            Dict with 'value' and 'params' keys.
        """
        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if not completed:
            logging.getLogger(__name__).error(
                "No trials completed successfully. "
                "All trials were pruned or failed. "
                "Inspect per-trial logs for root cause."
            )
            return {"value": float("inf"), "params": {}}

        return {
            "value": study.best_value,
            "params": study.best_params,
        }

    def _save_study_results(
        self, study: "optuna.Study", phase: str
    ) -> None:
        """
        Persists all trial results for a study phase to a JSON file.

        Uses an atomic write pattern (write to .tmp → os.replace to final path)
        to guarantee the output file is never in a partially-written state,
        even if 8 workers call this simultaneously. os.replace() is atomic
        on POSIX and near-atomic on Windows (the last writer wins, but the
        file is never corrupt).

        Individual worker results are tagged with the worker PID so they
        can be distinguished if needed. The final merged view comes from
        reading the Optuna study, which contains all workers' trials.

        Args:
            study: The completed Optuna study.
            phase: Phase label ('tokenizer' or 'basemodel').
        """
        trials_data = []
        for t in study.trials:
            if t.value is not None:
                trials_data.append({
                    "number":            t.number,
                    "value":             t.value,
                    "params":            t.params,
                    "state":             str(t.state),
                    "datetime_start":    (
                        t.datetime_start.isoformat()
                        if t.datetime_start else None
                    ),
                    "datetime_complete": (
                        t.datetime_complete.isoformat()
                        if t.datetime_complete else None
                    ),
                })

        completed = [t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE]
        payload = {
            "best_value":   study.best_value  if completed else None,
            "best_params":  study.best_params if completed else {},
            "n_trials":     len(study.trials),
            "n_completed":  len(completed),
            "n_pruned":     sum(
                1 for t in study.trials
                if t.state == optuna.trial.TrialState.PRUNED
            ),
            "n_failed":     sum(
                1 for t in study.trials
                if t.state == optuna.trial.TrialState.FAIL
            ),
            "trials":       trials_data,
            "timestamp":    datetime.datetime.now().isoformat(),
            "worker_tag":   self._worker_tag,
        }

        # Atomic write: temp file → rename
        final_path = os.path.join(self.results_dir, f"{phase}_trials.json")
        tmp_path   = f"{final_path}.{self._worker_tag}.tmp"

        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)

            # ── TASK 2.1: Windows race-condition backoff for atomic writes ──
            # OneDrive or other indexing services often hold transient read-locks
            # on newly created files. This ensures os.replace doesn't crash the worker.
            for attempt in range(5):
                try:
                    os.replace(tmp_path, final_path)  # Atomic on POSIX, near-atomic on Windows
                    self._logger.info(f"Results atomically written to: {final_path}")
                    print(f"📊 Results saved to: {final_path}")
                    break
                except PermissionError:
                    if attempt == 4:
                        raise  # Re-raise after exhausting retries
                    time.sleep(0.1 * (2 ** attempt))  # Exponential backoff: 0.1s, 0.2s, 0.4s, 0.8s

        except Exception as exc:
            self._logger.error(
                f"Failed to save results to {final_path}: {exc}", exc_info=True
            )
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        # ── Optional: Save visualisation HTML plots ────────────────────────
        self._save_visualisations(study, phase)

    def _save_visualisations(
        self, study: "optuna.Study", phase: str
    ) -> None:
        """
        Attempts to save Optuna HTML visualisation plots.

        Silently skips if plotly is not installed or no completed trials exist.

        Args:
            study: Completed Optuna study.
            phase: Phase label for file naming.
        """
        try:
            import optuna.visualization as vis
            completed = [t for t in study.trials
                         if t.state == optuna.trial.TrialState.COMPLETE]
            if len(completed) < 2:
                return  # Not enough trials for meaningful plots

            plots = {
                f"{phase}_param_importances.html": vis.plot_param_importances,
                f"{phase}_optimization_history.html": vis.plot_optimization_history,
                f"{phase}_parallel_coordinate.html": vis.plot_parallel_coordinate,
            }
            for filename, plot_fn in plots.items():
                try:
                    fig = plot_fn(study)
                    fig.write_html(os.path.join(self.results_dir, filename))
                except Exception:
                    pass  # Individual plot failure should not abort saving

            self._logger.info(
                f"Visualisation plots saved to: {self.results_dir}"
            )
            print(f"📈 Plots saved to: {self.results_dir}")
        except ImportError:
            pass  # plotly not installed — skip silently

    # ── Reporting ─────────────────────────────────────────────────────────────

    def _print_final_report(self, results: Dict[str, Any]) -> None:
        """Prints a formatted summary of all HPO results to stdout."""
        print(f"\n{'='*60}")
        print("HPO FINAL REPORT")
        print(f"{'='*60}")
        for phase, result in results.items():
            print(f"\n{'─'*40}")
            print(f"Phase: {phase.upper()}")
            if result["value"] == float("inf"):
                print("  ⚠️  No trials completed successfully.")
            else:
                print(f"  Best val loss: {result['value']:.6f}")
                print(f"  Best hyperparameters:")
                for k, v in sorted(result["params"].items()):
                    print(f"    {k:<40}: {v}")
        print(f"\n{'='*60}")

    def print_importance_report(self) -> None:
        """
        Prints a ranked hyperparameter importance table using Optuna's
        fANOVA estimator. Requires at least 4 completed trials and a
        persistent storage backend (not in-memory).
        """
        if not OPTUNA_AVAILABLE:
            return

        storage_uri = self.config.hpo_storage
        if storage_uri is None:
            print(
                "\n⚠️  Parameter importance report requires persistent storage "
                "(set hpo.storage in config). Skipping."
            )
            return

        print(f"\n{'='*60}")
        print("HYPERPARAMETER IMPORTANCE RANKING")
        print(f"{'='*60}")

        for phase in ("tokenizer", "basemodel"):
            study_name = f"{self.config.hpo_study_name}_{phase}"
            try:
                study = optuna.load_study(
                    study_name=study_name, storage=storage_uri
                )
                completed = [
                    t for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE
                ]
                if len(completed) < 4:
                    print(
                        f"\n{phase.upper()}: Not enough completed trials "
                        f"({len(completed)}/4 minimum) for importance analysis."
                    )
                    continue

                importances = optuna.importance.get_param_importances(study)
                print(f"\n{phase.upper()} importances:")
                for param, imp in sorted(
                    importances.items(), key=lambda x: x[1], reverse=True
                ):
                    bar = "█" * int(imp * 40)
                    print(f"  {param:<42} {imp:.4f}  {bar}")
            except Exception as exc:
                self._logger.debug(
                    f"Could not compute importances for {phase}: {exc}"
                )

    # ── Config Application ────────────────────────────────────────────────────

    def apply_best_to_config(
        self, output_config_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Writes the best found hyperparameters from all completed phases back
        into a new YAML config file. The original config is never modified.

        The `clip` parameter is a data-preprocessing scalar that belongs in
        the `data:` YAML block (where CustomFinetuneConfig reads it from).
        All other optimised parameters are written to the `training:` block.

        Args:
            output_config_path: Output path. Defaults to
                                <original_name>_hpo_best.yaml.

        Returns:
            Path to the written config file, or None if no results exist.
        """
        if not self.results:
            print(
                "No HPO results to apply. "
                "Run tune_tokenizer(), tune_basemodel(), or tune_full_pipeline() first."
            )
            return None

        # Load raw YAML to preserve comments and formatting as much as possible
        with open(self.config_path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)

        # ── Merge all phase best params into a single flat dict ────────────
        all_best_params: Dict[str, Any] = {}
        for phase_results in self.results.values():
            all_best_params.update(phase_results.get("params", {}))

        # ── FIX: Route `clip` to the `data` block, not `training` ─────────
        # `clip` is a data-preprocessing scalar read by CustomFinetuneConfig
        # exclusively from config["data"]["clip"]. Writing it into the
        # `training:` block causes it to be silently ignored at load time,
        # leaving the old (un-optimised) clip value in effect.
        #
        # .pop() removes `clip` from all_best_params so the subsequent
        # .update() call does NOT write a duplicate (and unreachable) `clip`
        # key into the `training:` block.
        if "clip" in all_best_params:
            optimized_clip = all_best_params.pop("clip")
            raw_config.setdefault("data", {})["clip"] = optimized_clip
            self._logger.info(
                f"Routed optimised clip={optimized_clip} → data.clip "
                f"(removed from training block to prevent shadowing)."
            )

        # ── Inject remaining parameters into the training block ────────────
        raw_config.setdefault("training", {}).update(all_best_params)

        # ── Stamp experiment metadata ──────────────────────────────────────
        raw_config.setdefault("experiment", {})["hpo_applied"] = True
        raw_config["experiment"]["hpo_applied_timestamp"] = (
            datetime.datetime.now().isoformat()
        )

        if output_config_path is None:
            base, ext = os.path.splitext(self.config_path)
            output_config_path = f"{base}_hpo_best{ext}"

        with open(output_config_path, "w", encoding="utf-8") as f:
            yaml.dump(
                raw_config, f,
                default_flow_style=False,
                allow_unicode=True,
                indent=2,
                sort_keys=False,
            )

        self._logger.info(f"Best config written to: {output_config_path}")
        print(f"\n✅ Best config written to: {output_config_path}")
        return output_config_path


# ══════════════════════════════════════════════════════════════════════════════
# Utility: Safe DataLoader Worker Count
# ══════════════════════════════════════════════════════════════════════════════

def _compute_safe_num_workers(n_gpu_workers: int = 1) -> int:
    """
    Computes a per-process DataLoader num_workers count that avoids
    CPU and disk I/O saturation when N GPU processes run simultaneously.

    With N=8 GPUs and num_workers=6 (from config), the system would spawn
    8×6=48 DataLoader processes competing for CPU cores and disk bandwidth,
    degrading all workers below single-process performance.

    Formula:
        available_cores = total_cpu_cores × 0.8  (20% reserved for OS/Optuna)
        per_process     = floor(available_cores / n_gpu_workers)
        safe_count      = clamp(per_process, 1, 4)

    The upper clamp of 4 reflects that DataLoader parallelism has strongly
    diminishing returns beyond 4 workers for most I/O workloads, particularly
    when reading from a shared NFS or NVMe that is already multi-reader bound.

    Args:
        n_gpu_workers: Number of concurrent GPU worker processes.

    Returns:
        Safe num_workers value for DataLoader in each worker process.
    """
    total_cores     = multiprocessing.cpu_count()
    available_cores = max(1, int(total_cores * 0.8))
    per_process     = max(1, available_cores // max(1, n_gpu_workers))
    safe_count      = min(per_process, 4)

    logging.getLogger(__name__).info(
        f"DataLoader num_workers: {safe_count} "
        f"(CPU cores={total_cores}, GPU workers={n_gpu_workers})"
    )
    return safe_count


# ══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hpo_tuner.py",
        description=(
            "Asynchronous Hyperparameter Tuning for Nos Model Finetuning.\n\n"
            "Single GPU:\n"
            "  python hpo_tuner.py --config configs/config.yaml\n\n"
            "Multi-GPU (8x A40 cluster):\n"
            "  ./launch_hpo.sh --config configs/config.yaml --n-trials 30"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="both",
        choices=["tokenizer", "basemodel", "both"],
        help="Which training phase to tune. Default: both.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Override n_trials from config.",
    )
    parser.add_argument(
        "--apply-best",
        action="store_true",
        help="Write best params back to a new config file after tuning.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Path to finetuned tokenizer (required for --phase basemodel).",
    )
    parser.add_argument(
        "--output-config",
        type=str,
        default=None,
        help="Output path for best config YAML (default: <config>_hpo_best.yaml).",
    )
    parser.add_argument(
        "--show-importance",
        action="store_true",
        help="Print hyperparameter importance ranking after tuning.",
    )
    return parser


def main() -> None:
    parser = _build_argument_parser()
    args   = parser.parse_args()

    if not OPTUNA_AVAILABLE:
        print(
            "ERROR: Optuna is not installed.\n"
            "Install it with: pip install optuna plotly"
        )
        sys.exit(1)

    # ── Validate phase/tokenizer-path combination ──────────────────────────
    if args.phase == "basemodel" and args.tokenizer_path is None:
        # Will fall back to config.finetuned_tokenizer_path in tune_basemodel;
        # log a warning but don't abort here since config may have it set.
        print(
            "⚠️  --phase basemodel without --tokenizer-path. "
            "Will use finetuned_tokenizer path from config."
        )

    # ── Initialise tuner ──────────────────────────────────────────────────
    try:
        tuner = NosHPOTuner(args.config)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print("\nHPO Worker Configuration:")
    print(f"  Config:      {args.config}")
    print(f"  Phase:       {args.phase}")
    print(f"  Trials:      {args.n_trials or tuner.config.hpo_n_trials}")
    print(f"  Sampler:     {tuner.config.hpo_sampler}")
    print(f"  Pruner:      {tuner.config.hpo_pruner}")
    print(f"  Storage:     {tuner.config.hpo_storage or 'in-memory (single worker only)'}")
    print(f"  Device:      {tuner.device}")
    print(f"  DL Workers:  {tuner._safe_num_workers} per process")
    print(f"  Worker tag:  {tuner._worker_tag}")

    wall_start = time.perf_counter()

    # ── Run HPO ───────────────────────────────────────────────────────────
    try:
        if args.phase == "both":
            results = tuner.tune_full_pipeline()
        elif args.phase == "tokenizer":
            results = tuner.tune_tokenizer(n_trials=args.n_trials)
        elif args.phase == "basemodel":
            results = tuner.tune_basemodel(
                tokenizer_path=args.tokenizer_path,
                n_trials=args.n_trials,
            )
    except KeyboardInterrupt:
        print(
            "\n⚠️  HPO interrupted by user. "
            "Completed trials have been saved to the study database."
        )
        sys.exit(0)

    wall_elapsed = time.perf_counter() - wall_start
    print(f"\n⏱  Total HPO wall time: {wall_elapsed / 60:.1f} minutes")

    # ── Post-run actions ──────────────────────────────────────────────────
    if args.apply_best:
        tuner.apply_best_to_config(args.output_config)

    if args.show_importance:
        tuner.print_importance_report()


if __name__ == "__main__":
    main()
```

---

### `launch_hpo.sh`

```bash
#!/usr/bin/env bash
# =============================================================================
# launch_hpo.sh
# Asynchronous Multi-GPU HPO Launcher for Nos Model Finetuning
#
# Architecture:
#   Spawns one independent hpo_tuner.py process per GPU. Each process gets
#   its own CUDA context via CUDA_VISIBLE_DEVICES, with no shared memory,
#   no DDP, and no synchronisation barriers. All workers share only the
#   Optuna SQLite/PostgreSQL study database.
#
# Compatibility:
#   - Linux        : bash 4.0+  (Ubuntu, Debian, CentOS, RHEL, Amazon Linux)
#   - Windows      : Git Bash, WSL2, Cygwin (all ship bash 4+)
#   - macOS        : bash 3.2+ via /bin/bash (limited testing)
#   - Cloud VMs    : AWS, GCP, Azure GPU instances (all Linux-based)
#   - Windows VMs  : Azure NV-series, AWS G-series with Git Bash / WSL2
#
# Prerequisites:
#   - nvidia-smi   accessible in PATH
#   - python       accessible in PATH (or set PYTHON_BIN below)
#   - hpo_tuner.py in the same directory as this script (or set SCRIPT_DIR)
#
# Usage:
#   chmod +x launch_hpo.sh
#
#   # Tune both phases across all GPUs (30 trials per worker):
#   ./launch_hpo.sh --config configs/config.yaml --phase both --n-trials 30
#
#   # Tune tokenizer only on GPUs 0,1,2 (GPU subset):
#   ./launch_hpo.sh --gpus 0,1,2 --config configs/config.yaml --phase tokenizer
#
#   # Dry run — print what would be launched without executing:
#   ./launch_hpo.sh --dry-run --config configs/config.yaml --phase both
#
#   # Resume a previously interrupted study:
#   ./launch_hpo.sh --config configs/config.yaml --phase both --n-trials 30
#   (Optuna's load_if_exists=True means workers automatically resume)
#
#   # Use a custom Python binary (e.g., conda env):
#   PYTHON_BIN=/opt/conda/envs/nos/bin/python ./launch_hpo.sh --config ...
#
#   # Limit VRAM per GPU (useful on shared machines):
#   GPU_MEMORY_FRACTION=0.8 ./launch_hpo.sh --config configs/config.yaml
#
# Environment Variables (all optional):
#   PYTHON_BIN              Path to python executable. Default: auto-detected.
#   GPU_MEMORY_FRACTION     CUDA memory fraction [0.1-1.0]. Default: unset.
#   HPO_TIMEOUT_SECONDS     Kill workers after N seconds. Default: unset (no limit).
#   HPO_WORKER_NICE         nice(1) priority for workers [−20 to 19]. Default: 0.
#   LOG_DIR                 Override log directory. Default: logs/hpo_workers.
#
# Exit Codes:
#   0   All workers completed successfully.
#   1   One or more workers failed (details in log files).
#   2   Configuration or environment error (bad args, missing deps).
#   3   Interrupted by user (SIGINT/SIGTERM). Partial results are preserved.
# =============================================================================

# ── Bash version guard ────────────────────────────────────────────────────────
# Arrays with negative indexing and associative arrays require bash 4.0+.
# macOS ships bash 3.2 at /bin/bash but Git Bash / Homebrew bash is 5.x.
if [ "${BASH_VERSINFO[0]}" -lt 4 ]; then
    echo "ERROR: bash 4.0 or later is required." >&2
    echo "       Detected: bash ${BASH_VERSION}" >&2
    echo "       macOS users: brew install bash && use /usr/local/bin/bash" >&2
    echo "       Windows users: use Git Bash 4+ or WSL2." >&2
    exit 2
fi

set -euo pipefail
# -e  : Exit on any unhandled error
# -u  : Treat unset variables as errors
# -o pipefail : Propagate pipe failures (not just last command)

# ── Script self-location (works with symlinks and spaces in paths) ─────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"

# ── Colour output (disabled automatically when not a terminal) ────────────────
if [ -t 1 ] && command -v tput &>/dev/null && tput colors &>/dev/null; then
    C_RESET="\033[0m"
    C_BOLD="\033[1m"
    C_RED="\033[31m"
    C_GREEN="\033[32m"
    C_YELLOW="\033[33m"
    C_CYAN="\033[36m"
    C_WHITE="\033[37m"
else
    C_RESET="" C_BOLD="" C_RED="" C_GREEN="" C_YELLOW="" C_CYAN="" C_WHITE=""
fi

# ── Logging helpers ───────────────────────────────────────────────────────────
_ts()    { date "+%Y-%m-%d %H:%M:%S"; }
_info()  { echo -e "${C_CYAN}[$(_ts)] [INFO]${C_RESET}  $*"; }
_ok()    { echo -e "${C_GREEN}[$(_ts)] [ OK ]${C_RESET}  $*"; }
_warn()  { echo -e "${C_YELLOW}[$(_ts)] [WARN]${C_RESET}  $*"; }
_error() { echo -e "${C_RED}[$(_ts)] [ERR ]${C_RESET}  $*" >&2; }
_die()   { _error "$*"; exit 2; }

# ── Default configuration ─────────────────────────────────────────────────────
GPU_SUBSET=""                          # Empty = use all GPUs
DRY_RUN=false
PYTHON_BIN="${PYTHON_BIN:-}"           # Auto-detected below if empty
GPU_MEMORY_FRACTION="${GPU_MEMORY_FRACTION:-}"
HPO_TIMEOUT_SECONDS="${HPO_TIMEOUT_SECONDS:-}"
HPO_WORKER_NICE="${HPO_WORKER_NICE:-0}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/hpo_workers}"
HPO_TUNER_SCRIPT="${SCRIPT_DIR}/hpo_tuner.py"
HPO_ARGS=()                            # Arguments forwarded to hpo_tuner.py

# ── Argument parser ───────────────────────────────────────────────────────────
# Separates launcher-specific flags from hpo_tuner.py pass-through arguments.
_usage() {
    cat <<EOF

${C_BOLD}Usage:${C_RESET}
  ${SCRIPT_NAME} [LAUNCHER OPTIONS] [HPO_TUNER OPTIONS]

${C_BOLD}Launcher Options:${C_RESET}
  --gpus <0,1,2,...>      Comma-separated GPU IDs to use. Default: all GPUs.
  --dry-run               Print launch commands without executing them.
  --log-dir <path>        Override log directory. Default: logs/hpo_workers/
  --help, -h              Show this help message.

${C_BOLD}HPO Tuner Options (passed through to hpo_tuner.py):${C_RESET}
  --config <path>         Path to YAML config. Default: configs/config.yaml
  --phase <phase>         tokenizer | basemodel | both. Default: both
  --n-trials <int>        Number of Optuna trials per worker.
  --apply-best            Write best params to a new config file.
  --tokenizer-path <path> Tokenizer path (for --phase basemodel).
  --output-config <path>  Output path for best config YAML.
  --show-importance       Print hyperparameter importance after tuning.

${C_BOLD}Environment Variables:${C_RESET}
  PYTHON_BIN              Python executable. Default: auto-detected.
  GPU_MEMORY_FRACTION     CUDA memory fraction [0.1-1.0].
  HPO_TIMEOUT_SECONDS     Worker timeout in seconds.
  HPO_WORKER_NICE         Process nice priority [-20 to 19]. Default: 0.
  LOG_DIR                 Log directory override.

${C_BOLD}Examples:${C_RESET}
  ${SCRIPT_NAME} --config configs/config.yaml --phase both --n-trials 30
  ${SCRIPT_NAME} --gpus 0,1,2 --config configs/config.yaml --phase tokenizer
  ${SCRIPT_NAME} --dry-run --config configs/config.yaml --phase both
  PYTHON_BIN=/opt/conda/bin/python ${SCRIPT_NAME} --config configs/config.yaml

EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)
            [[ $# -lt 2 ]] && _die "--gpus requires an argument."
            GPU_SUBSET="$2"; shift 2 ;;
        --dry-run)
            DRY_RUN=true; shift ;;
        --log-dir)
            [[ $# -lt 2 ]] && _die "--log-dir requires an argument."
            LOG_DIR="$2"; shift 2 ;;
        --help|-h)
            _usage; exit 0 ;;
        *)
            # Everything else is forwarded to hpo_tuner.py
            HPO_ARGS+=("$1"); shift ;;
    esac
done

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Environment Detection & Validation
# ══════════════════════════════════════════════════════════════════════════════

# ── OS detection ──────────────────────────────────────────────────────────────
_detect_os() {
    local os_name
    os_name="$(uname -s 2>/dev/null || echo 'Unknown')"
    case "${os_name}" in
        Linux*)   echo "linux"   ;;
        Darwin*)  echo "macos"   ;;
        CYGWIN*)  echo "windows" ;;
        MINGW*)   echo "windows" ;;
        MSYS*)    echo "windows" ;;
        *)
            # Final fallback: check for Windows-style paths
            if [[ "${OSTYPE:-}" == "msys" ]] || [[ "${OSTYPE:-}" == "cygwin" ]]; then
                echo "windows"
            else
                echo "unknown"
            fi
            ;;
    esac
}

OS_TYPE="$(_detect_os)"
_info "Detected OS: ${OS_TYPE}"

# ── Python detection ──────────────────────────────────────────────────────────
_detect_python() {
    # If user explicitly set PYTHON_BIN, validate and use it
    if [[ -n "${PYTHON_BIN}" ]]; then
        if command -v "${PYTHON_BIN}" &>/dev/null; then
            echo "${PYTHON_BIN}"
            return 0
        else
            _die "PYTHON_BIN='${PYTHON_BIN}' is not executable or not in PATH."
        fi
    fi

    # Auto-detection priority:
    # 1. python3   (preferred on Linux/macOS/WSL2)
    # 2. python    (Windows native, conda base envs)
    # 3. py -3     (Windows Python Launcher — py.exe)
    local candidates=("python3" "python" "py")
    for candidate in "${candidates[@]}"; do
        if command -v "${candidate}" &>/dev/null; then
            # Verify it is Python 3 (not Python 2)
            local version
            version="$("${candidate}" -c 'import sys; print(sys.version_info.major)' 2>/dev/null || echo '0')"
            if [[ "${version}" == "3" ]]; then
                echo "${candidate}"
                return 0
            fi
        fi
    done

    # Windows py.exe launcher
    if command -v py &>/dev/null; then
        local version
        version="$(py -3 -c 'import sys; print(sys.version_info.major)' 2>/dev/null || echo '0')"
        if [[ "${version}" == "3" ]]; then
            echo "py -3"
            return 0
        fi
    fi

    _die "No Python 3 interpreter found in PATH. " \
         "Set PYTHON_BIN=/path/to/python3 or activate your conda/venv."
}

PYTHON_BIN="$(_detect_python)"
_info "Python interpreter: ${PYTHON_BIN}"

# Validate Python version is 3.8+
PY_VERSION="$(${PYTHON_BIN} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo '0.0')"
PY_MAJOR="${PY_VERSION%%.*}"
PY_MINOR="${PY_VERSION#*.}"
if [[ "${PY_MAJOR}" -lt 3 ]] || { [[ "${PY_MAJOR}" -eq 3 ]] && [[ "${PY_MINOR}" -lt 8 ]]; }; then
    _die "Python 3.8 or later required. Found: Python ${PY_VERSION}"
fi
_info "Python version: ${PY_VERSION} ✓"

# ── nvidia-smi detection ──────────────────────────────────────────────────────
_detect_nvidia_smi() {
    # Standard PATH check
    if command -v nvidia-smi &>/dev/null; then
        echo "nvidia-smi"
        return 0
    fi

    # Windows-specific install paths (not always in PATH inside Git Bash)
    local win_paths=(
        "/c/Windows/System32/nvidia-smi.exe"
        "/c/Program Files/NVIDIA Corporation/NVSMI/nvidia-smi.exe"
        "/c/Windows/System32/nvidia-smi"
    )
    for path in "${win_paths[@]}"; do
        if [[ -x "${path}" ]]; then
            echo "${path}"
            return 0
        fi
    done

    return 1
}

NVIDIA_SMI=""
if ! NVIDIA_SMI="$(_detect_nvidia_smi)"; then
    _die "nvidia-smi not found in PATH or standard install locations." \
         "Ensure NVIDIA drivers are installed and nvidia-smi is accessible."
fi
_info "nvidia-smi: ${NVIDIA_SMI} ✓"

# ── hpo_tuner.py location check ───────────────────────────────────────────────
if [[ ! -f "${HPO_TUNER_SCRIPT}" ]]; then
    _die "hpo_tuner.py not found at: ${HPO_TUNER_SCRIPT}" \
         "Place launch_hpo.sh in the same directory as hpo_tuner.py."
fi
_info "HPO script: ${HPO_TUNER_SCRIPT} ✓"

# ── Optuna installation check ─────────────────────────────────────────────────
if ! ${PYTHON_BIN} -c "import optuna" &>/dev/null; then
    _die "Optuna is not installed in the detected Python environment." \
         "Run: ${PYTHON_BIN} -m pip install optuna plotly"
fi
OPTUNA_VERSION="$(${PYTHON_BIN} -c 'import optuna; print(optuna.__version__)' 2>/dev/null || echo 'unknown')"
_info "Optuna version: ${OPTUNA_VERSION} ✓"

# ── PyTorch + CUDA check ──────────────────────────────────────────────────────
TORCH_CUDA_AVAILABLE="$(${PYTHON_BIN} -c \
    'import torch; print("yes" if torch.cuda.is_available() else "no")' \
    2>/dev/null || echo 'no')"
if [[ "${TORCH_CUDA_AVAILABLE}" != "yes" ]]; then
    _warn "torch.cuda.is_available() returned False." \
          "Workers will run on CPU. Performance will be severely degraded."
fi

# ── nice availability (non-critical on Windows) ───────────────────────────────
NICE_CMD=""
if command -v nice &>/dev/null && [[ "${OS_TYPE}" != "windows" ]]; then
    NICE_CMD="nice -n ${HPO_WORKER_NICE}"
fi

# ── timeout availability ──────────────────────────────────────────────────────
TIMEOUT_CMD=""
if [[ -n "${HPO_TIMEOUT_SECONDS}" ]]; then
    if command -v timeout &>/dev/null; then
        TIMEOUT_CMD="timeout ${HPO_TIMEOUT_SECONDS}"
    elif command -v gtimeout &>/dev/null; then
        # macOS via coreutils: brew install coreutils
        TIMEOUT_CMD="gtimeout ${HPO_TIMEOUT_SECONDS}"
    else
        _warn "HPO_TIMEOUT_SECONDS=${HPO_TIMEOUT_SECONDS} set but 'timeout' not found." \
              "Workers will run without a time limit."
    fi
fi

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: GPU Enumeration
# ══════════════════════════════════════════════════════════════════════════════

# ── Enumerate all available GPUs ──────────────────────────────────────────────
_enumerate_gpus() {
    local raw_output
    # --query-gpu=index gives numeric IDs; csv,noheader strips the header row
    raw_output="$(${NVIDIA_SMI} --query-gpu=index,name,memory.total \
        --format=csv,noheader,nounits 2>/dev/null)" || {
        _die "nvidia-smi failed to enumerate GPUs. " \
             "Check driver installation: ${NVIDIA_SMI} --version"
    }

    if [[ -z "${raw_output}" ]]; then
        _die "nvidia-smi returned no GPU information."
    fi
    echo "${raw_output}"
}

# Build GPU arrays
declare -a ALL_GPU_IDS=()
declare -a ALL_GPU_NAMES=()
declare -a ALL_GPU_VRAM=()

while IFS=',' read -r gpu_id gpu_name gpu_vram; do
    # Trim whitespace (critical for Windows where nvidia-smi adds extra spaces)
    gpu_id="$(echo "${gpu_id}"   | tr -d '[:space:]')"
    gpu_name="$(echo "${gpu_name}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    gpu_vram="$(echo "${gpu_vram}" | tr -d '[:space:]')"

    ALL_GPU_IDS+=("${gpu_id}")
    ALL_GPU_NAMES+=("${gpu_name}")
    ALL_GPU_VRAM+=("${gpu_vram}")
done < <(_enumerate_gpus)

TOTAL_GPUS="${#ALL_GPU_IDS[@]}"
if [[ "${TOTAL_GPUS}" -eq 0 ]]; then
    _die "No GPUs detected. Check nvidia-smi output manually: ${NVIDIA_SMI}"
fi

# ── Apply GPU subset filter ───────────────────────────────────────────────────
declare -a SELECTED_GPU_IDS=()
declare -a SELECTED_GPU_NAMES=()
declare -a SELECTED_GPU_VRAM=()

if [[ -n "${GPU_SUBSET}" ]]; then
    # Parse comma-separated list: "0,1,2" → array
    IFS=',' read -ra REQUESTED_IDS <<< "${GPU_SUBSET}"
    for req_id in "${REQUESTED_IDS[@]}"; do
        req_id="$(echo "${req_id}" | tr -d '[:space:]')"
        local_found=false
        for i in "${!ALL_GPU_IDS[@]}"; do
            if [[ "${ALL_GPU_IDS[$i]}" == "${req_id}" ]]; then
                SELECTED_GPU_IDS+=("${ALL_GPU_IDS[$i]}")
                SELECTED_GPU_NAMES+=("${ALL_GPU_NAMES[$i]}")
                SELECTED_GPU_VRAM+=("${ALL_GPU_VRAM[$i]}")
                local_found=true
                break
            fi
        done
        if [[ "${local_found}" == false ]]; then
            _warn "Requested GPU ${req_id} not found in available GPUs [${ALL_GPU_IDS[*]}]. Skipping."
        fi
    done

    if [[ "${#SELECTED_GPU_IDS[@]}" -eq 0 ]]; then
        _die "No valid GPUs remain after applying --gpus '${GPU_SUBSET}'."
    fi
else
    # Use all available GPUs
    SELECTED_GPU_IDS=("${ALL_GPU_IDS[@]}")
    SELECTED_GPU_NAMES=("${ALL_GPU_NAMES[@]}")
    SELECTED_GPU_VRAM=("${ALL_GPU_VRAM[@]}")
fi

NUM_WORKERS="${#SELECTED_GPU_IDS[@]}"

# ── Check for existing HPO DB and warn if starting fresh ──────────────────────
_check_existing_db() {
    local storage_uri=""
    # Extract storage URI from HPO_ARGS if --config was passed
    local config_path=""
    for i in "${!HPO_ARGS[@]}"; do
        if [[ "${HPO_ARGS[$i]}" == "--config" ]]; then
            config_path="${HPO_ARGS[$((i+1))]:-}"
            break
        fi
    done

    if [[ -n "${config_path}" ]] && [[ -f "${config_path}" ]]; then
        # Simple grep — avoids requiring PyYAML at bash level
        storage_uri="$(grep -E '^\s*storage:' "${config_path}" \
            | sed 's/.*storage:[[:space:]]*//' \
            | tr -d '"'"'" \
            | tr -d '[:space:]' || echo '')"
        if [[ "${storage_uri}" == "null" ]] || [[ -z "${storage_uri}" ]]; then
            _warn "hpo.storage is null in config. Using in-memory Optuna storage." \
                  "Results will NOT be shared between workers or persisted across runs." \
                  "Set hpo.storage to a SQLite or PostgreSQL URI for multi-GPU HPO."
        else
            # Extract file path from sqlite:///path?timeout=60
            local db_path
            db_path="$(echo "${storage_uri}" \
                | sed 's|sqlite:///||' \
                | sed 's|?.*||')"
            if [[ -f "${db_path}" ]]; then
                _info "Existing Optuna DB found: ${db_path}"
                _info "Workers will resume the existing study (load_if_exists=True)."
            fi
        fi
    fi
}
_check_existing_db

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Signal Handling & Cleanup
# ══════════════════════════════════════════════════════════════════════════════

declare -a WORKER_PIDS=()
INTERRUPTED=false

_cleanup() {
    local signal="${1:-SIGTERM}"
    INTERRUPTED=true

    echo ""
    _warn "Received ${signal}. Sending SIGTERM to all worker processes..."

    local kill_failed=0
    for pid in "${WORKER_PIDS[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            # ── TASK 2.2: Clean Process Termination for Windows/WSL ──────────
            # Standard 'kill' leaves Python multiprocessing DataLoader workers orphaned
            # on Windows, trapping VRAM forever. We must kill the entire process tree.
            if [[ "${OS_TYPE}" == "windows" ]]; then
                taskkill //F //T //PID "${pid}" 2>/dev/null || kill_failed=$((kill_failed + 1))
            else
                kill -TERM "${pid}" 2>/dev/null || kill_failed=$((kill_failed + 1))
            fi
            _warn "  Sent termination signal to PID ${pid}"
        fi
    done

    # Give workers 10 seconds to shut down gracefully
    # Skip wait on Windows since taskkill //F is immediate
    if [[ "${OS_TYPE}" != "windows" ]]; then
        local grace_seconds=10
        _info "Waiting up to ${grace_seconds}s for graceful shutdown..."
        local elapsed=0
        while [[ ${elapsed} -lt ${grace_seconds} ]]; do
            local still_running=0
            for pid in "${WORKER_PIDS[@]}"; do
                kill -0 "${pid}" 2>/dev/null && still_running=$((still_running + 1))
            done
            [[ ${still_running} -eq 0 ]] && break
            sleep 1
            elapsed=$((elapsed + 1))
        done

        # Force kill any remaining workers (POSIX)
        for pid in "${WORKER_PIDS[@]}"; do
            if kill -0 "${pid}" 2>/dev/null; then
                _warn "  Force killing PID ${pid} (SIGKILL)"
                kill -KILL "${pid}" 2>/dev/null || true
            fi
        done
    fi

    _warn "Interrupted. Completed trials have been saved to the Optuna DB."
    _warn "Re-run the same command to resume from where you left off."
    exit 3
}

# Trap SIGINT (Ctrl+C), SIGTERM (kill), and SIGHUP (terminal close)
# Note: SIGHUP is not available on Windows but the trap is harmless there
trap '_cleanup SIGINT'  INT
trap '_cleanup SIGTERM' TERM
trap '_cleanup SIGHUP'  HUP 2>/dev/null || true

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Print Launch Plan
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo -e "${C_BOLD}╔══════════════════════════════════════════════════════════════╗${C_RESET}"
echo -e "${C_BOLD}║       Nos HPO — Asynchronous Multi-GPU Launcher              ║${C_RESET}"
echo -e "${C_BOLD}╠══════════════════════════════════════════════════════════════╣${C_RESET}"
printf  "║  %-20s : %-37s║\n" "OS"           "${OS_TYPE}"
printf  "║  %-20s : %-37s║\n" "Python"       "${PYTHON_BIN} (${PY_VERSION})"
printf  "║  %-20s : %-37s║\n" "Optuna"       "${OPTUNA_VERSION}"
printf  "║  %-20s : %-37s║\n" "Total GPUs"   "${TOTAL_GPUS} detected"
printf  "║  %-20s : %-37s║\n" "Worker GPUs"  "${NUM_WORKERS} selected"
printf  "║  %-20s : %-37s║\n" "Log dir"      "${LOG_DIR}"
[[ -n "${HPO_TIMEOUT_SECONDS}" ]] && \
printf  "║  %-20s : %-37s║\n" "Worker timeout"  "${HPO_TIMEOUT_SECONDS}s"
[[ "${DRY_RUN}" == true ]] && \
printf  "║  %-20s : %-37s║\n" "Mode"         "DRY RUN — no processes launched"
echo -e "${C_BOLD}╠══════════════════════════════════════════════════════════════╣${C_RESET}"
echo    "║  Selected GPUs:                                              ║"
for i in "${!SELECTED_GPU_IDS[@]}"; do
    printf "║    GPU %-3s : %-20s  %6s MiB VRAM          ║\n" \
        "${SELECTED_GPU_IDS[$i]}" \
        "${SELECTED_GPU_NAMES[$i]:0:20}" \
        "${SELECTED_GPU_VRAM[$i]}"
done
echo -e "${C_BOLD}╠══════════════════════════════════════════════════════════════╣${C_RESET}"
echo    "║  HPO arguments forwarded to hpo_tuner.py:                    ║"
printf  "║    %-58s║\n" "${HPO_ARGS[*]:0:58}"
echo -e "${C_BOLD}╚══════════════════════════════════════════════════════════════╝${C_RESET}"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Worker Launch
# ══════════════════════════════════════════════════════════════════════════════

mkdir -p "${LOG_DIR}"

# Per-run timestamp for log file naming (shared across all workers in this run)
RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"

declare -a LOG_FILES=()

for i in "${!SELECTED_GPU_IDS[@]}"; do
    GPU_ID="${SELECTED_GPU_IDS[$i]}"
    GPU_NAME="${SELECTED_GPU_NAMES[$i]}"
    LOG_FILE="${LOG_DIR}/worker_gpu${GPU_ID}_${RUN_TIMESTAMP}.log"
    LOG_FILES+=("${LOG_FILE}")

    # ── Build the full command ─────────────────────────────────────────────
    # Construct as an array to handle paths with spaces correctly
    CMD=()

    # Process priority (Linux/macOS only)
    if [[ -n "${NICE_CMD}" ]]; then
        CMD+=($NICE_CMD)
    fi

    # Timeout wrapper (if configured)
    if [[ -n "${TIMEOUT_CMD}" ]]; then
        CMD+=($TIMEOUT_CMD)
    fi

    # Python interpreter
    # Split PYTHON_BIN in case it contains flags (e.g., "py -3")
    read -ra PY_PARTS <<< "${PYTHON_BIN}"
    CMD+=("${PY_PARTS[@]}")

    # Unbuffered output: critical so log files get real-time writes,
    # not buffered writes that only flush when the process exits.
    CMD+=("-u")

    # The HPO script
    CMD+=("${HPO_TUNER_SCRIPT}")

    # Forward all HPO arguments
    CMD+=("${HPO_ARGS[@]}")

    # ── Environment for this worker ────────────────────────────────────────
    # CUDA_VISIBLE_DEVICES: Restricts this process to exactly one physical GPU.
    # CUDA remaps the assigned GPU to logical index cuda:0 inside the process.
    # PYTHONUNBUFFERED: Belt-and-suspenders for -u flag (some launchers ignore -u).
    # PYTHONFAULTHANDLER: Dumps C-level stack traces on segfaults (invaluable for debugging).
    declare -a WORKER_ENV=(
        "CUDA_VISIBLE_DEVICES=${GPU_ID}"
        "PYTHONUNBUFFERED=1"
        "PYTHONFAULTHANDLER=1"
    )

    # Optional VRAM fraction limiter
    if [[ -n "${GPU_MEMORY_FRACTION}" ]]; then
        WORKER_ENV+=("PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512")
        # Note: actual fraction limiting requires PyTorch code changes;
        # this env var limits the allocator's split size as a proxy.
    fi

    if [[ "${DRY_RUN}" == true ]]; then
        _info "[DRY RUN] GPU ${GPU_ID} (${GPU_NAME}) would run:"
        echo  "         env ${WORKER_ENV[*]} ${CMD[*]}"
        echo  "         >> ${LOG_FILE} 2>&1 &"
    else
        # Write a header into the log file before the process starts
        {
            echo "============================================================"
            echo "  Nos HPO Worker Log"
            echo "  Run timestamp : ${RUN_TIMESTAMP}"
            echo "  GPU ID        : ${GPU_ID} (${GPU_NAME})"
            echo "  GPU VRAM      : ${SELECTED_GPU_VRAM[$i]} MiB"
            echo "  Command       : ${CMD[*]}"
            echo "  Environment   : ${WORKER_ENV[*]}"
            echo "  Started at    : $(date)"
            echo "============================================================"
        } > "${LOG_FILE}"

        # Launch the worker process
        # env sets per-process environment without polluting the parent shell
        env "${WORKER_ENV[@]}" "${CMD[@]}" >> "${LOG_FILE}" 2>&1 &
        WORKER_PID=$!

        WORKER_PIDS+=("${WORKER_PID}")
        _ok "GPU ${GPU_ID} (${GPU_NAME}) → PID ${WORKER_PID} → ${LOG_FILE}"
    fi
done

# ── Dry run exits here ────────────────────────────────────────────────────────
if [[ "${DRY_RUN}" == true ]]; then
    echo ""
    _info "Dry run complete. No processes were launched."
    exit 0
fi

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: Live Monitoring
# ══════════════════════════════════════════════════════════════════════════════

echo ""
_info "All ${NUM_WORKERS} worker(s) launched."
echo ""
echo -e "  ${C_CYAN}Monitor logs (all workers):${C_RESET}"
echo    "    tail -f ${LOG_DIR}/worker_gpu*_${RUN_TIMESTAMP}.log"
echo ""
echo -e "  ${C_CYAN}Monitor individual worker:${C_RESET}"
for i in "${!SELECTED_GPU_IDS[@]}"; do
    echo "    tail -f ${LOG_FILES[$i]}"
done
echo ""
echo -e "  ${C_CYAN}Monitor GPU utilisation:${C_RESET}"
echo    "    watch -n 2 nvidia-smi"
echo ""
echo -e "  ${C_CYAN}Stop all workers:${C_RESET}"
echo    "    Ctrl+C  (graceful SIGTERM → waits 10s → SIGKILL)"
echo ""

# ── Optional: background VRAM monitor ─────────────────────────────────────────
# Logs nvidia-smi output every 60 seconds into a separate file for post-run
# VRAM analysis. Killed automatically when the script exits.
VRAM_LOG="${LOG_DIR}/vram_monitor_${RUN_TIMESTAMP}.log"
(
    while true; do
        {
            echo "--- $(date) ---"
            "${NVIDIA_SMI}" \
                --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu \
                --format=csv,noheader,nounits 2>/dev/null || true
            echo ""
        } >> "${VRAM_LOG}" 2>/dev/null
        sleep 60
    done
) &
VRAM_MONITOR_PID=$!
_info "VRAM monitor started (PID ${VRAM_MONITOR_PID}) → ${VRAM_LOG}"

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: Wait for All Workers & Collect Results
# ══════════════════════════════════════════════════════════════════════════════

declare -a EXIT_CODES=()
declare -a FAILED_GPUS=()
declare -a SUCCEEDED_GPUS=()
FAILED_COUNT=0
SUCCEEDED_COUNT=0

_info "Waiting for all ${NUM_WORKERS} worker(s) to complete..."
echo  "  (This may take several hours for large HPO runs)"
echo ""

for i in "${!WORKER_PIDS[@]}"; do
    PID="${WORKER_PIDS[$i]}"
    GPU_ID="${SELECTED_GPU_IDS[$i]}"
    GPU_NAME="${SELECTED_GPU_NAMES[$i]}"
    LOG_FILE="${LOG_FILES[$i]}"

    EXIT_CODE=0
    wait "${PID}" || EXIT_CODE=$?
    EXIT_CODES+=("${EXIT_CODE}")

    # Append completion footer to the worker's log
    {
        echo ""
        echo "============================================================"
        echo "  Worker exited at: $(date)"
        echo "  Exit code: ${EXIT_CODE}"
        echo "============================================================"
    } >> "${LOG_FILE}" 2>/dev/null || true

    if [[ "${EXIT_CODE}" -eq 0 ]]; then
        _ok  "GPU ${GPU_ID} (${GPU_NAME}) | PID ${PID} | EXIT 0 | SUCCESS"
        SUCCEEDED_GPUS+=("${GPU_ID}")
        SUCCEEDED_COUNT=$((SUCCEEDED_COUNT + 1))
    elif [[ "${EXIT_CODE}" -eq 3 ]]; then
        # Exit code 3 = interrupted by user (our own convention from hpo_tuner.py)
        _warn "GPU ${GPU_ID} (${GPU_NAME}) | PID ${PID} | EXIT 3 | INTERRUPTED"
        _warn "  Partial results are preserved in the Optuna DB."
        SUCCEEDED_GPUS+=("${GPU_ID}")  # Treat interruption as non-failure
        SUCCEEDED_COUNT=$((SUCCEEDED_COUNT + 1))
    else
        _error "GPU ${GPU_ID} (${GPU_NAME}) | PID ${PID} | EXIT ${EXIT_CODE} | FAILED"
        _error "  Last 20 lines of log:"
        tail -n 20 "${LOG_FILE}" | sed 's/^/    /' >&2
        FAILED_GPUS+=("${GPU_ID}")
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
done

# ── Kill the VRAM monitor ─────────────────────────────────────────────────────
kill "${VRAM_MONITOR_PID}" 2>/dev/null || true
wait "${VRAM_MONITOR_PID}" 2>/dev/null || true

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8: Final Report
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo -e "${C_BOLD}══════════════════════════════════════════════════════════════${C_RESET}"
echo -e "${C_BOLD}  HPO LAUNCHER — FINAL REPORT${C_RESET}"
echo    "══════════════════════════════════════════════════════════════"
printf  "  %-24s : %s\n" "Run timestamp"    "${RUN_TIMESTAMP}"
printf  "  %-24s : %s\n" "Total workers"    "${NUM_WORKERS}"
printf  "  %-24s : %s\n" "Succeeded"        "${SUCCEEDED_COUNT}"
printf  "  %-24s : %s\n" "Failed"           "${FAILED_COUNT}"
printf  "  %-24s : %s\n" "Log directory"    "${LOG_DIR}"
printf  "  %-24s : %s\n" "VRAM log"         "${VRAM_LOG}"
echo    "══════════════════════════════════════════════════════════════"

if [[ "${FAILED_COUNT}" -gt 0 ]]; then
    echo ""
    _error "Failed GPU workers: ${FAILED_GPUS[*]}"
    _error "Inspect their logs:"
    for fail_gpu in "${FAILED_GPUS[@]}"; do
        echo  "  ${LOG_DIR}/worker_gpu${fail_gpu}_${RUN_TIMESTAMP}.log"
    done
    echo ""
    _info "Completed trials from successful workers are preserved in the Optuna DB."
    _info "You can re-run the same command to launch replacement workers for the failed GPUs:"
    echo  "  ./launch_hpo.sh --gpus $(IFS=','; echo "${FAILED_GPUS[*]}") ${HPO_ARGS[*]}"
    echo ""
    exit 1
fi

echo ""
_ok "All ${NUM_WORKERS} worker(s) completed successfully."
echo ""
_info "Next steps:"
echo  "  1. Apply best params to config:"
echo  "     ${PYTHON_BIN} hpo_tuner.py ${HPO_ARGS[*]} --apply-best"
echo  "  2. View hyperparameter importance:"
echo  "     ${PYTHON_BIN} hpo_tuner.py ${HPO_ARGS[*]} --show-importance"
echo  "  3. Review VRAM usage over time:"
echo  "     cat ${VRAM_LOG}"
echo ""
exit 0
```

---

### `finetune_base_model.py`

```python
import os
import sys
import json
import time
import pickle
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import datetime
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from time import gmtime, strftime
import logging
from logging.handlers import RotatingFileHandler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append('../')
from model import Nos, NosTokenizer, NosPredictor
from config_loader import CustomFinetuneConfig


# ── Utilities ─────────────────────────────────────────────────────────────────

def format_time(seconds: float) -> str:
    """Convert a duration in seconds to a human-readable string."""
    return str(datetime.timedelta(seconds=int(seconds)))


def get_model_size(model: torch.nn.Module) -> str:
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if total_params >= 1_000_000_000:
        return f"{total_params / 1e9:.1f}B"
    elif total_params >= 1_000_000:
        return f"{total_params / 1e6:.1f}M"
    else:
        return f"{total_params / 1e3:.1f}K"


# ── Dataset ───────────────────────────────────────────────────────────────────

class CustomKlineDataset(Dataset):
    """
    Memory-mapped Kline dataset with Log Return transformation for price columns.
    """

    FEATURE_COLS      = ['open', 'high', 'low', 'close', 'volume', 'amount']
    TIME_FEATURE_COLS = ['minute', 'hour', 'weekday', 'day', 'month']

    def __init__(
        self,
        data_path: str,
        data_type: str = 'train',
        lookback_window: int = 90,
        predict_window: int = 10,
        clip: float = 5.0,
        seed: int = 100,
        train_ratio: float = 0.70,
        val_ratio: float  = 0.15,
        test_ratio: float = 0.15,
    ):
        self.data_path       = data_path
        self.data_type       = data_type
        self.lookback_window = lookback_window
        self.predict_window  = predict_window
        self.window          = lookback_window + predict_window + 1
        self.clip            = clip
        self.seed            = seed
        self.train_ratio     = train_ratio
        self.val_ratio       = val_ratio
        self.test_ratio      = test_ratio

        self.py_rng        = random.Random(seed)
        self.current_epoch = 0

        self._load_or_build_cache()

        self.n_samples = len(self.x_data) - self.window + 1
        if self.n_samples <= 0:
            raise ValueError(
                f"[{data_type.upper()}] Not enough data rows ({len(self.x_data)}) "
                f"for window size {self.window}."
            )
        print(
            f"[{data_type.upper()}] Rows: {len(self.x_data):,}  "
            f"Samples: {self.n_samples:,}"
        )

    def _load_or_build_cache(self):
        cache_x     = f"{self.data_path}.{self.data_type}.x.npy"
        cache_stamp = f"{self.data_path}.{self.data_type}.stamp.npy"
        lock_path   = f"{self.data_path}.{self.data_type}.lock"

        # Fast path: lock-free check for 99% of workers (cache already exists)
        if os.path.exists(cache_x) and os.path.exists(cache_stamp):
            self.x_data     = np.load(cache_x,     mmap_mode='r')
            self.stamp_data = np.load(cache_stamp, mmap_mode='r')
            return

        from filelock import FileLock
        with FileLock(lock_path, timeout=600):
            # Double-check inside lock
            if os.path.exists(cache_x) and os.path.exists(cache_stamp):
                self.x_data     = np.load(cache_x,     mmap_mode='r')
                self.stamp_data = np.load(cache_stamp, mmap_mode='r')
                return

            print(f"[{self.data_type.upper()}] Building mmap cache …")
            df = pd.read_csv(self.data_path)
            df['timestamps'] = pd.to_datetime(df['timestamps'])
            df = df.sort_values('timestamps').reset_index(drop=True)

            # Temporal features
            df['minute']  = df['timestamps'].dt.minute
            df['hour']    = df['timestamps'].dt.hour
            df['weekday'] = df['timestamps'].dt.weekday
            df['day']     = df['timestamps'].dt.day
            df['month']   = df['timestamps'].dt.month

            if df.isnull().any().any():
                df = df.ffill()

            # ── PROFESSIONAL UPGRADE: Log Returns ──
            price_cols = ['open', 'high', 'low', 'close']
            df[price_cols] = np.log(df[price_cols] / df[price_cols].shift(1))

            # Clean up the NaN created by shift on row 0, and any infinites
            df[price_cols] = df[price_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

            # Chronological split
            n = len(df)
            train_end = int(n * self.train_ratio)
            val_end   = int(n * (self.train_ratio + self.val_ratio))

            slices = {'train': slice(None, train_end),
                      'val':   slice(train_end, val_end),
                      'test':  slice(val_end, None)}
            df = df.iloc[slices[self.data_type]]

            x_arr     = df[self.FEATURE_COLS].values.astype(np.float32)
            stamp_arr = df[self.TIME_FEATURE_COLS].values.astype(np.float32)

            del df

            # Atomic writes
            tmp_x     = f"{cache_x}.tmp.{os.getpid()}.npy"
            tmp_stamp = f"{cache_stamp}.tmp.{os.getpid()}.npy"

            np.save(tmp_x, x_arr)
            np.save(tmp_stamp, stamp_arr)

            os.replace(tmp_x, cache_x)
            os.replace(tmp_stamp, cache_stamp)

        self.x_data     = np.load(cache_x,     mmap_mode='r')
        self.stamp_data = np.load(cache_stamp, mmap_mode='r')

    def set_epoch_seed(self, epoch: int):
        self.py_rng.seed(self.seed + epoch)
        self.current_epoch = epoch

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        max_start = len(self.x_data) - self.window
        start_idx = idx % (max_start + 1)
        end_idx = start_idx + self.window

        x       = self.x_data[start_idx:end_idx].copy()
        x_stamp = self.stamp_data[start_idx:end_idx].copy()

        # Calculate mean and std ONLY on the lookback window
        lookback_x = x[:self.lookback_window]
        x_mean = lookback_x.mean(axis=0)
        x_std  = lookback_x.std(axis=0)

        # Apply historical stats to the ENTIRE window
        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.clip, self.clip)

        return torch.from_numpy(x), torch.from_numpy(x_stamp)


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(exp_name: str, log_dir: str, rank: int = 0) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(f"basemodel_training_rank_{rank}")
    logger.setLevel(logging.INFO)

    # Idempotent – do not attach duplicate handlers on re-import
    if logger.handlers:
        return logger

    log_file     = os.path.join(log_dir, f"basemodel_training_rank_{rank}.log")
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler() if rank == 0 else None
    if console_handler:
        console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    file_handler.setFormatter(formatter)
    if console_handler:
        console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    if console_handler:
        logger.addHandler(console_handler)

    logger.info("=== Basemodel Training Started ===")
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Log dir:    {log_dir}")
    logger.info(f"Rank:       {rank}")
    logger.info(f"Timestamp:  {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")

    return logger


# ── DataLoaders ───────────────────────────────────────────────────────────────

def create_dataloaders(config):
    is_main = (
        not dist.is_available()
        or not dist.is_initialized()
        or dist.get_rank() == 0
    )
    if is_main:
        print("Creating data loaders …")

    shared_kw = dict(
        data_path       = config.data_path,
        lookback_window = config.lookback_window,
        predict_window  = config.predict_window,
        clip            = config.clip,
        train_ratio     = config.train_ratio,
        val_ratio       = config.val_ratio,
        test_ratio      = config.test_ratio,
    )

    train_dataset = CustomKlineDataset(data_type='train', seed=config.seed,     **shared_kw)
    val_dataset   = CustomKlineDataset(data_type='val',   seed=config.seed + 1, **shared_kw)

    use_ddp = dist.is_available() and dist.is_initialized()
    train_sampler = (
        DistributedSampler(train_dataset, shuffle=True)  if use_ddp else None
    )
    # FIX: Use drop_last=False - validation loop correctly masks padding via valid_mask.
    # drop_last=True discards real data when batch sizes don't divide evenly, causing
    # inconsistent val_loss across different GPU configurations.
    val_sampler = (
        DistributedSampler(val_dataset, shuffle=False, drop_last=False) if use_ddp else None
    )

    loader_kw = dict(
        num_workers=config.num_workers, 
        pin_memory=True,
        persistent_workers=config.persistent_workers if config.num_workers > 0 else False
    )

    train_loader = DataLoader(
        train_dataset,
        # FIX: Explicit cast from trial config to ensure HPO batch_size is respected
        batch_size = int(config.batch_size),
        shuffle    = (train_sampler is None),
        drop_last  = True,
        sampler    = train_sampler,
        **loader_kw,
    )
    val_loader = DataLoader(
        val_dataset,
        # FIX: Explicit cast from trial config to ensure HPO batch_size is respected
        batch_size = int(config.batch_size),
        shuffle    = False,
        drop_last  = False,
        sampler    = val_sampler,
        **loader_kw,
    )

    if is_main:
        print(
            f"Train samples: {len(train_dataset):,}  "
            f"Val samples: {len(val_dataset):,}"
        )

    return (
        train_loader, val_loader,
        train_dataset, val_dataset,
        train_sampler, val_sampler,
    )


# ── Training Loop ─────────────────────────────────────────────────────────────

def train_model(model, tokenizer, device, config, save_dir, logger, trial=None):
    """
    Fine-tunes *model* (NosPredictor / Nos) with the frozen *tokenizer*.

    True Gradient Accumulation
    ──────────────────────────
    Accumulates gradients across `accumulation_steps` DataLoader batches before
    performing an optimizer step. This achieves the effective batch size:
    effective_batch_size = batch_size * accumulation_steps

    The scheduler uses effective_steps_per_epoch to correctly count optimizer
    steps, not raw DataLoader batches.
    """
    logger.info("Starting base-model training …")

    use_ddp = dist.is_available() and dist.is_initialized()
    rank    = dist.get_rank() if use_ddp else 0

    (
        train_loader, val_loader,
        train_dataset, val_dataset,
        train_sampler, val_sampler,
    ) = create_dataloaders(config)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = config.predictor_learning_rate,
        betas        = (config.adam_beta1, config.adam_beta2),
        weight_decay = config.adam_weight_decay,
    )

    # ── Scheduler: use effective steps for true gradient accumulation ────────
    pct_start  = getattr(config, 'basemodel_pct_start',  0.03)
    div_factor = getattr(config, 'basemodel_div_factor', 10.0)
    accumulation_steps = getattr(config, 'accumulation_steps', 1)

    # Effective steps = ceil(total_batches / accumulation_steps)
    import math
    effective_steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        # FIX: Guarantee max_lr matches the trial's exact optimizer LR
        max_lr          = [group['lr'] for group in optimizer.param_groups],
        steps_per_epoch = effective_steps_per_epoch,   # 1 step == accumulation_steps batches
        epochs          = config.basemodel_epochs,
        pct_start       = pct_start,
        div_factor      = div_factor,
    )

    # ── DDP wrapping (after scheduler construction) ────────────────────────────
    if use_ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        model = DDP(
            model,
            device_ids          = [local_rank],
            output_device       = local_rank,
            find_unused_parameters = False,
        )

    # ── Hyper-parameters from config ──────────────────────────────────────────
    max_grad_norm     = getattr(config, 'basemodel_max_grad_norm', 3.0)

    # Convenience alias: unwrap DDP only once per call-site
    def raw_model():
        return model.module if use_ddp else model

    # ── Epoch loop ────────────────────────────────────────────────────────────
    best_val_loss   = float('inf')
    batch_idx_global = 0

    for epoch in range(config.basemodel_epochs):
        epoch_start = time.time()
        model.train()

        # ── NEW: Scheduled Sampling Probability ──
        # Linear decay from 100% to 0% over the first 75% of training.
        # The final 25% of epochs will be pure autoregressive training.
        tf_decay_epochs = max(1, int(config.basemodel_epochs * 0.75))
        tf_prob = max(0.0, 1.0 - (epoch / tf_decay_epochs))

        # Gumbel temperature decay: tau starts at 1.0, decays to 0.3 over training
        tau_decay_epochs = max(1, int(config.basemodel_epochs * 0.75))
        current_tau = max(0.3, 1.0 - (epoch / tau_decay_epochs) * 0.7)

        # Deterministic-but-varied shuffling per epoch
        train_dataset.set_epoch_seed(epoch * 10_000)
        val_dataset.set_epoch_seed(0)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_train_loss = 0.0
        train_batches    = 0
        current_accum_loss = 0.0

        # Zero gradients BEFORE the dataloader loop (true gradient accumulation)
        optimizer.zero_grad()

        # ── Dataloader loop (True Gradient Accumulation) ─────────────────────
        for batch_idx, (batch_x, batch_x_stamp) in enumerate(train_loader):
            batch_x       = batch_x.to(device, non_blocking=True)
            batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)

            # Tokenizer is frozen — skip its grad computation entirely
            with torch.no_grad():
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)

            token_in  = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
            token_out = [token_seq_0[:, 1:],  token_seq_1[:, 1:]]

            # ── Apply Scheduled Sampling (DDP Safe) ──
            if use_ddp:
                if rank == 0:
                    use_tf_tensor = torch.tensor([1.0 if random.random() < tf_prob else 0.0], device=device)
                else:
                    use_tf_tensor = torch.tensor([0.0], device=device)

                dist.broadcast(use_tf_tensor, src=0)
                use_tf = bool(use_tf_tensor.item())
            else:
                use_tf = random.random() < tf_prob

            logits = raw_model()(
                s1_ids=token_in[0],
                s2_ids=token_in[1],
                stamp=batch_x_stamp[:, :-1, :],
                use_teacher_forcing=use_tf,
                s1_targets=token_out[0] if use_tf else None,
                gumbel_tau=current_tau  # Injected dynamic temperature
            )

            loss, s1_loss, s2_loss = raw_model().head.compute_loss(
                logits[0], logits[1], token_out[0], token_out[1]
            )

            # ── TASK 1.2: Dynamic Micro-Batch & DDP Sync Storm Fix ──────────
            is_last_batch = (batch_idx + 1) == len(train_loader)
            is_step_time = (batch_idx + 1) % accumulation_steps == 0 or is_last_batch

            # Dynamic scalar to prevent inverse gradient accumulation on the tail batch
            current_micro_batches = (batch_idx % accumulation_steps) + 1 if is_last_batch else accumulation_steps
            loss_scaled = loss / current_micro_batches

            # DDP Context Manager to prevent all_reduce storms
            if use_ddp and not is_step_time:
                with model.no_sync():
                    loss_scaled.backward()
            else:
                loss_scaled.backward()

            current_accum_loss += loss.item()

            # ── Optimizer step (once per accumulation cycle) ────────────────
            if is_step_time:
                torch.nn.utils.clip_grad_norm_(
                    raw_model().parameters(),
                    max_norm=max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # Logging
                avg_loss = current_accum_loss / current_micro_batches
                epoch_train_loss += avg_loss
                train_batches    += 1

                if (batch_idx_global + 1) % config.log_interval == 0:
                    lr      = optimizer.param_groups[0]['lr']
                    log_msg = (
                        f"[Epoch {epoch+1}/{config.basemodel_epochs}, "
                        f"Step {batch_idx_global+1}/{effective_steps_per_epoch}] "
                        f"LR: {lr:.6f}  Loss: {avg_loss:.4f}"
                    )
                    logger.info(log_msg)
                    if rank == 0:
                        print(log_msg)

                current_accum_loss = 0.0
                batch_idx_global += 1

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_loss_sum   = 0.0
        val_sample_cnt = 0

        with torch.no_grad():
            for batch_x, batch_x_stamp in val_loader:
                batch_x       = batch_x.to(device, non_blocking=True)
                batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)

                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
                token_in  = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
                token_out = [token_seq_0[:, 1:],  token_seq_1[:, 1:]]

                logits = raw_model()(
                    s1_ids=token_in[0], 
                    s2_ids=token_in[1], 
                    stamp=batch_x_stamp[:, :-1, :],
                    use_teacher_forcing=False
                )
                loss, _, _ = raw_model().head.compute_loss(
                    logits[0], logits[1], token_out[0], token_out[1]
                )

                # Detect valid samples (non-padding) to ignore DDP dummy padding
                valid_mask = (batch_x != 0).any(dim=-1).any(dim=-1)
                valid_count = valid_mask.sum().item()
                val_loss_sum   += loss.item() * valid_count
                val_sample_cnt += valid_count

        # ── DDP aggregation ───────────────────────────────────────────────────
        if use_ddp:
            agg = torch.tensor(
                [epoch_train_loss, float(train_batches),
                 val_loss_sum,     float(val_sample_cnt)],
                dtype=torch.float64, device=device,
            )
            dist.all_reduce(agg, op=dist.ReduceOp.SUM)
            avg_train_loss = agg[0].item() / agg[1].item() if agg[1].item() > 0 else 0.0
            avg_val_loss   = agg[2].item() / agg[3].item() if agg[3].item() > 0 else 0.0
        else:
            avg_train_loss = epoch_train_loss / train_batches if train_batches > 0 else 0.0
            avg_val_loss   = val_loss_sum / val_sample_cnt    if val_sample_cnt > 0 else 0.0

        epoch_time    = time.time() - epoch_start
        epoch_summary = (
            f"\n--- Epoch {epoch+1}/{config.basemodel_epochs} Summary ---\n"
            f"  Train Loss : {avg_train_loss:.6f}\n"
            f"  Val   Loss : {avg_val_loss:.6f}\n"
            f"  Epoch Time : {format_time(epoch_time)}\n"
        )
        logger.info(epoch_summary)
        if rank == 0:
            print(epoch_summary)

        # ── TASK 1.1: Optuna Pruning Hook (Traceback OOM Fix) ─────────────────
        if trial is not None:
            import optuna
            trial.report(avg_val_loss, epoch)

            if trial.should_prune():
                prune_msg = f"Trial {trial.number} pruned at epoch {epoch} (val_loss: {avg_val_loss:.6f})"
                logger.info(prune_msg)
                if rank == 0:
                    print(prune_msg)

                # CRITICAL: Sever local variable references to prevent Traceback OOM leak
                locals().clear()
                raise optuna.exceptions.TrialPruned()    
        

        # ── Checkpoint (rank-0 only) ───────────────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if rank == 0:
                ckpt_dir = os.path.join(save_dir, "best_model")
                os.makedirs(ckpt_dir, exist_ok=True)
                raw_model().save_pretrained(ckpt_dir)
                save_msg = (
                    f"✓ Best model saved → {ckpt_dir}  "
                    f"(val loss: {best_val_loss:.6f})"
                )
                logger.info(save_msg)
                print(save_msg)

    return best_val_loss


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Nos Base-Model Fine-tuning')
    parser.add_argument(
        '--config', type=str, default='config.yaml',
        help='Path to YAML config file (default: config.yaml)',
    )
    args   = parser.parse_args()
    config = CustomFinetuneConfig(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(config.basemodel_save_path, exist_ok=True)

    log_dir = os.path.join(config.base_save_path, "logs")
    logger  = setup_logging(config.exp_name, log_dir, rank=0)

    # Reproducibility
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    logger.info("Loading tokenizer …")
    if getattr(config, 'pre_trained_tokenizer', True):
        tokenizer = NosTokenizer.from_pretrained(config.finetuned_tokenizer_path)
    else:
        logger.info("pre_trained_tokenizer=False — random init")
        cfg_path = os.path.join(config.pretrained_tokenizer_path, 'config.json')
        with open(cfg_path) as fh:
            arch = json.load(fh)
        tokenizer = NosTokenizer(
            d_in            = arch.get('d_in',             6),
            d_model         = arch.get('d_model',         256),
            n_heads         = arch.get('n_heads',           4),
            ff_dim          = arch.get('ff_dim',          512),
            n_enc_layers    = arch.get('n_enc_layers',      4),
            n_dec_layers    = arch.get('n_dec_layers',      4),
            ffn_dropout_p   = arch.get('ffn_dropout_p',  0.0),
            attn_dropout_p  = arch.get('attn_dropout_p', 0.0),
            resid_dropout_p = arch.get('resid_dropout_p',0.0),
            s1_bits         = arch.get('s1_bits',          10),
            s2_bits         = arch.get('s2_bits',          10),
            beta            = arch.get('beta',           0.05),
            gamma0          = arch.get('gamma0',          1.0),
            gamma           = arch.get('gamma',           1.1),
            zeta            = arch.get('zeta',           0.05),
            group_size      = arch.get('group_size',        4),
        )

    # ── Predictor ─────────────────────────────────────────────────────────────
    logger.info("Loading predictor …")
    if getattr(config, 'pre_trained_predictor', True):
        model = Nos.from_pretrained(config.pretrained_predictor_path)
    else:
        logger.info("pre_trained_predictor=False — random init")
        cfg_path = os.path.join(config.pretrained_predictor_path, 'config.json')
        with open(cfg_path) as fh:
            arch = json.load(fh)
        model = Nos(
            s1_bits         = arch.get('s1_bits',          10),
            s2_bits         = arch.get('s2_bits',          10),
            n_layers        = arch.get('n_layers',          12),
            d_model         = arch.get('d_model',          832),
            n_heads         = arch.get('n_heads',           16),
            ff_dim          = arch.get('ff_dim',          2048),
            ffn_dropout_p   = arch.get('ffn_dropout_p',   0.2),
            attn_dropout_p  = arch.get('attn_dropout_p',  0.0),
            resid_dropout_p = arch.get('resid_dropout_p', 0.2),
            token_dropout_p = arch.get('token_dropout_p', 0.0),
            learn_te        = arch.get('learn_te',        True),
        )

    tokenizer = tokenizer.to(device)
    model     = model.to(device)

    logger.info(f"Tokenizer size : {get_model_size(tokenizer)}")
    logger.info(f"Model size     : {get_model_size(model)}")
    print(f"Tokenizer: {get_model_size(tokenizer)}  |  Model: {get_model_size(model)}")

    # ── Config summary ────────────────────────────────────────────────────────
    logger.info("=== Training Configuration ===")
    for key, val in [
        ("data_path",           config.data_path),
        ("lookback_window",     config.lookback_window),
        ("predict_window",      config.predict_window),
        ("batch_size",          config.batch_size),
        ("accumulation_steps",  getattr(config, 'accumulation_steps', 1)),
        ("learning_rate",       config.predictor_learning_rate),
        ("epochs",              config.basemodel_epochs),
        ("device",              device),
        ("tokenizer_path",      config.finetuned_tokenizer_path),
        ("predictor_path",      config.pretrained_predictor_path),
    ]:
        logger.info(f"  {key:<22}: {val}")

    # ── Train ─────────────────────────────────────────────────────────────────
    best_val_loss = train_model(
        model, tokenizer, device, config,
        config.basemodel_save_path, logger,
    )

    final_msg = (
        f"Training complete.  Best val loss: {best_val_loss:.6f}\n"
        f"Checkpoint: {config.basemodel_save_path}"
    )
    logger.info(final_msg)
    print(final_msg)


if __name__ == "__main__":
    main()
```

---

### `finetune_tokenizer.py`

```python
import os
import sys
import json
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from time import gmtime, strftime
import datetime
import logging
from logging.handlers import RotatingFileHandler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append("../")
from model import NosTokenizer
from finetune_base_model import CustomKlineDataset
from config_loader import CustomFinetuneConfig


def set_seed(seed: int, rank: int = 0):
    actual_seed = seed
    random.seed(actual_seed)
    np.random.seed(actual_seed)
    torch.manual_seed(actual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actual_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_model_size(model: torch.nn.Module) -> str:
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if total_params >= 1e9:
        return f"{total_params / 1e9:.1f}B"
    elif total_params >= 1e6:
        return f"{total_params / 1e6:.1f}M"
    else:
        return f"{total_params / 1e3:.1f}K"


def format_time(seconds: float) -> str:
    return str(datetime.timedelta(seconds=int(seconds)))


def setup_logging(exp_name: str, log_dir: str, rank: int = 0) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(f"tokenizer_training_rank_{rank}")
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        return logger
    
    log_file = os.path.join(log_dir, f"tokenizer_training_rank_{rank}.log")
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    
    console_handler = None
    if rank == 0:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    if console_handler is not None:
        console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    if console_handler is not None:
        logger.addHandler(console_handler)
    
    logger.info(f"=== Tokenizer Training Started ===")
    logger.info(f"Experiment Name: {exp_name}")
    logger.info(f"Log Directory: {log_dir}")
    logger.info(f"Rank: {rank}")
    logger.info(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return logger


def create_dataloaders(config):
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        print("Creating tokenizer training data loaders...")
    
    train_dataset = CustomKlineDataset(
        data_path=config.data_path,
        data_type="train",
        lookback_window=config.lookback_window,
        predict_window=config.predict_window,
        clip=config.clip,
        seed=config.seed,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio
    )
    
    val_dataset = CustomKlineDataset(
        data_path=config.data_path,
        data_type="val",
        lookback_window=config.lookback_window,
        predict_window=config.predict_window,
        clip=config.clip,
        seed=config.seed + 1,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio
    )
    
    use_ddp = dist.is_available() and dist.is_initialized()
    train_sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True) if use_ddp else None
    # FIX: Use drop_last=False - validation loop correctly masks padding via valid_mask.
    # drop_last=True discards real data when batch sizes don't divide evenly, causing
    # inconsistent val_loss across different GPU configurations.
    val_sampler = DistributedSampler(val_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False, drop_last=False) if use_ddp else None

    train_loader = DataLoader(
        train_dataset,
        # FIX: Explicit cast from trial config to ensure HPO batch_size is respected
        batch_size=int(config.batch_size),
        shuffle=(train_sampler is None),
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
        drop_last=True,
        sampler=train_sampler
    )

    val_loader = DataLoader(
        val_dataset,
        # FIX: Explicit cast from trial config to ensure HPO batch_size is respected
        batch_size=int(config.batch_size),
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
        drop_last=False,
        sampler=val_sampler
    )
    
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        print(f"Training set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")
    
    return train_loader, val_loader, train_dataset, val_dataset, train_sampler, val_sampler



def train_tokenizer(model, device, config, save_dir, logger, trial=None):
    """
    All previously hardcoded values (pct_start, div_factor, max_norm)
    are now read from config with safe fallbacks.

    True Gradient Accumulation
    ──────────────────────────
    Accumulates gradients across `accumulation_steps` DataLoader batches before
    performing an optimizer step. This achieves the effective batch size:
    effective_batch_size = batch_size * accumulation_steps
    """
    logger.info("Starting tokenizer training...")
    use_ddp = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if use_ddp else 0

    train_loader, val_loader, train_dataset, val_dataset, \
        train_sampler, val_sampler = create_dataloaders(config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.tokenizer_learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay
    )

    # ── Scheduler: use effective steps for true gradient accumulation ────────
    pct_start = getattr(config, 'tokenizer_pct_start', 0.03)
    div_factor = getattr(config, 'tokenizer_div_factor', 10.0)
    accumulation_steps = getattr(config, 'accumulation_steps', 1)

    import math
    effective_steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        # FIX: Guarantee max_lr matches the trial's exact optimizer LR
        max_lr=[group['lr'] for group in optimizer.param_groups],
        steps_per_epoch=effective_steps_per_epoch,
        epochs=config.tokenizer_epochs,
        pct_start=pct_start,
        div_factor=div_factor
    )

    if use_ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        model = DDP(model, device_ids=[local_rank],
                    output_device=local_rank, find_unused_parameters=False)

    # ── Grad clip: read from config ───────────────────────────────
    max_grad_norm = getattr(config, 'tokenizer_max_grad_norm', 2.0)

    best_val_loss = float("inf")
    batch_idx_global = 0

    for epoch in range(config.tokenizer_epochs):
        epoch_start_time = time.time()
        model.train()

        train_dataset.set_epoch_seed(epoch * 10000)
        val_dataset.set_epoch_seed(0)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        current_accum_loss = 0.0

        # Zero gradients BEFORE the dataloader loop (true gradient accumulation)
        optimizer.zero_grad()

        # ── Dataloader loop (True Gradient Accumulation) ─────────────────────
        for batch_idx, (batch_x, _) in enumerate(train_loader):
            batch_x = batch_x.to(device, non_blocking=True)

            zs, bsq_loss, _, _ = (model.module if use_ddp else model)(batch_x)
            z_pre, z = zs

            recon_loss_pre = F.mse_loss(z_pre, batch_x)
            recon_loss_all = F.mse_loss(z, batch_x)

            weighted_recon = (
                (config.tokenizer_recon_pre_weight * recon_loss_pre) +
                (config.tokenizer_recon_all_weight * recon_loss_all)
            )

            loss = (config.tokenizer_recon_weight * weighted_recon) + (config.tokenizer_bsq_weight * bsq_loss)

            # ── TASK 1.2: Dynamic Micro-Batch & DDP Sync Storm Fix ──────────
            is_last_batch = (batch_idx + 1) == len(train_loader)
            is_step_time = (batch_idx + 1) % accumulation_steps == 0 or is_last_batch

            # Dynamic scalar to prevent inverse gradient accumulation on the tail batch
            current_micro_batches = (batch_idx % accumulation_steps) + 1 if is_last_batch else accumulation_steps
            loss_scaled = loss / current_micro_batches

            # DDP Context Manager to prevent all_reduce storms
            if use_ddp and not is_step_time:
                with model.no_sync():
                    loss_scaled.backward()
            else:
                loss_scaled.backward()

            current_accum_loss += loss.item()

            # ── Optimizer step (once per accumulation cycle) ────────────────
            if is_step_time:
                # Grad clip from config
                torch.nn.utils.clip_grad_norm_(
                    (model.module if use_ddp else model).parameters(),
                    max_norm=max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # Logging
                avg_loss = current_accum_loss / current_micro_batches

                if (batch_idx_global + 1) % config.log_interval == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    log_msg = (
                        f"[Epoch {epoch+1}/{config.tokenizer_epochs}, "
                        f"Step {batch_idx_global+1}/{effective_steps_per_epoch}] "
                        f"LR: {lr:.6f}, Loss: {avg_loss:.4f}"
                    )
                    logger.info(log_msg)
                    if rank == 0:
                        print(log_msg)

                current_accum_loss = 0.0
                batch_idx_global += 1

        # ── Validation ────────────────────────────────────────────
        model.eval()
        tot_val_loss_sum_rank = 0.0
        val_sample_count_rank = 0

        with torch.no_grad():
            for ori_batch_x, _ in val_loader:
                ori_batch_x = ori_batch_x.to(device, non_blocking=True)
                zs, _, _, _ = (model.module if use_ddp else model)(ori_batch_x)
                _, z = zs
                val_loss_item = F.mse_loss(z, ori_batch_x)
                # Detect valid samples (non-padding) to ignore DDP dummy padding
                valid_mask = (ori_batch_x != 0).any(dim=-1).any(dim=-1)
                valid_count = valid_mask.sum().item()
                tot_val_loss_sum_rank += val_loss_item.item() * valid_count
                val_sample_count_rank += valid_count

        if use_ddp:
            tensor_sum = torch.tensor(
                [tot_val_loss_sum_rank, val_sample_count_rank],
                dtype=torch.float64, device=device
            )
            dist.all_reduce(tensor_sum, op=dist.ReduceOp.SUM)
            avg_val_loss = (tensor_sum[0].item() / tensor_sum[1].item()
                            if tensor_sum[1].item() > 0 else 0.0)
        else:
            avg_val_loss = (tot_val_loss_sum_rank / val_sample_count_rank
                            if val_sample_count_rank > 0 else 0)

        epoch_time = time.time() - epoch_start_time
        epoch_summary = (
            f"\n--- Epoch {epoch+1}/{config.tokenizer_epochs} Summary ---\n"
            f"Validation Loss: {avg_val_loss:.6f}\n"
            f"Epoch Time: {format_time(epoch_time)}\n"
        )
        logger.info(epoch_summary)
        if rank == 0:
            print(epoch_summary)

        # ── TASK 1.1: Optuna Pruning Hook (Traceback OOM Fix) ─────────────────
        if trial is not None:
            import optuna
            trial.report(avg_val_loss, epoch)

            if trial.should_prune():
                prune_msg = f"Trial {trial.number} pruned at epoch {epoch} (val_loss: {avg_val_loss:.6f})"
                logger.info(prune_msg)
                if rank == 0:
                    print(prune_msg)

                # CRITICAL: Sever local variable references to prevent Traceback OOM leak
                locals().clear()
                raise optuna.exceptions.TrialPruned()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if rank == 0:
                model_save_path = os.path.join(save_dir, "best_model")
                os.makedirs(model_save_path, exist_ok=True)
                (model.module if use_ddp else model).save_pretrained(model_save_path)
                save_msg = (f"Best model saved: {model_save_path} "
                            f"(val loss: {best_val_loss:.6f})")
                logger.info(save_msg)
                print(save_msg)

    return best_val_loss


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Nos Tokenizer Fine-tuning Training')
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='Configuration file path (default: config.yaml)')
    args = parser.parse_args()
    
    config = CustomFinetuneConfig(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    config = CustomFinetuneConfig(args.config)
    
    os.makedirs(config.tokenizer_save_path, exist_ok=True)
    
    log_dir = os.path.join(config.base_save_path, "logs")
    logger = setup_logging(config.exp_name, log_dir, 0)
    
    set_seed(config.seed)
    
    # 加载预训练tokenizer
    if getattr(config, 'pre_trained_tokenizer', True):
        logger.info("Loading pretrained tokenizer...")
        print("Loading pretrained tokenizer...")
        tokenizer = NosTokenizer.from_pretrained(config.pretrained_tokenizer_path)
    else:
        print("pre_trained_tokenizer=False, randomly initializing Tokenizer architecture")
        import json, os
        cfg_path = os.path.join(config.pretrained_tokenizer_path, 'config.json')
        with open(cfg_path, 'r') as f:
            arch = json.load(f)
        tokenizer = NosTokenizer(
            d_in=arch.get('d_in', 6),
            d_model=arch.get('d_model', 256),
            n_heads=arch.get('n_heads', 4),
            ff_dim=arch.get('ff_dim', 512),
            n_enc_layers=arch.get('n_enc_layers', 4),
            n_dec_layers=arch.get('n_dec_layers', 4),
            ffn_dropout_p=arch.get('ffn_dropout_p', 0.0),
            attn_dropout_p=arch.get('attn_dropout_p', 0.0),
            resid_dropout_p=arch.get('resid_dropout_p', 0.0),
            s1_bits=arch.get('s1_bits', 10),
            s2_bits=arch.get('s2_bits', 10),
            beta=arch.get('beta', 0.05),
            gamma0=arch.get('gamma0', 1.0),
            gamma=arch.get('gamma', 1.1),
            zeta=arch.get('zeta', 0.05),
            group_size=arch.get('group_size', 4)
        )
    tokenizer = tokenizer.to(device)
    
    model_size = get_model_size(tokenizer)
    logger.info(f"Tokenizer parameters: {model_size}")
    print(f"Tokenizer parameters: {model_size}")
    
    logger.info("=== Training Configuration ===")
    logger.info(f"Data path: {config.data_path}")
    logger.info(f"Lookback window: {config.lookback_window}")
    logger.info(f"Predict window: {config.predict_window}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.tokenizer_learning_rate}")
    logger.info(f"Training epochs: {config.tokenizer_epochs}")
    logger.info(f"Device: {device}")
    logger.info(f"Distributed training: False")
    
    logger.info("Starting tokenizer fine-tuning training...")
    print("Starting tokenizer fine-tuning training...")
    best_val_loss = train_tokenizer(tokenizer, device, config, config.tokenizer_save_path, logger)
    
    final_msg = f"Tokenizer training completed! Best validation loss: {best_val_loss:.4f}\nModel saved to: {config.tokenizer_save_path}"
    logger.info(final_msg)
    print(final_msg)


if __name__ == "__main__":
    main()
    

```

---

### `train_sequential.py`

```python
"""
train_sequential.py

Sequential fine-tuning pipeline for the Nos model.
Runs tokenizer and basemodel training phases in order, with optional
HPO parameter injection via apply_bsq_overrides and apply_dropout_overrides.

Launch (single GPU):
    python train_sequential.py --config config.yaml

Launch (multi-GPU via torchrun):
    torchrun --nproc_per_node=8 train_sequential.py --config config.yaml
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist

sys.path.append('../')
from model import Nos, NosTokenizer, NosPredictor

from config_loader import CustomFinetuneConfig
from finetune_tokenizer import train_tokenizer, set_seed, setup_logging as setup_tokenizer_logging
from finetune_base_model import train_model, create_dataloaders, setup_logging as setup_basemodel_logging

# ── HPO override helpers ───────────────────────────────────────────────────────
# Reusing these functions from hpo_tuner.py guarantees that the dropout and BSQ
# parameter injection logic is identical during HPO search and final training.
from hpo_tuner import apply_dropout_overrides, apply_bsq_overrides


class SequentialTrainer:
    """
    Orchestrates sequential fine-tuning of the Nos tokenizer and basemodel.

    Handles device setup, distributed training initialisation, directory
    creation, and delegates to the phase-specific training functions.
    """

    def __init__(self, config_path: str = None):
        self.config = CustomFinetuneConfig(config_path)
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(
            os.environ.get(
                "LOCAL_RANK",
                str(self.config.device_id if hasattr(self.config, 'device_id') else 0)
            )
        )
        self.device = self._setup_device()

        self.config.print_config_summary()

    # ── Device / Distributed Setup ─────────────────────────────────────────────

    def _setup_device(self):
        if self.config.use_cuda and torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            device = torch.device(f"cuda:{self.local_rank}")
        else:
            device = torch.device("cpu")

        if self.rank == 0:
            print(
                f"Using device: {device} "
                f"(rank={self.rank}, world_size={self.world_size}, "
                f"local_rank={self.local_rank})"
            )
        return device

    def _setup_distributed(self):
        if self.world_size > 1 and torch.cuda.is_available():
            backend = os.environ.get("DIST_BACKEND", "nccl").lower()
            if not dist.is_initialized():
                dist.init_process_group(backend=backend)
            if self.rank == 0:
                print(
                    f"Distributed training initialised: "
                    f"backend={backend}, world_size={self.world_size}"
                )
        else:
            if self.rank == 0:
                print(
                    "Distributed training not enabled, "
                    "using single GPU/CPU training"
                )

    # ── Directory / Model Existence Helpers ────────────────────────────────────

    def _check_existing_models(self):
        tokenizer_exists = os.path.exists(self.config.tokenizer_best_model_path)
        basemodel_exists = os.path.exists(self.config.basemodel_best_model_path)

        print(f"Tokenizer model exists: {tokenizer_exists}")
        print(f"Basemodel model exists: {basemodel_exists}")

        return tokenizer_exists, basemodel_exists

    def _create_directories(self):
        os.makedirs(self.config.tokenizer_save_path, exist_ok=True)
        os.makedirs(self.config.basemodel_save_path, exist_ok=True)
        print(f"Created directory: {self.config.tokenizer_save_path}")
        print(f"Created directory: {self.config.basemodel_save_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 1: Tokenizer Fine-tuning
    # ══════════════════════════════════════════════════════════════════════════

    def train_tokenizer_phase(self):
        """
        Fine-tunes the NosTokenizer.

        Model is loaded (pretrained or randomly initialised), HPO-derived
        architecture overrides are injected *before* the model is moved to
        the GPU, then standard tokenizer training proceeds.
        """
        print("\n" + "=" * 60)
        print("Starting Tokenizer Fine-tuning Phase")
        print("=" * 60)

        tokenizer_exists, _ = self._check_existing_models()
        if tokenizer_exists and self.config.skip_existing:
            print("Tokenizer model already exists, skipping training")
            return True

        log_dir = os.path.join(self.config.base_save_path, "logs")
        logger = setup_tokenizer_logging(self.config.exp_name, log_dir, self.rank)

        set_seed(self.config.seed)

        # ── Model instantiation ────────────────────────────────────────────
        if getattr(self.config, 'pre_trained_tokenizer', True):
            logger.info("Loading pretrained tokenizer...")
            if self.rank == 0:
                print("Loading pretrained tokenizer...")
            tokenizer = NosTokenizer.from_pretrained(
                self.config.pretrained_tokenizer_path
            )
        else:
            if self.rank == 0:
                print(
                    "pre_trained_tokenizer=False, "
                    "randomly initialising Tokenizer architecture"
                )
            import json
            cfg_path = os.path.join(
                self.config.pretrained_tokenizer_path, 'config.json'
            )
            with open(cfg_path, 'r') as f:
                arch = json.load(f)
            tokenizer = NosTokenizer(
                d_in=arch.get('d_in', 6),
                d_model=arch.get('d_model', 256),
                n_heads=arch.get('n_heads', 4),
                ff_dim=arch.get('ff_dim', 512),
                n_enc_layers=arch.get('n_enc_layers', 4),
                n_dec_layers=arch.get('n_dec_layers', 4),
                ffn_dropout_p=arch.get('ffn_dropout_p', 0.0),
                attn_dropout_p=arch.get('attn_dropout_p', 0.0),
                resid_dropout_p=arch.get('resid_dropout_p', 0.0),
                s1_bits=arch.get('s1_bits', 10),
                s2_bits=arch.get('s2_bits', 10),
                beta=arch.get('beta', 0.05),
                gamma0=arch.get('gamma0', 1.0),
                gamma=arch.get('gamma', 1.1),
                zeta=arch.get('zeta', 0.05),
                group_size=arch.get('group_size', 4),
            )

        # ── HPO parameter injection (must happen BEFORE .to(device)) ──────
        # Applying overrides on CPU avoids device-mismatch errors that can
        # occur when modifying module attributes after GPU placement,
        # particularly inside DDP wrappers.
        tokenizer = apply_bsq_overrides(tokenizer, self.config)
        tokenizer = apply_dropout_overrides(tokenizer, self.config)

        # ── Move to device ─────────────────────────────────────────────────
        tokenizer = tokenizer.to(self.device)

        # ── Diagnostics ───────────────────────────────────────────────────
        model_size = sum(p.numel() for p in tokenizer.parameters())
        logger.info(f"Tokenizer parameters: {model_size:,}")
        if self.rank == 0:
            print(f"Tokenizer parameters: {model_size:,}")

        logger.info("=== Training Configuration ===")
        logger.info(f"Data path: {self.config.data_path}")
        logger.info(f"Lookback window: {self.config.lookback_window}")
        logger.info(f"Predict window: {self.config.predict_window}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Learning rate: {self.config.tokenizer_learning_rate}")
        logger.info(f"Training epochs: {self.config.tokenizer_epochs}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Distributed training: False")

        # ── Training ───────────────────────────────────────────────────────
        logger.info("Starting tokenizer fine-tuning training...")
        if self.rank == 0:
            print("Starting tokenizer fine-tuning training...")

        start_time = time.time()
        best_val_loss = train_tokenizer(
            tokenizer,
            self.device,
            self.config,
            self.config.tokenizer_save_path,
            logger,
        )
        training_time = time.time() - start_time

        final_msg = (
            f"Tokenizer training completed! "
            f"Best validation loss: {best_val_loss:.4f}\n"
            f"Training time: {training_time / 60:.2f} minutes\n"
            f"Model saved to: {self.config.tokenizer_save_path}"
        )
        logger.info(final_msg)
        if self.rank == 0:
            print(f"\n{final_msg}")

        return True

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 2: Basemodel (Predictor) Fine-tuning
    # ══════════════════════════════════════════════════════════════════════════

    def train_basemodel_phase(self):
        """
        Fine-tunes the Nos predictor with a (optionally frozen) tokenizer.

        Both the tokenizer and predictor receive HPO-derived overrides before
        being moved to the GPU.  The tokenizer receives both BSQ and dropout
        overrides; the predictor receives only dropout overrides (BSQ lives
        exclusively in the tokenizer architecture).
        """
        print("\n" + "=" * 60)
        print("Starting Basemodel Fine-tuning Phase")
        print("=" * 60)

        if getattr(self.config, 'pre_trained_tokenizer', True):
            if not os.path.exists(self.config.finetuned_tokenizer_path):
                raise FileNotFoundError(
                    f"Fine-tuned tokenizer does not exist: "
                    f"{self.config.finetuned_tokenizer_path}"
                )

        _, basemodel_exists = self._check_existing_models()
        if basemodel_exists and self.config.skip_existing:
            print("Basemodel model already exists, skipping training")
            return True

        log_dir = os.path.join(self.config.base_save_path, "logs")
        logger = setup_basemodel_logging(self.config.exp_name, log_dir, self.rank)

        set_seed(self.config.seed)

        # ── Tokenizer instantiation ────────────────────────────────────────
        if getattr(self.config, 'pre_trained_tokenizer', True):
            logger.info("Loading fine-tuned tokenizer...")
            if self.rank == 0:
                print("Loading fine-tuned tokenizer...")
            tokenizer = NosTokenizer.from_pretrained(
                self.config.finetuned_tokenizer_path
            )
        else:
            if self.rank == 0:
                print(
                    "pre_trained_tokenizer=False, "
                    "randomly initialising Tokenizer architecture "
                    "for Predictor training"
                )
            import json
            cfg_path = os.path.join(
                self.config.pretrained_tokenizer_path, 'config.json'
            )
            with open(cfg_path, 'r') as f:
                arch = json.load(f)
            tokenizer = NosTokenizer(
                d_in=arch.get('d_in', 6),
                d_model=arch.get('d_model', 256),
                n_heads=arch.get('n_heads', 4),
                ff_dim=arch.get('ff_dim', 512),
                n_enc_layers=arch.get('n_enc_layers', 4),
                n_dec_layers=arch.get('n_dec_layers', 4),
                ffn_dropout_p=arch.get('ffn_dropout_p', 0.0),
                attn_dropout_p=arch.get('attn_dropout_p', 0.0),
                resid_dropout_p=arch.get('resid_dropout_p', 0.0),
                s1_bits=arch.get('s1_bits', 10),
                s2_bits=arch.get('s2_bits', 10),
                beta=arch.get('beta', 0.05),
                gamma0=arch.get('gamma0', 1.0),
                gamma=arch.get('gamma', 1.1),
                zeta=arch.get('zeta', 0.05),
                group_size=arch.get('group_size', 4),
            )

        # ── HPO injection into the frozen tokenizer (CPU, before .to()) ───
        tokenizer = apply_bsq_overrides(tokenizer, self.config)
        tokenizer = apply_dropout_overrides(tokenizer, self.config)

        # ── Move tokenizer to device ───────────────────────────────────────
        tokenizer = tokenizer.to(self.device)

        # ── Predictor instantiation ────────────────────────────────────────
        if getattr(self.config, 'pre_trained_predictor', True):
            logger.info("Loading pretrained predictor...")
            if self.rank == 0:
                print("Loading pretrained predictor...")
            model = Nos.from_pretrained(self.config.pretrained_predictor_path)
        else:
            if self.rank == 0:
                print(
                    "pre_trained_predictor=False, "
                    "randomly initialising Predictor architecture"
                )
            import json
            cfg_path = os.path.join(
                self.config.pretrained_predictor_path, 'config.json'
            )
            with open(cfg_path, 'r') as f:
                arch = json.load(f)
            print("model_config: ", arch)
            model = Nos(
                s1_bits=arch.get('s1_bits', 10),
                s2_bits=arch.get('s2_bits', 10),
                n_layers=arch.get('n_layers', 12),
                d_model=arch.get('d_model', 832),
                n_heads=arch.get('n_heads', 16),
                ff_dim=arch.get('ff_dim', 2048),
                ffn_dropout_p=arch.get('ffn_dropout_p', 0.2),
                attn_dropout_p=arch.get('attn_dropout_p', 0.0),
                resid_dropout_p=arch.get('resid_dropout_p', 0.2),
                token_dropout_p=arch.get('token_dropout_p', 0.0),
                learn_te=arch.get('learn_te', True),
            )

        # ── HPO dropout injection into predictor (CPU, before .to()) ──────
        # BSQ overrides are intentionally omitted here — the BSQ quantizer
        # lives inside the tokenizer, not the predictor.
        model = apply_dropout_overrides(model, self.config)

        # ── Move predictor to device ───────────────────────────────────────
        model = model.to(self.device)

        # ── Diagnostics ───────────────────────────────────────────────────
        model_size = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {model_size:,}")
        if self.rank == 0:
            print(f"Model parameters: {model_size:,}")

        logger.info("=== Training Configuration ===")
        logger.info(f"Data path: {self.config.data_path}")
        logger.info(f"Lookback window: {self.config.lookback_window}")
        logger.info(f"Predict window: {self.config.predict_window}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Learning rate: {self.config.predictor_learning_rate}")
        logger.info(f"Training epochs: {self.config.basemodel_epochs}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Tokenizer path: {self.config.finetuned_tokenizer_path}")
        logger.info(f"Pretrained model path: {self.config.pretrained_predictor_path}")

        # ── Training ───────────────────────────────────────────────────────
        logger.info("Starting fine-tuning training...")
        if self.rank == 0:
            print("Starting fine-tuning training...")

        start_time = time.time()
        best_val_loss = train_model(
            model,
            tokenizer,
            self.device,
            self.config,
            self.config.basemodel_save_path,
            logger,
        )
        training_time = time.time() - start_time

        final_msg = (
            f"Basemodel training completed! "
            f"Best validation loss: {best_val_loss:.4f}\n"
            f"Training time: {training_time / 60:.2f} minutes\n"
            f"Model saved to: {self.config.basemodel_save_path}"
        )
        logger.info(final_msg)
        if self.rank == 0:
            print(f"\n{final_msg}")

        return True

    # ══════════════════════════════════════════════════════════════════════════
    # Top-level Orchestration
    # ══════════════════════════════════════════════════════════════════════════

    def run_training(self):
        if self.rank == 0:
            print("Starting Nos model sequential fine-tuning training")
            print(f"Experiment name: {self.config.experiment_name}")
            print(f"Experiment description: {self.config.experiment_description}")

        self._setup_distributed()
        self._create_directories()

        total_start_time = time.time()

        try:
            if self.config.train_tokenizer:
                success = self.train_tokenizer_phase()
                if not success:
                    print("Tokenizer training failed, terminating training")
                    return False
            else:
                print("Skipping Tokenizer training phase")

            if self.config.train_basemodel:
                success = self.train_basemodel_phase()
                if not success:
                    print("Basemodel training failed, terminating training")
                    return False
            else:
                print("Skipping Basemodel training phase")

            total_time = time.time() - total_start_time

            if self.rank == 0:
                print("\n" + "=" * 60)
                print("Training completed!")
                print("=" * 60)
                print(f"Total training time: {total_time / 60:.2f} minutes")
                print(f"Tokenizer model: {self.config.tokenizer_best_model_path}")
                print(f"Basemodel model: {self.config.basemodel_best_model_path}")
                print("=" * 60)

            return True

        except Exception as e:
            if self.rank == 0:
                print(f"Error occurred during training: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

        finally:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Nos Model Sequential Fine-tuning Training'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Configuration file path (default: config.yaml)',
    )
    parser.add_argument(
        '--skip-tokenizer',
        action='store_true',
        help='Skip tokenizer training phase',
    )
    parser.add_argument(
        '--skip-basemodel',
        action='store_true',
        help='Skip basemodel training phase',
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip training for existing models',
    )

    args = parser.parse_args()

    trainer = SequentialTrainer(args.config)

    if args.skip_tokenizer:
        trainer.config.train_tokenizer = False
    if args.skip_basemodel:
        trainer.config.train_basemodel = False
    if args.skip_existing:
        trainer.config.skip_existing = True

    success = trainer.run_training()

    if success:
        print("Training completed successfully!")
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        sys.exit(0)
    else:
        print("Training failed!")
        if dist.is_available() and dist.is_initialized():
            try:
                dist.barrier()
                dist.destroy_process_group()
            except Exception:
                pass
        sys.exit(1)


if __name__ == "__main__":
    main()
```

---

### `full_scale_backtest.py`

```python
import os
import logging
from typing import List, Dict, Any, Optional, Union
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from finetune_base_model import CustomFinetuneConfig
from model import Nos, NosTokenizer, NosPredictor

# =============================================================================
# Constants & Configuration
# =============================================================================

CONFIG_PATH = "configs/config_t40_hpo_winner.yaml"  # Make sure this points to your active config
RESULTS_DIR = "backtest_results"

PRICE_COLS = frozenset(['open', 'high', 'low', 'close'])
VOL_AMT_COLS = frozenset(['volume', 'amount'])

class VerdictThresholds:
    """Constants for signal verdict thresholds."""
    ALPHA_EXCEPTIONAL = 57.0
    ALPHA_STRONG = 55.0
    ALPHA_BASE = 53.0
    ALPHA_WEAK = 51.0
    ALPHA_MARGINAL = 50.0

    CORR_STRONG = 0.5
    CORR_MODERATE = 0.3
    CORR_WEAK = 0.1

    EPSILON = 1e-10
    DIR_MOVE_THRESHOLD = 0.0001

# =============================================================================
# Logging Setup
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("nos_backtest_engine")

# =============================================================================
# Utility Functions
# =============================================================================

def _sort_extra_cols(cols: List[str]) -> List[str]:
    """Sort extra_* columns numerically when possible, alphabetically otherwise."""
    def _sort_key(c: str) -> tuple:
        try:
            return (0, int(c.split('_')[1]))
        except (ValueError, IndexError):
            return (1, c)
    return sorted(cols, key=_sort_key)


def get_all_feature_cols(df: pd.DataFrame) -> List[str]:
    """
    Return the ordered list of all feature columns exactly as the model sees them.
    """
    base_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
    extra_cols = [c for c in df.columns if c.startswith('extra_')]
    extra_cols = _sort_extra_cols(extra_cols)
    return base_cols + extra_cols


def _get_column_category(col: str) -> str:
    """Determine the logical category of a feature column."""
    if col in PRICE_COLS:
        return "PRICE"
    elif col in VOL_AMT_COLS:
        return "VOL/AMT"
    elif col.startswith('extra_'):
        return "EXTRA"
    return "OTHER"


def _get_column_verdict(acc: float, corr: float, category: str) -> str:
    """Generate a categorical verdict string based on calculated metrics."""
    if category in ("PRICE", "VOL/AMT"):
        if np.isnan(acc):
            if corr > VerdictThresholds.CORR_MODERATE:
                return "🟡 CORRELATION SIGNAL"
            return "⚪ INSUFFICIENT DATA"
            
        if acc > VerdictThresholds.ALPHA_EXCEPTIONAL:
            return "🟢🟢 EXCEPTIONAL ALPHA"
        elif acc > VerdictThresholds.ALPHA_STRONG:
            return "🟢 STRONG ALPHA"
        elif acc > VerdictThresholds.ALPHA_BASE:
            return "🟢 MODEL HAS ALPHA"
        elif acc > VerdictThresholds.ALPHA_WEAK:
            return "🟡 WEAK SIGNAL"
        elif acc > VerdictThresholds.ALPHA_MARGINAL:
            return "🟡 MARGINAL"
        return "🔴 NO SIGNAL"
    else:
        if np.isnan(acc):
            if corr > VerdictThresholds.CORR_STRONG:
                return "🟢 STRONG CORRELATION"
            elif corr > VerdictThresholds.CORR_MODERATE:
                return "🟡 MODERATE CORRELATION"
            elif corr > VerdictThresholds.CORR_WEAK:
                return "🟡 WEAK CORRELATION"
            return "🔴 NO CORRELATION"
            
        if acc > VerdictThresholds.ALPHA_STRONG and corr > VerdictThresholds.CORR_MODERATE:
            return "🟢🟢 EXCEPTIONAL (DIR+CORR)"
        elif acc > VerdictThresholds.ALPHA_BASE:
            return "🟢 STRONG SIGNAL"
        elif acc > VerdictThresholds.ALPHA_WEAK:
            return "🟡 WEAK SIGNAL"
        elif acc > VerdictThresholds.ALPHA_MARGINAL:
            return "🟡 MARGINAL"
        return "🔴 NO SIGNAL"

# =============================================================================
# Mathematical & Metrics Engine
# =============================================================================

class MetricsCalculator:
    """Encapsulates statistical operations for evaluating model predictions."""

    @staticmethod
    def compute_rmse(pred: np.ndarray, true: np.ndarray) -> float:
        return float(np.sqrt(np.mean((pred - true) ** 2)))

    @staticmethod
    def compute_mape(pred: np.ndarray, true: np.ndarray) -> float:
        denom = np.abs(true) + VerdictThresholds.EPSILON
        return float(np.mean(np.abs((pred - true) / denom)) * 100)

    @staticmethod
    def compute_r2(pred: np.ndarray, true: np.ndarray) -> float:
        ss_res = np.sum((pred - true) ** 2)
        ss_tot = np.sum((true - np.mean(true)) ** 2)
        return float(1.0 - (ss_res / (ss_tot + VerdictThresholds.EPSILON))) if ss_tot > VerdictThresholds.EPSILON else 0.0

    @staticmethod
    def compute_vectorized_path_accuracy(pred: np.ndarray, true: np.ndarray, initial_price: float) -> float:
        """Vectorized approach to calculate full-path directional accuracy."""
        if len(pred) < 1:
            return np.nan

        shifted_true = np.roll(true, 1)
        shifted_true[0] = initial_price

        t_moves = true - shifted_true
        p_moves = pred - shifted_true

        valid_mask = (np.abs(t_moves) / (np.abs(shifted_true) + VerdictThresholds.EPSILON)) > VerdictThresholds.DIR_MOVE_THRESHOLD
        
        if not np.any(valid_mask):
            return np.nan

        correct_dir = np.sign(p_moves[valid_mask]) == np.sign(t_moves[valid_mask])
        return float(np.mean(correct_dir))

def compute_metrics_all_columns(task: Dict[str, Any], pred_df: pd.DataFrame, feature_cols: List[str]) -> Optional[Dict[str, Any]]:
    """
    Compute directional accuracy, correlation, RMSE, MAPE for EVERY column.
    """
    truth_df = task['truth']
    min_len = min(len(pred_df), len(truth_df))
    
    if min_len == 0:
        return None

    available_cols = [c for c in feature_cols if c in pred_df.columns and c in truth_df.columns]
    if not available_cols:
        return None

    result = {
        'start_time': task['y_ts'].iloc[0] if not task['y_ts'].empty else None,
        'end_time': task['y_ts'].iloc[-1] if not task['y_ts'].empty else None,
    }

    col_corrs, col_rmses, col_mapes, col_accs = [], [], [], []

    for col in available_cols:
        pred_vals = pred_df[col].values[:min_len]
        true_vals = truth_df[col].values[:min_len]
        last_price = task['context'][col].iloc[-1] if col in task['context'].columns else None

        # Base error metrics
        rmse = MetricsCalculator.compute_rmse(pred_vals, true_vals)
        mape = MetricsCalculator.compute_mape(pred_vals, true_vals)
        mae = float(np.mean(np.abs(pred_vals - true_vals)))
        r_squared = MetricsCalculator.compute_r2(pred_vals, true_vals)
        
        mean_true = np.mean(np.abs(true_vals)) + VerdictThresholds.EPSILON
        nrmse = (rmse / mean_true) * 100

        # Correlation
        if min_len < 2 or np.std(pred_vals) < VerdictThresholds.EPSILON or np.std(true_vals) < VerdictThresholds.EPSILON:
            corr = 0.0
        else:
            corr = np.corrcoef(pred_vals, true_vals)[0, 1]
            if np.isnan(corr): corr = 0.0

        # Directional Accuracy (T+1 and Path)
        correct_dir = None
        path_acc = np.nan
        
        if last_price is not None and not np.isnan(last_price):
            pred_move = pred_vals[0] - last_price
            true_move = true_vals[0] - last_price
            
            if abs(true_move) / (abs(last_price) + VerdictThresholds.EPSILON) > VerdictThresholds.DIR_MOVE_THRESHOLD:
                correct_dir = 1 if (np.sign(pred_move) == np.sign(true_move)) else 0

            if min_len > 1:
                path_acc = MetricsCalculator.compute_vectorized_path_accuracy(pred_vals, true_vals, last_price)

        # Build column results
        result.update({
            f'{col}_corr': corr,
            f'{col}_rmse': rmse,
            f'{col}_nrmse': nrmse,
            f'{col}_mape': mape,
            f'{col}_mae': mae,
            f'{col}_r2': r_squared,
            f'{col}_path_acc': path_acc,
            f'{col}_last_price': float(last_price) if last_price is not None else np.nan,
            f'{col}_pred_t1': float(pred_vals[0]),
            f'{col}_true_t1': float(true_vals[0])
        })

        if correct_dir is not None:
            result[f'{col}_acc'] = correct_dir
            col_accs.append(correct_dir)

        col_corrs.append(corr)
        col_rmses.append(rmse)
        col_mapes.append(mape)

    # Global Aggregates
    result['avg_corr_all'] = np.mean(col_corrs) if col_corrs else 0.0
    result['avg_rmse_all'] = np.mean(col_rmses) if col_rmses else 0.0
    result['avg_mape_all'] = np.mean(col_mapes) if col_mapes else 0.0
    result['avg_acc_all'] = np.mean(col_accs) if col_accs else np.nan

    # Group Aggregates (PRICE vs VOL/AMT)
    price_accs = [result[f'{c}_acc'] for c in PRICE_COLS if f'{c}_acc' in result]
    result['avg_acc_price'] = np.mean(price_accs) if price_accs else np.nan

    va_accs = [result[f'{c}_acc'] for c in VOL_AMT_COLS if f'{c}_acc' in result]
    result['avg_acc_va'] = np.mean(va_accs) if va_accs else np.nan

    return result

# =============================================================================
# Reporting & I/O Engine
# =============================================================================

class DataExporter:
    """Handles writing backtest results to disk efficiently."""
    
    @staticmethod
    def save_results(df_res: pd.DataFrame, col_summary_rows: List[Dict[str, Any]], feature_cols: List[str]) -> None:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        try:
            full_path = os.path.join(RESULTS_DIR, "backtest_results_full.csv")
            df_res.to_csv(full_path, index=False)
            logger.info(f"Saved full results: {full_path}")

            if col_summary_rows:
                summary_path = os.path.join(RESULTS_DIR, "backtest_column_summary.csv")
                pd.DataFrame(col_summary_rows).to_csv(summary_path, index=False)
                logger.info(f"Saved column summary: {summary_path}")

            if len(df_res) > 20:
                rolling_path = os.path.join(RESULTS_DIR, "backtest_rolling_accuracy.csv")
                DataExporter._save_rolling_metrics(df_res, feature_cols, rolling_path)

            deepdive_dir = os.path.join(RESULTS_DIR, "column_deep_dives")
            os.makedirs(deepdive_dir, exist_ok=True)
            for col in feature_cols:
                col_keys = [k for k in df_res.columns if k.startswith(f'{col}_')]
                if col_keys:
                    col_df = df_res[['start_time', 'end_time'] + col_keys]
                    col_df.to_csv(os.path.join(deepdive_dir, f"deepdive_{col}.csv"), index=False)
            
        except OSError as e:
            logger.exception(f"I/O Error while saving results: {e}")

    @staticmethod
    def _save_rolling_metrics(df_res: pd.DataFrame, feature_cols: List[str], target_path: str) -> None:
        rolling_df = pd.DataFrame()
        rolling_df['start_time'] = df_res['start_time']
        
        # Calculate dynamic window and strictly enforce min_periods <= window_size
        window_size = max(1, min(50, len(df_res) // 4))
        min_periods = min(window_size, max(1, window_size // 2))

        for col in feature_cols:
            acc_key, corr_key = f'{col}_acc', f'{col}_corr'
            if acc_key in df_res.columns:
                rolling_df[f'{col}_rolling_acc'] = df_res[acc_key].rolling(
                    window=window_size, min_periods=min_periods).mean() * 100
            if corr_key in df_res.columns:
                rolling_df[f'{col}_rolling_corr'] = df_res[corr_key].rolling(
                    window=window_size, min_periods=min_periods).mean()

        rolling_df.to_csv(target_path, index=False)


class ReportingEngine:
    """Manages the generation of terminal output and metrics summarization."""
    
    @staticmethod
    def print_detailed_report(results: List[Dict[str, Any]], feature_cols: List[str]) -> None:
        if not results:
            logger.warning("No results generated to report.")
            return

        df_res = pd.DataFrame(results)
        logger.info(f"Generated {len(results)} samples across {len(feature_cols)} feature columns.")
        
        col_summary_rows = []

        # ── 1. Print Detailed Column-by-Column Deep Dives ──
        for idx, col in enumerate(feature_cols):
            corr_key, acc_key = f'{col}_corr', f'{col}_acc'
            if corr_key not in df_res.columns:
                continue
                
            cat = _get_column_category(col)
            corr_vals = df_res[corr_key].dropna()
            acc_vals = df_res[acc_key].dropna() if acc_key in df_res.columns else pd.Series(dtype=float)

            mean_corr = corr_vals.mean() if not corr_vals.empty else 0.0
            mean_acc = acc_vals.mean() * 100 if not acc_vals.empty else np.nan
            verdict = _get_column_verdict(mean_acc, mean_corr, cat)
            
            # Print the deep dive log
            logger.info(f"[{idx + 1}/{len(feature_cols)}] DEEP DIVE: {col.upper():<8} ({cat:<8}) - Accuracy: {mean_acc:>6.2f}% | Verdict: {verdict}")

            col_summary_rows.append({
                'column': col,
                'category': cat,
                'accuracy_%': mean_acc,
                'mean_corr': mean_corr,
                'n_samples': len(corr_vals),
                'n_directional': len(acc_vals),
                'verdict': verdict,
            })

        # ── 2. Print Summary Table ──
        logger.info("=" * 85)
        logger.info(f"{'COLUMN':<15} | {'CATEGORY':<10} | {'DIR. ACCURACY':<15} | {'CORRELATION':<12} | {'VERDICT'}")
        logger.info("-" * 85)
        
        for row in col_summary_rows:
            acc_str = f"{row['accuracy_%']:.2f}%" if not np.isnan(row['accuracy_%']) else "N/A"
            corr_str = f"{row['mean_corr']:.4f}"
            
            logger.info(f"{row['column'].upper():<15} | {row['category']:<10} | {acc_str:<15} | {corr_str:<12} | {row['verdict']}")
        logger.info("=" * 85)

        # ── 3. Print Category Aggregates ──
        overall_acc = df_res['avg_acc_all'].mean() * 100 if 'avg_acc_all' in df_res.columns else np.nan
        price_acc = df_res['avg_acc_price'].mean() * 100 if 'avg_acc_price' in df_res.columns else np.nan
        va_acc = df_res['avg_acc_va'].mean() * 100 if 'avg_acc_va' in df_res.columns else np.nan

        logger.info("--- CATEGORY AGGREGATES ---")
        logger.info(f"Avg Accuracy [ALL COLUMNS]:   {overall_acc:.2f}%" if not np.isnan(overall_acc) else "Avg Accuracy [ALL COLUMNS]:   N/A")
        logger.info(f"Avg Accuracy [PRICE ONLY]:    {price_acc:.2f}%" if not np.isnan(price_acc) else "Avg Accuracy [PRICE ONLY]:    N/A")
        logger.info(f"Avg Accuracy [VOL/AMT ONLY]:  {va_acc:.2f}%" if not np.isnan(va_acc) else "Avg Accuracy [VOL/AMT ONLY]:  N/A")
        logger.info("=====================================================================================")

        DataExporter.save_results(df_res, col_summary_rows, feature_cols)

# =============================================================================
# Main Execution
# =============================================================================

def main() -> None:
    logger.info("Initializing Nos FULL-SCALE INSTITUTIONAL MULTI-COLUMN BACKTEST")

    if not os.path.exists(CONFIG_PATH):
        logger.error(f"Config not found: {CONFIG_PATH}")
        return

    config = CustomFinetuneConfig(CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Loaded config parameters: Lookback={config.lookback_window}, Forecast={config.predict_window}, BatchSize={config.batch_size}, Device={device}")

    try:
        tokenizer_path = config.finetuned_tokenizer_path if os.path.exists(config.finetuned_tokenizer_path) else config.tokenizer_best_model_path
        tokenizer = NosTokenizer.from_pretrained(tokenizer_path)
        model = Nos.from_pretrained(config.basemodel_best_model_path).to(device)

        predictor = NosPredictor(
            model=model,
            tokenizer=tokenizer,
            device=str(device),
            max_context=config.max_context,
            clip=config.clip
        )
        logger.info("Models loaded successfully.")
    except Exception as e:
        logger.exception("Failed to load models.")
        return

    logger.info(f"Loading dataset from: {config.data_path}")
    full_df = pd.read_csv(config.data_path)
    full_df['timestamps'] = pd.to_datetime(full_df['timestamps'])

    if 'volume' not in full_df.columns:
        full_df['volume'] = 0.0
    if 'amount' not in full_df.columns:
        full_df['amount'] = full_df['volume'] * full_df[['open', 'high', 'low', 'close']].mean(axis=1) if 'volume' in full_df.columns else 0.0

    feature_cols = get_all_feature_cols(full_df)
    
    if len(feature_cols) != tokenizer.d_in:
        logger.error(f"FATAL: Tokenizer expects {tokenizer.d_in} features, found {len(feature_cols)}. Exiting.")
        return

    train_end = int(len(full_df) * config.train_ratio)
    val_df = full_df.iloc[train_end:].reset_index(drop=True)
    val_df[feature_cols] = val_df[feature_cols].ffill().fillna(0.0)

    logger.info("Generating backtest windows...")
    tasks = []
    max_start = len(val_df) - config.lookback_window - config.predict_window

    for start_idx in range(0, max_start + 1, config.predict_window):
        hist_end = start_idx + config.lookback_window
        future_end = hist_end + config.predict_window

        ctx = val_df.iloc[start_idx:hist_end].copy().reset_index(drop=True)
        tgt = val_df.iloc[hist_end:future_end].copy().reset_index(drop=True)

        if len(ctx) == config.lookback_window and len(tgt) == config.predict_window:
            tasks.append({
                'context': ctx,
                'x_ts': ctx['timestamps'].reset_index(drop=True),
                'y_ts': tgt['timestamps'].reset_index(drop=True),
                'truth': tgt,
            })

    if not tasks:
        logger.warning("No scenarios generated. Exiting.")
        return

    logger.info(f"Running inference on {len(tasks)} tasks.")
    results, errors = [], 0

    for i in tqdm(range(0, len(tasks), config.batch_size), desc="Backtesting"):
        batch_tasks = tasks[i: i + config.batch_size]
        
        try:
            preds_batch = predictor.predict_batch(
                df_list=[t['context'] for t in batch_tasks],
                x_timestamp_list=[t['x_ts'] for t in batch_tasks],
                y_timestamp_list=[t['y_ts'] for t in batch_tasks],
                pred_len=config.predict_window,
                T=0.01, top_k=0, top_p=0.99, sample_count=1, verbose=False
            )

            for j, pred_df in enumerate(preds_batch):
                try:
                    res = compute_metrics_all_columns(batch_tasks[j], pred_df, feature_cols)
                    if res: results.append(res)
                except ValueError as ve:
                    errors += 1
                    if errors <= 5: logger.exception(f"Metric calculation error in task {i + j}: {ve}")
                    
        except RuntimeError as re:
            errors += 1
            if errors <= 5: logger.exception(f"Batch prediction error at index {i}: {re}")

    if errors > 0:
        logger.warning(f"Encountered {errors} total errors during prediction pipeline.")

    logger.info("Generating reports...")
    ReportingEngine.print_detailed_report(results, feature_cols)

if __name__ == "__main__":
    main()
```

---

### `requirements.txt`

```text
# Core Deep Learning & Math
torch>=2.0.0
numpy
pandas
einops

# Model Hub & Progress Tracking
huggingface-hub
tqdm

# Configuration Parsing
PyYAML

# Hyperparameter Optimization (HPO)
optuna
plotly
SQLAlchemy
```

---

### `CLAUDE.md`

```markdown
# CLAUDE.md — Project Context for Claude Code

## Project Overview

This is **uNOS** (micro Neural Operational System), a financial time-series prediction system built on a custom transformer architecture. The project finetunes pretrained NOS models on OHLCV (Open-High-Low-Close-Volume) market data for next-step price prediction.

**Project Type**: Python ML/Deep Learning project  
**Domain**: Financial time-series forecasting  
**Primary Use**: Training and running inference on custom finetuned models for cryptocurrency/stock price prediction

---

## Architecture Summary

### The Two-Stage Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE FLOW                           │
│                                                                     │
│  Raw CSV Data (OHLCV time series)                                   │
│      │                                                              │
│      ▼                                                              │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │  PHASE 1: Tokenizer Finetuning                           │       │
│  │  (NosTokenizer)                                          │       │
│  │  Input: Raw OHLCV → learns domain-specific codes        │       │
│  │  Loss: MSE reconstruction + BSQ entropy                 │       │
│  │  Output: Finetuned tokenizer → finetuned/tokenizer/      │       │
│  └─────────────────────────────────────────────────────────┘       │
│      │                                                              │
│      ▼  (tokenizer frozen, used for encoding)                      │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │  PHASE 2: Basemodel (Predictor) Finetuning              │       │
│  │  (Nos / NosPredictor)                                   │       │
│  │  Input: Tokenized sequences → learns temporal patterns  │       │
│  │  Loss: CrossEntropy(s1_tokens) + CrossEntropy(s2_tokens) │       │
│  │  Output: Finetuned predictor → finetuned/basemodel/      │       │
│  └─────────────────────────────────────────────────────────┘       │
│      │                                                              │
│      ▼                                                              │
│  Inference: predictor.predict(df, timestamps, pred_len)            │
└─────────────────────────────────────────────────────────────────────┘
```

### Model Components

| Component | File | Purpose |
|-----------|------|---------|
| **NosTokenizer** | `model/nos.py:13-179` | Encodes OHLCV into discrete tokens using BSQ |
| **Nos** | `model/nos.py:181-336` | Main transformer with hierarchical embedding |
| **NosPredictor** | `model/nos.py:463-633` | High-level inference wrapper |
| **BSQuantizer** | `model/module.py` | Binary Spherical Quantization |
| **HierarchicalEmbedding** | `model/module.py` | S1+S2 token embeddings |
| **DependencyAwareLayer** | `model/module.py` | S2 conditioned on S1 |

---

## Key Scripts

| Script | Purpose |
|--------|---------|
| `train_sequential.py` | Main training orchestrator — runs tokenizer then basemodel |
| `hpo_tuner.py` | Optuna-based hyperparameter optimization |
| `config_loader.py` | YAML config parsing with defaults |
| `finetune_tokenizer.py` | Tokenizer-specific training loop |
| `finetune_base_model.py` | Basemodel-specific training loop |
| `codebase.py` | Generates CODE.md documentation |
| `train.md` | Comprehensive training guide (281KB+) |

---

## Configuration

### Default Config Location
`configs/config.yaml`

### Key Config Sections

```yaml
data:
  data_path: "data/1h.csv"        # CSV with timestamps, OHLCV columns
  lookback_window: 512            # Historical context length
  predict_window: 48              # Prediction horizon
  clip: 5.0                       # Normalization clipping

training:
  tokenizer_epochs: 30
  basemodel_epochs: 20
  batch_size: 32
  tokenizer_learning_rate: 0.0002
  predictor_learning_rate: 0.000001   # Keep very low to avoid forgetting

model_paths:
  pretrained_tokenizer: "models/nos_tokenizer_2k"
  pretrained_predictor: "models/nos_mini"

hpo:
  enabled: false                  # Set true to activate Optuna
  n_trials: 30
  storage: null                   # Use null for in-memory (dev), SQLite for persistence
```

### Required CSV Format
```csv
timestamps,open,high,low,close,volume,amount
2024-01-01 00:00:00,45000.0,45100.0,44900.0,45050.0,100.0,4500000.0
...
```

---

## Common Commands

### Quick Start (Default Config)
```bash
# Train both tokenizer and basemodel
python train_sequential.py --config configs/config.yaml

# Train only tokenizer (Phase 1)
python train_sequential.py --config configs/config.yaml --skip-basemodel

# Train only basemodel (Phase 2, requires finetuned tokenizer)
python train_sequential.py --config configs/config.yaml --skip-tokenizer

# Skip models that already exist
python train_sequential.py --config configs/config.yaml --skip-existing
```

### Hyperparameter Optimization
```bash
# Run HPO for tokenizer only
python hpo_tuner.py --config configs/config.yaml --phase tokenizer --n-trials 40

# Run HPO for basemodel only (requires finetuned tokenizer)
python hpo_tuner.py --config configs/config.yaml --phase basemodel --tokenizer-path finetuned/test_1h/tokenizer/best_model --n-trials 40

# Apply best HPO params to new config
python hpo_tuner.py --config configs/config.yaml --phase tokenizer --apply-best --output-config configs/config_best.yaml
```

### Inference Example
```python
import pandas as pd
from model import NosPredictor, Nos, NosTokenizer
from transformers import AutoModel

# Load finetuned models
tokenizer = NosTokenizer.from_pretrained("finetuned/test_1h/tokenizer/best_model")
model = Nos.from_pretrained("finetuned/test_1h/basemodel/best_model")

# Create predictor
predictor = NosPredictor(model, tokenizer, device="cuda:0", max_context=512, clip=5.0)

# Load data
df = pd.read_csv("data/1h.csv")
df['timestamps'] = pd.to_datetime(df['timestamps'])
df = df.sort_values('timestamps').reset_index(drop=True)

# Prepare timestamps
x_timestamp = df['timestamps'].iloc[-512:]
y_timestamp = pd.date_range(start=x_timestamp.iloc[-1], periods=49, freq='h')[1:]

# Predict
preds = predictor.predict(
    df.iloc[-512:],
    x_timestamp,
    y_timestamp,
    pred_len=48,
    T=1.0,           # temperature
    top_k=0,
    top_p=0.9,
    sample_count=5   # number of samples to average
)
print(preds)
```

---

## Training Tips & Pitfalls

### Critical Rules

1. **Predictor LR must be very low** (1e-6 to 5e-6)
   - Pretrained model is valuable — too high causes catastrophic forgetting
   - If val loss increases sharply during training, reduce LR 10x

2. **Always use pretrained models** (`pre_trained: true`)
   - Never train from scratch — the pretrained tokenizerlearns domain-agnostic representations
   - Path: `models/nos_tokenizer_2k` and `models/nos_mini`

3. **Tokenize first, then predict** (never skip tokenizer)
   - Tokenizer quality directly limits predictor ceiling
   - Run tokenizer quality check after Phase 1:
   ```python
   # Reconstruction MSE should be < 0.05
   ```

4. **Data validation before training**
   - Check for gaps in time series (>2h for 1h data)
   - Verify no zero/negative prices
   - Ensure sufficient validation samples (>200 usable windows)

### Common Issues

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| CUDA OOM | Batch too large | Reduce `batch_size` or increase `accumulation_steps` |
| Val loss flat | LR too low | Increase `tokenizer_learning_rate` to 5e-4 |
| Val loss spikes | Catastrophic forgetting | Reduce `predictor_learning_rate` to 1e-7 |
| Codebook collapse | BSQ params wrong | Increase `bsq_gamma` to 1.5 |
| HPO all pruned | Too few epochs | Increase `hpo_tokenizer_epochs` to 15 |

---

## Development Patterns

### Adding New Data Sources
1. Ensure CSV has: `timestamps`, `open`, `high`, `low`, `close`, `volume`, `amount`
2. Update `data.data_path` in config
3. Validate with data quality check script (see train.md Part 1)

### Modifying Model Architecture
- Tokenizer changes: Edit `model/nos.py` class `NosTokenizer`
- Predictor changes: Edit `model/nos.py` class `Nos`
- Core modules: Edit `model/module.py`

### Adding New Training Scenarios
1. Copy `configs/config.yaml` to new file
2. Modify `model_paths.exp_name` for isolation
3. Update hyperparameters as needed

---

## File Structure

```
uNOS/
├── model/
│   ├── __init__.py      # Exports: NosTokenizer, Nos, NosPredictor
│   ├── nos.py           # Core model classes (634 lines)
│   └── module.py        # Building blocks: TransformerBlock, BSQuantizer, etc.
├── models/              # Pretrained weights
│   ├── nos_tokenizer_2k/
│   ├── nos_tokenizer_base/
│   ├── nos_tokenizer_2k/
│   ├── nos_mini/
│   └── nos_base/
├── configs/             # YAML configs
│   ├── config.yaml     # Default config
│   └── ...
├── data/                # Training data (user-provided CSVs)
├── finetuned/           # Output: trained models
│   └── {exp_name}/
│       ├── tokenizer/best_model/
│       └── basemodel/best_model/
├── train_sequential.py  # Main training entry point
├── hpo_tuner.py        # HPO with Optuna
├── config_loader.py    # Config parsing
├── finetune_tokenizer.py
├── finetune_base_model.py
├── train.md            # Comprehensive guide (281KB)
├── CODE.md             # Auto-generated code docs
└── CLAUDE.md           # This file
```

---

## Dependencies

```
torch
pandas
numpy
pyyaml
huggingface_hub
optuna
plotly
kaleido
tqdm
```

Install: `pip install torch pandas numpy pyyaml huggingface_hub optuna plotly kaleido tqdm`

---

## User Preferences

- **Working Directory**: `c:\Users\kashy\OneDrive\Documents\uNOS`
- **Git User**: Mayank Kashyap
- **Platform**: Windows 11

---

## Notes

- The `train.md` file contains an extremely detailed 950+ line guide covering every aspect of the pipeline
- Use `python codebase.py` to regenerate CODE.md if files change
- HPO results persist in SQLite (when storage is configured) — can resume interrupted runs
- Batch inference supported via `predictor.predict_batch()` for parallel prediction
```

---
