To download and clean lightcurve files
1) Create directories "asassn_lcs" and "asassn_lcs_clean"
2) Right click each one and mark as excluded (If you don't do this, Pycharm will try to index all files)
3) Run LightCurveDownload.py (Change PER_CLASS to something more reasonable if testing)
4) Delete index.csv file from "asassn_lcs" folder
5) Run LightCurveClean.py

To create model
1) Download and clean lightcurves
2) Run LightCurveCNN.py

To use the model for prediction
1) Create model first
2) Change line 11 of LightCurvedDownloadUnclassified.py to desired number of lightcurves
3) Run LightCurveDownloadUnclassified.py
4) Run LightCurvePredict.py

To plot a lightcurve
1) Download lightcurves and clean if desired
2) Change filepath in line 32 of LightCurveGraph.py to desired file (Works with cleaned and uncleaned files)
3) Run LightCurveGraph.py
