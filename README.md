# Cost curve test implementation

`costCurveData_script.py` generates data for an ROC curve, from which the cost curve can then be constructed using the `costCurve.m` script, located in `./output/G_P`, `./output/GvR` and `./output/R_F`. Currently `costCurveData_script.py` is configured for the GvR algorithm. To configure it for GP or RF, set the correct path in line 7 and comment and uncomment the appropriate expressions in lines 118-120.

Data obtained from `costCurve.m` was then used for the full cost curve comparison, listed in `./comparativeTestImgs/b2b/comparison.m` and `./comparativeTestImgs/ISO/comparison.m`. The main purpose of this script was to generate a lower envelope among all the obtained results.