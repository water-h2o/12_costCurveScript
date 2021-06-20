# ======================================================================
# == 0. PARAMETERS =====================================================
# ======================================================================


directory_path = './data'
out_dir = './output/GvR/'
n_tests = 10

# IN THE FUTURE, ALL THE CLASSIFIER PARAMETERS WILL GO HERE AS WELL



# ======================================================================
# == 1. IMPORTS ========================================================
# ======================================================================



# ------ 1.1. General Imports ------------------------------------------


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split #sklearn...
                                                     #.cross_validation 
                                                     # is deprecated

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.utils import shuffle

from os import listdir
from os.path import isfile, join


# ------ 1.2. Specialized Imports --------------------------------------


# - - - - -  1.2.1. kNN imports  - - - - - - - - - - - - - - - - - - - -

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# - - - - -  1.2.2. RBF SVM imports  - - - - - - - - - - - - - - - - - -

from sklearn.svm import SVC

# - - - - -  1.2.3. Gaussian process imports - - - - - - - - - - - - - -

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

# - - - - -  1.2.3. Gaussian process imports - - - - - - - - - - - - - -

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

# - - - - -  1.2.4. Random forest imports  - - - - - - - - - - - - - - - 

from sklearn.ensemble import RandomForestClassifier


# ======================================================================
# == 2. FUNCTION DEFINITIONS ===========================================
# ======================================================================


# ------ 2.1. High Level Functions -------------------------------------

# - - - - -  2.0.1 Set dataset size, equal label proportions - - - - - -

def configure_dataset(b2b_D, one_D, n_samples = -1):
    
    if n_samples == -1:
        
        n_samples = 2 * min(b2b_D.shape[0],one_D.shape[0])
    
    #b2b_D = b2b_D[0:int(n_samples / 2)]
    #one_D = one_D[0:int(n_samples / 2)]
    X_features_D = pd.concat([b2b_D, one_D], ignore_index = True)
    X_features = X_features_D.to_numpy()
    
    b2b_labels = np.ones( (b2b_D.shape[0],), dtype=int)# 1 == found 0vbb
    one_labels = np.zeros((one_D.shape[0],), dtype=int)# 0 == found 1e
    Y_labels = np.concatenate((b2b_labels,one_labels), axis=None)
    
    print('C_D() --> sample size, X_features = ' + str(len(X_features)))
    print('C_D() --> sample size, Y_labels = ' + str(len(Y_labels)))
    
    return (X_features, Y_labels)


# - - - - -  2.0.2 Perform ensemble test - - - - - - - - - - - - - - - -

def ROC_test(X_features, Y_labels, n_tests, c_thr):
    
    fp_s = np.empty([n_tests,1])
    tp_s = np.empty([n_tests,1])
    
    print('E_T() --> sample size, X_features = ' + str(len(X_features)))
    print('E_T() --> sample size, Y_labels = ' + str(len(Y_labels)))
    
    for i in range(0, n_tests):
        
        print('i = ' + str(i))
        
        x_train, x_test, \
        y_train, y_test  = train_test_split(X_features,
                                            Y_labels,
                                            test_size = 0.2)
        
        print('generated test and training sets')
        
#        fp_s[i], tp_s[i] = run_G_P(x_train,x_test,y_train,y_test,c_thr)
#        fp_s[i], tp_s[i] = run_R_F(x_train,x_test,y_train,y_test,c_thr)
        fp_s[i], tp_s[i] = run_GvR(x_train,x_test,y_train,y_test,c_thr)

        print('fp = ' + str(fp_s[i]) + ' , tp = ' + str(tp_s[i]))
    
    return (fp_s, tp_s)

# ------ 2.2. Mid Level Functions --------------------------------------


# - - - - -  2.1.1. Gaussian Process Classification and Plotting - - - -

def run_G_P(x_train, x_test, y_train, y_test, c_thr):
    
    # procedure

    gpc = GaussianProcessClassifier(1.0 * RBF(1.0))
    gpc.fit(x_train, y_train)
    pred_prob = gpc.predict_proba(x_test)
    label_pred = (pred_prob [:,1] >= c_thr).astype('int')
    
    # generating output
    
    np.set_printoptions(precision = 5)
    return give_fp_tp(y_test, label_pred, normalize = True)



# - - - - -  2.1.2. Random Forest Classification and Plotting  - - - - -

def run_R_F(x_train, x_test, y_train, y_test, c_thr):
    
    # procedure
    
    rand_frst = RandomForestClassifier(n_estimators = 100)
    rand_frst.fit(x_train, y_train)
    pred_prob = rand_frst.predict_proba(x_test)
    label_pred = (pred_prob [:,1] >= c_thr).astype('int')
    
    # generating output
    
    np.set_printoptions(precision = 5)
    return give_fp_tp(y_test, label_pred, normalize = True)
    


# - - - - -  2.1.3. gaus V. ranf Classification and Plotting  - - - - -

def run_GvR(x_train, x_test, y_train, y_test, c_thr):
    
    # procedure
    
    rand_frst = RandomForestClassifier(n_estimators = 100)
    rand_frst.fit(x_train, y_train)
    pred_prob_ranf = rand_frst.predict_proba(x_test)
    label_pred_ranf = (pred_prob_ranf [:,1] >= c_thr).astype('int')
    
    gpc = GaussianProcessClassifier(1.0 * RBF(1.0))
    gpc.fit(x_train, y_train)
    pred_prob_gaus = gpc.predict_proba(x_test)
    label_pred_gaus = (pred_prob_gaus [:,1] >= c_thr).astype('int')
    
    label_pred = label_pred_ranf * label_pred_gaus
    
    # generating output
    
    np.set_printoptions(precision = 5)
    return give_fp_tp(y_test, label_pred, normalize = True)


# ------ 2.3 Low Level Functions ---------------------------------------



# - - - - -  2.2.1. FP rate / TP rate  - - - - - - - - - - - - - - - - -


def give_fp_tp(y_true, y_pred, normalize = False):

    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    # Compute confusion matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Only use the labels that appear in the data
    
    
    if normalize:
        
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
        print("Normalized confusion matrix")
        
    else:
        
        print('Confusion matrix, without normalization')

    cm_fp = -9999
    cm_tp = -9999

    if len(cm) == 2:

        cm_fp = cm[0,1]
        cm_tp = cm[1,1]
    
    return (cm_fp, cm_tp)


# ======================================================================
# == 3. RUNNING SCRIPT =================================================
# ======================================================================


# ------ 3.1. Importing Files from Data Folder -------------------------



files = [ f for f in listdir(directory_path)
            if isfile( join(directory_path, f) ) ]

files_b2b = sorted( [f for f in files if f[4:6] == '0v'] )
files_one = sorted( [f for f in files if f[4:6] == '1e'] )

files_both_list = [files_b2b, files_one]

files_both_DF = pd.DataFrame(files_both_list)
files_both_DF = files_both_DF.transpose()
files_both_DF = (directory_path + '/') + files_both_DF

print(files_both_DF)

# ------ 3.2. Performing All Classifications on All Imported Files -----



for i in range(files_both_DF.shape[0]) :
    
    print(' ')
    print('current files:')
    print(files_both_DF.at[i,0])
    print(files_both_DF.at[i,1])
    print(' ')
    
    b2b_D = pd.read_excel(files_both_DF.at[i,0])
    one_D = pd.read_excel(files_both_DF.at[i,1])
    
    
    # defining filename
    
    scenario = files_both_DF.at[i,0].split('_')[0].split('/')[-1] + '_'
    
    sans = "sans_"
    brem = ""
    npks = ""
    nplt = ""
    
    if len(files_both_DF.at[i,0].split('_')) >= 6 : # has categorization
    
        sans = ""
        brem = files_both_DF.at[i,0].split('_')[2] + '_'
        npks = files_both_DF.at[i,0].split('_')[3] + '_'
        nplt = files_both_DF.at[i,0].split('_')[4] + '_'
    
    
    dataStyle = files_both_DF.at[i,0].split('_')[-1].split('.')[0]+'_'
    #    "SFFS", "MDS", etc.
    
    outfile = scenario + sans + brem + npks + nplt + dataStyle
    
    # the tests themselves
    
    # so the idea now is to make tests like so:
    #     *  10 tests for each dataset size, sampling with replacement
    #     *  train size --> 80% , test size --> 20%
    #     0. 100
    #     1. 200
    #     2. 500
    #     3. 1000
    #     4. 2000
    #     5. 5000
    #     6. All data
    
    # - - - - - A. Prepare containers for output file  - - - - - - - - -
    
    print ('preparing containers')
    
    n_largest = min(b2b_D.shape[0],one_D.shape[0]) * 2
    
    print('max samples = ' + str(n_largest))
    
    idx_row = np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    
    fp_s = np.empty([4 + n_tests, len(idx_row)])
    tp_s = np.empty([4 + n_tests, len(idx_row)])
    
    # 1st row   --> n_samples
    # next rows --> data
    # empty space
    # mean
    # stdev
    
    fp_s[0,:] = idx_row
    tp_s[0,:] = idx_row
    
    fp_s[n_tests + 1,:] = -np.ones(len(idx_row))
    tp_s[n_tests + 1,:] = -np.ones(len(idx_row))
    
    # - - - - - 0. P(0vbb) =   0%  - - - - - - - - - - - - - - - - - - -
    
    print (outfile + ' --> threshold = 0%')
    
    fp_s[1:(n_tests+1),0] = np.ones([1, n_tests])
    tp_s[1:(n_tests+1),0] = np.ones([1, n_tests])
     
    # - - - - - 0. P(0vbb) =  10%  - - - - - - - - - - - - - - - - - - -
    
    print (outfile + ' --> threshold = ' + str(idx_row[1]*100) + '%')
    
    X_features, Y_labels = configure_dataset(b2b_D, one_D, n_largest)
    fp_s_1, tp_s_1 = ROC_test(X_features,Y_labels,n_tests,idx_row[1])
    
    fp_s[1:(n_tests+1),1] = np.transpose(fp_s_1)
    tp_s[1:(n_tests+1),1] = np.transpose(tp_s_1)
    
    # - - - - - 0. P(0vbb) =  20%  - - - - - - - - - - - - - - - - - - -
    
    print (outfile + ' --> threshold = ' + str(idx_row[2]*100) + '%')
    
    X_features, Y_labels = configure_dataset(b2b_D, one_D, n_largest)
    fp_s_2, tp_s_2 = ROC_test(X_features,Y_labels,n_tests,idx_row[2])
    
    fp_s[1:(n_tests+1),2] = np.transpose(fp_s_2)
    tp_s[1:(n_tests+1),2] = np.transpose(tp_s_2)
    
    # - - - - - 0. P(0vbb) =  30%  - - - - - - - - - - - - - - - - - - -
    
    print (outfile + ' --> threshold = ' + str(idx_row[3]*100) + '%')
    
    X_features, Y_labels = configure_dataset(b2b_D, one_D, n_largest)
    fp_s_3, tp_s_3 = ROC_test(X_features,Y_labels,n_tests,idx_row[3])
    
    fp_s[1:(n_tests+1),3] = np.transpose(fp_s_3)
    tp_s[1:(n_tests+1),3] = np.transpose(tp_s_3)
    
    # - - - - - 0. P(0vbb) =  40%  - - - - - - - - - - - - - - - - - - -
    
    print (outfile + ' --> threshold = ' + str(idx_row[4]*100) + '%')
    
    X_features, Y_labels = configure_dataset(b2b_D, one_D, n_largest)
    fp_s_4, tp_s_4 = ROC_test(X_features,Y_labels,n_tests,idx_row[4])
    
    fp_s[1:(n_tests+1),4] = np.transpose(fp_s_4)
    tp_s[1:(n_tests+1),4] = np.transpose(tp_s_4)
    
    # - - - - - 0. P(0vbb) =  50%  - - - - - - - - - - - - - - - - - - -
    
    print (outfile + ' --> threshold = ' + str(idx_row[5]*100) + '%')
    
    X_features, Y_labels = configure_dataset(b2b_D, one_D, n_largest)
    fp_s_5, tp_s_5 = ROC_test(X_features,Y_labels,n_tests,idx_row[5])
    
    fp_s[1:(n_tests+1),5] = np.transpose(fp_s_5)
    tp_s[1:(n_tests+1),5] = np.transpose(tp_s_5)
    
    # - - - - - 0. P(0vbb) =  60%  - - - - - - - - - - - - - - - - - - -
    
    print (outfile + ' --> threshold = ' + str(idx_row[6]*100) + '%')
    
    X_features, Y_labels = configure_dataset(b2b_D, one_D, n_largest)
    fp_s_6, tp_s_6 = ROC_test(X_features,Y_labels,n_tests,idx_row[6])
    
    fp_s[1:(n_tests+1),6] = np.transpose(fp_s_6)
    tp_s[1:(n_tests+1),6] = np.transpose(tp_s_6)
    
    # - - - - - 0. P(0vbb) =  70%  - - - - - - - - - - - - - - - - - - -
    
    print (outfile + ' --> threshold = ' + str(idx_row[7]*100) + '%')
    
    X_features, Y_labels = configure_dataset(b2b_D, one_D, n_largest)
    fp_s_7, tp_s_7 = ROC_test(X_features,Y_labels,n_tests,idx_row[7])
    
    fp_s[1:(n_tests+1),7] = np.transpose(fp_s_7)
    tp_s[1:(n_tests+1),7] = np.transpose(tp_s_7)
    
    # - - - - - 0. P(0vbb) =  80%  - - - - - - - - - - - - - - - - - - -
    
    print (outfile + ' --> threshold = ' + str(idx_row[8]*100) + '%')
    
    X_features, Y_labels = configure_dataset(b2b_D, one_D, n_largest)
    fp_s_8, tp_s_8 = ROC_test(X_features,Y_labels,n_tests,idx_row[8])
    
    fp_s[1:(n_tests+1),8] = np.transpose(fp_s_8)
    tp_s[1:(n_tests+1),8] = np.transpose(tp_s_8)
    
    # - - - - - 0. P(0vbb) =  90%  - - - - - - - - - - - - - - - - - - -
    
    print (outfile + ' --> threshold = ' + str(idx_row[9]*100) + '%')
    
    X_features, Y_labels = configure_dataset(b2b_D, one_D, n_largest)
    fp_s_9, tp_s_9 = ROC_test(X_features,Y_labels,n_tests,idx_row[9])
    
    fp_s[1:(n_tests+1),9] = np.transpose(fp_s_9)
    tp_s[1:(n_tests+1),9] = np.transpose(tp_s_9)
    
    # - - - - - 0. P(0vbb) = 100%  - - - - - - - - - - - - - - - - - - -
    
    print (outfile + ' --> threshold = ' + str(idx_row[10]*100) + '%')
    
    X_features, Y_labels = configure_dataset(b2b_D, one_D, n_largest)
    fp_s_10, tp_s_10 = ROC_test(X_features,Y_labels,n_tests,idx_row[10])
    
    fp_s[1:(n_tests+1),10] = np.transpose(fp_s_10)
    tp_s[1:(n_tests+1),10] = np.transpose(tp_s_10)
    
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    #  - write labels of logfile  -  -  -  -  -  -  -  -  -  -  -  -  -
    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    
    for j in range(len(idx_row)):
        
        fp_s[n_tests + 2,j] = np.average(fp_s[1:(n_tests+1),j])
        fp_s[n_tests + 3,j] = np.std(    fp_s[1:(n_tests+1),j])
        
        tp_s[n_tests + 2,j] = np.average(tp_s[1:(n_tests+1),j])
        tp_s[n_tests + 3,j] = np.std(    tp_s[1:(n_tests+1),j])
    
    fp_s_D = pd.DataFrame(fp_s)
    tp_s_D = pd.DataFrame(tp_s)
    
    outaddr_fp = out_dir + outfile + 'FP.xlsx'
    outaddr_tp = out_dir + outfile + 'TP.xlsx'
    
    fp_s_D.to_excel(outaddr_fp, index = False)
    tp_s_D.to_excel(outaddr_tp, index = False)
    
print("script finished running.")

