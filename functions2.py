import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import shap
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from yellowbrick.classifier import ClassPredictionError
from io import BytesIO
import base64
import joblib 
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve
import re

@st.cache_data
def show_data_to_upload():
    data_to_upload = pd.read_csv("data_test.csv")
    csv_file = data_to_upload.to_csv(index=False)
    return csv_file

def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    href = f'<a href="data:file/csv;base64,{b64}">Click here to download table</a>'
    return href

def get_image_download_link(buffered, filename, text):
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href =  f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href

def extract_weeks(s):
    m = re.search(r'(\d+)', s)
    if m:
        return int(m.group(0))
    else:
        return None

@st.cache_data
def preprocessing(data): 
    data.drop(columns=['Unnamed: 0', 'member_id', 'pymnt_plan', 'funded_amnt', 'funded_amnt_inv', 'emp_title', 'title', 'zip_code', 'desc', 'verification_status_joint', 'mths_since_last_record', 'mths_since_last_major_derog', 'application_type'], axis=1, inplace=True) 
    data['last_week_pay'] = data['last_week_pay'].apply(extract_weeks)
    data['home_ownership'] = ['OTHER'  if i not in ['OWN', 'RENT'] else i for i in data['home_ownership']]
    data['purpose'] = ['other' if i not in ['debt_consolidation', 'credit_card', 'home_improvement'] else i for i in data['purpose']]

    cat_cols = [col for col in data if data[col].dtype == 'object']
    num_cols = [col for col in data if data[col].dtype != 'object']

    data[num_cols] = data[num_cols].fillna(data[num_cols].median())
    data[cat_cols] = data[cat_cols].fillna("Unknown")
    data[cat_cols] = data[cat_cols].replace(" ", "Unknown")

    data['emp_length'] = data['emp_length'].replace({'9 years': 9, '< 1 year': 0, '2 years': 2, '10+ years': 10, '5 years': 5, '8 years': 8, '7 years': 7, '4 years': 4, 'Unknown': 999, '1 year': 1, '3 years': 3, '6 years': 6}).astype('int')


    cols_to_check = ['annual_inc', 'dti', 'inq_last_6mths', 'open_acc', 'total_acc','revol_bal', 'revol_util', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_week_pay', 'tot_cur_bal', 'total_rev_hi_lim', 'loan_status']
    Q1 = data[cols_to_check].quantile(0.25)
    Q3 = data[cols_to_check].quantile(0.75)
    IQR = Q3 - Q1
    upper_threshold = Q3 + 1.5 * IQR

    outliers = data[cols_to_check][((data[cols_to_check] > upper_threshold).any(axis=1))]
    outliers_to_drop = outliers[(outliers > upper_threshold).sum(axis=1) >=4]
    data.drop(outliers_to_drop.index, inplace = True)

    data = data[(data['annual_inc'] < 2000000) 
                            & (data['dti'] < 40)
                            & (data['delinq_2yrs'] < 20)
                            & (data['mths_since_last_delinq'] < 150)
                            & (data['open_acc'] < 60)
                            & (data['pub_rec'] < 20)
                            & (data['revol_bal'] < 1000000)
                            & (data['revol_util'] < 200)
                            & (data['total_acc'] < 125)
                            & (data['total_rec_late_fee'] < 150)
                            & (data['recoveries'] < 15000)
                            & (data['collection_recovery_fee'] < 2000)
                            & (data['collections_12_mths_ex_med'] < 5.0)
                            & (data['acc_now_delinq'] < 4)
                            & (data['tot_coll_amt'] < 100000)
                            & (data['tot_cur_bal'] < 3000000)
                            & (data['total_rev_hi_lim'] < 1000000)]
    return data 
    
@st.cache_data
def best_performances(X, y):
    target_names=['Non Defaulters', 'Defaulters']
    final_pipeline = joblib.load("lgbmfitt.joblib")
    #final_pipeline = joblib.load("final_pipeline_fit.joblib")
    pred = final_pipeline.predict(X)
    score = final_pipeline.predict_proba(X)[:, 1]
    len_s = np.zeros(len(y))
    len_s[score > 0.52] = 1.

    report = classification_report(y, len_s, target_names=target_names, output_dict=True)
    df_report = pd.DataFrame(report).transpose().round(2)
    acc_score = accuracy_score(y, len_s)
    df_report['accuracy'] = round(acc_score, 2)
    df_report['Best Model'] = 'LGBMClassifier'
    df_report = df_report.loc[target_names, ['Best Model', 'precision', 'recall', 'accuracy', 'f1-score']]
    df_report = df_report.rename_axis("Class").reset_index(drop=True)
    conf_matrix = confusion_matrix(y, len_s)
    return df_report, conf_matrix, target_names, final_pipeline, len_s, score

@st.cache_resource
def plots(conf_matrix, _final_pipeline, target_names, X, y, len_s, score):

    optimal_proba_cutoffs = {}

    fig1, ax1= plt.subplots(figsize=(25, 15))
    sns.heatmap(conf_matrix, annot=True, fmt='d', ax=ax1, annot_kws={"size": 30})
    ax1.set_yticklabels(labels=["Non Défaillants", "Défaillants"], fontsize=30)
    ax1.set_xticklabels(labels=["Non Défaillants", "Défaillants"], fontsize=30)
    ax1.collections[0].colorbar.ax.tick_params(labelsize=30)

    fig2, ax2 = plt.subplots(figsize=(25, 15))
    visualizer = ClassPredictionError(_final_pipeline.named_steps['model'], classes=target_names, normalize=True, ax=ax2)
    visualizer.score(X, y)
    visualizer.finalize()
    visualizer.ax.set_xticklabels(ax2.get_xticklabels(), fontsize=30)
    visualizer.ax.set_title('Class Prediction Error for LGBMClassifier', fontsize=30)
    visualizer.ax.legend(loc="right", fontsize=30)
    visualizer.ax.set_xlabel(ax2.get_xlabel(), fontsize=30)
    visualizer.ax.set_ylabel(ax2.get_ylabel(), fontsize=30)
    plt.tight_layout()

    fig3, ax3 = plt.subplots(figsize=(25, 15))
    roc_auc = roc_auc_score(y, len_s)
    false_positive_rate, true_positive_rate, threshold = roc_curve(y, len_s)
    ax3.plot(false_positive_rate, true_positive_rate, color='b', label='ROC curve (area = %0.3f)' % roc_auc)
    ax3.plot([0, 1], [0, 1], color='r', linestyle='--')
    ax3.set_xlim([0.0, 1.0])
    ax3.xaxis.set_tick_params(labelsize=30)
    ax3.set_ylim([0.0, 1.0])
    ax3.yaxis.set_tick_params(labelsize=30)
    ax3.set_xlabel('False Positive Rate', fontsize=30)
    ax3.set_ylabel('True Positive Rate', fontsize=30)
    ax3.set_title('Receiver Operating Characteristic (ROC) for LGBMClassifier', fontsize=30)
    ax3.legend(loc='lower right', fontsize=30)

    fig4, ax4 = plt.subplots(figsize=(25, 15))
    precision, recall, threshold = precision_recall_curve(y, score)
    tst_prt = pd.DataFrame({
        "threshold": threshold,
        "recall": recall[1:],
        "precision": precision[1:]
    })
    tst_prt_melted = pd.melt(tst_prt, id_vars=["threshold"], value_vars=[   
        "recall", "precision"])
    sns.lineplot(x="threshold", y="value", hue="variable", data=tst_prt_melted)
    optimal_proba_cutoff = sorted(list(zip(np.abs(precision - recall), threshold)), key=lambda i: i[0], reverse=False)[0][1]
    optimal_proba_cutoffs['LGBMClassifier'] = optimal_proba_cutoff
    title = '(Threshold: {:.2f})'.format(optimal_proba_cutoff)
    ax4.legend(title='variable', fontsize=30)
    ax4.set_xlabel(title, fontsize=30)
    ax4.set_ylabel('value', fontsize=30)
    ax4.set_title('Precision Recall Curve', fontsize=30)

    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    return fig1, fig2, fig3, fig4, optimal_proba_cutoffs

@st.cache_data
def shap_plots(X, y, data):
    estimator = joblib.load("LGBM_train_fit.joblib")
    np.random.seed(42)
    sample_indices = np.random.choice(data.index, size = 5000, replace = False)
    X_sample = X.loc[sample_indices].reset_index(drop=True)
    y_sample = y.loc[sample_indices].reset_index(drop=True)
    data_sample = data.loc[sample_indices].reset_index(drop=True)
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X_sample)

    num_features = X_sample.shape[1]
    fig, ax = plt.subplots(5, 2, figsize=(30, 25))
    ax = ax.flatten()
    for i, col in enumerate(X_sample.columns):
        shap.dependence_plot(i, shap_values[1], X_sample, show=False, ax=ax[i])
        plt.subplots_adjust(hspace=0.4)
    return estimator, explainer, shap_values, X_sample, y_sample, data_sample, fig