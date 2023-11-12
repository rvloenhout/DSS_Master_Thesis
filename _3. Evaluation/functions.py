from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix, recall_score, precision_score, f1_score, roc_curve
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#https://mikulskibartosz.name/how-to-reduce-memory-usage-in-pandas
def rm(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


#results = pd.DataFrame(columns=["Model", "AUROC", "Balanced Accuracy", "F1", "Recall", "Precision"])

def evaluation(y_test_input, y_pred_input, model_name, y_pred_proba_input, results):

    auroc = round(roc_auc_score(y_test_input, y_pred_proba_input),3)
    bal_acc = round(balanced_accuracy_score(y_test_input, y_pred_input),3)
    f1 = round(f1_score(y_test_input, y_pred_input),3)
    recall = round(recall_score(y_test_input, y_pred_input),3)
    precis = round(precision_score(y_test_input, y_pred_input),3)

    results.loc[len(results)+1] = [type(model_name).__name__, auroc, bal_acc, f1, recall, precis]

    print("AUROC Score: ", auroc)
    print("Balanced Accuracy Score: ", bal_acc)
    print("F1 Score: ", f1)
    print("Recall Score: ", recall)
    print("Precision Score: ", precis)
    cm = confusion_matrix(y_test_input, y_pred_input)

    fig, ax = plt.subplots(figsize=(4, 6))
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')
    
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()


def auroc_curve(y_test, y_pred_prob_logreg_tuned_imbalance, y_pred_prob_xgb_tuned_imbalance, y_pred_prob_tabnet_tuned_imbalance, y_pred_prob_rf_tuned_imbalanced):
    #Plot AUROC for LogReg
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob_logreg_tuned_imbalance)
    auc = round(roc_auc_score(y_test, y_pred_prob_logreg_tuned_imbalance), 3)
    plt.plot(fpr, tpr, linestyle='--', label="Logistic Regression, AUC="+str(auc))

    #Plot AUROC for XGBoost
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob_xgb_tuned_imbalance)
    auc = round(roc_auc_score(y_test, y_pred_prob_xgb_tuned_imbalance), 3)
    plt.plot(fpr, tpr, linestyle='--', label="Gradient Boosting, AUC="+str(auc))

    #Plot AUROC for TabNet
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob_tabnet_tuned_imbalance)
    auc = round(roc_auc_score(y_test, y_pred_prob_tabnet_tuned_imbalance), 3)
    plt.plot(fpr, tpr, linestyle='--', label="TabNet, AUC="+str(auc))

    #Plot AUROC for Random Forests
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob_rf_tuned_imbalanced)
    auc = round(roc_auc_score(y_test, y_pred_prob_rf_tuned_imbalanced), 3)
    plt.plot(fpr, tpr, linestyle='--', label="Random Forests, AUC="+str(auc))

    #Add baseline of 0.5 random guesser
    plt.plot([0, 1], [0, 1], color='gray')

    #Add legend
    plt.legend()

    #Add labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')


    plt.figure(figsize=(14, 12)) 
    plt.show()