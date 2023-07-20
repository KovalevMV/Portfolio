# Телеком 📞: исследование оттока клиентов

[ipynb](https://github.com/KovalevMV/Portfolio/blob/main/Telecom/telecom_customer_churn_for_git.ipynb)

## Описание проекта

Оператор связи поставил задачу научиться прогнозировать отток клиентов при условии выяснения, что пользователь планирует уйти, ему будут предложены промокоды и специальные условия.

## Навыки и инструменты

- **python**
- **pandas**
- **numpy**
- sklearn.model_selection import train_test_split
- catboost import CatBoostClassifier
- xgboost import XGBClassifier #
- lightgbm import LGBMClassifier #
- RandomizedSearchCV, GridSearchCV, cross_val_score, ShuffleSplit
- sklearn.metrics: roc_auc_score, auc, roc_curve, classification_report, confusion_matrix, accuracy_score, RocCurveDisplay, ConfusionMatrixDisplay
- sklearn.preprocessing: OneHotEncoder,  OrdinalEncoder,  StandardScaler

## 

## Общий вывод

1. Машинное обучение позволяет видеть метрики вероятности ухода клиента и наиболее выжные признаки, т.е. то на что стоит обратить внимание.  
2. Обученная модель (AUC-ROC test:  0.928) полностью справляется с поставленной заказчиком задачи (AUC-ROC >= 0.85)
