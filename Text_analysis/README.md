# Анализ текстов
[ipynb](https://github.com/KovalevMV/Portfolio/blob/main/Telecom/telecom_customer_churn_for_git.ipynb)


## Описание проекта

(NLP) Классификация с выделением токсичных комментариев на английском язык для дольнейшей модерации.



## Навыки и инструменты

- **python**
- **pandas**
- **numpy**
- nltk.stem.**WordNetLemmatizer**
- sklearn.ensemble.**RandomForestClassifier**
- nltk.tokenize 
- nltk.corpus 
- LogisticRegression
- TfidfVectorizer  
- train_test_split, RandomizedSearchCV
- DecisionTreeClassifier
- f1_score



## Вывод

Была проведена исследовательская работа по обработке текстов и обучению и выбору модели для определения токсичных комментариев.  Исходя из полученных метрик качества моделей, лучшая модель на RandomizedSearchCV - LightGBM c параметрами max_depth: 25, learning_rate: 0.3. Необходимые метрики достигнуты, модель LightGBM, обученная через RandomizedSearchCV, предсказывает с необходимой метрикой: F1 > 0.75.
