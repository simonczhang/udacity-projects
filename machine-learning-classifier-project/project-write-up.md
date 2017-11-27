# Project Free-Response​ ​Questions

##### 1. Summarize​ ​for​ ​us​ ​the​ ​goal​ ​of​ ​this​ ​project​ ​and​ ​how​ ​machine learning​ ​is​ ​useful​ ​in​ ​trying​ ​to​ ​accomplish​ ​it.​ ​As​ ​part​ ​of​ ​your answer,​ ​give​ ​some​ ​background​ ​on​ ​the​ ​dataset​ ​and​ ​how​ ​it​ ​can​ ​be​ ​used​ ​to answer​ ​the​ ​project​ ​question.​ ​Were​ ​there​ ​any​ ​outliers​ ​in​ ​the​ ​data​ ​when you​ ​got​ ​it,​ ​and​ ​how​ ​did​ ​you​ ​handle​ ​those?​ ​​

The​ ​goal​ ​of​ ​this​ ​project​ ​is​ ​to​ ​try​ ​and​ ​find​ ​out​ ​who​ ​are​ ​the POI’s​ ​(persons​ ​of​ ​interest)​ ​at
Enron​ ​who​ ​were​ ​involved​ ​in​ ​the​ ​scandal.​ ​By​ ​using​ ​a​ ​supervised​ ​machine learning​ ​algorithm,​ ​I​ ​can​ ​build​ ​a​ ​model​ ​to​ ​help​ ​predict​ ​these​ ​POI’s.

The​ ​Enron​ ​dataset​ ​contains​ ​146​ ​data​ ​points​ ​of​ ​individuals​ ​who​ ​worked at​ ​Enron.​ ​There​ ​are​ ​a​ ​number​ ​of​ ​financial​ ​and​ ​email​ ​feature​ ​data​ ​that
￼
￼goes​ ​along​ ​with​ ​each​ ​data​ ​point.​ ​

Below​ ​is​ ​a​ ​quick​ ​breakdown​ ​of​ ​the dataset:
* Total​ ​number​ ​of​ ​data​ ​points:​ ​146 Number​ ​of​ ​Poi’s:​ ​18
* Number​ ​of​ ​Non-Poi’s:​ ​128
* Number​ ​of​ ​Features​ ​Used:​ ​20​ ​(all​ ​except​ ​email_address)
* Total​ ​missing​ ​values​ ​(NaN)​ ​in​ ​dataset:
```
{'salary':​ ​51,​ ​
 'to_messages':​ ​60,​ ​
 'deferral_payments':​ ​107,
 'total_payments':​ ​21,​ ​
 'loan_advances':​ ​142,​ ​
 'bonus':​ ​64,
 'restricted_stock_deferred':​ ​128,​ ​
 'total_stock_value':​ ​20,
 'shared_receipt_with_poi':​ ​60,​ ​
 'long_term_incentive':​ ​80,
 'exercised_stock_options':​ ​44,​ ​
 'from_messages':​ ​60,​ ​
 'other':​ ​53,
 'from_poi_to_this_person':​ ​60,​ ​
 'from_this_person_to_poi':​ ​60,
 'deferred_income':​ ​97,​ ​
 'expenses':​ ​51,​ ​
 'restricted_stock':​ ​36,
 'director_fees':​ ​129}
```
After​ ​exploring​ ​the​ ​data​ ​a​ ​bit,​ ​I​ ​found​ ​that​ ​there​ ​was​ ​a​ ​data​ ​point included​ ​with​ ​key​ ​name​ ​```‘TOTAL’```​ ​which​ ​turned​ ​out​ ​to​ ​be​ ​an​ ​outlier​ ​with ```‘salary’​``` ​value​ ​much​ ​higher​ ​than​ ​the​ ​rest​ ​of​ ​the​ ​data​ ​points.​ ​This​ ​is most​ ​likely​ ​an​ ​error​ ​in​ ​making​ ​the​ ​dataset​ ​as​ ​I​ ​don’t​ ​believe​ ​there’s a​ ​person​ ​named​ ​```‘TOTAL’```​ ​that​ ​worked​ ​at​ ​Enron​ ​with​ ​no​ ​email​ ​address either.​ ​I​ ​decided​ ​to​ ​get​ ​rid​ ​of​ ​this​ ​data​ ​point.

###### 2. What​ ​features​ ​did​ ​you​ ​end​ ​up​ ​using​ ​in​ ​your​ ​POI​ ​identifier,​ ​and what​ ​selection​ ​process​ ​did​ ​you​ ​use​ ​to​ ​pick​ ​them?​ ​Did​ ​you​ ​have​ ​to​ ​do any​ ​scaling?​ ​Why​ ​or​ ​why​ ​not?​ ​As​ ​part​ ​of​ ​the​ ​assignment,​ ​you​ ​should attempt​ ​to​ ​engineer​ ​your​ ​own​ ​feature​ ​that​ ​does​ ​not​ ​come​ ​ready-made​ ​in the​ ​dataset​ ​--​ ​explain​ ​what​ ​feature​ ​you​ ​tried​ ​to​ ​make,​ ​and​ ​the rationale​ ​behind​ ​it.​ ​

I​ ​ended​ ​up​ ​using​ ​all​ ​of​ ​the​ ​features​ ​that​ ​came​ ​with​ ​the​ ​dataset except​ ​```'email_address'```.​ ​I​ ​knew​ ​that​ ​I​ ​was​ ​going​ ​to​ ​use​ ​PCA​ ​so​ ​I​ ​wanted to​ ​include​ ​all​ ​the​ ​features.​ ​The​ ​reason​ ​I​ ​chose​ ​to​ ​use​ ​PCA​ ​is​ ​because with​ ​20​ ​features​ ​in​ ​the​ ​dataset,​ ​PCA​ ​would​ ​allow​ ​me​ ​to​ ​automatically
￼compress​ ​the​ ​features​ ​into​ ​a​ ​smaller​ ​number​ ​of​ ​principal​ ​components that​ ​are​ ​the​ ​most​ ​useful​ ​when​ ​looking​ ​for​ ​POI’s.​

​I​ ​also​ ​used MinMaxScaler​ ​before​ ​PCA​ ​in​ ​my​ ​pipeline.​ ​I​ ​decided​ ​to​ ​scale​ ​because there​ ​were​ ​some​ ​features​ ​that​ ​had​ ​values​ ​around​ ​1​ ​or​ ​2​ ​like ```‘from_poi_to_this_person’​``` ​while​ ​other​ ​features​ ​over​ ​5,000,000​ ​like ```‘salary’​``` ​and​ ​```‘bonus’```.​

​I​ ​created​ ​2​ ​new​ ​features​ ​called ```'fraction_from_poi'​``` ​and​ ​```'fraction_to_poi'​``` ​which​ ​is​ ​the​ ​proportion​ ​of emails​ ​to​ ​and​ ​from​ ​poi​ ​divided​ ​by​ ​all​ ​emails​ ​to​ ​and​ ​from​ ​a​ ​given person,​ ​respectively.​ ​I​ ​thought​ ​this​ ​might​ ​be​ ​useful​ ​to​ ​use​ ​these​ ​new features​ ​instead​ ​of​ ​having​ ​the​ ​absolute​ ​number​ ​of​ ​emails​ ​to​ ​poi​ ​and from​ ​poi​ ​since​ ​you​ ​can​ ​see​ ​the​ ​relative​ ​frequency​ ​of​ ​emails​ ​by​ ​using the​ ​fraction.​ ​I​ ​tested​ ​the​ ​new​ ​features​ ​in​ ​tester.py​ ​and​ ​it​ ​turns​ ​out that​ ​the​ ​results​ ​were​ ​worse​ ​across​ ​the​ ​board​ ​for​ ​recall,​ ​precision, and​ ​f1,​ ​compared​ ​to​ ​the​ ​original​ ​dataset​ ​without​ ​the​ ​new​ ​features​ ​so I​ ​decided​ ​to​ ​keep​ ​those​ ​features​ ​out​ ​of​ ​my​ ​final​ ​algorithm.
```
#PCA,​ ​GaussianNB​ ​results:
Pipeline(memory=None, ​​​​​​​​​​steps=[('scaler',​​MinMaxScaler(copy=True,​​feature_range=(0, 1))),​ ​('pca',​ ​PCA(copy=True,​ ​iterated_power='auto',​ ​n_components=9, random_state=None,
​ ​​ ​svd_solver='auto',​ ​tol=0.0,​ ​whiten=False)),​ ​('clf', GaussianNB(priors=None))])
​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​Accuracy:​ ​0.82140​ ​​ ​​ ​​ ​​ ​​ ​​ ​Precision:​ ​0.35584​ ​​ ​​ ​​ ​​ ​​ ​Recall: 0.41900​ ​F1:​ ​0.38485​ ​​ ​​ ​​ ​​ ​F2:​ ​0.40464
​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​Total​ ​predictions:​ ​15000​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​True​ ​positives:​ ​​ ​838​ ​​ ​​ ​​ ​False positives:​ ​1517​ ​​ ​​ ​False​ ​negatives:​ ​1162​ ​​ ​​ ​True​ ​negatives:​ ​11483

#PCA,​ ​GaussianNB​ ​with​ ​'fraction_from_poi'​ ​and​ ​'fraction_to_poi' features results:
Pipeline(memory=None,
​ ​​ ​​ ​​ ​​ ​steps=[('scaler',​ ​MinMaxScaler(copy=True,​ ​feature_range=(0, 1))),​ ​('pca',​ ​PCA(copy=True,​ ​iterated_power='auto',​ ​n_components=9, random_state=None,
​ ​​ ​svd_solver='auto',​ ​tol=0.0,​ ​whiten=False)),​ ​('clf', GaussianNB(priors=None))])
​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​Accuracy:​ ​0.82013​ ​​ ​​ ​​ ​​ ​​ ​​ ​Precision:​ ​0.32374​ ​​ ​​ ​​ ​​ ​​ ​Recall: 0.32050​ ​F1:​ ​0.32211​ ​​ ​​ ​​ ​​ ​F2:​ ​0.32114
​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​Total​ ​predictions:​ ​15000​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​True​ ​positives:​ ​​ ​641​ ​​ ​​ ​​ ​False positives:​ ​1339​ ​​ ​​ ​False​ ​negatives:​ ​1359​ ​​ ​​ ​True​ ​negatives:​ ​11661
```
￼
##### 3. What​ ​algorithm​ ​did​ ​you​ ​end​ ​up​ ​using?​ ​What​ ​other​ ​one(s)​ ​did​ ​you try?​ ​How​ ​did​ ​model​ ​performance​ ​differ​ ​between​ ​algorithms?​

I​ ​tried​ ​SVC​ ​with​ ​rbf​ ​kernel,​ ​Adaboost,​ ​Random​ ​Forest,​ ​kNN,​ ​and GaussianNB,​ ​all​ ​with​ ​base​ ​parameters,​ ​and​ ​also​ ​with​ ​PCA​ ​in​ ​a​ ​pipeline initially​ ​to​ ​see​ ​which​ ​one​ ​would​ ​best​ ​fit​ ​this​ ​dataset.​ ​I​ ​found​ ​that GaussianNB​ ​with​ ​PCA​ ​pipeline​ ​and​ ​Adaboost​ ​with​ ​base​ ​parameters performed​ ​the​ ​best​ ​so​ ​I​ ​decided​ ​to​ ​fine​ ​tune​ ​the​ ​two​ ​and​ ​see​ ​which one​ ​would​ ​give​ ​me​ ​the​ ​optimal​ ​results.​ ​Since​ ​I​ ​used​ ​PCA​ ​with GaussianNB​ ​I​ ​used​ ​all​ ​the​ ​features​ ​in​ ​my​ ​features​ ​list​ ​to​ ​create​ ​the pipeline.​ ​For​ ​the​ ​Adaboost​ ​pipeline,​ ​I​ ​decided​ ​to​ ​do​ ​some​ ​feature selection​ ​with​ ​SelectKBest​ ​since​ ​I​ ​wasn’t​ ​using​ ​PCA​ ​in​ ​that​ ​pipeline. I​ ​threw​ ​both​ ​pipelines​ ​into​ ​GridSearchCV​ ​to​ ​find​ ​the​ ​optimal combination​ ​of​ ​parameters​ ​to​ ​use​ ​for​ ​both​ ​algorithms.​ ​However,​ ​after tuning,​ ​GaussianNB​ ​was​ ​the​ ​one​ ​I​ ​decided​ ​to​ ​use​ ​in​ ​the​ ​end.
```
#Tester.py​ ​results

#BASE​ ​PARAMETERS​ ​TEST​ ​ONLY:

SVC:
​ ​​ ​​ ​​ ​​0​ ​True​ ​positives​
Adaboost:
​ ​​ ​​ ​​ ​Precision:​ ​0.39974 ​ ​​ ​​ ​​ ​Recall:​ ​0.30200
​ ​​ ​​ ​​ ​F1:​ ​0.34406
​ ​​ ​​ ​​ ​Accuracy:​ ​0.84647
NB:
​ ​​ ​​ ​​ ​Precision:​ ​0.23578 ​ ​​ ​​ ​​ ​Recall:​ ​0.40000
​ ​​ ​​ ​​ ​F1:​ ​0.29668
​ ​​ ​​ ​​ ​Accuracy:​ ​0.74713
kNN:
​ ​​ ​​ ​​ ​Precision:​ ​0.65278 ​ ​​ ​​ ​​ ​Recall:​ ​0.21150
​ ​​ ​​ ​​ ​F1:​ ​0.31949
​ ​​ ​​ ​​ ​Accuracy:​ ​0.87987
RandomForest:
​ ​​ ​​ ​​ ​Precision:​ ​0.34720
￼​ ​​ ​​ ​​ ​Recall:​ ​0.10850
​ ​​ ​​ ​​ ​F1:​ ​0.16533
​ ​​ ​​ ​​ ​Accuracy:​ ​0.85393

#BASE​ ​PARAMETERS​ ​TEST​ ​W/PCA(n_estimators=10)​ ​PIPELINE:

 SVC:
​ ​​ ​​ ​​ ​​​0​ ​True​ ​positives​
Adaboost:
​ ​​ ​​ ​​ ​Precision:​ ​0.19007 ​ ​​ ​​ ​​ ​Recall:​ ​0.13400
​ ​​ ​​ ​​ ​F1:​ ​0.15718
​ ​​ ​​ ​​ ​Accuracy:​ ​0.80840
NB: ​​​​​​​​Precision:​​0.42621 ​ ​​ ​​ ​​ ​Recall:​ ​0.30900
​ ​​ ​​ ​​ ​F1:​ ​0.35826
​ ​​ ​​ ​​ ​Accuracy:​ ​0.85240
kNN:
​ ​​ ​​ ​​ ​Precision:​ ​0.67626 ​ ​​ ​​ ​​ ​Recall:​ ​0.23500
​ ​​ ​​ ​​ ​F1:​ ​0.34879
​ ​​ ​​ ​​ ​Accuracy:​ ​0.88300
RandomForest:
​ ​​ ​​ ​​ ​Precision:​ ​0.35325 ​ ​​ ​​ ​​ ​Recall:​ ​0.13600
​ ​​ ​​ ​​ ​F1:​ ​0.19639
​ ​​ ​​ ​​ ​Accuracy:​ ​0.85160
```
##### 4. What​ ​does​ ​it​ ​mean​ ​to​ ​tune​ ​the​ ​parameters​ ​of​ ​an​ ​algorithm,​ ​and what​ ​can​ ​happen​ ​if​ ​you​ ​don’t​ ​do​ ​this​ ​well?​ ​​ ​How​ ​did​ ​you​ ​tune​ ​the parameters​ ​of​ ​your​ ​particular​ ​algorithm?​ ​What​ ​parameters​ ​did​ ​you tune?​
Tuning​ ​the​ ​parameters​ ​of​ ​an​ ​algorithm​ ​means​ ​to​ ​try​ ​and​ ​find​ ​the optimal​ ​parameters​ ​for​ ​an​ ​algorithm​ ​that​ ​gives​ ​you​ ​the​ ​best​ ​results you’re​ ​looking​ ​for​ ​whether​ ​that​ ​be​ ​accuracy,​ ​f1,​ ​or​ ​another evaluation​ ​metric.​ ​If​ ​the​ ​parameters​ ​aren’t​ ​tuned​ ​well,​ ​the​ ​algorithm could​ ​have​ ​high​ ​variance​ ​or​ ​high​ ​bias​ ​which​ ​means​ ​the​ ​algorithm​ ​could be​ ​over​ ​or​ ​underfitting​ ​to​ ​the​ ​dataset.​

​I​ ​used​ ​GridSearchCV​ ​to​ ​tune the​ ​most​ ​relevant​ ​parameters​ ​for​ ​each​ ​algorithm​ ​in​ ​my​ ​pipeline​ ​which allows​ ​me​ ​to​ ​input​ ​a​ ​matrix​ ​of​ ​different​ ​parameter​ ​tunes​ ​and​ ​output the​ ​optimal​ ​tuning​ ​automatically.​ ​For​ ​GaussianNB,​ ​I​ ​only​ ​tuned​ ​PCA’s n_components​ ​parameter​ ​since​ ​GaussianNB​ ​doesn’t​ ​need​ ​to​ ​be​ ​tuned.​ ​For Adaboost,​ ​I​ ​tuned​ ​the​ ​SelectKBest’s​ ​k​ ​parameter​ ​as​ ​well​ ​as​ ​Adaboost’s n_estimators​ ​parameter.
Below​ ​are​ ​the​ ​parameters​ ​I​ ​tuned​ ​for​ ​each​ ​algorithm:
```
#GaussianNB​ ​parameter​ ​grid​ ​with​ ​PCA:
nb_param_grid​ ​=​ ​{
​ ​​  ​​ ​'pca__n_components':​ ​(7,​ ​9,​ ​10,​ ​15),}

#Adaboost​ ​parameter​ ​grid​ ​with​ ​SelectKBest:
ada_param_grid​ ​=​ ​{
​ ​​ ​​ ​​ ​'feature_selection__k':​ ​(5,​ ​10,​ ​15),
​  ​​ ​​ ​'clf__n_estimators':​ ​(25,​ ​50​ ​,​ ​75,​ ​100),}
```
Oddly​ ​enough,​ ​GaussianNB​ ​performed​ ​better​ ​in​ ​the​ ​tester​ ​function​ ​once I​ ​tuned​ ​the​ ​parameters;​ ​however,​ ​Adaboost​ ​performed​ ​significantly worse​ ​than​ ​the​ ​base​ ​parameters​ ​no​ ​matter​ ​what​ ​parameter​ ​tuning​ ​I used.​ ​I​ ​used​ ​the​ ​default​ ​KFolds​ ​cross​ ​evaluation​ ​as​ ​well​ ​as StratifiedShuffleSplit​ ​in​ ​the​ ​cv​ ​parameter​ ​for​ ​GridSearchCV​ ​and​ ​with both​ ​evaluation​ ​methods,​ ​Adaboost​ ​did​ ​worse​ ​when​ ​tuned.​ ​In​ ​the​ ​end, GaussianNB​ ​tuned​ ​performed​ ​better​ ​than​ ​Adaboost​ ​tuned​ ​and​ ​untuned.
```
#Untuned:
Adaboost:
​ ​​ ​​ ​​ ​Precision:​ ​0.39974 ​ ​​ ​​ ​​ ​Recall:​ ​0.30200
​ ​​ ​​ ​​ ​F1:​ ​0.34406
​ ​​ ​​ ​​ ​Accuracy:​ ​0.84647
￼NB​ ​with​ ​PCA:
​ ​​ ​​ ​​ ​Precision:​ ​0.42621 ​ ​​ ​​ ​​ ​Recall:​ ​0.30900
​ ​​ ​​ ​​ ​F1:​ ​0.35826
​ ​​ ​​ ​​ ​Accuracy:​ ​0.85240

#Tuned:
SelectKBest,​ ​Adaboost​ ​results:
Pipeline(memory=None,
​ ​​ ​​ ​​ ​​ ​steps=[('feature_selection',​ ​SelectKBest(k=10, score_func=<function​ ​f_classif​ ​at​ ​0x1a104e0a28>)),​ ​('clf', AdaBoostClassifier(algorithm='SAMME.R',​ ​base_estimator=None,
​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​learning_rate=1.0,​ ​n_estimators=25,​ ​random_state=None))])
​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​Accuracy:​ ​0.83913​ ​​ ​​ ​​ ​​ ​​ ​​ ​Precision:​ ​0.35365​ ​​ ​​ ​​ ​​ ​​ ​Recall: 0.24950​ ​F1:​ ​0.29258​ ​​ ​​ ​​ ​​ ​F2:​ ​0.26512
​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​Total​ ​predictions:​ ​15000​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​True​ ​positives:​ ​​ ​499​ ​​ ​​ ​​ ​False positives:​ ​​ ​912​ ​​ ​​ ​False​ ​negatives:​ ​1501​ ​​ ​​ ​True​ ​negatives:​ ​12088

PCA,​ ​GaussianNB​ ​results:
Pipeline(memory=None,
​ ​​ ​​ ​​ ​​ ​steps=[('scaler',​ ​MinMaxScaler(copy=True,​ ​feature_range=(0, 1))),​ ​('pca',​ ​PCA(copy=True,​ ​iterated_power='auto',​ ​n_components=9, random_state=None,
​ ​​ ​svd_solver='auto',​ ​tol=0.0,​ ​whiten=False)),​ ​('clf', GaussianNB(priors=None))])
​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​Accuracy:​ ​0.82140​ ​​ ​​ ​​ ​​ ​​ ​​ ​Precision:​ ​0.35584​ ​​ ​​ ​​ ​​ ​​ ​Recall: 0.41900​ ​F1:​ ​0.38485​ ​​ ​​ ​​ ​​ ​F2:​ ​0.40464
​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​Total​ ​predictions:​ ​15000​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​True​ ​positives:​ ​​ ​838​ ​​ ​​ ​​ ​False positives:​ ​1517​ ​​ ​​ ​False​ ​negatives:​ ​1162​ ​​ ​​ ​True​ ​negatives:​ ​11483
```
##### 5. What​ ​is​ ​validation,​ ​and​ ​what’s​ ​a​ ​classic​ ​mistake​ ​you​ ​can​ ​make if​ ​you​ ​do​ ​it​ ​wrong?​ ​How​ ​did​ ​you​ ​validate​ ​your​ ​analysis?​ ​
Validation​ ​is​ ​evaluating​ ​the​ ​performance​ ​of​ ​your​ ​algorithm​ ​by splitting​ ​the​ ​data​ ​into​ ​different​ ​subsets​ ​and​ ​training​ ​your​ ​algorithm on​ ​a​ ​training​ ​set​ ​and​ ​testing​ ​your​ ​algorithm​ ​on​ ​a​ ​different​ ​testing set.​ ​It​ ​is​ ​important​ ​to​ ​not​ ​use​ ​any​ ​data​ ​from​ ​your​ ​testing​ ​set​ ​in your​ ​training​ ​set​ ​when​ ​training​ ​your​ ​algorithm​ ​because​ ​it​ ​could​ ​cause ‘bleeding’​ ​and​ ​produce​ ​inaccurate​ ​results​ ​in​ ​your​ ​evaluation​ ​metrics.

￼When​ ​I​ ​first​ ​tried​ ​out​ ​the​ ​initial​ ​5​ ​algorithms​ ​I​ ​used train_test_split(features,​ ​labels,​ ​test_size=0.3,​ ​random_state=42)​ ​to train​ ​and​ ​test​ ​my​ ​algorithms.​ ​Once​ ​I​ ​found​ ​the​ ​top​ ​two​ ​algorithms​ ​I put​ ​them​ ​inside​ ​GridSearchCV​ ​which​ ​automatically​ ​uses​ ​KFolds​ ​cross validation​ ​to​ ​test​ ​the​ ​algorithms.​ ​
```
#My​ ​original​ ​PCA(n_estimators=10), GuassianNB​ ​pipeline​ ​performance​ ​results​ ​​using​ ​test_train_split:
NB:
​ ​​ ​​ ​​ ​Precision:​ ​0.42621
​ ​​ ​​ ​​ ​Recall:​ ​0.30900
​ ​​ ​​ ​​ ​F1:​ ​0.35826
​ ​​ ​​ ​​ ​Accuracy:​ ​0.85240

#My​ ​final​ ​results​ ​after​ ​tuning:
PCA,​ ​GaussianNB​ ​results:
Pipeline(memory=None,
​ ​​ ​​ ​​ ​​ ​steps=[('scaler',​ ​MinMaxScaler(copy=True,​ ​feature_range=(0, 1))),​ ​('pca',​ ​PCA(copy=True,​ ​iterated_power='auto',​ ​n_components=9, random_state=None,
​ ​​ ​svd_solver='auto',​ ​tol=0.0,​ ​whiten=False)),​ ​('clf', GaussianNB(priors=None))])
​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​Accuracy:​ ​0.82140​ ​​ ​​ ​​ ​​ ​​ ​​ ​Precision:​ ​0.35584​ ​​ ​​ ​​ ​​ ​​ ​Recall: 0.41900​ ​F1:​ ​0.38485​ ​​ ​​ ​​ ​​ ​F2:​ ​0.40464
​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​Total​ ​predictions:​ ​15000​ ​​ ​​ ​​ ​​ ​​ ​​ ​​ ​True​ ​positives:​ ​​ ​838​ ​​ ​​ ​​ ​False positives:​ ​1517​ ​​ ​​ ​False​ ​negatives:​ ​1162​ ​​ ​​ ​True​ ​negatives:​ ​11483
```
##### 6. Give​ ​at​ ​least​ ​2​ ​evaluation​ ​metrics​ ​and​ ​your​ ​average​ ​performance for​ ​each​ ​of​ ​them.​ ​​ ​Explain​ ​an​ ​interpretation​ ​of​ ​your​ ​metrics​ ​that says​ ​something​ ​human-understandable​ ​about​ ​your​ ​algorithm’s performance.​
I​ ​performed​ ​multi-metric​ ​evaluation​ ​using​ ​the​ ​scoring​ ​parameter in​ ​GridSearchCV:
```
#scoring​ ​list
scoring​ ​=​ ​['f1',​ ​'recall',​ ​'precision']
```
I​ ​found​ ​the​ ​optimal​ ​the​ ​three​ ​optimal​ ​parameter​ ​tuning​ ​combinations for​ ​each​ ​of​ ​the​ ​scoring​ ​methods​ ​in​ ​the​ ​scoring​ ​list:​ ​f1,​ ​recall,​ ​and precision.​ ​Then​ ​from​ ​there,​ ​I​ ​ran​ ​those​ ​three​ ​optimal​ ​parameter​ ​tunes in​ ​tester.py​ ​and​ ​checked​ ​to​ ​see​ ​which​ ​one​ ​gave​ ​me​ ​the​ ​best​ ​overall results.

￼My​ ​initial​ ​results​ ​when​ ​using​ ​just​ ​PCA​ ​and​ ​GaussianNB​ ​in​ ​a​ ​pipeline gave​ ​me​ ​a​ ​higher​ ​precision​ ​and​ ​lower​ ​recall​ ​than​ ​the​ ​tuned​ ​algorithm. The​ ​f1​ ​was​ ​higher​ ​for​ ​the​ ​tuned​ ​algorithm​ ​as​ ​well.​ ​My​ ​main​ ​concern once​ ​I​ ​was​ ​over​ ​0.3​ ​for​ ​both​ ​precision​ ​and​ ​recall​ ​was​ ​which​ ​one should​ ​I​ ​optimize​ ​for.​ ​I​ ​decided​ ​in​ ​the​ ​case​ ​of​ ​the​ ​Enron​ ​dataset​ ​and determining​ ​POI’s​ ​that​ ​it​ ​would​ ​be​ ​more​ ​important​ ​to​ ​optimize​ ​recall over​ ​precision.​

​By​ ​optimizing​ ​for​ ​recall,​ ​my​ ​algorithm​ ​is​ ​more​ ​likely to​ ​flag​ ​POI’s​ ​whenever​ ​a​ ​POI​ ​shows​ ​up​ ​but​ ​at​ ​the​ ​cost​ ​of​ ​also flagging​ ​people​ ​who​ ​sometimes​ ​may​ ​not​ ​be​ ​POI’s.​ ​If​ ​I​ ​had​ ​optimized for​ ​precision,​ ​when​ ​my​ ​algorithm​ ​flags​ ​someone​ ​as​ ​a​ ​POI,​ ​I​ ​know​ ​with more​ ​confidence​ ​that​ ​that​ ​person​ ​is​ ​in​ ​fact​ ​a​ ​POI,​ ​but​ ​my​ ​algorithm sometimes​ ​misses​ ​real​ ​POI’s​ ​because​ ​it​ ​is​ ​more​ ​conservative​ ​when flagging.​ ​The​ ​reason​ ​I​ ​chose​ ​recall​ ​is​ ​because​ ​I​ ​think​ ​that​ ​having​ ​a bigger​ ​pool​ ​of​ ​POI’s​ ​to​ ​choose​ ​from​ ​is​ ​better​ ​than​ ​having​ ​a​ ​smaller pool​ ​of​ ​POI’s​ ​since​ ​we​ ​are​ ​investigating​ ​a​ ​corporate​ ​scandal​ ​and​ ​the more​ ​information​ ​the​ ​better.​ ​I​ ​envision​ ​investigators​ ​having​ ​a​ ​larger group​ ​of​ ​persons​ ​of​ ​interest​ ​initially,​ ​then​ ​doing​ ​a​ ​more​ ​thorough analysis​ ​of​ ​each​ ​of​ ​these​ ​people​ ​to​ ​find​ ​the​ ​real​ ​POI’s.​ ​If​ ​I​ ​had optimized​ ​for​ ​precision,​ ​investigators​ ​would​ ​have​ ​a​ ​smaller​ ​group​ ​of people​ ​who​ ​are​ ​most​ ​likely​ ​POI’s​ ​but​ ​could​ ​definitely​ ​be​ ​missing​ ​more people​ ​and​ ​potentially​ ​more​ ​pieces​ ​of​ ​the​ ​overall​ ​puzzle.​ ​Plus,​ ​in America,​ ​you​ ​are​ ​innocent​ ​until​ ​proven​ ​guilty​ ​which​ ​means​ ​even​ ​if​ ​a person​ ​is​ ​flagged​ ​as​ ​a​ ​POI,​ ​it​ ​doesn’t​ ​mean​ ​they​ ​will​ ​get​ ​convicted of​ ​anything​ ​if​ ​they​ ​really​ ​are​ ​innocent.
