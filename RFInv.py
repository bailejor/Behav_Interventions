from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, make_scorer, roc_auc_score, balanced_accuracy_score, confusion_matrix, precision_score, recall_score, fbeta_score
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, cross_validate, StratifiedKFold
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from numpy import mean
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.metrics import geometric_mean_score
from imblearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFE
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier


dataframe = pd.read_csv("FCTnoST2.csv", header = 0)
dataset = dataframe.values

X_train = dataframe.iloc[:,1:18]
y_train = dataset[:,18:19].astype(float)

y_vals = dataframe.iloc[:, 18:19]
print(y_vals.value_counts())

pca = PCA(n_components = 3)
#X_train =pca.fit_transform(X_train)

y_train = y_train.ravel()
seed = 6


#GET CATERGORICAL FEATURES SEPARATED FROM CONTINUOUS, scale continuous, smotenc with all
smote_value = 0.55
print("smote value is " + str(smote_value))

sm = KMeansSMOTE(random_state = seed, sampling_strategy = smote_value, cluster_balance_threshold = 0.3)
rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select= 17)
inpt = 17
def create_model(x):
  def bm():
    clf = Sequential()
    clf.add(Dense(9, activation='relu', input_dim = x))
    clf.add(Dense(9, activation='relu'))
    clf.add(Dense(2, activation = 'sigmoid'))
    clf.compile(loss = 'categorical_crossentropy', optimizer = SGD())
    return model
  return bm

#Scale continuous features only
preprocessor = ColumnTransformer(transformers = [('scale_cont', StandardScaler(), [0, 1, 14, 15])], remainder = 'passthrough')
print(preprocessor)
clf = KerasClassifier(build_fn = create_model(inpt), verbose = 0)
clf_params = {'clf__epochs': [50, 100]}
pipeline20 = Pipeline([('preprocessor', preprocessor), ('rfe', rfe), ('clf', clf)])

kNN = KNeighborsClassifier()
pipeline1 = Pipeline([
	('preprocessor', preprocessor), ('kNN', kNN)])
kNN_par= {'kNN__n_neighbors':[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 20, 22, 28], 'kNN__weights': ['uniform', 'distance'],
'kNN__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 'kNN__leaf_size': [10, 20, 30, 40, 50],
'kNN__metric': ['minkowski']}

rfc =RandomForestClassifier()
pipeline2 = Pipeline([('preprocessor', preprocessor), ('rfc', rfc)])
param_grid = {'rfc__bootstrap': [True],
              'rfc__n_jobs':[-1],
              'rfc__criterion' :['gini'],
              'rfc__max_depth': [2, 4, 6],
              'rfc__max_features': ['auto'],
              'rfc__min_samples_leaf': [1, 2, 4],
              'rfc__min_samples_split': [2, 5, 10],
              'rfc__n_estimators': [100, 400],
              'rfc__class_weight':[{0:2.46, 1:1}]}

lr = LogisticRegression()
pipeline15 = Pipeline([('preprocessor', preprocessor), ('sm', sm), ('lr', lr)])
lr_params = {'lr__penalty':['none'], 'lr__max_iter': [2000]}


pipeline16 = Pipeline([('preprocessor', preprocessor), ('lr', lr)])
lr_params2 = {'lr__penalty':['elasticnet'], 'lr__solver':['saga'], 'lr__l1_ratio':[0.5], 'lr__max_iter': [2000], 'lr__class_weight':[{0:1, 1:1}, {0:2, 1:1}, {0:3, 1:1}]}

mlp = MLPClassifier(random_state=seed)
pipeline8 = Pipeline([('preprocessor', preprocessor), ('mlp', mlp)])
parameter_space = {'mlp__hidden_layer_sizes': [(10, 10, 10), (5, 5), (10,10), (10,10,10, 10, 10)],
     'mlp__activation': ['relu'],
     'mlp__solver': ['adam', 'sgd'],
     'mlp__max_iter': [10000, 15000, 20000],
     'mlp__alpha': [0.1, 0.01, 0.001],
     'mlp__learning_rate': ['constant','adaptive'],
     'mlp__early_stopping':[False]}

gbm = GradientBoostingClassifier()
pipeline3 = Pipeline([ ('preprocessor', preprocessor), ('sm', sm), ('gbm', gbm)])
param = {"gbm__loss":["deviance"],
    "gbm__learning_rate": [0.1, 0.05, 0.01],
    "gbm__min_samples_split": [2, 3, 4, 5],
    "gbm__min_samples_leaf": [1, 2, 3],
    "gbm__max_depth":[10, 20, 30],
    "gbm__max_features":['auto'],
    "gbm__criterion": ["friedman_mse"],
    "gbm__n_estimators":[100, 200, 300, 400]
    }

dt = DecisionTreeClassifier()
pipeline11 = Pipeline([('preprocessor', preprocessor), ('dt', dt)])
dt_params = {'dt__max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10],
      'dt__criterion':['gini', 'entropy'],
      'dt__max_features':['auto'],
      'dt__class_weight':[{0:1.5, 1:1}, {0:2.2, 1:1}, {0:2, 1:1}, {0:2.4, 1:1}, {0:3, 1:1}],
      'dt__random_state':[seed],
      'dt__min_samples_split':[2, 6, 4],
      'dt__min_samples_leaf':[12, 26, 4, 10]
      }


svm = SVC()
pipeline4 = Pipeline([('preprocessor', preprocessor), ('svm', svm)])
tuned_parameters = {'svm__kernel':('linear', 'rbf'), 'svm__C':(2, 1.5, 1, 0.25, 0.5, 0.75, 0.6, 0.8, 0.9), 'svm__gamma': [0.1, 1, 10, 50, 100, 1000],
'svm__class_weight': [{1:1, 0: 0.6}]}

NB = GaussianNB(priors = [0.37, 0.63])
pipeline5 = Pipeline([('preprocessor', preprocessor), ('NB', NB)])
nb_params = {'NB__var_smoothing': [0.9, 0.5, 0.3, 0.1, 0.01, 0.001]}


dc = DummyClassifier()
pipeline6 = Pipeline([('sm', sm), ('dc', dc)])
dummy_params = {'dc__strategy': ['constant'], 'dc__constant':[1]}

dc2 = DummyClassifier()
pipeline10 = Pipeline([('sm', sm), ('dc', dc)])
dummy_params2 = {'dc__strategy': ['uniform']}

dc3 = DummyClassifier()
pipeline9 = Pipeline([('sm', sm), ('dc', dc)])
dummy_params3 = {'dc__strategy': ['constant'], 'dc__constant':[0]}

ada = AdaBoostClassifier()
pipeline7 = Pipeline([('preprocessor', preprocessor), ('ada', ada)])
ada_params = {'ada__n_estimators': [50, 100, 200], 'ada__learning_rate':[1, 0.01, 0.001, 0.5]}


inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)


def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]

f1_scorer = make_scorer(f1_score, pos_label = 0)
roc_auc = make_scorer(roc_auc_score)
balanced_accuracy = make_scorer(balanced_accuracy_score) 
geo = make_scorer(geometric_mean_score) 
accuracy = make_scorer(accuracy_score)
fbeta = make_scorer(fbeta_score, beta = 2, pos_label=0)




models = []
#models.append(('gbm', GridSearchCV(pipeline3, param, cv=inner_cv, n_jobs=-1, scoring = geo)))
#models.append(('rfc', GridSearchCV(pipeline2, param_grid, cv=inner_cv, n_jobs=-1, scoring = fbeta)))
#models.append(('kNN', GridSearchCV(pipeline1, kNN_par, cv=inner_cv, n_jobs=-1, scoring = accuracy)))
#models.append(('dt', GridSearchCV(pipeline11, dt_params, cv=inner_cv, n_jobs = -1, scoring = accuracy)))
#models.append(('svm', GridSearchCV(pipeline4, tuned_parameters, cv=inner_cv, n_jobs=-1, scoring = accuracy)))
#models.append(('MLP', GridSearchCV(pipeline8, parameter_space, cv=inner_cv, n_jobs=-1, scoring = accuracy)))
#models.append(('NB', GridSearchCV(pipeline5, nb_params, cv=inner_cv, n_jobs=-1, scoring = accuracy)))
#models.append(('Always Predicts FCT Success', GridSearchCV(pipeline6, dummy_params, cv=inner_cv, n_jobs=-1)))
#models.append(('Balanced FCT Prediction', GridSearchCV(pipeline10, dummy_params2, cv=inner_cv, n_jobs=-1)))
#models.append(('Always Predicts FCT Failure', GridSearchCV(pipeline9, dummy_params3, cv=inner_cv, n_jobs=-1)))
#models.append(('ada', GridSearchCV(pipeline7, ada_params, cv=inner_cv, n_jobs=-1, scoring = accuracy)))
#models.append(('lr', GridSearchCV(pipeline15, lr_params, cv=inner_cv, n_jobs=-1, scoring = fbeta)))
#models.append(('lr', GridSearchCV(pipeline16, lr_params2, cv=inner_cv, n_jobs=-1, scoring = jaccard)))
models.append(('clf', GridSearchCV(pipeline20, clf_params, cv=inner_cv, n_jobs=-1, scoring = accuracy)))



scoring = {'accuracy': make_scorer(accuracy_score), 
'f1': make_scorer(f1_score, pos_label = 0), 
'BalancedAcc': make_scorer(balanced_accuracy_score),
'Geometric Mean' : make_scorer(geometric_mean_score),
'tp' : make_scorer(tp), 'tn' : make_scorer(tn),
'fp' : make_scorer(fp), 'fn' : make_scorer(fn),
'FCT_fail_Precision': make_scorer(precision_score, pos_label=0),
'FCT_fail_Recall' : make_scorer(recall_score, pos_label=0),
'FCT_Suc_Precision': make_scorer(precision_score, pos_label=1),
'FCT_Suc_Recall' : make_scorer(recall_score, pos_label=1),
'AUC_ROC': make_scorer(roc_auc_score)
}




names =[]
hyperparams = []
results = []
#X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=6)

for name, model in models:
  print(name)
  results= cross_validate(model, X_train, y_train, cv=outer_cv, scoring= scoring)
  #results.append(nested_cv_results)
  #names.append(name)

  for metric_name in results.keys():
    average_score = np.average(results[metric_name])
    print('%s : %f' % (metric_name, average_score))


tang_tp = 58
tang_fn = 13
tang_fp = 16
tang_tn = 16

tang_precis = (tang_tp)/(tang_tp + tang_fp)
tang_recall = (tang_tp)/(tang_tp + tang_fn)
tang_f1 = (2* tang_precis * tang_recall) / (tang_precis + tang_recall)

neg_recall = (tang_tn)/(tang_tn + tang_fp)
#Neg recall is same thing as specificity or true negative rate
#Recall is same thing as true positive rate

balanced_falligant = (tang_recall + neg_recall) / 2
print("Precision for Falligant et al. was " + str(tang_precis))
print("Recall for Falligant et al. was " + str(tang_recall))
print("The F1 score for Falligan et al. was " + str(tang_f1))
print("The balanced accuracy for Falligant et al. was " + str(balanced_falligant))
print('NPV average for Falligant et al. was 0.56, average PPV was 0.79')



