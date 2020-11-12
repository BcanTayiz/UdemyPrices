import streamlit as st 
import pandas as pd
import numpy as np
import re
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import datetime
import random
from sklearn import preprocessing
from sklearn import metrics
from matplotlib.pyplot import figure
from sklearn.metrics import r2_score


random.seed(10)


st.title("Streamlit example")

st.write("""
Udemy Courses Subscriber Total Value Prediction
""")

#Data Feature Eng.
data = pd.read_csv('udemy_courses.csv')
data = data.loc[data['is_paid'] == 1]
st.write(data.head())


data['total_value'] = data['price'] * data['num_subscribers']

data['titleLen'] = data['course_title'].str.len()

data['time'] = pd.to_datetime(data['published_timestamp'], errors='coerce')
data['month'] = data['time'].dt.month




regressorTypes = ("Random Forests","Support Vector","XGBoost","Neural Net")

regressor_name = st.sidebar.selectbox("Select Regressor",regressorTypes)




st.header("All Na Values")
st.table(data.isna().sum())





#NLP KEYWORD COUNT

from sklearn.feature_extraction.text import TfidfVectorizer
corpus = data.course_title.tolist()
vectorizer = TfidfVectorizer()
transformedCorpus = vectorizer.fit_transform(corpus)

countKeywords = []
for title in data.course_title.tolist():  
    count = 0
    for word in vectorizer.get_feature_names():
        if word in title:
            count += 1

    countKeywords.append(count)


data['Keyword_Count'] = countKeywords



#NLP KEYWORD COUNT


data = data.drop(['course_id','is_paid', 'price','course_title','num_subscribers','time','num_reviews','url','published_timestamp'], axis=1)


catCategories = []
for name in data.columns:
    if data[name].dtype == "object":
        data[name] = data[name].astype('category')
        catCategories.append(dict(enumerate(data[name].astype('category').cat.categories)))
        data[name] = data[name].cat.codes
        
        
st.header("Select Target Value")
target_name = st.selectbox("Select Target Value",tuple(data.columns))
st.write(data.head())


FeatureNames = data.drop(target_name,axis=1).columns   
X = data.drop(target_name,axis=1)
y = data[target_name]

x = X #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
X = pd.DataFrame(x_scaled)

st.header("Normalized Train Data Frame")
st.write(X)


def parameter_ui(regressor_name):
    params = dict()
    if regressor_name == regressorTypes[0]:
        n_estimators = st.sidebar.slider("n_estimators",10,100)
        criterion = st.sidebar.selectbox("criterion",("mse", "mae"))
        max_depth = st.sidebar.slider("max_depth",1,100)
        min_samples_split = st.sidebar.slider("min_samples_split",2,100)
        min_samples_leaf = st.sidebar.slider("min_samples_leaf",1,100)
        max_features = st.sidebar.selectbox("criterion",("auto", "sqrt", "log2"))
        max_leaf_nodes = st.sidebar.slider("max_leaf_nodes",2,100)
        min_impurity_decrease = st.sidebar.slider("min_impurity_decrease",0,1000)
        bootstrap = st.sidebar.selectbox("bootstrap",("True", "False"))
        oob_score = st.sidebar.selectbox("oob_score",("True", "False"))
        

        min_impurity_decrease = min_impurity_decrease / 10

        params["n_estimators"] = n_estimators
        params["criterion"] = criterion
        params["max_depth"] = max_depth
        params["min_samples_split"] = min_samples_split
        params["min_samples_leaf"] = min_samples_leaf
        params["max_features"] = max_features
        params["max_leaf_nodes"] = max_leaf_nodes
        params["min_impurity_decrease"] = min_impurity_decrease
        params["bootstrap"] = bootstrap
        params["oob_score"] = oob_score


    elif regressor_name == regressorTypes[1]:
        kernel = st.sidebar.selectbox("criterion",("linear", "poly", "rbf","sigmoid","precomputed"))
        degree = st.sidebar.slider("degree",1,10)
        gamma = st.sidebar.selectbox("gamma",("scale", "auto"))
        C = st.sidebar.slider("C",1,20)

        params["kernel"] = kernel
        params["degree"] = degree
        params["gamma"] = gamma
        params["C"] = C
    elif regressor_name == regressorTypes[2]:
        booster  = st.sidebar.selectbox("booster",("gbtree", "gblinear","dart"))
        verbosity  = st.sidebar.selectbox("verbosity",(0,1,2,3))
        nthread  = st.sidebar.slider("nthread ",1,100)
        eta = st.sidebar.slider("eta ",0,100,1)
        gamma  = st.sidebar.slider("gamma ",0,100)
        max_depth  = st.sidebar.slider("max_depth ",1,100)

        eta = eta / 100

        params["booster"] = booster
        params["verbosity"] = verbosity
        params["nthread"] = nthread
        params["eta"] = eta
        params["gamma"] = gamma
        params["max_depth"] = max_depth
    elif regressor_name == regressorTypes[3]:
        hidden_layer_sizes = st.sidebar.slider("hidden_layer_sizes",1,20)
        activation = st.sidebar.selectbox("activation",("identity", "logistic","tanh","relu"))
        solver = st.sidebar.selectbox("solver",("lbfgs", "sgd", "adam"))
        alpha = st.sidebar.slider("alpha",0,100,1)
        batch_size = st.sidebar.slider("batch_size",1,100)
        max_iter = st.sidebar.slider("max_iter",1,100)
        learning_rate = st.sidebar.selectbox("learning_rate",("constant", "invscaling", "adaptive"))

        alpha = alpha / 100

        params["hidden_layer_sizes"] = hidden_layer_sizes
        params["activation"] = activation
        params["solver"] = solver
        params["alpha"] = alpha
        params["batch_size"] = batch_size
        params["max_iter"] = max_iter
        params["learning_rate"] = learning_rate

    return params


params = parameter_ui(regressor_name)

def get_regressor(regressor_name,params):
    if regressor_name == regressorTypes[0]:
         reg = RandomForestRegressor(n_estimators=params["n_estimators"], criterion=params["criterion"],max_depth=params["max_depth"]
         ,min_samples_split = params["min_samples_split"],min_samples_leaf = params["min_samples_leaf"],max_features = params["max_features"],
          max_leaf_nodes=params["max_leaf_nodes"],min_impurity_decrease = params["min_impurity_decrease"],bootstrap=params["bootstrap"],
          oob_score = params["oob_score"])
    elif regressor_name == regressorTypes[1]:
        reg =  SVR(kernel=params["kernel"], degree=params["degree"],gamma=params["gamma"],C = params["C"])
    elif regressor_name == regressorTypes[2]:
        reg =  xgb.XGBRegressor(booster=params["booster"], verbosity=params["verbosity"],nthread=params["nthread"],eta=params["eta"],gamma=params["gamma"],max_depth=params["max_depth"])
    elif regressor_name == regressorTypes[3]:
        reg = MLPRegressor(hidden_layer_sizes=params["hidden_layer_sizes"],activation=params["activation"],solver=params["solver"],
        alpha=params["alpha"],batch_size=params["batch_size"],learning_rate=params["learning_rate"],max_iter = params["max_iter"])

    return reg

reg = get_regressor(regressor_name,params)

st.write(reg)       


#Regression
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 42)

reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)



pyplot.scatter(y_pred,y_test)

pyplot.title('Prediction Results')
pyplot.xlabel('Y Prediction Data')
pyplot.ylabel('Y Test Data')

st.pyplot()
st.header("R2 Score")
st.write("R2 score shows correlation of results. Make sure it is close to -1 or close to 1")
st.write(r2_score(y_test, y_pred))

st.set_option('deprecation.showPyplotGlobalUse', False)

#MODEL IMPORTANCE
try:
    st.header("Feature Importance Figure")
    importances = reg.feature_importances_
    # summarize feature importance
    for i,v in enumerate(importances):
        print('Feature: %0d, Score: %.5f' % (i,v))
    # plot feature importance
  
    figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
    pyplot.bar([FeatureNames[x] for x in range(len(importances))], importances)

    st.pyplot()
except:
    pass



st.header("Results")
MSE = mean_squared_error(y_test,y_pred)

st.write("*****Try to find lowest MSE so that you will reach the probably best accurate result*****")

st.write(f"Regressor = {regressor_name}")
st.write(f"MSE = {MSE}")

st.header("Feature Information for Input")
for cat in catCategories:
    st.write(f"learn here the categories {name},{cat}")

st.header("Enter Input for Prediction")

num_of_lectures = st.text_input('Enter number of lectures:') 
level = st.text_input('Enter your course level:') 
contentDuration = st.text_input('Enter your content duration as hours:') 
subject = st.text_input('Enter your subject here:') 
titleLen = st.text_input('Estimated title length:') 
month = st.text_input('Which month is it created?:')
keyword = st.text_input('How many keyword did you use on title?:')


df = pd.DataFrame(columns = X_train.columns)
df.loc[0] = [0 for i in X_train.columns ]
df.loc[1] = [num_of_lectures,level,contentDuration,subject,titleLen,month,keyword]


if  num_of_lectures and level and contentDuration and subject and titleLen and month:
    prediction = reg.predict(df.astype(float))
    if target_name == "total_value":
        st.header("Probable {0} from course {1}".format(target_name,'${:,.2f}'.format(prediction[1])))
    else:
        st.header("Probable {0} from course {1}".format(target_name,int(prediction[1])))