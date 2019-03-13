# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import math
from pandas import DataFrame 
from collections import Counter
from pprint import pprint
from sklearn.model_selection import train_test_split # Import train_test_split function

df_tennis = pd.read_csv('C:\\Users\\wwwmu\\Downloads\\tenis.csv')

def entropy(propbs):
    return sum([-x*math.log2(x) for x in propbs])


def entropylist(list_a):
    classAttr = Counter(x for x in list_a)
    print(classAttr)
    numberOfInstanceInClass = len(list_a)*1.0
    print(numberOfInstanceInClass)
    probpsForevryAttr = [x/numberOfInstanceInClass for x in classAttr.values()]
    print(min(probpsForevryAttr),max(probpsForevryAttr))
    print('_________________________EntropyList_____________________')
    return entropy(probpsForevryAttr)

def informationGain(dataset,Sname,targetName):
    values = dataset.groupby(targetName)
    for name,group in values:
        print("Name:\n",name)
        print("Group:\n",group)
    nobs = len(dataset)*1.0
    Sv=values.agg({Sname:[entropylist,lambda x :len(x)/nobs]})[Sname]
    Sv.columns = ['Entropy','Prop']
    print(Sv)
    
   
    return (entropylist(dataset[Sname])-sum(Sv['Entropy'] * Sv['Prop']))
    #entropylist(s) - sum((len(target)/len(S))*entropylist(target))    

def ID3(dataset,Sname,features, default_class=None):
    cn = Counter(x for x in dataset[Sname])
    if len(cn)==1:
        return next(iter(cn))
    elif dataset.empty or (not features):
        return default_class
    else:
        default_class = max(cn)
        print(default_class)
        
        gainForAll = [ informationGain(dataset,Sname,x) for x in features]
        indexOfMaxGain = gainForAll.index(max(gainForAll))
        nameOfAttr = features[indexOfMaxGain]
        tree ={nameOfAttr:{}}
        remaningFeatures = [x for x in features if x!=nameOfAttr]
        for attr_name,subDataSet in dataset.groupby(nameOfAttr):
           subtree= ID3(subDataSet,Sname,remaningFeatures,default_class)
           tree[nameOfAttr][attr_name] = subtree
        return tree
####################################################################################        
attribute_names = list(df_tennis.columns)
attribute_names.remove('PlayTennis')
X = df_tennis[attribute_names] # Features
Y = df_tennis.PlayTennis # Target variable
X_train, X_test  = train_test_split(X, test_size=0.2)

tree = ID3(df_tennis.iloc[1:-6],'PlayTennis',attribute_names)

pprint(tree)
####################################################################################
def classify(instance,tree,default=None):
    #print("Instance:")
    #print(instance)
    #print(default)
    attribute = next(iter(tree)) 
    #print("Key:",tree.keys()) 
    #print("Attribute:",attribute)
    #print("Insance of Attribute :",instance[attribute],attribute)
    if instance[attribute] in tree[attribute].keys():
        result = tree[attribute][instance[attribute]]
        #print("Instance Attribute:",instance[attribute],"TreeKeys :",tree[attribute].keys())
        if isinstance(result,dict):
            return classify(instance,result)
        else:
            return result
    else:
        return default

a= (df_tennis.iloc[-3:]).apply(classify,axis=1,args=(tree,'no'))
from sklearn.metrics import accuracy_score
print(accuracy_score(df_tennis.iloc[-3:]['PlayTennis'],a ))