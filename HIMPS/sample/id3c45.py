
import pandas as pd
import numpy as np
import json
import operator
from sklearn.datasets import load_iris


global tree_tr
global petal_length, petal_width, sepal_width, sepal_length

#-------------------FUNGSI UMUM------------------------

def globalEntropy(df):
	entropy = 0
	vals = df.iloc[:,-1].unique()
	len_vals = len(df.iloc[:,-1])
	for val in vals:
		p = df.iloc[:,-1].value_counts()[val]/len_vals
		entropy = entropy + -p*safe_log2(p)
	return entropy

def safe_log2(x):
    if x <= 0:
        return 0
    return np.log2(x)

def attrEntropy(df,attrName):
	attrs = df.iloc[:,-1].unique()
	attrVals = df[attrName].unique()
	
	entropy = 0
	for attrVal in attrVals:
		ent = 0
		for attr in attrs:
			sv = len(df[attrName][df.iloc[:,-1] == attr][df[attrName] == attrVal])
			s = len(df[attrName][df[attrName] == attrVal])
			frac = sv/(s)
			ent += -frac*safe_log2(frac)
		t = s/len(df)
		entropy += -t*ent
	
	return (abs(entropy))

def informationGain(rootEntropy, attrEntropy):
	return rootEntropy - attrEntropy

#-------------------FUNGSI ID3------------------------

def bestAttr(df, col_a, col_b):
	gains = {}
	for col in df.columns[col_a:col_b]:
		gains[col] = informationGain(globalEntropy(df), attrEntropy(df, col))
	
	return max(gains, key=gains.get)

def filterTab(df, attr, val):
	return (df.loc[df[attr]==val])

def check_all_attr(data):
	if len(data.unique())==1:
		return data.unique()
	return None

def id3 (df,col_a,col_b, tree=None):
	if tree is None:
		tree = {}
	if globalEntropy(df)==0:
		tree = {df.iloc[:,-1][0]}
		return tree
	else:
		root = bestAttr(df,col_a,col_b)
		tree[root] = {}
		for attrVal in df[root].unique():
			new_df = filterTab(df, root, attrVal)
			vals = new_df.iloc[:,-1].unique()
			if len(vals)==1:
				tree[root][attrVal] = vals[0]
			else:
				tree[root][attrVal] = id3(new_df,col_a,col_b)
		return tree

#-------------------FUNGSI C4.5------------------------

def gainratio(df, cols, gain):
	if gain == 0 :
		return 0
	gainRatio = 0
	splitInformation = 0
	vals = df[cols].unique()
	len_vals = len(df[cols])
	for val in vals:
		p = df[cols].value_counts()[val]/len_vals
		splitInformation = splitInformation + -p*safe_log2(p)

	gainRatio = gain / splitInformation
	return gainRatio

def bestAttrc45(df, is_gain_ratio,col_a, col_b):
	gains = {}
	nonobject = {}
	gainratios = {}

	for col in df.columns[col_a:col_b]:
		if (len(df.loc[df[col] == '?']) > 0):
			df = missingValueHandling(df, col)

		if df[col].dtype != 'object' :
				nonobject[col], gains[col] = c45ContinousHandling(df, col)
		else :
			gains[col] = informationGain(globalEntropy(df), attrEntropy(df, col))

		gainratios[col] = gainratio(df, col, gains[col])

	if all(value == 0 for value in gains.values()) :
		return None
	if is_gain_ratio :
		maxc45 = max(gainratios, key=gainratios.get)
	else :
		maxc45 = max(gains, key=gains.get)
	
	if maxc45 in nonobject :
		df.loc[:,maxc45] = nonobject[maxc45]
		
	return maxc45

def missingValueHandling(df, col_name):
	idx = df.loc[df[col_name] == "?"].index[0]

	target = df.iloc[idx,-1]
	val_attr = df[col_name].unique()
	new_df = df.loc[df.iloc[:,-1] == target]

	modus = 0
	for val in val_attr:
		p = new_df[col_name].value_counts()[val]
		if modus < p:
			modus = p
			value = val
		
	df.loc[idx,col_name] = value

	return df

def c45ContinousHandling(df, column_name):
    values = sorted(df[column_name].unique())

    if len(values) == 1:
        threshold = values[0]
    
    else :
        gains = [0 for i in range (len(values)-1)]
        for i in range(len(values)-1) :
            threshold = values[i]
            
            subset1 = df[df[column_name] <= threshold]
            subset2 =  df[df[column_name] > threshold]

            subset1_prob = len(subset1) / len(df[column_name])
            subset2_prob = len(subset2) / len(df[column_name])
            
            gains[i] = globalEntropy(df) - subset1_prob*globalEntropy(subset1) - subset2_prob*globalEntropy(subset2)
            

    winner_gain = max(gains)
    threshold = values[gains.index(max(gains))] 
    temp = np.where(df[column_name] <= threshold, "<="+str(threshold), ">"+str(threshold))
    
    return temp, winner_gain

def c45 (df,is_gain_ratio,col_a,col_b,tree=None):
	if tree is None:
		tree = {}
	if globalEntropy(df)==0:
		tree = {df.iloc[:,-1][0]}
		return tree
	else:
		root = bestAttrc45(df, is_gain_ratio, col_a, col_b)
		if root is None :
			tree = df.iloc[:,-1].value_counts().idxmax() 
			return tree

		tree[root] = {}
		for attrVal in df[root].unique():
			new_df = filterTab(df, root, attrVal)
			vals = new_df.iloc[:,-1].unique()
			if len(vals)==1:
				tree[root][attrVal] = vals[0]
			else:
				tree[root][attrVal] = c45(new_df, is_gain_ratio, col_a, col_b)
		return tree

def splitData(df, length):
	training = pd.DataFrame()
	evaluation = pd.DataFrame()

	n = round(length/5)
	training = df.loc[0:n-1]
	evaluation = df.loc[n:]
	
	return (training,evaluation)

def check_tree(instance, tree, target_vals, latest_attr = None):
	try:	
		l = list(tree.keys())
	except AttributeError:
		l = tree
	
	if instance.iloc[-1] in target_vals:
		if instance.iloc[-1] == l:
			return True
		else:
			return False
	if len(l) == 1:
		check_tree(instance, tree[l[0]], target_vals, l)
	else:
		print(latest_attr)
		attr_val = instance[latest_attr]
		t =attr_val.values[0]
		print(attr_val.values[0])
		check_tree(instance, tree[t], target_vals)


def akurasi(evaluation, tree = None):
	global tree_tr
	benar = 0
	if tree is None:
		for row in range(len(evaluation)):
			if check_tree(evaluation.loc[row], tree_tr, evaluation.iloc[:,-1].unique()):
				benar = benar + 1
	else:
		evaluation = pd.DataFrame.from_dict(evaluation)
		for row in range(len(evaluation)):
			if check_tree(evaluation.loc[row], tree, evaluation.iloc[:,-1].unique()):
				benar = benar + 1
		
	hasilAkurasi = benar/len(evaluation)

	return hasilAkurasi

def try_prune(tree, attr):
	if tree is attr:
		each_attr = list(attr.keys())
		for i in each_attr:
			tree.pop(i)
		return
	else:
		for j in tree:
			if isinstance(j, dict):
				try_prune(tree[j], attr)

def prunningNode(tree_training, evaluation, saved_tree = None):
	for l in tree_training.keys():
		if not isinstance(tree_training[l], dict):
			akurasiNode = akurasi(evaluation)
			new_tree = saved_tree
			try_prune(new_tree, tree_training)

			if akurasiNode < akurasi(evaluation, new_tree):
				try_prune(saved_tree, tree_training)
			return tree_training
		else:
			prunningNode(tree_training[l], evaluation, saved_tree)
	return tree_training

def prunning(df, is_gain_ratio, col_a, col_b):
	ev, tr = splitData(df, len(df))
	tree_training = c45(tr, is_gain_ratio, col_a, col_b)
	final_tree = prunningNode(tree_training, ev, tree_training)
	return final_tree

#-------------------SAVE MODEL------------------------

def save_model(tree) : 
	with open('model.json', 'w') as file_model :
		json.dump(tree, file_model)

#-------------------LOAD MODEL------------------------

def load_model() :
	with open('model.json', 'r') as file_model :
		new_tree = json.load(file_model)

	return new_tree

#----------------BUAT INSTANS BARU----------------------

def getList(dict): 
    list = [] 
    for key in dict.keys(): 
        list.append(key)    
    return list

def create_instance(tree):
	# global petal_length, petal_width, sepal_width, sepal_length

	# print('type', type(tree))
	# print("tree1", tree)
	if (str(type(tree)) == "<class 'float'>"):
		return (tree)
	elif (str(type(tree)) == "<class 'numpy.float64'>"):
		return (tree.item())
		# print("tree2", tree)
		
	else:
		# print("tree", tree)
		k = list(tree.keys())[0]
		# print(k)
		# input()
		if (len(k) != 0):
			if (k == 'petal width (cm)'):
				for val in tree[k].keys():
					if (val[0] == '<'):
						if (petal_width <= float(val[-3:])):
							return create_instance(tree[k][val])
					elif (val[0] == '>'):
						if (petal_width > float(val[-3:])):
							return create_instance(tree[k][val])
			elif (k == 'petal length (cm)'):
				for val in tree[k].keys():
					if (val[0] == '<'):
						if (petal_length <= float(val[-3:])):
							return create_instance(tree[k][val])
					elif (val[0] == '>'):
						if (petal_length > float(val[-3:])):
							return create_instance(tree[k][val])
			elif (k == 'sepal width (cm)'):
				for val in tree[k].keys():
					if (val[0] == '<'):
						if (sepal_width <= float(val[-3:])):
							return create_instance(tree[k][val])
					elif (val[0] == '>'):
						if (sepal_width > float(val[-3:])):
							return create_instance(tree[k][val])
			elif (k == 'sepal length (cm)'):
				for val in tree[k].keys():
					if (val[0] == '<'):
						if (sepal_length <= float(val[-3:])):
							return create_instance(tree[k][val])
					elif (val[0] == '>'):
						if (sepal_length > float(val[-3:])):
							return create_instance(tree[k][val])

def predict(data, model):
	predic = []
	global petal_length, petal_width, sepal_width, sepal_length
	
	for row in data.iterrows():
		sepal_length = row[1][1]
		sepal_width = row[1][2]
		petal_length = row[1][3]
		petal_width = row[1][4]
		pred = create_instance(model)
		predic.append(pred)

	return predic
		

#-------------------MAIN PROGRAM------------------------


# print("Data:")
# print("1. Play Tennis (pastikan terdapat play-tennis.csv dalam folder yang sama)")
# print("2. Iris")
# data = input("Choose data (1/2) : ")

# if data == '1' :
# 	data_input = pd.read_csv("play-tennis.csv")
# 	col_a = 1
# 	col_b = -1
# elif data == '2' :
# 	iris = load_iris()
# 	data_input = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
#                      columns= iris['feature_names'] + ['target'])
# 	col_a = 0
# 	col_b = -1

# print()
# print("Methods:")
# print("1. ID3")
# print("2. C4.5")
# method = input("Choose method (1/2) : ")
# if method == '1' :
# 	tree = id3(data_input,col_a,col_b)
# elif method == '2' :
# 	ev, tr = splitData(data_input, len(data_input))
	
# 	gain_ratio = input("do you want to use gain ratio? (yes/no) ")
# 	if gain_ratio == "yes" :
# 		is_gain_ratio = True
# 	elif gain_ratio == "no" :
# 		is_gain_ratio = False
# 	tree_tr = c45(tr, is_gain_ratio, col_a, col_b)
# 	tree = prunning(data_input, is_gain_ratio, col_a, col_b) 

# print()
# print("Decision Tree : ")
# print(tree)
# print()

# print("--------saving model-----------")
# save_model(tree)

# print("--------loading model-----------")
# print()

# new_tree = load_model()
# print(new_tree)

# sepal_length = float(input("Sepal length : "))
# sepal_width = float(input("Sepal width : "))
# petal_length = float(input("Petal length : "))
# petal_width = float(input("Petal width : "))

# sepal_length = 7.7
# sepal_width = 2.6
# petal_length = 6.9
# petal_width = 2.3

# print(create_instance(new_tree))
# iris = load_iris()
# X = iris.data
# y = iris.target
# class_names = iris.target_names

# # Split the data into a training set and a test set


# DATA TRAIN
# col_a = 0
# col_b = -1
# is_gain_ratio = 1
# iris = load_iris()
# data_input = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
#                       columns= iris['feature_names'] + ['target'])
# ev, tr = splitData(data_input, len(data_input))
# tree = c45(tr, is_gain_ratio, col_a, col_b)
# print(predict(data_input, tree))

