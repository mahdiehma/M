import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from numpy import set_printoptions
import random
import math,time,sys, os
import matplotlib.pyplot as plt 
from matplotlib import pyplot
from sklearn import preprocessing
from scipy.stats import spearmanr
from datetime import datetime
from sklearn.metrics import recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

# load the Santander customer satisfaction dataset from Kaggle

X_train = pd.read_csv('')
#X_train.drop (['s03r','03r','su1category'], inplace = True , axis=1)

# separate dataset into train and test

# remove constant features
constant_features = [
    feat for feat in X_train.columns if X_train[feat].std() == 0
]

X_train.drop(labels=constant_features, axis=1, inplace=True)


# remove quasi-constant features
sel = VarianceThreshold(
    threshold=0.01)  # 0.1 indicates 99% of observations approximately

sel.fit(X_train)  # fit finds the features with low variance

sum(sel.get_support()) # how many not quasi-constant?
features_to_keep = X_train.columns[sel.get_support()]
# we can then remove the features like this
X_train = sel.transform(X_train)

# sklearn transformations lead to numpy arrays
# here I transform the arrays back to dataframes
# please be mindful of getting the columns assigned
# correctly

X_train= pd.DataFrame(X_train)
X_train.columns = features_to_keep

# check for duplicated features in the training set
duplicated_feat = []
for i in range(0, len(X_train.columns)):
    if i % 10 == 0:  # this helps me understand how the loop is going
        print(i)

    col_1 = X_train.columns[i]

    for col_2 in X_train.columns[i + 1:]:
        if X_train[col_1].equals(X_train[col_2]):
            duplicated_feat.append(col_2)
            
len(duplicated_feat)
X_train.drop(labels=duplicated_feat, axis=1, inplace=True)

df=X_train.drop



df=X_train

print('df',df)

class NaiveBayesClassifier():
    '''
    Bayes Theorem form
    P(y|X) = P(X|y) * P(y) / P(X)
    '''
    def calc_prior(self, features, target):
        '''
        prior probability P(y)
        calculate prior probabilities
        '''
        self.prior = (features.groupby(target).apply(lambda x: len(x)) / self.rows).to_numpy()

        return self.prior
    
    def calc_statistics(self, features, target):
        '''
        calculate mean, variance for each column and convert to numpy array
        ''' 
        self.mean = features.groupby(target).apply(np.mean).to_numpy()
        self.var = features.groupby(target).apply(np.var).to_numpy()
              
        return self.mean, self.var
    
    def gaussian_density(self, class_idx, x):     
        '''
        calculate probability from gaussian density function (normally distributed)
        we will assume that probability of specific target value given specific class is normally distributed 
        
        probability density function derived from wikipedia:
        (1/√2pi*σ) * exp((-1/2)*((x-μ)^2)/(2*σ²)), where μ is mean, σ² is variance, σ is quare root of variance (standard deviation)
        '''
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp((-1/2)*((x-mean)**2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        prob = numerator / denominator
        return prob
    
    def calc_posterior(self, x):
        posteriors = []

        # calculate posterior probability for each class
        for i in range(self.count):
            prior = np.log(self.prior[i]) ## use the log to make it more numerically stable
            conditional = np.sum(np.log(self.gaussian_density(i, x))) # use the log to make it more numerically stable
            posterior = prior + conditional
            posteriors.append(posterior)
        # return class with highest posterior probability
        return self.classes[np.argmax(posteriors)]
     

    def fit(self, features, target):
        self.classes = np.unique(target)
        self.count = len(self.classes)
        self.feature_nums = features.shape[1]
        self.rows = features.shape[0]
        
        self.calc_statistics(features, target)
        self.calc_prior(features, target)
     
    def predict(self, features):
        preds = [self.calc_posterior(f) for f in features.to_numpy()]
        return preds

    def accuracy(self, y_test, y_pred):
        accuracy = np.sum(y_test == y_pred) / len(y_test)
        return accuracy

y=df.iloc[:,-1].values
df
dataset = df.drop(df.columns[[0,6]], axis = 1)
df.info()
#Menampilkan kolom dataset
columns = list(df.columns)
columns
df.duplicated().sum()
#Missing Values
df.isna().any()
# Cek null values
df.isnull().sum()
plt.figure(figsize=(5,7))
df['Target'].value_counts().plot.pie(autopct='%1.1f%%', colors = ['blue','lime'])
plt.title("Presentase Pasien Diabetes", fontdict={'fontsize': 18})

plt.tight_layout()
import statsmodels.api as sm 
X = np.append(arr = np.ones((350,0)).astype(int), values = dataset, axis = 1) 
X_opt = X[:]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

def backwardElimination(X, sl):
    numVars = len(X[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, X).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    X = np.delete(X, j, 1)
                    regressor_OLS.summary()
    return X

SL = 0.05
X_train_opt = X[:]
X_Modeled = backwardElimination(X_train_opt, SL)
df = pd.DataFrame(X)
df
df = pd.DataFrame(X_Modeled)
df
import statsmodels.api as sm 
sl = 0.05
regressor_OLS = sm.OLS(endog = y, exog = X_Modeled).fit()
regressor_OLS.summary()
import statsmodels.api as sm 
sl = 0.05
regressor_OLS = sm.OLS(endog = y, exog = dataset).fit()
regressor_OLS.summary()
dataset = dataset.drop(dataset.columns[[2]], axis = 1)
import statsmodels.api as sm 
sl = 0.05
regressor_OLS = sm.OLS(endog = y, exog = dataset).fit()
regressor_OLS.summary()
dataset = dataset.drop(dataset.columns[[3]], axis = 1)
import statsmodels.api as sm 
sl = 0.05
regressor_OLS = sm.OLS(endog = y, exog = dataset).fit()
regressor_OLS.summary()

# shuffle dataset with sample

df_seleksi=pd.read_csv()
df_seleksi = df_seleksi.drop(df_seleksi.columns[[0,3,5]], axis = 1)
print(df_seleksi.shape)
# set features and target
X, y = df_seleksi.iloc[:, :-1], df_seleksi.iloc[:, -1]
result = pd.concat([X, y], axis=1).reindex(X.index)
print('best:',result)


#genetic

def initialize(popSize,dim):
    population=np.zeros((popSize,dim))
    minn = 1
    maxx = math.floor(0.8*dim)
    if maxx<minn:
        minn = maxx
        
    for i in range(popSize):
        random.seed(i**3 + 10 + time.time() ) 
        no = random.randint(minn,maxx)
        if no == 0:
            no = 1
        random.seed(time.time()+ 100)
        pos = random.sample(range(0,dim-1),no)
        for j in pos:
            population[i][j]=1

		# print(population[i])  
    return population


def fitness(solution, trainX, trainy, testX,testy):
	cols=np.flatnonzero(solution)
	val=1
	if np.shape(cols)[0]==0:
		return val	
	clf=RandomForestClassifier(n_estimators=300)
    #clf=KNeighborsClassifier(n_neighbors=5)
	train_data=trainX[:,cols]
	test_data=testX[:,cols]
	clf.fit(train_data,trainy)
	error=1-clf.score(test_data,testy)

	#in case of multi objective  []
	featureRatio = (solution.sum()/np.shape(solution)[0])
	val=omega*error+(1-omega)*featureRatio
	# print(error,featureRatio,val)
	return val

def allfit(population, trainX,  trainy,testX, testy):
	x=np.shape(population)[0]
	acc=np.zeros(x)
	for i in range(x):
		acc[i]=fitness(population[i],trainX,trainy,testX,testy)     
		#print(acc[i])
	return acc

def selectParentRoulette(popSize,fitnList):
	maxx=max(fitnList)
	fitnList = np.array(fitnList)
	fitnList = fitnList/maxx
	minn = min(fitnList)
	fitnList = 1- fitnList/fitnList.sum()

	print("data:",fitnList)
	random.seed(time.time()+19)
	val = random.uniform(0,fitnList.sum())
	for i in range(popSize):
		if val <= fitnList[i]:
			return i
		val -= fitnList[i]
	return -1

def randomwalk(agent,agentFit):
	percent = 30
	percent /= 100
	neighbor = agent.copy()
	size = np.shape(agent)[0]
	upper = int(percent*size)
	if upper <= 1 or upper>size:
		upper = size
	x = random.randint(1,upper)
	pos = random.sample(range(0,size - 1),x)
	for i in pos:
		neighbor[i] = 1 - neighbor[i]
	return neighbor

def adaptiveBeta(agent,agentFit, trainX, trainy,testX,testy):
	bmin = 0.1 #parameter: (can be made 0.01)
	bmax = 1
	maxIter = 10 # parameter: (can be increased )
	maxIter = int(max(10,10*agentFit))


	for curr in range(maxIter):
		neighbor = agent.copy()
		size = np.shape(agent)[0]
		neighbor = randomwalk(neighbor,agentFit)

		beta = bmin + (curr / maxIter)*(bmax - bmin)
		for i in range(size):
			random.seed( time.time() + i )
			if random.random() <= beta:
				neighbor[i] = agent[i]
		neighFit = fitness(neighbor,trainX,trainy,testX,testy)
		if neighFit <= agentFit:
			agent = neighbor.copy()
			agentFit = neighFit
	return (agent,agentFit)


def geneticAlgo(dataset,popSize,maxIter,randomstate):

	#--------------------------------------------------------------------
	#df.drop (['tcp','private','REJ'], inplace = True , axis=1)
    
    #=========================================================================
# 	scaler = preprocessing.MinMaxScaler() #normalize
# 	names = df.columns
# 	d = scaler.fit_transform(df)
# 	df = pd.DataFrame(d, columns=names)
    #============================================================================
	
	(a,b)=np.shape(result)
	print(a,b)
	data = result.values[:,0:b-1]
	label = result.values[:,b-1]
	dimension = np.shape(data)[1] #solution dimension
	#---------------------------------------------------------------------

	cross = 5
	test_size = (1/cross)
	trainX, testX, trainy, testy = train_test_split(data, label,stratify=label ,test_size=test_size,random_state=randomstate) #
	print(np.shape(trainX),np.shape(trainy),np.shape(testX),np.shape(testy))

	clf=MLPClassifier(random_state=1, max_iter=300)
    #clf=KNeighborsClassifier(n_neighbors=5)
	clf.fit(trainX,trainy)
	val=clf.score(testX,testy)
	whole_accuracy = val
	print("Total Acc: ",val)

	x_axis = []
	y_axis = []
	population = initialize(popSize,dimension)
	GBESTSOL = np.zeros(np.shape(population[0]))
	GBESTFIT = 1000

	start_time = datetime.now()

	for currIter in range(1,maxIter):
		newpop = np.zeros((popSize,dimension))
		# intermediate = np.zeros((popSize,dimension))
		fitList = allfit(population,trainX,trainy,testX,testy)
		arr1inds = fitList.argsort()
		population = population[arr1inds]
		fitList= fitList[arr1inds]
# 		print(fitList)

# 		for i in range(popSize):
# 			print('here',i,fitList[i])
# 		print('sum:',fitList.sum())
# 		if currIter==1:
# 			y_axis.append(min(fitList))
# 		else:
# 			y_axis.append(min(min(fitList),y_axis[len(y_axis)-1]))
# 		x_axis.append(currIter)

		bestInx = np.argmin(fitList)
		fitBest = min(fitList)
		print(fitBest)
 		#print(population[bestInx])
		print(population[bestInx])
		if fitBest<GBESTFIT:
			GBESTSOL = population[bestInx].copy()
			GBESTFIT = fitBest

		for selectioncount in range(int(popSize/2)):
			parent1 =   selectParentRoulette(popSize,fitList)
			parent2 = parent1
			while parent2 == parent1:
				random.seed(time.time())
				# parent2 = random.randint(0,popSize-1)
				parent2 = selectParentRoulette(popSize,fitList)

				# print(parent2)
			# print('parents:',parent1,parent2)
			parent1 = population[parent1].copy()
			parent2 = population[parent2].copy()
			#cross over between parent1 and parent2
			child1 = parent1.copy()
			child2 = parent2.copy()
			for i in range(dimension):
				random.seed(time.time())
				if random.uniform(0,1)<crossoverprob:
					child1[i]=parent2[i]
					child2[i]=parent1[i]
			i = selectioncount
			j = int(i+(popSize/2))
			# print(i,j)
			newpop[i]=child1.copy()
			newpop[j]=child2.copy()

		#mutation
		mutationprob = muprobmin + (muprobmax - muprobmin)*(currIter/maxIter)
		for index in range(popSize):
			for i in range(dimension):
				random.seed(time.time()+dimension+popSize)
				if random.uniform(0,1)<mutationprob:
					newpop[index][i]= 1- newpop[index][i]
		# for i in range(popSize):
			# print('before:',newpop[i].sum(),fitList[i])
			# newpop[i],fitList[i] = adaptiveBeta(newpop[i],fitList[i],trainX,trainy,testX,testy)
			# newpop[i],fitList[i] = deepcopy(mutation(newpop[i],fitList[i],trainX,trainy,testX,testy))
			# print('after:',newpop[i].sum(),fitList[i])

		population = newpop.copy()
# 	pyplot.plot(x_axis,y_axis)
# 	pyplot.show()

	#test accuracy
	cols = np.flatnonzero(GBESTSOL)
	val = 1
	if np.shape(cols)[0]==0:
		return GBESTSOL
# 	clf = KNeighborsClassifier(n_neighbors=5)
    #clf = RandomForestClassifier(n_estimators=300)
    #clf = MLPClassifier(random_state=1, max_iter=300)
    #clf = DecisionTreeClassifier()
	clf = RandomForestClassifier(n_estimators=300)
	train_data = trainX[:,cols]
	test_data = testX[:,cols]
	clf.fit(train_data,trainy)
	val = clf.score(test_data,testy)
    
	return GBESTSOL,val,fitBest

popSize = 10
maxIter = 31
omega = 0.85
alpha = 0.8
beta = 0.2
crossoverprob = 0.6
muprobmin = 0.01
muprobmax = 0.3
datasetList = ["csv"]

randomstateList=[15,5,15,26,12,7,10,8,37,19,35,2,49,26,1,25,47,12]

for datasetinx in range(len(datasetList)): #
	dataset=datasetList[datasetinx]
	best_accuracy = -100
	best_no_features = 100
	best_answer = []
	accuList = []
	featList = []
	for count in range(10):
		if (dataset == "WaveformEW" or dataset == "KrvskpEW") : #and count>2 :
			break
		print(count)
		answer,testAcc = geneticAlgo(""+dataset+"",popSize,maxIter,randomstateList[datasetinx])
		print(testAcc,answer.sum())
		accuList.append(testAcc)
		featList.append(answer.sum())
		if testAcc>=best_accuracy and answer.sum()<best_no_features:
			best_accuracy = testAcc
			best_no_features = answer.sum()
			best_answer = answer.copy()
		if testAcc>best_accuracy:
			best_accuracy = testAcc
			best_no_features = answer.sum()
			best_answer = answer.copy()



	print(dataset,"best:",best_accuracy,best_no_features)
	#inx = np.argmax(accuList)
    #best_accuracy = accuList[inx]
	#best_no_features = featList[inx]
	#print(dataset,"best:",accuList[inx],featList[inx])
	#with open("result_GA.csv","a") as f:
	#	print(dataset,"%.2f"%(100*best_accuracy),best_no_features,sep=',',file=f)
	#with open("result_SMOXarrayA.csv","a") as f:
	#    print(dataset,end=',',file=f)
    #for i in accuList:
	#    print("%.2f"%(100*i),end=',',file=f)
    #print('',file=f)

	#with open("result_SMOXarrayF.csv","a") as f:
    #	print(dataset,end=',',file=f)
    
	#for i in featList:
	#print('',file=f)



print(dataset,"best:",best_accuracy,best_no_features)
fig = plt.figure(figsize=(6,4))
Iter = [i for i in range(10)]
plt.plot(label = 'BGA', color = 'red')

plt.title('dataset',fontsize =14, color = 'navy')
plt.ylabel('FitnList',fontsize=14)
plt.xlabel('#Iteration',fontsize=14)
plt.legend()
plt.show()