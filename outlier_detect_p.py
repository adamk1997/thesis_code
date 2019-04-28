#Outlier Detection!

import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, mixture,metrics
#from kernel_ref import KernelKMeans as ker_ref
import time
import matplotlib.patches as mpatches
import csv
from kkmeans import kkmeans, rbf, make_kernel
import functools as ft
import multiprocessing as mp


def test_ucirv(attacks,benign_mod=None):

	np.random.seed(10)

	train_size= 1000  #Number of training samples from normal data
	test_size = 1000 #Amount of data being tested from testing data attack file 
	kclust = 4 #number of clusters assumed in data
	param_size = 15
	
	sets = len(attacks)
	
	size = train_size
	sizes = [train_size+test_size,test_size] #array of data needed from normal 3nd attack data
	true = np.zeros((train_size,)) # placeholder true labels array for training dataset
	true_param = true[0:param_size] # placeholder true labels array for parameter training dataset
	true[(4*size-1):(sets*size-1)] = 1 
	row_set = np.array([0,1,2,24,25,26,41,42,43,58,59,60]) #columns from data used as features
	
############################################################################################

	with open('thesis/traffic_1.csv', newline='') as csvfile:
		traffic1 = csv.reader(csvfile)
		for row in traffic1:
			### REMOVE These next two lines FOR ROW specification TO WORK
			row_set=np.array(range(len(row)))
			data_width = len(row_set)
			break
#############################################################################################
			

	data_width = len(row_set)
		
	data = np.zeros((sum(sizes),data_width))
	
	
	sizessum=0

	#reads in the data. 
	for i in range(0,sets):
		

		filename= "thesis/"+ str(attacks[i])+".csv"
		
		with open(filename, newline='') as csvfile:
			traffic1 = csv.reader(csvfile)
			count = 0
			rand_count = 1+np.random.randint(0,5)
			real_count = 0 
			for row in traffic1:
				
				if count < rand_count:
					count=count+1
					pass
				else:
					real_count = real_count+1
					#add the commented part to randomize data selection. 
					rand_count = rand_count + 1#np.random.randint(0,5)
					count2 = 0
					count_subset = 0
					for element in row:
						if count2 == row_set[count_subset]:
							data[real_count+sizessum,count_subset] = float(element)
							count_subset=count_subset+1
							if count_subset == len(row_set):
								break
						count2=count2+1
					count = count+1
					if count>=sizes[i]:
						break
		sizessum=sizessum+sizes[i]

	# obtain training data, testing data, and true sets from the read in data
	train_data = data[0:train_size,:]
	param_data = train_data[0:param_size,]
	test_data = data[train_size:(2*test_size+train_size),:]
	true = np.zeros(2*test_size,)
	true[0:test_size] = 1
	tol_n = 15
	#if there is no existing benign model, train one.
	if benign_mod is None:
		#make euclidean distances for parameter range, and create parameter range
		K = make_kernel(test_data.T,"euclid")
		
		Kmin = np.where(K==0,math.inf,K)
		tol_low = -np.log10(np.max(K))
		tol_high = np.log10(np.max(K))*4
		tol_range_em = np.logspace(tol_low,tol_high,num = tol_n, base = 10)
		true_train = np.zeros(train_size,)
		max_ami = -math.inf
		#initializes arrays for scores
		ami_scores = np.zeros(tol_n,)
		dist_scores = np.zeros(tol_n,)
		clust_scores = np.zeros(tol_n,)
		ami_comps = np.zeros(tol_n-1,)
		count = 0 
		#set tol_range to best possible for this dataset.
		#tol_range_em = np.array([10**(26)])
		#iterates over each tol_range and computes the value of model, and the minmax score for the model
		for i in tol_range_em:
			print("I :", 1/i)
			st = time.time()
			for j in range(0,1):
				benign_mod = kkmeans(test_data.T,2,"rbf",init_type = None,true = true, gamma = 1/i)
				
				ami = benign_mod.ami
				ami_scores[count] = ami
				benign_mod = kkmeans(param_data.T,kclust,"rbf",init_type = None,true = true_param, gamma = 1/i)
				assignments = benign_mod.assignments
				if count == 0:
					assign_orig = assignments

				else:
					ami_comps[count-1] = metrics.adjusted_mutual_info_score(assignments,assign_orig)
				
				Klin = make_kernel(param_data.T,"linear")
				ami = minmax_score(Klin,assignments,kclust)
				#normalize the cluster score heuristic
				aCS = np.abs(ami)
				if aCS  <10**(-10):
					nrm_cs = 0
				else:
					nrm_cs = 1- aCS/(10**round(np.log10(aCS)))
				clust_scores[count] = ami
				
			count = count+1
			if ami>max_ami and abs(ami) >10**(-10) :
				max_ami = ami
				gamma = 1/i
				
		#compute tol_range and benign model based on results for the best gamma
		gamma = 1/tol_range_em[0]
		tol_low = np.min(Kmin)
		tol_high = np.max(K)


		tol_range_em = np.linspace(0,tol_high*10,num = tol_n) #, base = 10)
		
		benign_mod = kkmeans(train_data.T,kclust,"rbf",init_type = None,true = true_train, gamma = gamma)
		assignments = benign_mod.assignments
		ami = benign_mod.ami

	else:
		assignments = benign_mod.assignments
		ami = benign_mod.ami
		gamma = benign_mod.gamma
		K = make_kernel(train_data.T,"rbf",gamma)

		tol_range_keep = np.linspace(0,np.max(K),tol_n)
	#holds clust norms to reduce computation time in outlier detections	
	clust_norms = benign_mod.clust_norms
	
	#Compute tol_range, and the roc_curve object, so that it displays a graph. All graphs are shown at end of all comparisons
	K = make_kernel(train_data.T,"rbf",gamma)
	
	tol_low = 0
	tol_high = np.max(K)

	tol_range_keep = np.linspace(0,tol_high,num = tol_n)#,base=10)
	outlier_kk_roc = ft.partial(outlier_kk, assignments = assignments,data = train_data.T,K=K,gamma=gamma,kclust=kclust,cn = benign_mod.clust_norms)
	
	rratio_k_1 = roc_curve(outlier_kk_roc,test_data,test_size,test_size,tol_range_keep,'g',len(tol_range_keep),1)
	
	#returns the benign model used, and the false rate and true rate so they can be stored as a whole for all comparisons

	return benign_mod,rratio_k_1.false_rate,rratio_k_1.true_rate





	










# outlier metric for kernel k-means model
def outlier_kk(point,assignments,data,K,gamma,kclust,tol,cn=None):
	mindist = math.inf
	minclust = -1
	for i in range(kclust):
		dist = dist_euclid(point,i,assignments,data,K,gamma,cn)

		if dist< mindist:
			mindist = dist
			minclust = i
	return tol-mindist

# outlier metric for EM model
def outlier_em(point,means,covs,tol):
	kclust = means.shape[0]
	mindist = math.inf
	minclust = -1
	for i in range(kclust):
		diffs = np.subtract(point,means[i,:])
		covarmat = np.eye(len(diffs))*covs[i]
		invcov = np.linalg.solve(covarmat,diffs)
		dist = np.sqrt(np.dot(diffs.T,invcov))

		if dist< mindist:
			mindist = dist
			minclust = i
	return tol-mindist

def dist_euclid(point,j,assignments,data,K,gamma,cn =None):


	clust_size = np.sum(np.where(assignments == j,1,0))
	
	out_dist = 0
	clust_ind = (assignments == j)
	out_time = time.time()
	for i in range(len(assignments)):
		if clust_ind[i]:
			out_dist = out_dist + rbf(point,data[:,i],gamma = gamma)
	

	if cn is None:
		Kclust = K[clust_ind,:]
		cn = np.sum(Kclust[:,clust_ind])/clust_size/clust_size
	else:
		cn = cn[j]
	
	dist = rbf(point,point,gamma = gamma)-2*out_dist/clust_size+cn

	if dist < -(10**(-10)):
		print("DIST", dist)
		print("OUTDIST",2*out_dist/clust_size)
		print("CN+ K(point,point)", rbf(point,point,gamma = gamma)+cn)
		print("NEEDED TO STOP")



	return dist


#claculates ROC curve with given outlier metric. 
class roc_curve:
	def __init__(self,outlier_fun,data,false_num,true_num,tol_range=None,c='r',size = 50,win = 1,show= True):
		if tol_range is None:
			K = make_kernel(data.T,"euclid")
			Kmin = np.where(K==0,math.inf,K)
			tol_low = np.min(Kmin)
			tol_high = np.max(K)
			tol_range = np.linspace(tol_low,tol_high,num = size)
		
		false_rates = [0]*size
		true_rates = [0]*size
		count = 0
		assert(data.shape[0] == false_num+true_num)
		tol_time = time.time()
		for i in tol_range:
			
			tol = i
			sum_false=0
			sum_true=0
			countj = 0
			win_count = 0
			anom_win = 0
			anoms = [0]*data.shape[0]
			minsubtols = [0]*data.shape[0]
			data_time = time.time()
			#iterates through all training data to produce the deisred results for rates on this tolerance
			for j in range(data.shape[0]):
				minsubtol = outlier_fun(data[countj,:],tol = tol)
				minsubtols[j] = minsubtol
				if tol == 0 and minsubtol>0:
					print("STOP", minsubtol)
					#exit(0)

				if minsubtol<=math.pow(10,-10) and j < false_num:
					sum_false = sum_false + 1
					anoms[countj] = 1
					
				elif minsubtol <=math.pow(10,-10) and j>=false_num:
					sum_true = sum_true + 1
					anoms[countj] = 1
				countj = countj+1
				if countj>=win:
					if np.sum(anoms[(countj-win):(countj+1)]) > (win/2):
						anom_win = anom_win+1
						
			print("data time", time.time()-data_time)

				


			if false_num != 0:
				falses = sum_false/false_num
			else:
				falses = 0
			if true_num != 0:
				trues = sum_true/true_num
			else:
				trues = 0
			false_rates[count] = falses
			true_rates[count] = trues
			count = count+1
		print("time to go through entire tol_range",time.time()-tol_time)
		#plot the rates in ROC curve if desired.
		if show:
			plt.plot(false_rates,true_rates,c=c)
			plt.ylabel("True Positive Rates")
			plt.xlabel("False Positive Rates")
		#store false rates as object
		self.false_rate = false_rates
		self.true_rate = true_rates
		ratio_rates = np.subtract(true_rates,false_rates)
		best_ind = np.argmax(ratio_rates)

		ob_size = false_num+true_num
		
		#store other interesting sliding window statistics.
		self.ratio_k = np.max(ratio_rates)
		self.anom_win = anom_win
		self.best_tol = tol_range[best_ind]
		self.win_num = (ob_size-win+1)/2
		
		if self.anom_win > (self.win_num):
			self.anomalous = True
		else:
			self.anomalous = False

		return 

#Calculates heuristic for optimal gamma parameter

def minmax_score(K,assignments,kclust):
	tot = 0
	dist_mat = dist_matrix(K)
	for i in range(kclust):
		clust_ind = (assignments == i)
		if np.sum(clust_ind) == 0:
			return -math.inf
		#calculate minimum for each cluster
		Kclust = dist_mat[clust_ind,:][:,clust_ind]
		#print("KClust index", Kclust.shape)
		mins = np.min(Kclust,axis=0)
		intra_max_min = np.mean(mins)
		tot = tot + intra_max_min
	avg_intra = tot/kclust
	tot = 0
	for i in range(kclust):
		clust_not = (assignments != i)
		clust_ind = (assignments == i)
		Kclust = dist_mat[clust_ind,:][:,clust_not]
		small_dist = np.mean(np.min(Kclust,axis = 1))
		tot = tot+small_dist
	avg_inter = tot/kclust
	return avg_inter-avg_intra

#computes euclidean distance matrix based on gram matrix of some sort.
#For linear distances this just does Euclidean distances
def dist_matrix(K):

	dist_mat=np.multiply(-2,K)
	Kvert = K[:,:]
	Khorz = K[:,:]
	for i in range(K.shape[0]):
		dist_mat[i,i] = math.inf
		Khorz[i,:] = (K[i,i]*K.shape[0])
		Kvert[:,i] = (K[i,i]*K.shape[0]).T

	dist_mat = dist_mat+Khorz+Kvert
	return dist_mat

