
import math
import numpy as np
from sklearn import cluster,metrics
from sklearn.metrics.pairwise import pairwise_kernels
class kkmeans:
	def __init__(self,X,k,dist_type,init_type="random",true = None,gamma = None):
	
		n = X.shape[1] #number of samples in data we are clustering
		kclust  = k #number of clusters
		K = make_kernel(X,dist_type,gamma=gamma) #creates distance matrix with given kernel and
		MAX_ITER = 100 # Max number of iterations performed by algorithm
		tol = 1e-5 # tolerance that stops the algorithm
		

		
		#random initialization option
		if init_type == "random":
			
			assignments = np.random.choice(range(0,kclust),n)

		#kmeans++ initialization option
		else:
			
			assignments = np.zeros((n,))

			assignments = cluster.KMeans(kclust, init='k-means++', max_iter=1).fit(X.T).labels_
			
		outlier_metric = np.zeros((n,))
		
		tot_dist = -1 #holds old total distance for comparison
		tot_dist_new = -100 #stores new distance for comparison
		count = 0  #stores number fo iterations of algorithm
		base_assignments = assignments.copy() #holds initial assignments while assignments changes
		
		#runs either till the algorithm reaches maximum iterations or total
		center_norm = np.zeros(kclust,)
		while abs(tot_dist-tot_dist_new)>tol and count<MAX_ITER:
			
			#increments count, sets total distance from centers and distances^2 to 0, sets new assignments
			count = count+1
			tot_dist = tot_dist_new
			tot_dist_new = 0
			tot_dist_sq = 0

			# calculates the cluster inner products for this iteration
			for j in range(0,kclust):
				clust_ind = (base_assignments == j)
				clust_size = np.sum(np.sum(np.where(base_assignments == j,1,0)))
				Kclust = K[clust_ind,:]	
				center_norm[j] = np.sum(Kclust[:,clust_ind])/clust_size/clust_size

			# calulates new assingments
			for i in range(0,len(base_assignments)):
				outlier_metric[i] = 0
				min_dist = math.inf
				min_center = -1

				#calculates each distance
				for j in range(0,kclust):
					dist = dist_euclid(i,j,base_assignments,K,cn = center_norm[j])
					if dist <= min_dist:
						min_dist = dist
						min_center = j

				tot_dist_new = tot_dist_new + dist
				tot_dist_sq = tot_dist_sq + dist**2
				
				
				outlier_metric[i] = min_dist
				
				assignments[i] = min_center

			base_assignments = assignments.copy()

		#calculates new clusters
		for j in range(kclust):
			clust_ind = (base_assignments == j)
			clust_size = np.sum(np.sum(np.where(base_assignments == j,1,0)))
			Kclust = K[clust_ind,:]	
			center_norm[j] = np.sum(Kclust[:,clust_ind])/clust_size/clust_size

		#assign results of the clustering algorithm so they can be accessed
		self.assignments = base_assignments
		self.ami = metrics.adjusted_mutual_info_score(assignments,true)
		self.dist = tot_dist_new
		self.dist2 = tot_dist_sq
		self.gamma = gamma
		self.clust_norms = center_norm

		return  

# calculates euclidean distance between two points in the explicit mapping using Kernel matrix
def dist_euclid(i,j,assignments,K, cn = None):
	clust_size = np.sum(np.sum(np.where(assignments == j,1,0)))
	clust_ind = (assignments == j)

	if cn is None:
		Kclust = K[clust_ind,:]
		cn = np.sum(Kclust[:,clust_ind])/clust_size/clust_size

	outlier_dist = np.sum(np.where(assignments == j,K[i,:],0))/clust_size
		
	
	dist = K[i,i]-2*outlier_dist+cn
	return dist#,cn

#computes different kernel matrices, depending on distance metric
def make_kernel(X,dist_type,gamma=None):
	n = X.shape[1]
	if dist_type == "linear":
		f = lin_dist
		gamma = None
	elif dist_type == "rbf":

		K = pairwise_kernels(X.T,metric = 'rbf',gamma = gamma)
		return K
	elif dist_type == "euclid":
		f = euclid
		gamma = None
	K = np.zeros((n,n))
	
	for i in range(n):
		for j in range(i+1):
			if gamma is not None:
				val = f(X[:,i],X[:,j],gamma)
			else:
				val = f(X[:,i],X[:,j])
			K[i,j] = val
			K[j,i] = val
	return K

#calculates Euclidean distance
def euclid(x,y):

	diff = np.subtract(x,y)
	dist = np.sum(np.sqrt(np.multiply(diff,diff)))
	return dist

#calculates dot product for linear Kernel matrix.
def lin_dist(x,y):

	dist = np.dot(x.T,y)
	return dist

#calculates rbf Kernel function
def rbf(x,y,gamma=50):
	dist = math.pow(np.linalg.norm(np.subtract(x,y),2),2)
	dist = np.exp(-gamma*dist)
	return dist

