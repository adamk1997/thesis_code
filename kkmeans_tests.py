import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture, metrics

import time
import matplotlib.patches as mpatches
import csv
from kkmeans import kkmeans, make_kernel




def big_test(attacks,num_samples = 100):

	# read in files form working directory. Assumes files from insert URL here have been
	#downloaded into thesis
	sets = len(attacks)
	clusts = len(attacks)
	size = num_samples
	true = np.zeros((sets*size,))
	with open('thesis/traffic_1.csv', newline='') as csvfile:
		traffic1 = csv.reader(csvfile)
		for row in traffic1:
			### REMOVE THese next two lines FOR ROW specification TO WORK
			row_set=np.array(range(len(row)))
			data_width = len(row_set)
			break
#############################################################################################
			

	
	
		
	data = np.zeros((sets*size,data_width))
	
	#read in the number of samples for each dataset being compared.
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
					rand_count = rand_count + np.random.randint(0,5)
					count2 = 0
					count_subset = 0
					for element in row:
						if count2 == row_set[count_subset]:
							data[real_count+size*i,count_subset] = float(element)
							count_subset=count_subset+1
							if count_subset >= len(row_set):
								break
						count2=count2+1
					count = count+1
					if count>=size:
						break
		true[(i*size):(size*(i+1))]=np.multiply(i,np.ones(size))	
	
		true[(i*size):(size*(i+1))]=np.multiply(i,np.ones(size))
	
	#makes kernel matrix
	K = make_kernel(data.T,"euclid",)
	Kmin = np.where(K==0,math.inf,K)
	#calculates min and max for above diagonal in Kernel Matrix for gamma range tested
	N = K.shape[0]
	Ntri = int((N*N-N)/2)
	diff_dist = np.zeros((Ntri,))
	count = 0
	for i in range(0,N):
		for j in range(0,i):
			diff_dist[count] = K[i,j]
			count=count+1
	diff_dist = np.sort(diff_dist)
	Ntrismall = round(Ntri)
	
	small_distances = diff_dist[0:Ntrismall]
	

	indexnine = np.percentile(K,90)
	# These can be determined by either max and min with a multiplier, or just max 
	#and negative of max with multipliers
	min_val = -np.log10(np.max(small_distances))*1

	max_val = np.log10(np.max(small_distances))*4
	
	gammas = np.logspace(min_val,max_val,15)
	values = np.zeros((len(gammas),))
	

	#computes assignments for each tol_range
	count = 0
	trials = 2 
	# number of trials can be changed to produce average AMIs over didfferent random initializations
	for i in gammas:
		values[count] = None
		for j in range(trials):
			if j==0:
				# first clustering attempt
				clustering = kkmeans(data.T,clusts,'rbf',init_type=None,true = true, gamma = 1/i)
				assignments = clustering.assignments
				ami = clustering.ami

				values[count] = ami
				
			else:
				# subsequent clustering attempts
				clustering = kkmeans(data.T,clusts,'rbf',init_type=None,true = true, gamma = 1/i)
				assignments = clustering.assignments
				ami = clustering.ami
				vals = ami


				values[count] = values[count] + vals
		values[count] = values[count]/trials
		count = count+1
	return gammas, values 


