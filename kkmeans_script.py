from kkmeans_tests import big_test
import numpy as np
import matplotlib.pyplot as plt
import time
import csv

#MASTERLIST OF ATTACKS - "syn", "ack", "junk_g", "scan_g","udp_g","udp","tcp_g","scan","udpplain"
# URL where data can be found. file names need to be replicated as well, benign traffic is 
#traffic_0-traffic_8, and gafgyt or mirai_attacks0-mirai_attacks8 and mirai_attacks8
def big_test_run(attacks, trials = 2,sets = 2,data=500):
	
	datasize = np.zeros((sets,1))
	data_ami = np.zeros((sets,1))
	for j in range(0,sets):
		sum_ami = 0
		data = data#+0
		
		

		gamma=None

		for i in range(0,trials):
			if gamma is None:
				gamma,values=big_test(attacks,num_samples = data) #,ref_values,comp_values
				
			else:
				gamma1,values1= big_test(attacks,num_samples = data) #, ref_values1,comp_values1 
				gamma = np.add(gamma,gamma1)
				values = np.add(values,values1)
				
			
		gamma = np.divide(gamma,trials)
		values = np.divide(values,trials)
		
	return np.max(gamma),np.max(values)


st = time.time()
sub_attacks = ["/udp","/syn", "/ack", "/scan","/udpplain","/junk", "/scan","/udp","/tcp"]
attack_num = np.array(range(9))
devices = ["Danmini_Doorbell","Ecobee_Thermostat","Ennio_Doorbell","Philips_B120N10_Baby_Monitor","Provision_PT_737E_Security_Camera","Provision_PT_838_Security_Camera","Samsung_SNH_1011_N_Webcam","SimpleHome_XCS7_1002_WHT_Security_Camera","SimpleHome_XCS7_1003_WHT_Security_Camera"]
device_num = np.array(range(9))
gammas = np.zeros(71,)
values = np.zeros(71,)
device_ind = np.zeros(71,)
attack_ind = np.zeros(71,)
count = 0

# iterates over all different sets of device combinations and prints the results. 

# commented out code produces the same results for devices and attack combos
for i in range(9):
	#####
	for j in range(i+1,9):
		if i==j:
			skip

		######
		#remove and replace with
		'''for j in range(9):
			# continues for missing data, so algorithm does not stop. Mirai for device 2 and 6 do not exist
			if j < 5 and i == 2:
				continue
			if j < 5 and i == 6:
				continue'''
		
		benign = "traffic_"+str(i)
		####
		attack = "traffic_"+str(j)
		attacks = [benign,benign_j]
		####
		#replace with 
		'''botnet = "mirai_attacks"+str(i)+sub_attacks[j]
		if j >=5:
			botnet = "gafgyt_attacks"+str(i)+sub_attacks[j]
		attacks = [benign,botnet]'''
		print("USING THESE ::", devices[j], devices[i])
		device_ind[count] = device_num[i] 
		attack_ind[count] = attack_num[j]
		start_kk = time.time()
		gammas[count],values[count] = big_test_run(attacks)
		
		count = count+1
		

# saves all results in current directory
np.savetxt("AMIvalues.csv", values, delimiter=",")
np.savetxt("gamma.csv", gammas, delimiter=",")
np.savetxt("device_nums.csv", device_ind, delimiter=",")
np.savetxt("attack_nums.csv", attack_ind, delimiter=",")
print("time elapsed ", time.time()-st)
		

