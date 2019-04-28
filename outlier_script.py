from outlier_detect import test_ucirv
import matplotlib.pyplot as plt
import time
import numpy as np
if __name__=='__main__':
	st = time.time()
	# all attack and device names
	sub_attacks = ["/udp","/syn", "/ack", "/scan","/udpplain","/junk", "/scan","/udp","/tcp"]
	devices = ["Danmini_Doorbell","Ecobee_Thermostat","Ennio_Doorbell","Philips_B120N10_Baby_Monitor","Provision_PT_737E_Security_Camera","Provision_PT_838_Security_Camera","Samsung_SNH_1011_N_Webcam","SimpleHome_XCS7_1002_WHT_Security_Camera","SimpleHome_XCS7_1003_WHT_Security_Camera"]
	tol_n = 15 # sets number of tolerances to be tested over. Needs to be same in other file for assignments to work
	false_rates = np.zeros((71,tol_n)) #initializes false rates matrix.
	true_rates = np.zeros((71,tol_n)) #initializes true rates matrix.
	count = 0
	for i in range(9):
		for j in range(9):
			# continues for missing data, so algorithm does not stop. Mirai for device 2 and 6 do not exist
			if j < 5 and i == 2:
				continue
			if j < 5 and i == 6:
				continue

			# based on iteration number, decides which device and attack to compare.
			botnet = "mirai_attacks"+str(i)+sub_attacks[j]
			if j >=5:
				botnet = "gafgyt_attacks"+str(i)+sub_attacks[j]
			benign = "traffic_"+str(i)
			attacks = [benign,botnet]
			print("USING THESE ::", attacks, devices[i])
			# either calculates benign model or uses benign model if the device has been compared before. 
			# calculation of benign_model is independent of attacks.
			if j == 0:
				benign_model,false_rates_1,true_rates_1 = test_ucirv(attacks)
			else:
				benign_model,false_rates_1,true_rates_1 = test_ucirv(attacks,benign_mod = benign_model)
			#stores false and true rates from this iteration in larger rates matrix.

			false_rates[count,:] = false_rates_1
			true_rates[count,:] = true_rates_1
			count = count+1
			print("time elapsed so far at iter "+str(count), time.time()-st)
	# stores the rates in 2 files
	np.savetxt("fr.csv", false_rates, delimiter=",")
	np.savetxt("tr.csv", true_rates, delimiter=",")

	#show the full graph of data.
	plt.show()
	print("time elapsed ", time.time()-st)