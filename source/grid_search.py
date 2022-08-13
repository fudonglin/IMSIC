import experiment as exp
import params
import time
import numpy as np


l_con_list = np.linspace(0.1, 1, 10)  # find the optimal l_con, can change based on the machine state
c_con_list = range(4, 5) # modify graph to go higher

for c_con in c_con_list:
    for l_con in l_con_list:
        print("Experiment | c: %f | lambda (c): %f" % (c_con, l_con))
        t0 = time.time()
        params.lambda_con = l_con
        params.opt.code_dim = c_con
        exp.experiment()
        t1 = time.time()
        print("\t Duration: " + str(t1-t0))

print("-Done-")
