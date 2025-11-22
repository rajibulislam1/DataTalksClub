# Load The model
import pickle
output = 'logisticmodel.bin'

with open(output, 'wb') as f_out:
    pickle.dump((dv, log), f_out)
