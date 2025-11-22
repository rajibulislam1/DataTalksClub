# Load The model
import pickle
output = 'logisticmodel.bin'

with open(output, 'rb') as f_in:
    dv, log = pickle.load(f_in)




new_data = {
    'parents':'usual', 
    'has_nurs':'proper', 
    'form':'completed',
    'children':3,
    'housing':'convenient', 
    'finance':'convenient',
    'social':'slightly_prob', 
    'health':'priority'
}


X = dv.transform([new_data])


y_pred = log.predict_proba(X)[0, 1]

print('input', new_data)
print('Class recommended',y_pred)
