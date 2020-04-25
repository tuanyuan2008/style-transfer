import pickle

losses = []
with open('dg_losses.txt', 'r') as f:
    for line in f:
        losses.append(float(line[15:23]))
f.close()

with open('dg_losses.data', 'wb') as g:
    # store the data as binary data stream
    pickle.dump(losses, g)
g.close()

losses = []
with open('drg_losses.txt', 'r') as f:
    for line in f:
        losses.append(float(line[15:23]))
f.close()

with open('drg_losses.data', 'wb') as g:
    # store the data as binary data stream
    pickle.dump(losses, g)
g.close()

losses = []
with open('bert_class_losses.txt', 'r') as f:
    for line in f:
        losses.append(float(line[15:23]))
f.close()

with open('bert_class_losses.data', 'wb') as g:
    # store the data as binary data stream
    pickle.dump(losses, g)
g.close()