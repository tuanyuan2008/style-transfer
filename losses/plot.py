import matplotlib.pyplot as plt
import pickle

# with open('dg_losses.data', 'rb') as f:
#     # read the data as binary data stream
#     dg_losses = pickle.load(f)
#     plt.title('Delete-Generate Model Losses Per Iteration')
#     plt.xlabel('Iteration')
#     plt.ylabel('Loss')
#     plt.plot(dg_losses)
#     plt.savefig('dg_loss.png')

# f.close()

# with open('drg_losses.data', 'rb') as f:
#     # read the data as binary data stream
#     dg_losses = pickle.load(f)
#     plt.title('Delete-Retrieve-Generate Model Losses Per Iteration')
#     plt.xlabel('Iteration')
#     plt.ylabel('Loss')
#     plt.plot(dg_losses)
#     plt.savefig('drg_loss.png')

# f.close()

with open('bert_class_losses.data', 'rb') as f:
    # read the data as binary data stream
    dg_losses = pickle.load(f)
    plt.title('BERT Classifier Losses Per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.plot(dg_losses)
    plt.savefig('bert_class_loss.png')

f.close()