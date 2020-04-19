# roc curve and auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# precision-recall curve and f1
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc

# Matplotlib
from matplotlib import pyplot as plt

# Pytorch
import torch
from torch.utils.data import DataLoader

# Numpy
import numpy as np

# Custom
from Model import SiameseConvNet, distance_metric
from Dataset import TestDataset

def calculate_results(predictions, labels):
    
    treshold_max = np.max(predictions)
    treshold_min = np.min(predictions)
    P = np.sum(labels == 1)
    N = np.sum(labels == 0)
    step = 0.001

    TPR_full = []
    FPR_full = []
    Precision_full = []
    Recall_full = []
    print('*****************************************************************************')
    for treshold in np.arange(treshold_min, treshold_max + step, step):
        
        print(f'Treshold = {treshold:.4f}')
        idx1 = predictions.ravel() <= treshold
        idx2 = predictions.ravel() > treshold

        TP = np.sum(labels[idx1] == 1)
        FN = P - TP             
        TN = np.sum(labels[idx2] == 0)
        FP = N - TN
        print(f'TP = {TP:.0f}, FN = {FN:.0f}, TN = {TN:.0f}, FP = {FP:.0f}')
        
        # roc curve
        TPR = float(TP/P)
        TPR_full.append(TPR)
        TNR = float(TN/N)
        FPR = 1-TNR
        FPR_full.append(FPR)
        ROC_ACC = (TP + TN)/(P + N)
        print(f'TPR = {TPR:.4f}, FPR = {FPR:.4f}, ROC_ACC = {ROC_ACC:.4f}')
        
        # precision-recall curve and F1 score
        if TP>0:
            Precision = float(TP / (TP + FP))
            Precision_full.append(Precision)
            Recall = float(TP / (TP + FN))
            Recall_full.append(Recall)
            F1_score = (2*Precision*Recall) / (Precision+Recall)
            print(f'Precision = {Precision:.4f}, Recall = {Recall:.4f}, F1_score = {F1_score:.4f}')

    return TPR_full, FPR_full, Precision_full, Recall_full

if __name__ == "__main__":
    
    # version control
    print('Torch', torch.__version__, 'CUDA', torch.version.cuda)
    
    # device config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device = " + str(device)) 

    # Load model
    model = SiameseConvNet().to(device)
    model.load_state_dict(torch.load(open('./Models/checkpoint_epoch_20', 'rb'), map_location=device))
    model.eval()
    
    # Prepare test data
    test_dataset = TestDataset()
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Calculate accuracy
    with torch.no_grad():
        
        total_labels = []
        total_dist =  []
        for index, (images1, images2, labels) in enumerate(test_loader):
            
            images1 = images1.to(device)
            images2 = images2.to(device)
            features1, features2 = model.forward(images1, images2)
            
            temp1 = distance_metric(features1, features2).to('cpu').numpy()
            total_dist = np.concatenate([total_dist, temp1])
            
            temp2 = labels.to('cpu').numpy()
            total_labels = np.concatenate([total_labels, temp2]).astype(int)
            
        print("total dist = " + str(len(total_dist)))
        print("total labels = " + str(len(total_labels)))
        
        TPR, FPR, Precision, Recall = calculate_results(total_dist, total_labels)
        #print(f'Total Accuracy = {accuracy:.4f}, d {d:.4f}')
    
        plt.subplot(1, 2, 1)
        # plot the roc curve for the model
        plt.plot(FPR, TPR, marker='.', label='ROC (TPR-FPR)')
        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid()
        # show the legend
        plt.legend()
        
        plt.subplot(1, 2, 2)
        # plot the Precision-Recall curve for the model
        plt.plot(Recall, Precision, marker='.', label='Precision-Recall')
        # axis labels
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid()
        # show the legend
        plt.legend()
        
        # save and show plot
        plt.savefig('Test Results.png')
        plt.show()
        
        
        


