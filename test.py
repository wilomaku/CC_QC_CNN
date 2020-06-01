import random, time, json, cv2, glob

from tabulate import tabulate

import pandas as pd
import nibabel as nib
import numpy as np

import scipy.misc as scp_misc
import scipy.ndimage as ndimage

from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

try:
    from sklearn.utils.fixes import signature
except:
    from inspect import signature

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import datasets, models, transforms

from aux import func as dl

from PIL import Image

start = time.time()

# define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Load the pretrained model from pytorch
        self.ext_ftr = models.resnet18(pretrained=True)

        self.ext_ftr = nn.Sequential(*list(self.ext_ftr.children())[:-2])

        # Freeze training for all "features" layers
        for param in self.ext_ftr[:-1].parameters():
            param.requires_grad = False

        # Freeze training for all "features" layers
        for param in self.ext_ftr[-1][0].parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(18432, 1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.5)
        #self.fc3 = nn.Linear(512, 1)
        self.fc3 = nn.Linear(512, 2)
        #self.sigmout = nn.Sigmoid()

    def forward(self, x):

        #print(x.size())
        y = self.ext_ftr(x)
        y = y.view(-1, 18432)
        #print(y.size())
        y = self.dropout1(F.relu(self.fc1(y)))
        y = self.dropout2(F.relu(self.fc2(y)))
        return self.fc3(y)

def test(data_loader, model):
    print('Testing...')

    model.eval()

    for i, sample_batched in enumerate(data_loader):
        inputs = sample_batched['image'].type(torch.FloatTensor)
        targets = sample_batched['label'].type(torch.LongTensor)
        print(inputs.size(),targets.size())
        if device == 'cuda':
            inputs, targets = inputs.cuda(), targets.cuda()

        with torch.no_grad():
            outputs = model(Variable(inputs))
        #####################################3
        sftm = nn.Softmax(dim=1)
        sftm_out = sftm(outputs)
        #print(sftm_out)
        fpr, tpr, thresholds = metrics.roc_curve(targets, sftm_out[:,1])
        auc_vec = metrics.auc(fpr, tpr)
        print('Final AUC: ', auc_vec)

        #opt_th = 0.5
        opt_th = 0.397485576115
        print('Best separation threshold: {}'.format(opt_th))

        y_pred = sftm_out[:,1] > opt_th

        mx_conf = confusion_matrix(targets, y_pred)

        ind_err_rand = np.where(np.logical_xor(targets, y_pred))

        accuracy, recall, precision, f1 = dl.report_metrics(mx_conf)

        print('===== Final Report =====')
        print(tabulate([['AUC', auc_vec],
                        ['Accuracy', accuracy],
                        ['Recall', recall],
                        ['Precision', precision],
                        ['F1', f1],],
                       ['Metric', 'Value'], tablefmt='grid'))
        ##########################################
        _, predicted = outputs.topk(1, 1, True)

        test_acc = calculate_accuracy(outputs, targets)

        print('Acc {}'.format(test_acc))

        return sftm_out[:,1], y_pred

DIR_BAS = './dataset/'
DIR_SAVE = './saves/'

labeled_file = '{}output.csv'.format(DIR_SAVE)
all_labels = pd.read_csv(labeled_file,sep=',')
all_labels['Subject'] = all_labels['file'].str.split('/').apply(lambda x: x[-2][3:]).astype(int)
all_labels['Label'] = all_labels['true_label'].astype(int)

dirs_all = glob.glob('{}*'.format(DIR_BAS,))

dirs_all = [dir_all+'/' for dir_all in dirs_all]
print('Found dirs:',len(dirs_all))

ValsNormalize = {'mean': np.array([0.485, 0.456, 0.406]),'std': np.array([0.229, 0.224, 0.225])}

transformed_data_test = dl.MRIDataset_test(list_dirs=dirs_all, df_labels=all_labels,
                                           transform=transforms.Compose([dl.ToNormalize(ValsNormalize),
                                                                         dl.ToCrop(12),
                                                                         dl.ToResize((176,176)),
                                                                         dl.ToTensor(),
                                                                        ]))
test_loader = DataLoader(transformed_data_test, batch_size=len(dirs_all), shuffle=False)

model = Net()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
if device == 'cuda':
    model.cuda()

model_save = 'model_da_80.pth'
path_m = '{}{}'.format(DIR_SAVE,model_save)
model_dic = torch.load(path_m, map_location='cpu')#, map_location='cpu')

model.load_state_dict(model_dic['model_state_dict'])
probs, outs = test(test_loader, model)

out = pd.DataFrame([])
out['file'] = dirs_all
out['true_label'] = all_labels['Label']
out['QC_score'] = probs
out['output_label'] = outs

save_output = '{}output_cnn.csv'.format(temp_save)
out.to_csv(save_output,index=False)
print('Saved output at:', save_output)

end = time.time()
print('Total time: {}'.format(end - start))
