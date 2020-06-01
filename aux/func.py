import random, time, glob

from tabulate import tabulate

import pandas as pd
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

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

from PIL import Image

"""
Deep learning training and testing Module
"""
def crop_reg(path_masks):
    """Computes a bounding box containing all the masks"""
    crop_coord = np.array([[999,999,999],[0,0,0]])
    for folder in path_masks:
        try:
            file_T1 = '{}/T1.nii.gz'.format(folder)
            vol_T1 = nib.load(file_T1).get_data()

            vol_T1 = np.swapaxes(vol_T1, 0, -1)
            vol_T1 = np.swapaxes(vol_T1, 0, 1)

            tagfilename = '{}/mask_man.tag'.format(folder)
            tag = np.loadtxt(tagfilename, skiprows=4, comments=';')
            seg_mask_t1 = createMaskSegOr(vol_T1.shape, tag)

            img_base, mask = np.rot90(vol_T1), np.rot90(seg_mask_t1)

            nonzero_px = np.nonzero(mask)
            max_coord, min_coord = np.amax(nonzero_px,axis=1), np.amin(nonzero_px,axis=1)
            #print(min_coord, max_coord)
            crop_coord[0] = np.minimum(crop_coord[0],min_coord)
            crop_coord[1] = np.maximum(crop_coord[1],max_coord)
        except:
            pass

    return crop_coord

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vc = []

    def update(self, val, n=1):
        self.vc.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def createMaskSegOr(shape, file_tag):
    "Reconstruction of txt manual mask from annotated points"
    points = np.vstack([file_tag[:,:3]])

    pts_calculados  = np.zeros(file_tag.shape).astype('int')
    pts_calculados[:,0] = points[:,0] + (shape[2] / 2)
    pts_calculados[:,1] = points[:,1] + (shape[0] / 2)
    pts_calculados[:,2] = points[:,2] + (shape[1] / 2)

    esquerda_direita    = pts_calculados[:,0]
    anterior_posterior  = pts_calculados[:,1]
    inferior_superior   = pts_calculados[:,2]

    seg_mask = np.zeros(shape).astype('bool')
    seg_mask[anterior_posterior, inferior_superior, esquerda_direita] = True

    return seg_mask

class MRIDataset_list(Dataset):

    def __init__(self, list_dirs, df_labels, transform=None, stab_size = (240, 240)):
        """
        Args:
            root_dir (string): Directory with all the images.
            n_slice (int): Slice number. If -1, mid slice is passed.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.list_dirs = list_dirs
        self.df_labels = df_labels
        self.transform = transform
        self.stab_size = stab_size

    def __len__(self):
        return len(self.list_dirs)

    def __getitem__(self, idx):
        suj_folder = self.list_dirs[idx]
        pre_msp = 'msp_points_reg'
        msp_points_reg = glob.glob('{}{}.nii.gz'.format(suj_folder,pre_msp))
        gen_img_path = glob.glob('{}*_msp.nii'.format(suj_folder))
        if msp_points_reg != []:
            in_img_msp = nib.load(msp_points_reg[0]).get_data()
            msp = np.argmax(np.sum(np.sum(in_img_msp,axis=-1),axis=-1))
            gen_img = nib.load('{}t1_reg.nii.gz'.format(suj_folder)).get_data()[msp]
            gen_mask = nib.load('{}mask_reg.nii.gz'.format(suj_folder)).get_data()[msp]
        else:
            gen_img = nib.load(gen_img_path[0]).get_data()[::-2,::-2,0]
            gen_mask_path = glob.glob('{}*corrected.cc.nii'.format(suj_folder))[0]
            gen_mask = nib.load(gen_mask_path).get_data()[::-2,::-2,0]

        gen_mask = (gen_mask > 0.5).astype('uint8')

        if gen_img.shape == self.stab_size:
            pass
        elif (gen_img.shape == (224,224)) or (gen_img.shape == (256,256)):
            gen_img = np.array(Image.fromarray(gen_img).resize(self.stab_size, Image.ANTIALIAS))
            gen_mask = np.array(Image.fromarray(gen_mask).resize(self.stab_size, Image.ANTIALIAS))
        else:
            raise NameError('Image shape is not as expected')

        mask_smooth = gen_mask.copy().astype('float64')

        img_fore = gen_img*mask_smooth
        img_back = mask_smooth

        Subject = int(suj_folder.split('/')[-2][-6:])
        label = int(self.df_labels.loc[self.df_labels.Subject==Subject,'Label'])

        sample = {'image': np.stack((gen_img,img_fore,img_back),axis=0), 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

class MRIDataset_test(Dataset):

    def __init__(self, list_dirs, df_labels, transform=None, stab_size = (240, 240)):
        """
        Args:
            root_dir (string): Directory with all the images.
            n_slice (int): Slice number. If -1, mid slice is passed.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.list_dirs = list_dirs
        self.df_labels = df_labels
        self.transform = transform
        self.stab_size = stab_size

    def __len__(self):
        return len(self.list_dirs)

    def __getitem__(self, idx):
        suj_folder = self.list_dirs[idx]

        image_nii = glog.glob('{}*image*.nii.gz'.format(suj_folder))[0]
        gen_img = nib.load(image_nii).get_data()
        msp = int(gen_img.shape[0]/2)
        gen_img = gen_img[msp,::-1]
        mask_nii = glog.glob('{*}mask*.nii.gz'.format(suj_folder))[0]
        gen_mask = nib.load(mask_nii).get_data()
        gen_img = gen_img[::-1]

        gen_mask = (gen_mask > 0.5).astype('uint8')

        mask_smooth = gen_mask.copy().astype('float64')

        img_fore = gen_img*mask_smooth
        #img_back = gen_img*neg_mask_smooth
        img_back = mask_smooth

        Subject = int(suj_folder.split('/')[-2][-6:])
        label = int(self.df_labels.loc[self.df_labels.Subject==Subject,'Label'])

        sample = {'image': np.stack((gen_img,img_fore,img_back),axis=0), 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToResize(object):
    """Resize image"""
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        image_res = np.empty((image.shape[0],self.shape[0],self.shape[1])).astype(float)
        for chann in range(image.shape[0]):
            image_res[chann] = cv2.resize(image[chann].astype(float), dsize=self.shape, interpolation=cv2.INTER_CUBIC)

        return {'image': image_res,'label': label}

class ToTensor(object):
    """Numpy-arrays to Tensors."""
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        return {'image': torch.from_numpy(image).type(torch.float),'label': torch.tensor(label, dtype=torch.float)}

class ToNormalize(object):
    """Normalization between 0 and 1."""
    def __init__(self, vals2norm):
        self.mean = vals2norm['mean'].reshape(3,1,1)
        self.std = vals2norm['std'].reshape(3,1,1)

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        max_vals = image.max(axis=(1,2))
        image = image/max_vals.reshape(3,1,1)

        return {'image': (image - self.mean)/self.std,'label': label}

class ToRotate(object):
    """Apply  random rotation."""
    def __init__(self, lims2rot=2):
        if isinstance(lims2rot, tuple):
            self.min_angle = int(lims2rot[0]*10)
            self.max_angle = int(lims2rot[-1]*10)
        else:
            self.min_angle = int(lims2rot*-10)
            self.max_angle = int(lims2rot*10)

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        ang2rot = (random.randrange(self.min_angle, self.max_angle))/10.0

        for ch in range(image.shape[0]):
            image[ch] = Image.fromarray(image[ch]).rotate(ang2rot)

        return {'image': image,'label': label}

class ToCrop(object):
    """Crop image from center."""
    def __init__(self, bord=8):
        self.bord = bord

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        return {'image': image[:,self.bord:-self.bord,self.bord:-self.bord],'label': label}

class ToIntensity(object):
    """Apply brightness and contrast random transforms."""
    def __init__(self, brightness=0.05, contrast=0.02):
        self.brightness = brightness*100
        self.contrast = contrast*100

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        b2apply = (random.randrange(-self.brightness, self.brightness))/100.0
        c2apply = (random.randrange(1, self.contrast))/100.0
        return {'image': (image*c2apply + b2apply),'label': label}

def assert_dataset(list_folders, stab_size = (240, 240)):
    new_folders = []
    for suj_folder in list_folders:
        try:
            pre_msp = 'msp_points_reg'
            msp_points_reg = glob.glob('{}{}.nii.gz'.format(suj_folder,pre_msp))
            gen_img_path = glob.glob('{}*_msp.nii'.format(suj_folder))
            if msp_points_reg != []:
                in_img_msp = nib.load(msp_points_reg[0]).get_data()
                msp = np.argmax(np.sum(np.sum(in_img_msp,axis=-1),axis=-1))
                gen_img = nib.load('{}t1_reg.nii.gz'.format(suj_folder)).get_data()[msp]
            else:# gen_img_path != []:
                gen_img = nib.load(gen_img_path[0]).get_data()[::-2,::-2,0]

            shape_T1 = gen_img.shape

            if ((shape_T1 == (224,224)) or (shape_T1 == (256,256)) or (shape_T1 == stab_size)):
                new_folders.append(suj_folder)
            else:
                print(suj_folder,' != ', shape_T1)
        except Exception as e:
            print(suj_folder,' Problem: {}'.format(e))
    return new_folders

def transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"
    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0, 0.2, N+4)
    return mycmap

def plot_img_mask(img_stack, label, predict=None):
    "Plot image and mask side by side"
    img_base, img_mask = img_stack[0], img_stack[1]
    y, x = np.mgrid[0:img_base.shape[0], 0:img_base.shape[1]]

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,3*5))
    ax1.imshow(img_base,cmap='gray')
    ax2.imshow(img_base,cmap='gray')

    cb = ax2.contourf(x, y, (img_mask), 15, cmap=mycmap)
    ax1.set_axis_off()
    ax2.set_axis_off()
    if predict == None:
        plt.title('sample = {}'.format(label))
    else:
        plt.title('sample = {} / predicted = {}'.format(label,predict[0]))
    plt.show()

def plot_channels(img_stack, label, predict=None):
    "Plot image and mask side by side"
    img_base = img_stack[0]
    img_pond = img_stack[1]
    img_mask = img_stack[2]

    fig, axis = plt.subplots(1,3,figsize=(10,3*5))
    axis[0].imshow(img_base,cmap='gray')
    axis[1].imshow(img_pond,cmap='gray')
    axis[2].imshow(img_mask,cmap='gray')

    axis[0].set_axis_off()
    axis[1].set_axis_off()
    axis[2].set_axis_off()
    if predict == None:
        plt.title('sample = {}'.format(label))
    else:
        plt.title('sample = {} / predicted = {}'.format(label,predict[0]))
    plt.show()

def plot_matrix(cm, classes, normalize=False, title='Confusion matrix', fig_size=8, cmap=plt.cm.Blues, opt_bar=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(fig_size, fig_size))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    if opt_bar:
        plt.colorbar()
    tick_marks = np.arange(len(list(classes)))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if not opt_bar:
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], fmt), size=fig_size*3, ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")

    plt.show()

def report_metrics(cm):
    tp, tn = cm[1,1], cm[0,0]
    fp, fn = cm[0,1], cm[1,0]

    acc = (tp+tn)/np.sum(cm)
    rec = tp/(tp+fn)
    prec = tp/(tp+fp)
    f1 = 2*tp/(2*tp+fp+fn)

    return acc, rec, prec, f1

def calculate_accuracy(outputs, targets):
    _, pred = outputs.topk(1, 1, True)
    equality = pred == targets.view(*pred.shape)

    return torch.mean(equality.type(torch.FloatTensor)).item()

def train_epoch(tr_loader,vl_loader,model,criterion,optimizer,model_path='./model.pth'
                ,epochs=10,max_patience=5):
    epoch_time = AverageMeter()
    patience = 0
    try:
        print('Loading model...')
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_ld = checkpoint['epoch']
        tr_loss_ep = checkpoint['tr_loss']
        vl_loss_ep = checkpoint['vl_loss']
        min_loss = vl_loss_ep[-1]
        print('Ok!')
        print('Model trained by {} epochs, Training for +{} epochs'.format(epoch_ld,epochs))

    except:
        print('There is no checkpoint to be loaded')
        print('Training for {} epochs'.format(epochs))
        min_loss = np.inf
        epoch_ld = 0
        tr_loss_ep, vl_loss_ep = AverageMeter(), AverageMeter()
    for epoch in range(epoch_ld,epoch_ld+epochs):

        tr_loss, tr_accuracy = AverageMeter(), AverageMeter()
        vl_loss, vl_accuracy = AverageMeter(), AverageMeter()
        tr_accuracy, vl_accuracy = AverageMeter(), AverageMeter()

        end_time = time.time()
        for i, sample_batched in enumerate(tr_loader):
            inputs = sample_batched['image'].type(torch.FloatTensor)
            targets = sample_batched['label'].type(torch.LongTensor)
            if device == 'cuda':
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)
            tr_loss.update(loss.item(), inputs.size(0))
            tr_accuracy.update(acc, inputs.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                model.eval()
                for i, sample_batched in enumerate(vl_loader):
                    inputs = sample_batched['image'].type(torch.FloatTensor)
                    targets = sample_batched['label'].type(torch.LongTensor)
                    if device == 'cuda':
                        inputs, targets = inputs.cuda(), targets.cuda()
                    inputs, targets = Variable(inputs), Variable(targets)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    acc = calculate_accuracy(outputs, targets)
                    vl_loss.update(loss.item(), inputs.size(0))
                    vl_accuracy.update(acc, inputs.size(0))
                tr_loss_ep.update(tr_loss.avg)
                vl_loss_ep.update(vl_loss.avg)

                epoch_time.update(time.time() - end_time)
                end_time = time.time()

        model.train()

        print('Epoch: [{0}/{1}]\t'
              'Time: {2:.3f}\t'
              'Loss: [train>{3:.3f}/val>{4:.3f}]\t'
              'Acc {5:.3f}'.format(
                  epoch+1, epochs, epoch_time.val, tr_loss.avg, vl_loss.avg, vl_accuracy.avg))

        if vl_loss_ep.vc[-1] <= min_loss:
            print('Validation loss decreased: {:.6f} --> {:.6f}'.format(min_loss,vl_loss_ep.vc[-1]))
            print('Saving model...')
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'loss': loss,
                        'tr_loss': tr_loss_ep,
                        'vl_loss': vl_loss_ep,
                        }, model_path)
            print('Model saved at {}'.format(model_path))
            patience = 0
            min_loss = vl_loss_ep.vc[-1]
        else:
            patience += 1
        if patience == max_patience:
            print('oops the model seems to be not improving... break!')
            break

    print('Total train for {0} epochs, Total time: {1:.3f}'.format(epoch+1, epoch_time.sum))
    return tr_loss_ep, vl_loss_ep

def plot_prc(y_true, y_pred_prob):
    """
    This function plots the Precision Recall F1 curve.
    """
    precision, recall, threshold = metrics.precision_recall_curve(y_true, y_pred_prob)
    f1 = 2*(precision*recall)/(precision+recall)
    f1_max_ix = threshold[np.argmax(f1)]
    average_precision = metrics.average_precision_score(y_true, y_pred_prob)
    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})

    fig, ax = plt.subplots(1,2,figsize=(12,5))

    ax[0].step(recall, precision, color='b', alpha=0.2, where='post')
    ax[0].fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
    ax[0].set_xlabel('Recall')
    ax[0].set_ylabel('Precision')
    ax[0].set_ylim([0.0, 1.05])
    ax[0].set_xlim([0.0, 1.0])
    ax[0].set_title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))

    ax[1].set_title('Precision and Recall Scores as a function of the decision threshold')
    ax[1].plot(threshold, precision[:-1], 'b-', label='Precision')
    ax[1].plot(threshold, recall[:-1], 'g-', label='Recall')
    ax[1].plot(threshold, f1[:-1], 'r-', label='f1')
    ax[1].axvline(x=f1_max_ix, label='Th at = {0:.2f}'.format(f1_max_ix), c='r', linestyle='--')
    ax[1].set_ylabel('Score')
    ax[1].set_xlabel('Decision Threshold')
    ax[1].legend(loc='best')
    plt.show()

    return average_precision, f1_max_ix

def plot_prc_simple(y_true, y_pred_prob):
    """
    This function plots the Precision Recall F1 curve.
    """
    precision, recall, threshold = metrics.precision_recall_curve(y_true, y_pred_prob)
    f1 = 2*(precision*recall)/(precision+recall)
    f1_max_ix = threshold[np.argmax(f1)]
    average_precision = metrics.average_precision_score(y_true, y_pred_prob)

    return average_precision, f1_max_ix

def test(data_loader, model):
    print('Testing...')

    model.eval()

    data_time = AverageMeter()

    end_time = time.time()
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

        plt.title('Receiver Operating Characteristic Val')
        plt.plot(fpr, tpr, 'black', label = 'AUC = %0.2f' % auc_vec)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1], 'gray', linestyle='--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        print('------------------------------------')
        print('ROC Curve Teste:')
        plt.show()
        ####################################

        #opt_th = 0.5
        __, opt_th = plot_prc(targets, sftm_out[:,1])
        print('Best separation threshold: {}'.format(opt_th))

        y_pred = sftm_out[:,1] > opt_th

        mx_conf = confusion_matrix(targets, y_pred)
        plot_matrix(mx_conf, classes=np.unique(targets), fig_size=4, cmap=plt.cm.Greys,
                       title='Confussion matrix', opt_bar=False)

        ind_err_rand = np.where(np.logical_xor(targets, y_pred))

        accuracy, recall, precision, f1 = report_metrics(mx_conf)

        print('===== Final Report =====')
        print(tabulate([['AUC', auc_vec],
                        ['Accuracy', accuracy],
                        ['Recall', recall],
                        ['Precision', precision],
                        ['F1', f1],],
                       ['Metric', 'Value'], tablefmt='grid'))
        ##########################################
        _, predicted = outputs.topk(1, 1, True)
        print('Predict: ',predicted.view(1,-1))
        print('Targets: ',targets)

        test_acc = calculate_accuracy(outputs, targets)

        data_time.update(time.time() - end_time)
        end_time = time.time()

        print('Time: {0:.3f}\t'
              'Acc {1:.3f}'.format(data_time.val, test_acc))

        for elem in range(inputs.size(0)):
            plot_img_mask(inputs.cpu().numpy()[elem],
                          int(targets.cpu().numpy()[elem]),
                          predicted.cpu().numpy()[elem])
