#!/usr/bin/env python
# coding: utf-8

# In[1]:


#All imports for the model
import numpy as np
import pandas as pd
import xarray as xr
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[2]:


#For plotting. 
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# In[3]:


import datetime
from datetime import timedelta


# In[4]:


import cftime
import torchvision.transforms as T
import dask


# In[5]:


print("CUDA Version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())


# In[6]:


#Bring in the Data: 
total_training_features = xr.open_mfdataset('/home/mseiler1/scratch.pickerin-prj/total_training_features.nc')
total_training_flashes = xr.open_mfdataset('/home/mseiler1/scratch.pickerin-prj/total_training_flashes.nc')
total_training_LWI = xr.open_mfdataset('/home/mseiler1/scratch.pickerin-prj/total_training_LWI.nc')


# In[8]:


# In[9]:


testing_features_usa = xr.open_mfdataset('/home/mseiler1/scratch.pickerin-prj/testing_features_us.nc')
testing_features_amazon = xr.open_mfdataset('/home/mseiler1/scratch.pickerin-prj/testing_features_amazon.nc')
testing_flashes_usa = xr.open_mfdataset('/home/mseiler1/scratch.pickerin-prj/testing_flashes_us.nc')
testing_flashes_amazon = xr.open_mfdataset('/home/mseiler1/scratch.pickerin-prj/testing_flashes_amazon.nc')
testing_LWI_usa = xr.open_mfdataset('/home/mseiler1/scratch.pickerin-prj/testing_LWI_us.nc')
testing_LWI_amazon =  xr.open_mfdataset('/home/mseiler1/scratch.pickerin-prj/testing_LWI_amazon.nc')
testing_Lopez_usa = xr.open_mfdataset('/home/mseiler1/scratch.pickerin-prj/testing_Lopez_us.nc')
testing_Lopez_amazon = xr.open_mfdataset('/home/mseiler1/scratch.pickerin-prj/testing_Lopez_amazon.nc')


# In[26]:


# # 1. Define what you want to exclude
vars_to_drop = ['pretot']

# 2. Create the master list of variables to keep (from training data)
features_to_use = [var for var in total_training_features.data_vars if var not in vars_to_drop]

# 3. Apply this list to filter the Training data
total_training_features_filtered = total_training_features[features_to_use]

# 4. Apply this EXACT same list to filter your Testing datasets
total_testing_features_USA = testing_features_usa[features_to_use]
total_testing_features_Amazon = testing_features_amazon[features_to_use]


# In[27]:


#Standardizing the Features. So, we want to only pass in the training features here. 
scalers = {}  # keep scalers for each variable so you can inverse_transform later if needed

train_scaled = {}

for var in total_training_features_filtered.data_vars:
    # Extract the training data for this variable (one variable at a time.)
    train_data = total_training_features_filtered[var].values  # shape: (Datetime, Latitudes, Longitudes) #Training
    # Ensure consistent order
    train_data = total_training_features_filtered[var].transpose("Datetime", "Latitudes", "Longitudes").values
    
    # Flatten time & spatial dims so StandardScaler sees asamples × features
    train_flat = train_data.reshape(train_data.shape[0], -1)  # (time, lat*lon)
    
    # Fit scaler on training and transform 
    scaler = StandardScaler()
    train_scaled_flat = scaler.fit_transform(train_flat)
    
    # Save scaler for later use (inverse_transform, apply to new data, etc.)
    scalers[var] = scaler
    
    # Reshape back to original (time, lat, lon)
    train_scaled[var] = (("Datetime", "Latitudes", "Longitudes"),
                         train_scaled_flat.reshape(train_data.shape))

# Rebuild standardized datasets
train_features_scaled = xr.Dataset(train_scaled, coords=total_training_features.coords)


# In[ ]:


import joblib
# 2. Save the fitted scalers using joblib
scaler_path = 'Regression_NOTotPrecip_scalers_new.joblib'
joblib.dump(scalers, scaler_path)

import random
import os

def set_seed(seed=42):
    # 1. Base Python randomness
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 2. NumPy randomness
    np.random.seed(seed)

    # 3. PyTorch CPU & GPU randomness
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # For multi-GPU setups

    # 4. Force CuDNN to be deterministic (Crucial for CNNs)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Call the function immediately
set_seed(42)

# In[28]:


# --- Transform TEST (USA & Amazon) --- this is a function 
def transform_test(test_ds, scalers):
    test_scaled = {}
    for var in test_ds.data_vars:
        test_data = test_ds[var].transpose("Datetime", "Latitudes", "Longitudes").values
        test_flat = test_data.reshape(test_data.shape[0], -1)
        
        # Use training scaler to transform
        scaler = scalers[var]
        test_scaled_flat = scaler.transform(test_flat)
        
        test_scaled[var] = (("Datetime", "Latitudes", "Longitudes"),
                            test_scaled_flat.reshape(test_data.shape))
    
    return xr.Dataset(test_scaled, coords=test_ds.coords)


# In[29]:


# Apply the transform_test function on the FILTERED data for the USA and the Amazon 
testing_features_scaled_usa     = transform_test(total_testing_features_USA, scalers)
testing_features_scaled_amazon  = transform_test(total_testing_features_Amazon, scalers)


# #-----#-----#-----#
# 
# This is a check point. All the data has been brought in. 
# The TRAINING features have been standardized.
# The TESTING features have been standardized. 
# The flash data, LWI data, and the Lopez data have all been brought in. 
# Now, they need to be made into dataarrays so they can be put into the custom dataset for training. 
# 
# #-----#-----#-----#

# In[32]:


#Making it into a dataarray to be used for training
train_features_scaled_array = xr.concat(
    [train_features_scaled[var] for var in train_features_scaled.data_vars],
    dim="features"
)
train_features_scaled_array = train_features_scaled_array.transpose(
    "Datetime", "features", "Latitudes", "Longitudes"
)


# In[33]:


#Making the LWI dataset into a Dataarray 
LWI_array = xr.concat(
    [total_training_LWI[var] for var in total_training_LWI.data_vars],
    dim="features"
) 
LWI_array = LWI_array.transpose("Datetime", "features", "Latitudes", "Longitudes")


# In[34]:


#Making sure the Flash data is in a dataarray and in the correct format 
flashes_array = xr.concat(
    [total_training_flashes[var] for var in total_training_flashes.data_vars],
    dim="features"
) 


# In[35]:


#I will print all shapes here. 

# ALL DATA PRE-PROCESSING IS COMPLETE. 

# In[36]:


#This is a list of new functions: 
# new funcs for help
def get_onehot(labels_):
    """one hot helper"""
    labels_t = torch.from_numpy(labels_)              # [B,1,H,W]
    labels_idx = (labels_t.squeeze(1) > 0).long()       # [B,H,W] in {0,1}
    one_hot = F.one_hot(labels_idx, num_classes=2).permute(0, 3, 1, 2).float()  # [B,2,H,W]train_labels
    return one_hot

##This function is used to help with one-hot-encoding. This will be used for the classification part of the Multi-Task and the Hurdle 
# models. It takes in the labels (our flash data). Anything above 1, it will assign as "True" and anything lower than that (or zero) will
# be assigned as "False". These will be saved as a pytorch tensor, which can be passed through the model. Good for: Feeding labels to 
#cross-entropy or softmax networks

def get_classlbls(labels_):
    """one hot helper"""
    labels_t = torch.from_numpy(labels_)              # [B,1,H,W]
    labels_idx = (labels_t.squeeze(1) > 0).long()       # [B,H,W] in {0,1}
    return labels_idx

##This function is like the one above, however it just keeps the labels as values of 0 or 1. Converts the labels to a pytorch tensor. 
#Then, removes the channel dimension (using squeeze) and makes everything 1 or above equal to 1 and everyhting below equal to 0. 
# Good for, Use when you just need class indices, e.g., PyTorch’s nn.CrossEntropyLoss which expects integer labels

def get_regression_output(model_out):
    return model_out["regression"] if isinstance(model_out, dict) else model_out

# A small helper function that checks to see if the model output is a single tensor or a dictionary of tensors (Like if the model
# predicts multiple things, i.e. regression and classification). If the model_out is a dictionary, it grabs the tensor stored under
# "regression". If it is already a tensor, it just returns as is. Essentially helps so that no matter if i am doing multi-task or 
# hurdle, it will return exactly what I need without manually having to change it each time. 

def get_cls_logits(model_out):
    return model_out.get("cls_logits", None) if isinstance(model_out, dict) else None

#This is a small helper function. It is similar to the one above, however this time, if called, it will return just the classification 
# logits. If the model is not a dictionary, however, it will not return anything. Again, more of a convenience helper no matter what model
# is being run. 

def fmt_losses(total, reg, cls):
    if cls is None:
        return f"{total:.4f} (reg {reg:.4f})"
    return f"{total:.4f} (reg {reg:.4f}, cls {cls:.4f})"

#This is  function that returns the loss values in a neat way. Will either return total loss with regression only or will return 
# with both regression and classification. 

def update_confusion_matrix(confmat, preds, targets, num_classes=2):
    preds = preds.view(-1)
    targets = targets.view(-1)
    mask = (targets >= 0) & (targets < num_classes)
    preds = preds[mask]
    targets = targets[mask]
    idx = targets * num_classes + preds
    binc = torch.bincount(idx, minlength=num_classes * num_classes)
    confmat += binc.view(num_classes, num_classes)

#This is a function that updates a confusion matrix during the evaluation of a classification model. bins the occurence of each instance 
# so that it will be able to display as a matrix. correct positive, correct negatives, incorrect positives, incorrect negatives. 

def get_device():
    """
    Grab GPU (cuda).
    #This is where I should grab Zaratan
    """
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device


# In[37]:


#### Maria: helper for various run modes
CLS_KEYS = ("cls_labels", "class_labels", "seg_labels", "cls_label")

RUN_MODES = {
    "regression_only": {
        "use_cls_labels": False,
        "return_reg_only": True,      # MultiTaskUNet returns tensor
        "hurdle_train": False,
        "hurdle_thresh": None,
        "lambda_cls": 0.0,
        "active_class": 1,
        "cls_criterion": None,        # ignored because no cls labels / no cls loss
        "cls_key_candidates": CLS_KEYS,
    },
    "multitask_no_hurdle": {
        "use_cls_labels": True,
        "return_reg_only": False,     # MultiTaskUNet returns dict
        "hurdle_train": False,
        "hurdle_thresh": None,
        "lambda_cls": 1.0,
        "active_class": 1,
        "cls_criterion": nn.CrossEntropyLoss(),
        "cls_key_candidates": CLS_KEYS,
    },
    "multitask_hurdle_infer": {
        "use_cls_labels": True,
        "return_reg_only": False,
        "hurdle_train": False,        # hurdle only during eval/inference
        "hurdle_thresh": 0.50,
        "lambda_cls": 1.0,
        "active_class": 1,
        "cls_criterion": nn.CrossEntropyLoss(),
        "cls_key_candidates": CLS_KEYS,
    },
    "multitask_hurdle_train": {
        "use_cls_labels": True,
        "return_reg_only": False,
        "hurdle_train": True,         # hurdle during training
        "hurdle_thresh": 0.50,         # also monitor gated performance
        "lambda_cls": 1.0,
        "active_class": 1,
        "cls_criterion": nn.CrossEntropyLoss(),
        "cls_key_candidates": CLS_KEYS,
    },
}

#This is a configuration dictionary. The CLS_Keys: These are used to find the right labels dynamically in multi-task setups. 
# When you train a model, you pick a mode. And the script will come here and this dictionary of settings describes how to handle each mode. 
# You still have the same Model, you’re just telling the code how to interpret inputs, outputs, and losses.


# In[38]:


####Choices of RUN_Modes 

RUN_MODE = "regression_only"  # change only this
#RUN_MODE = "multitask_no_hurdle"  # change only this
#RUN_MODE = "multitask_hurdle_infer"  # change only this
#RUN_MODE = "multitask_hurdle_train"  # change only this
 
cfg = RUN_MODES[RUN_MODE]     #Stays the same. 


# In[39]:


# Check if multiple GPUs are available
device = get_device()

NUM_INPUT_CHANNELS = 12 # number of variables
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
BATCH_SIZE = 50

#Compute a threshold for the rare event weighting: 
alpha = 1.0
gamma = 0.75


# In[40]:


#This is where I make the split between training and validataion. 
#This is being split from the standardized features data. 
train_feats, valid_feats, train_labels, valid_labels, train_LWI, valid_LWI = train_test_split(
    train_features_scaled_array.values,
    flashes_array.values,
    LWI_array.values,
    test_size=0.3,
    shuffle=True,
    random_state=0
)
# In[53]:

# In[41]:


#### Maria: created classification labels here from original regression labels

# get onehot classes for multitask
#train_labels_onehot = get_onehot(train_labels) # not needed but leaving for ref
#valid_labels_onehot = get_onehot(valid_labels)


train_labels_class = get_classlbls(train_labels)
valid_labels_class = get_classlbls(valid_labels)


# In[43]:


#### Maria: added print statements for class labels

#print(train_feats.dims)
print("Training features: ", train_feats.shape)
print("Validation features: ", valid_feats.shape)
print("Training Labels: ", train_labels.shape)
print("Training Labels: ", train_labels_class.shape) # classification labels (B, H, W)
print("Validation Labels: ", valid_labels.shape)
print("Validation Labels: ", valid_labels_class.shape) # classification labels
print("Training LWI: ", train_LWI.shape)
print("Validation LWI: ", valid_LWI.shape)


# In[45]:


#### Maria: updated these classes and functions for UNet -- updated down stride

class DoubleConv(nn.Module):
    """(conv => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding="same"),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool (stride=2) then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2), #### stride 2 update -- Maria
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """
    Upscaling then double conv.

    Explicitly parameterized by:
      - deep_in: channels coming from deeper layer
      - skip_in: channels from skip connection
      - out_channels: output channels after fusion
    """
    def __init__(self, deep_in, skip_in, out_channels, bilinear=True):
        super().__init__()
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            up_out = deep_in  # channels unchanged
        else:
            # Typically halves channels on upsampling
            self.up = nn.ConvTranspose2d(deep_in, deep_in // 2, kernel_size=2, stride=2)
            up_out = deep_in // 2

        self.conv = DoubleConv(up_out + skip_in, out_channels)

    def forward(self, x_deep, x_skip):
        x_deep = self.up(x_deep)

        # Pad if needed (handles odd input sizes)
        diffY = x_skip.size(2) - x_deep.size(2)
        diffX = x_skip.size(3) - x_deep.size(3)
        x_deep = F.pad(
            x_deep,
            [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2],
        )

        x = torch.cat([x_skip, x_deep], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# In[46]:


#### Maria: new multitask unet based on previous classic unet

class MultiTaskUNet(nn.Module):
    """
    U-Net (4 downs) with two dense heads:
      - Regression: [B, C_reg, H, W]
      - Classification logits: [B, 2, H, W]

    To get classic UNetRegression-like behavior:
      yhat = model(x, return_reg_only=True)   # returns Tensor [B, C_reg, H, W]

    Default behavior:
      out = model(x)  # returns dict {"regression": ..., "cls_logits": ...}
    """
    def __init__(
        self,
        n_channels: int,
        n_regression_out: int = 1,
        base_channels: int = 32,
        bilinear: bool = False,
        positive_regression: bool = True,
        n_classes: int = 2,   # keep 2 for your setup
    ):
        super().__init__()
        self.positive_regression = positive_regression
        self.bilinear = bilinear
        self.n_classes = n_classes

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        # Match the same channel plan we used earlier:
        # If bilinear, reduce bottleneck for efficiency (common in U-Net variants)
        bottleneck = c5 // 2 if bilinear else c5

        # Encoder
        self.inc   = DoubleConv(n_channels, c1)
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)
        self.down4 = Down(c4, bottleneck)

        # Decoder (shared trunk)
        if bilinear:
            self.up1 = Up(bottleneck, c4, c4 // 2, bilinear=True)
            self.up2 = Up(c4 // 2, c3, c3 // 2, bilinear=True)
            self.up3 = Up(c3 // 2, c2, c2 // 2, bilinear=True)
            self.up4 = Up(c2 // 2, c1, c1,      bilinear=True)
            trunk_out_ch = c1
        else:
            self.up1 = Up(bottleneck, c4, c4, bilinear=False)
            self.up2 = Up(c4,        c3, c3, bilinear=False)
            self.up3 = Up(c3,        c2, c2, bilinear=False)
            self.up4 = Up(c2,        c1, c1, bilinear=False)
            trunk_out_ch = c1

        # Heads
        self.reg_head = OutConv(trunk_out_ch, n_regression_out)
        self.cls_head = OutConv(trunk_out_ch, n_classes)

    def forward(self, x, return_reg_only: bool = False):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder trunk
        x = self.up1(x5, x4)
        x = self.up2(x,  x3)    #Notice what is being passed through 
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)

        # Regression head
        reg_logits = self.reg_head(x)
        reg_out = F.softplus(reg_logits) if self.positive_regression else reg_logits

        if return_reg_only:    #if specified in Run_Mode I am guessing? 
            return reg_out

        # Classification head (logits)
        cls_logits = self.cls_head(x)

        return {"regression": reg_out, "cls_logits": cls_logits}


# In[48]:


### Maria: this is the new call to the multitask model

net1 = MultiTaskUNet(
    n_channels=NUM_INPUT_CHANNELS,
    n_regression_out=1, # dont need to change
    base_channels=32, # could be something to decrease if overfitting or increase if underfitting
    bilinear=False,
    positive_regression=True, # keep true for positive flash prediction
    n_classes=2 # dont need to change
    ).to(device)


# In[49]:


# 1. The Standard Tweedie (for Volume & Totals)
class TweedieLoss(nn.Module):
    def __init__(self, p=1.5, reduction='none'): # reduction='none' is CRITICAL
        super().__init__()
        self.p = p
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        p = self.p
        loss = (
            (torch.pow(y_true, 2 - p) / ((1 - p) * (2 - p)))
            - (y_true * torch.pow(y_pred, 1 - p)) / (1 - p)
            + (torch.pow(y_pred, 2 - p)) / (2 - p)
        )
        if self.reduction == 'mean': return loss.mean()
        elif self.reduction == 'sum': return loss.sum()
        else: return loss # Returns per-pixel loss map


# In[50]:


class CustomDataset(Dataset):
    def __init__(self, features, labels, cls_labels=None, LWI=None, transform=None):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)  # regression: e.g., [N,1,H,W]

        # cls_labels: expected [N,H,W] (class indices). Keep dtype as-is; we’ll cast in __getitem__.
        self.cls_labels = torch.as_tensor(cls_labels) if cls_labels is not None else None

        self.LWI = torch.as_tensor(LWI, dtype=torch.float32) if LWI is not None else None
        self.transform = transform

        if len(self.features) != len(self.labels):
            raise ValueError(f"features and labels length mismatch: {len(self.features)} vs {len(self.labels)}")
        if self.cls_labels is not None and len(self.cls_labels) != len(self.features):
            raise ValueError(f"cls_labels length mismatch: {len(self.cls_labels)} vs {len(self.features)}")
        if self.LWI is not None and len(self.LWI) != len(self.features):
            raise ValueError(f"LWI length mismatch: {len(self.LWI)} vs {len(self.features)}")

    def __len__(self):
        return len(self.features)

    @staticmethod
    def _flip_feature_chw(x):
        # x is [C,H,W]
        return torch.flip(x, dims=[1])

    @staticmethod
    def _flip_label_chw(x):
        # regression label is typically [1,H,W] (still CHW)
        return torch.flip(x, dims=[1])

    @staticmethod
    def _flip_hw(x):
        # cls_labels stored as [H,W]
        return torch.flip(x, dims=[0])

    def __getitem__(self, idx):
        feature = self.features[idx]             # [C,H,W]
        label = self.labels[idx]                 # [1,H,W] (or [C_reg,H,W])
        cls_label = self.cls_labels[idx] if self.cls_labels is not None else None  # [H,W]
        lwi = self.LWI[idx] if self.LWI is not None else None                     # [C2,H,W] or [1,H,W]

        # Apply transform (random vertical flip)
        if self.transform and torch.rand(1).item() < 0.5:
            feature = self._flip_feature_chw(feature)
            label = self._flip_label_chw(label)

            if cls_label is not None:
                cls_label = self._flip_hw(cls_label)

            if lwi is not None:
                # assumes LWI is CHW
                lwi = self._flip_feature_chw(lwi)

        sample = {"features": feature, "labels": label, "LWI": lwi}

        if cls_label is not None:
            # CrossEntropyLoss expects Long targets with class indices
            sample["cls_labels"] = cls_label.long()

        return sample


# In[51]:


#### Maria: New create dataset instances for training and testing
flip_transform = T.RandomVerticalFlip(p=0.5)

train_dataset = CustomDataset(
    train_feats,
    train_labels,
    cls_labels = (train_labels_class if cfg["use_cls_labels"] else None),  # change handled by cfg
    LWI = train_LWI,
    transform = flip_transform  # random vertical flipping
)

valid_dataset = CustomDataset(
    valid_feats,
    valid_labels,
    cls_labels = (valid_labels_class if cfg["use_cls_labels"] else None),  # change handled by cfg
    LWI = valid_LWI,
    transform = None  # no augmentation
)


# In[52]:


# Create DataLoader instances
train_loader = DataLoader(train_dataset,
               batch_size=BATCH_SIZE, 
               shuffle=True, 
               drop_last=True
)
valid_loader= DataLoader(valid_dataset, 
              batch_size=BATCH_SIZE, 
              shuffle=False, 
              drop_last=False           
)


# In[53]:


for data in valid_loader:
    print(data['features'].shape)
    print(data['labels'].shape)
    print(data['LWI'].shape)
    break


# In[54]:


for data in train_loader:
    print(data['features'].shape)
    print(data['labels'].shape)
    print(data['LWI'].shape)
    break


# In[34]:


#data['cls_labels'].shape # looks good (should throw keyerror if regression only -- no MT)


# In[55]:


### Maria: added this helper here

def forward_model(model, x, return_reg_only: bool):
    # Safest explicit: only pass the kwarg to MultiTaskUNet
    if isinstance(model, MultiTaskUNet):
        return model(x, return_reg_only=return_reg_only)
    return model(x)


# In[33]:


def train(
    model,
    dataloader,
    optimizer,
    reg_criterion,              # MUST be TweedieLoss(reduction='none')
    device,
    alpha=0.0,
    gamma=0.0,
    lambda_cls=1.0,
    cls_criterion=None,         
    cls_key_candidates=("cls_labels", "class_labels", "seg_labels", "cls_label"),
    hurdle_train=False,         
    active_class=1,
    return_reg_only=False,
    # --- CHANGED: We now just need the kernel size, not a separate criterion ---
    pool_kernel_size=5          
):
    model.train()
    running_total = 0.0
    running_reg = 0.0
    running_cls = 0.0
    cls_steps = 0

    # --- DEFINE POOLER ---
    # We define it here to ensure it uses the correct kernel size
    pooler = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=1, padding=pool_kernel_size//2).to(device)

    if cls_criterion is None:
        cls_criterion = nn.CrossEntropyLoss()

    eps = 1e-6

    for batch in dataloader:
        x = batch["features"].to(device).float()
        y_reg = batch["labels"].to(device).float() 

        out = model(x)

        if isinstance(out, dict):
            yhat_reg = out["regression"]
            cls_logits = out.get("cls_logits", None)
        else:
            yhat_reg = out
            cls_logits = None

        # ====================================================
        # NEW LOGIC: FUZZY TWEEDIE
        # ====================================================

        # 1. POOL FIRST (The "Fuzzy" Step)
        # Instead of pixel-vs-pixel, we compare neighborhood-vs-neighborhood
        y_pred_pooled = pooler(yhat_reg)
        y_true_pooled = pooler(y_reg)

        # 2. CALCULATE WEIGHTS ON THE POOLED TARGETS
        # We weight the loss based on the MAGNITUDE of the neighborhood peak.
        mean_pooled_label = y_true_pooled.mean()
        weights_pooled = 1 + alpha * ((y_true_pooled + eps) / (mean_pooled_label + eps)) ** gamma

        # 3. CALCULATE TWEEDIE LOSS ON POOLED MAPS
        # This checks: "Did you predict the correct MAX intensity in this 5x5 box?"
        per_elem_loss = reg_criterion(y_pred_pooled, y_true_pooled)

        # 4. APPLY WEIGHTS
        reg_weighted = weights_pooled * per_elem_loss
        
        # ====================================================

        # Classification Logic
        cls_loss = None
        cls_idx = None
        if cls_logits is not None:
             for k in cls_key_candidates:
                if k in batch:
                    cls_idx = batch[k].to(device)
                    break
        
        if cls_logits is not None and cls_idx is not None:
            if cls_idx.ndim == 4 and cls_idx.size(1) == 1:
                cls_idx = cls_idx.squeeze(1)
            cls_idx = cls_idx.long()
            cls_loss = cls_criterion(cls_logits, cls_idx)

        # Aggregate Regression Loss
        if hurdle_train and (cls_idx is not None):
            # Pool the mask too! A neighborhood is active if ANY pixel is active.
            gt_mask = (cls_idx == active_class).unsqueeze(1).float()
            gt_mask_pooled = pooler(gt_mask)
            gt_mask_pooled = (gt_mask_pooled > 0.01).float() 

            reg_loss_val = (reg_weighted * gt_mask_pooled).sum() / (gt_mask_pooled.sum() + eps)
        else:
            reg_loss_val = reg_weighted.mean()

        # Total Loss (Note: No sharp_loss added here, it's baked into reg_loss_val)
        total_loss = reg_loss_val + (lambda_cls * cls_loss if cls_loss is not None else 0.0)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        running_total += total_loss.item()
        running_reg += reg_loss_val.item()

        if cls_loss is not None:
            running_cls += cls_loss.item()
            cls_steps += 1

    total_avg = running_total / len(dataloader)
    reg_avg = running_reg / len(dataloader)
    cls_avg = (running_cls / cls_steps) if cls_steps > 0 else None

    # Returns 3 values (Total, Reg, Cls)
    return total_avg, reg_avg, cls_avg


# In[57]:


def validate(
    model,
    dataloader,
    device,
    reg_criterion,              # MUST be TweedieLoss(reduction='none')
    alpha=0.0,
    gamma=0.0,
    lambda_cls=1.0,
    cls_criterion=None,
    cls_key_candidates=("cls_labels", "class_labels", "seg_labels", "cls_label"),
    hurdle_train=False,
    active_class=1,
    return_reg_only=False,
    # --- CHANGED: Just need kernel size now ---
    pool_kernel_size=5
):
    model.eval()
    running_total = 0.0
    running_reg = 0.0
    running_cls = 0.0
    cls_steps = 0

    # --- DEFINE POOLER ---
    pooler = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=1, padding=pool_kernel_size//2).to(device)

    if cls_criterion is None:
        cls_criterion = nn.CrossEntropyLoss()

    eps = 1e-6

    with torch.no_grad():
        for batch in dataloader:
            x = batch["features"].to(device).float()
            y_reg = batch["labels"].to(device).float()

            out = model(x)

            if isinstance(out, dict):
                yhat_reg = out["regression"]
                cls_logits = out.get("cls_logits", None)
            else:
                yhat_reg = out
                cls_logits = None

            # ====================================================
            # NEW LOGIC: FUZZY TWEEDIE (Validation)
            # ====================================================

            # 1. POOL FIRST
            y_pred_pooled = pooler(yhat_reg)
            y_true_pooled = pooler(y_reg)

            # 2. CALCULATE WEIGHTS ON POOLED TARGETS
            mean_pooled_label = y_true_pooled.mean()
            weights_pooled = 1 + alpha * ((y_true_pooled + eps) / (mean_pooled_label + eps)) ** gamma

            # 3. CALCULATE TWEEDIE LOSS ON POOLED MAPS
            per_elem_loss = reg_criterion(y_pred_pooled, y_true_pooled)

            # 4. APPLY WEIGHTS
            reg_weighted = weights_pooled * per_elem_loss

            # ====================================================

            # Classification Logic
            cls_loss = None
            cls_idx = None

            if cls_logits is not None:
                for k in cls_key_candidates:
                    if k in batch:
                        cls_idx = batch[k].to(device)
                        break

            if cls_logits is not None and cls_idx is not None:
                if cls_idx.ndim == 4 and cls_idx.size(1) == 1:
                    cls_idx = cls_idx.squeeze(1)
                cls_idx = cls_idx.long()
                cls_loss = cls_criterion(cls_logits, cls_idx)

            # Aggregate Regression Loss
            if hurdle_train and (cls_idx is not None):
                # Pool mask to match dimensions
                gt_mask = (cls_idx == active_class).unsqueeze(1).float()
                gt_mask_pooled = pooler(gt_mask)
                gt_mask_pooled = (gt_mask_pooled > 0.01).float() 

                reg_loss_val = (reg_weighted * gt_mask_pooled).sum() / (gt_mask_pooled.sum() + eps)
            else:
                reg_loss_val = reg_weighted.mean()

            # Total Loss
            total_loss = reg_loss_val + (lambda_cls * cls_loss if cls_loss is not None else 0.0)

            running_total += total_loss.item()
            running_reg += reg_loss_val.item()
            
            if cls_loss is not None:
                running_cls += cls_loss.item()
                cls_steps += 1

    total_avg = running_total / len(dataloader)
    reg_avg = running_reg / len(dataloader)
    cls_avg = (running_cls / cls_steps) if cls_steps > 0 else None

    # Returns 3 values (Total, Reg, Cls) - Sharpness is implicit!
    return total_avg, reg_avg, cls_avg


# In[58]:


# maria: new helper for class and reg metrics

def compute_val_metrics_hurdle(
    model,
    dataloader,
    device,
    threshold=0.5,
    active_class=1,
    cls_key_candidates=("cls_labels", "class_labels", "seg_labels", "cls_label"),
    return_reg_only=False
):
    model.eval()

    mse = MeanSquaredError().to(device)
    r2 = R2Score().to(device)

    mse_g = MeanSquaredError().to(device)
    r2_g = R2Score().to(device)

    num_classes = 2
    confmat = torch.zeros((num_classes, num_classes), device=device, dtype=torch.int64)
    correct_pixels = 0
    total_pixels = 0
    did_cls = False

    with torch.no_grad():
        for batch in dataloader:
            x = batch["features"].to(device)
            y_reg = batch["labels"].to(device)

            out = forward_model(model, x, return_reg_only)
            
            if isinstance(out, dict):
                yhat_reg = out["regression"]
                cls_logits = out.get("cls_logits", None)
            else:
                yhat_reg = out
                cls_logits = None

            # regression metrics (ungated)
            yhat_flat = yhat_reg.view(yhat_reg.size(0), -1)
            y_flat = y_reg.view(y_reg.size(0), -1)
            mse.update(yhat_flat, y_flat)
            r2.update(yhat_flat, y_flat)

            # classification + gated regression metrics (only if cls_logits and labels exist)
            if cls_logits is not None:
                cls_idx = None
                for k in cls_key_candidates:
                    if k in batch:
                        cls_idx = batch[k].to(device)
                        break

                if cls_idx is not None:
                    did_cls = True
                    if cls_idx.ndim == 4 and cls_idx.size(1) == 1:
                        cls_idx = cls_idx.squeeze(1)
                    cls_idx = cls_idx.long()  # 0/1

                    preds = cls_logits.argmax(dim=1)  # [B,H,W]
                    correct_pixels += (preds == cls_idx).sum().item()
                    total_pixels += cls_idx.numel()
                    update_confusion_matrix(confmat, preds, cls_idx, num_classes=num_classes)

                    # gated regression (predicted mask)
                    p_active = torch.softmax(cls_logits, dim=1)[:, active_class:active_class+1, :, :]  # [B,1,H,W]
                    pred_mask = (p_active >= threshold).float()
                    yhat_g = yhat_reg * pred_mask

                    yhat_g_flat = yhat_g.view(yhat_g.size(0), -1)
                    mse_g.update(yhat_g_flat, y_flat)
                    r2_g.update(yhat_g_flat, y_flat)

    rmse = torch.sqrt(mse.compute()).item()
    r2_val = r2.compute().item()

    if did_cls and total_pixels > 0:
        pixel_acc = correct_pixels / total_pixels

        tp = torch.diag(confmat).float()
        fp = confmat.sum(dim=0).float() - tp
        fn = confmat.sum(dim=1).float() - tp
        denom = tp + fp + fn
        iou = tp / torch.clamp(denom, min=1.0)

        mean_iou = iou.mean().item()
        iou_bg = iou[0].item()
        iou_fg = iou[1].item()

        rmse_gated = torch.sqrt(mse_g.compute()).item()
        r2_gated = r2_g.compute().item()
    else:
        pixel_acc = mean_iou = iou_bg = iou_fg = None
        rmse_gated = r2_gated = None

    return {
        "rmse": rmse,
        "r2": r2_val,
        "rmse_gated": rmse_gated,
        "r2_gated": r2_gated,
        "pixel_acc": pixel_acc,
        "mean_iou": mean_iou,
        "iou_bg": iou_bg,
        "iou_fg": iou_fg,
    }


# In[59]:


for batch in train_loader:
    print(batch.keys())  # Show available keys
    break


# In[60]:


for batch in train_loader:
    inputs = batch["features"]  # Use the 'features' key for inputs
    targets = batch["labels"]   # Use the 'labels' key for targets
    LWI = batch['LWI']
    print(inputs.dtype, targets.dtype, LWI.dtype)  # Check their data types
    break


# In[61]:


for param in net1.parameters():
    print(param.dtype)
    break


# In[62]:


# 1. The Optimizer (Unchanged)
optimizer = optim.Adam(net1.parameters(), lr=LEARNING_RATE, amsgrad=False)

# 2. Tweedie Loss (Unchanged)
# Keep reduction='none' so we can apply weights to the pooled map
reg_criterion = TweedieLoss(p=1.1, reduction='none')

# ADD: Just define the kernel size variable you want to use
pool_kernel_size = 3


# In[40]:


from torchmetrics import MeanSquaredError, R2Score

# Define this outside the loop
pool_kernel_size = 3

epoch_metrics = []

for epoch in range(NUM_EPOCHS):

    # --- train ---
    # Back to 3 values (Total, Reg, Cls)
    t_total, t_reg, t_cls = train(       
        net1, train_loader, optimizer, 
        reg_criterion=reg_criterion,          # Must be TweedieLoss(reduction='none')
        device=device,
        alpha=alpha, gamma=gamma,
        lambda_cls=cfg["lambda_cls"],
        cls_criterion=cfg["cls_criterion"],
        cls_key_candidates=cfg["cls_key_candidates"],
        hurdle_train=cfg["hurdle_train"],
        active_class=cfg["active_class"],
        return_reg_only=cfg["return_reg_only"],
        # New Argument (Replaces sharpness_criterion)
        pool_kernel_size=pool_kernel_size     # <--- PASS 5 HERE
    )
    
    # --- validate ---
    # Back to 3 values
    v_total, v_reg, v_cls = validate(    
        net1, valid_loader, device, 
        reg_criterion=reg_criterion,          
        alpha=alpha, gamma=gamma,
        lambda_cls=cfg["lambda_cls"],
        cls_criterion=cfg["cls_criterion"],
        cls_key_candidates=cfg["cls_key_candidates"],
        hurdle_train=cfg["hurdle_train"],
        active_class=cfg["active_class"],
        return_reg_only=cfg["return_reg_only"],
        # New Argument
        pool_kernel_size=pool_kernel_size     # <--- PASS 5 HERE
    )

    # --- validation metrics ---
    metrics = compute_val_metrics_hurdle(
        net1, valid_loader, device,
        threshold=cfg["hurdle_thresh"] if cfg["hurdle_thresh"] is not None else 0.5,
        active_class=cfg["active_class"],
        return_reg_only=cfg["return_reg_only"],
    )

    # Build log line (Removed separate Sharpness metric)
    msg = (
        f"Epoch {epoch+1}/{NUM_EPOCHS} | "
        f"Train: {t_total:.4f} (Reg:{t_reg:.4f}) | " 
        f"Val: {v_total:.4f} (Reg:{v_reg:.4f}) | "   
        f"RMSE: {metrics['rmse']:.4f}"
    )

    if metrics["pixel_acc"] is not None:
        msg += (
            f" | PixAcc: {metrics['pixel_acc']:.4f}"
            f" | mIoU: {metrics['mean_iou']:.4f}"
        )

    print(msg)

    # Store results (Removed separate Sharpness metric)
    row = {
        "epoch": epoch + 1,

        "train_total_loss": float(t_total),
        "train_reg_loss": float(t_reg),
        "train_cls_loss": (float(t_cls) if t_cls is not None else None),

        "val_total_loss": float(v_total),
        "val_reg_loss": float(v_reg),
        "val_cls_loss": (float(v_cls) if v_cls is not None else None),

        "rmse": float(metrics["rmse"]),
        "r2": float(metrics["r2"]),

        "pixel_acc": metrics["pixel_acc"],
        "mean_iou": metrics["mean_iou"],
        "iou_bg": metrics["iou_bg"],
        "iou_fg": metrics["iou_fg"],
    }

    if cfg["hurdle_thresh"] is not None:
        row["rmse_gated"] = metrics["rmse_gated"]
        row["r2_gated"] = metrics["r2_gated"]
        row["hurdle_thresh"] = cfg["hurdle_thresh"]

    row["hurdle_train"] = cfg["hurdle_train"]
    epoch_metrics.append(row)

df = pd.DataFrame(epoch_metrics)
# df.to_csv("training_metrics_NEW_101.csv", index=False)


# Save weights and architecture
save_path = "Regression_NOTotPrecip_new.pth"
torch.save(net1.state_dict(), save_path)
print(f"Model saved to {save_path}")


# In[64]:


def test(
    model,
    dataloader,
    device,
    reg_criterion,              # MUST be TweedieLoss(reduction='none')
    alpha=0.0,
    gamma=0.0,
    lambda_cls=1.0,
    cls_criterion=None,
    cls_key_candidates=("cls_labels", "class_labels", "seg_labels", "cls_label"),
    hurdle_train=False,         # mask regression loss on GT active pixels
    threshold=0.5,              # gate predictions by predicted P(active)
    active_class=1,
    return_reg_only=False,
    # --- CHANGED: Added pool kernel size ---
    pool_kernel_size=5          
):
    model.eval()

    # --- DEFINE POOLER ---
    # We define it here to ensure it uses the correct kernel size
    pooler = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=1, padding=pool_kernel_size//2).to(device)

    if cls_criterion is None:
        cls_criterion = nn.CrossEntropyLoss()

    eps = 1e-6

    running_total = 0.0
    running_reg = 0.0
    running_cls = 0.0
    cls_steps = 0

    # metrics
    mse = MeanSquaredError().to(device)
    r2 = R2Score().to(device)
    mse_g = MeanSquaredError().to(device)
    r2_g = R2Score().to(device)

    num_classes = 2
    confmat = torch.zeros((num_classes, num_classes), device=device, dtype=torch.int64)
    correct_pixels = 0
    total_pixels = 0
    did_cls = False

    with torch.no_grad():
        for batch in dataloader:
            x = batch["features"].to(device).float()
            y_reg = batch["labels"].to(device).float()

            # CORRECTED LINE
            out = forward_model(model, x, return_reg_only)

            if isinstance(out, dict):
                yhat_reg = out["regression"]
                cls_logits = out.get("cls_logits", None)
            else:
                yhat_reg = out
                cls_logits = None

            # ====================================================
            # NEW LOGIC: FUZZY TWEEDIE LOSS (Option C)
            # ====================================================
            
            # 1. POOL FIRST
            y_pred_pooled = pooler(yhat_reg)
            y_true_pooled = pooler(y_reg)

            # 2. CALCULATE WEIGHTS ON POOLED TARGETS
            mean_pooled_label = y_true_pooled.mean()
            weights_pooled = 1 + alpha * ((y_true_pooled + eps) / (mean_pooled_label + eps)) ** gamma

            # 3. CALCULATE TWEEDIE LOSS ON POOLED MAPS
            per_elem_loss = reg_criterion(y_pred_pooled, y_true_pooled)
            reg_weighted = weights_pooled * per_elem_loss

            # ====================================================

            cls_loss = None
            cls_idx = None

            if cls_logits is not None:
                for k in cls_key_candidates:
                    if k in batch:
                        cls_idx = batch[k].to(device)
                        break

            if cls_logits is not None and cls_idx is not None:
                if cls_idx.ndim == 4 and cls_idx.size(1) == 1:
                    cls_idx = cls_idx.squeeze(1)
                cls_idx = cls_idx.long()
                cls_loss = cls_criterion(cls_logits, cls_idx)

            if hurdle_train and (cls_idx is not None):
                # Pool the mask too for the loss calculation
                gt_mask = (cls_idx == active_class).unsqueeze(1).float()
                gt_mask_pooled = pooler(gt_mask)
                gt_mask_pooled = (gt_mask_pooled > 0.01).float() 
                
                reg_loss = (reg_weighted * gt_mask_pooled).sum() / (gt_mask_pooled.sum() + eps)
            else:
                reg_loss = reg_weighted.mean()

            total_loss = reg_loss + (lambda_cls * cls_loss if cls_loss is not None else 0.0)

            running_total += total_loss.item()
            running_reg += reg_loss.item()
            if cls_loss is not None:
                running_cls += cls_loss.item()
                cls_steps += 1

            # --- METRICS (Keep these Pixel-Wise/Unpooled) ---
            # We want to measure how good the raw prediction is against the raw ground truth
            yhat_flat = yhat_reg.view(yhat_reg.size(0), -1)
            y_flat = y_reg.view(y_reg.size(0), -1)
            mse.update(yhat_flat, y_flat)
            r2.update(yhat_flat, y_flat)

            # --- classification + gated regression metrics ---
            if cls_logits is not None and cls_idx is not None:
                did_cls = True

                preds = cls_logits.argmax(dim=1)
                correct_pixels += (preds == cls_idx).sum().item()
                total_pixels += cls_idx.numel()
                # Assuming update_confusion_matrix is defined in your utils
                update_confusion_matrix(confmat, preds, cls_idx, num_classes=num_classes)

                p_active = torch.softmax(cls_logits, dim=1)[:, active_class:active_class+1, :, :]
                pred_mask = (p_active >= threshold).float()
                yhat_g = yhat_reg * pred_mask

                yhat_g_flat = yhat_g.view(yhat_g.size(0), -1)
                mse_g.update(yhat_g_flat, y_flat)
                r2_g.update(yhat_g_flat, y_flat)

    # average losses
    total_avg = running_total / len(dataloader)
    reg_avg = running_reg / len(dataloader)
    cls_avg = (running_cls / cls_steps) if cls_steps > 0 else None

    # finalize metrics
    rmse = torch.sqrt(mse.compute()).item()
    r2_val = r2.compute().item()

    if did_cls and total_pixels > 0:
        pixel_acc = correct_pixels / total_pixels

        tp = torch.diag(confmat).float()
        fp = confmat.sum(dim=0).float() - tp
        fn = confmat.sum(dim=1).float() - tp
        denom = tp + fp + fn
        iou = tp / torch.clamp(denom, min=1.0)

        mean_iou = iou.mean().item()
        iou_bg = iou[0].item()
        iou_fg = iou[1].item()

        rmse_gated = torch.sqrt(mse_g.compute()).item()
        r2_gated = r2_g.compute().item()
    else:
        pixel_acc = mean_iou = iou_bg = iou_fg = None
        rmse_gated = r2_gated = None

    return {
        "test_total_loss": total_avg,
        "test_reg_loss": reg_avg,
        "test_cls_loss": cls_avg,
        "rmse": rmse,
        "r2": r2_val,
        "rmse_gated": rmse_gated,
        "r2_gated": r2_gated,
        "pixel_acc": pixel_acc,
        "mean_iou": mean_iou,
        "iou_bg": iou_bg,
        "iou_fg": iou_fg,
    }


#This is for the PFI later on: 
# --- CREATE AND SAVE PFI BUNDLE ---

pfi_bundle = {
    # The raw arrays needed to shuffle the 13 variables (excluding LWI)
    'valid_feats': valid_feats,
    'valid_labels': valid_labels,
    'valid_LWI': valid_LWI,
    
    # Conditionally save class labels based on your RUN_MODE
    'valid_labels_class': valid_labels_class if cfg.get("use_cls_labels") else None,
    
    # Metadata required to rebuild the dataloader and evaluation metrics
    'cfg': cfg,
    'BATCH_SIZE': BATCH_SIZE
}

bundle_path = 'pfi_data_bundle_NOTotPrecip.pt'
torch.save(pfi_bundle, bundle_path)
print(f"PFI data bundle successfully saved to {bundle_path}")



# #Make sure that LWI is a numpy array before putting into a custom dataset. 
# Convert xarray to numpy
us_LWI_test_array = testing_LWI_usa['lwi'].values  # shape (2774, 64, 64)
# # Add channel dimension: (samples, 1, lat, lon)
us_LWI_test_array = np.transpose(us_LWI_test_array, (2, 0, 1))
us_LWI_test_array = np.expand_dims(us_LWI_test_array, axis=1)  # shape (2774, 1, 64, 64)

# In[43]:


#Make sure that LWI is a numpy array before putting into a custom dataset. 
# Convert xarray to numpy
amazon_LWI_test_array = testing_LWI_amazon['lwi'].values  # shape (2199, 64, 64)

# Add channel dimension: (samples, 1, lat, lon)
# # Add channel dimension: (samples, 1, lat, lon)
amazon_LWI_test_array = np.transpose(amazon_LWI_test_array, (2, 0, 1))
amazon_LWI_test_array = np.expand_dims(amazon_LWI_test_array, axis=1)  # shape (2199, 1, 64, 64)


# In[44]:


#We need to make sure that the testing features are correct. 
us_test_features = testing_features_scaled_usa.to_array(dim="features")  # shape: (features, Datetime, Lat, Lon)

# # Reorder dimensions to match PyTorch (samples, channels, height, width)
us_test_features = us_test_features.transpose("Datetime", "features", "Latitudes", "Longitudes")

# Extract the underlying NumPy array
us_test_features = us_test_features.values  # shape: (15547, 13, 64, 64)


# In[45]:


#this is for the amazon data 
#We need to make sure that the testing features are correct. 
amazon_test_features = testing_features_scaled_amazon.to_array(dim="features")  # shape: (features, Datetime, Lat, Lon)

# # Reorder dimensions to match PyTorch (samples, channels, height, width)
amazon_test_features = amazon_test_features.transpose("Datetime", "features", "Latitudes", "Longitudes")

# Extract the underlying NumPy array
amazon_test_features = amazon_test_features.values  # shape: (15547, 13, 64, 64)



# In[72]:


# plot_lightning_importance(0.4091, 0.1540, df_pfi_final, df_csi_final)


# In[65]:


# Optional: get classification labels if your model expects them
amazon_test_labels_class = get_classlbls(testing_flashes_amazon['flashes_log'].values)

# Create the CustomDataset for Amazon
amazon_test_dataset = CustomDataset(
    features=amazon_test_features,
    labels=testing_flashes_amazon['flashes_log'].values,
    cls_labels=amazon_test_labels_class if cfg["use_cls_labels"] else None,
    LWI=amazon_LWI_test_array,
    transform=None  # no augmentation for test set
)

# DataLoader
amazon_test_loader = DataLoader(
    amazon_test_dataset,
    batch_size=BATCH_SIZE,  # can keep same batch size
    shuffle=False,          # do NOT shuffle test set
    drop_last=False
)


# In[48]:


# Optional: get classification labels if your model expects them
us_test_labels_class = get_classlbls(testing_flashes_usa['flashes_log'].values)

# Create the CustomDataset for USA
us_test_dataset = CustomDataset(
    features=us_test_features,
    labels=testing_flashes_usa['flashes_log'].values,
    cls_labels=us_test_labels_class if cfg["use_cls_labels"] else None,
    LWI=us_LWI_test_array,
    transform=None  # no augmentation for test set
)

# DataLoader
us_test_loader = DataLoader(
    us_test_dataset,
    batch_size=BATCH_SIZE,  # same as training/validation
    shuffle=False,          # do NOT shuffle test set
    drop_last=False
)


# In[72]:


# usa_loader = DataLoader(usa_test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
# amazon_loader = DataLoader(amazon_test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)


# In[51]:


test_results = test(
    net1, us_test_loader, device, reg_criterion=reg_criterion,
    alpha=alpha, gamma=gamma,
    lambda_cls=cfg["lambda_cls"],
    cls_criterion=cfg["cls_criterion"],
    cls_key_candidates=cfg["cls_key_candidates"],
    hurdle_train=cfg["hurdle_train"],
    threshold=cfg["hurdle_thresh"],
    active_class=cfg["active_class"],
    return_reg_only=cfg["return_reg_only"],
    # --- NEW ARGUMENT ---
    pool_kernel_size=5 
)


# In[52]:


# --- USA predictions ---
predictions_usa = []

with torch.no_grad():
    for batch in us_test_loader:
        inputs = batch['features'].float().to(device)
        # Use forward_model to handle MultiTaskUNet
        outputs = forward_model(net1, inputs, return_reg_only=cfg["return_reg_only"])
        
        # outputs could be dict or Tensor
        if isinstance(outputs, dict):
            yhat = outputs["regression"]
        else:
            yhat = outputs
        
        predictions_usa.append(yhat.cpu().numpy())

predictions_usa = np.concatenate(predictions_usa, axis=0)

# squeeze channel if it's 1
if predictions_usa.shape[1] == 1:
    predictions_usa = predictions_usa.squeeze(1)


# --- Amazon predictions ---
predictions_amazon = []

with torch.no_grad():
    for batch in amazon_test_loader:
        inputs = batch['features'].float().to(device)
        outputs = forward_model(net1, inputs, return_reg_only=cfg["return_reg_only"])
        
        if isinstance(outputs, dict):
            yhat = outputs["regression"]
        else:
            yhat = outputs
        
        predictions_amazon.append(yhat.cpu().numpy())

predictions_amazon = np.concatenate(predictions_amazon, axis=0)

if predictions_amazon.shape[1] == 1:
    predictions_amazon = predictions_amazon.squeeze(1)

print("USA predictions:", predictions_usa.shape)
print("Amazon predictions:", predictions_amazon.shape)


# In[53]:


# Save predictions for USA
predictions_usa_da = xr.DataArray(
    predictions_usa,
    dims=("Datetime", "Latitudes", "Longitudes"),
    coords={
        "Datetime": testing_features_scaled_usa["Datetime"].values,
        "Latitudes": testing_features_scaled_usa["Latitudes"].values,
        "Longitudes": testing_features_scaled_usa["Longitudes"].values,
    },
    name="predicted_flashes"
)
predictions_usa_da.to_netcdf("Regression_NOTotPrecip_USA_new.nc")

# Save predictions for Amazon
predictions_amazon_da = xr.DataArray(
    predictions_amazon,
    dims=("Datetime", "Latitudes", "Longitudes"),
    coords={
        "Datetime": testing_features_scaled_amazon["Datetime"].values,
        "Latitudes": testing_features_scaled_amazon["Latitudes"].values,
        "Longitudes": testing_features_scaled_amazon["Longitudes"].values,
    },
    name="predicted_flashes"
)
predictions_amazon_da.to_netcdf("Regression_NOTotPrecip_Amazon_new.nc")

print("✅ Predictions saved for USA and Amazon as NetCDF files.")


