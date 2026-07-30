#!/usr/bin/env python
# coding: utf-8

# In[1]:

#PFI_MAE script
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import xarray as xr  # Assuming you are still loading your NetCDF files!

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchmetrics import MeanSquaredError, R2Score, MeanAbsoluteError
# Set up your hardware device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

import sys
import torch.optim as optim
import torchvision.transforms as transforms

print("imports in")

# In[2]:
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


# In[3]:


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



# In[19]:
print("still good")

class CustomDataset(Dataset):
    def __init__(self, features, labels, cls_labels=None, LWI=None, transform=None):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)  

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
        return torch.flip(x, dims=[1])

    @staticmethod
    def _flip_label_chw(x):
        return torch.flip(x, dims=[1])

    @staticmethod
    def _flip_hw(x):
        return torch.flip(x, dims=[0])

    def __getitem__(self, idx):
        feature = self.features[idx]             
        label = self.labels[idx]                 
        cls_label = self.cls_labels[idx] if self.cls_labels is not None else None  
        lwi = self.LWI[idx] if self.LWI is not None else None                     

        # Apply transform (random vertical flip)
        if self.transform and torch.rand(1).item() < 0.5:
            feature = self._flip_feature_chw(feature)
            label = self._flip_label_chw(label)

            if cls_label is not None:
                cls_label = self._flip_hw(cls_label)

            if lwi is not None:
                lwi = self._flip_feature_chw(lwi)

        # THE FIX: Only add to the dictionary if they exist!
        sample = {
            "features": feature, 
            "labels": label
        }

        if lwi is not None:
            sample["LWI"] = lwi

        if cls_label is not None:
            sample["cls_labels"] = cls_label.long()

        return sample


# In[20]:


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
    mae = MeanAbsoluteError().to(device)
    r2 = R2Score().to(device)

    mse_g = MeanSquaredError().to(device)
    mae_g = MeanAbsoluteError().to(device)
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
            mae.update(yhat_flat, y_flat)
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
                    mae_g.update(yhat_g_flat, y_flat)
                    r2_g.update(yhat_g_flat, y_flat)

    rmse = torch.sqrt(mse.compute()).item()
    mae_val = mae.compute().item()
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
        mae_gated = mae_g.compute().item()
        r2_gated = r2_g.compute().item()
    else:
        pixel_acc = mean_iou = iou_bg = iou_fg = None
        rmse_gated = r2_gated = mae_gated = None

    return {
        "rmse": rmse,
        "mae": mae_val,
        "r2": r2_val,
        "rmse_gated": rmse_gated,
        "mae_gated": mae_gated,
        "r2_gated": r2_gated,
        "pixel_acc": pixel_acc,
        "mean_iou": mean_iou,
        "iou_bg": iou_bg,
        "iou_fg": iou_fg,
    }


# In[21]:


from torchmetrics import MeanSquaredError, R2Score

def forward_model(model, x, return_reg_only=False):
    # This replaces the missing helper function in your evaluation code
    return model(x, return_reg_only=return_reg_only)


# In[30]:

# 1. Create the model instance (12 channels for your 12 bundle features)
model = MultiTaskUNet(n_channels=12, n_regression_out=1)  
model.to(device)

# 2. Path to your saved weights (.pt file from your actual training run)
# Update this path to where your trained weights actually live!
checkpoint_path = '/home/mseiler1/scratch.pickerin-prj/variable_shuffle/NOTotPrecip/Regression_NOTotPrecip_new.pth' 

# 3. Load the weights into the model shell
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

# If you saved just the state_dict, use this:
model.load_state_dict(checkpoint)

# If your training script saved a dictionary structure like {'model_state_dict': ...}, 
# uncomment the line below instead:
# model.load_state_dict(checkpoint['model_state_dict'])

# 4. Set to evaluation mode
model.eval()

# =====================================================================
# 1. LOAD THE PRE-PROCESSED PFI BUNDLE
# =====================================================================
print("Loading PFI data bundle...")
bundle_path_usa = '/home/mseiler1/scratch.pickerin-prj/variable_shuffle/NOTotPrecip/pfi_bundle_USA.pt' # Ensure this path is correct
pfi_bundle_usa = torch.load(bundle_path_usa, map_location=device)
bundle_path_amazon = '/home/mseiler1/scratch.pickerin-prj/variable_shuffle/NOTotPrecip/pfi_bundle_Amazon.pt' # Ensure this path is correct
pfi_bundle_amazon = torch.load(bundle_path_amazon, map_location=device)

feature_names = [
    'cape', 'cldfrac', 'cldht', 'crh', 'iceflux', 'iwc440', 'lcl', 
    'massflux', 'maxcloudfrac', 'mseratio', 'precon', 'tlapse'
]

# =====================================================================
# 2. UNPACK BUNDLES & CREATE DATALOADERS
# =====================================================================

# USA DataLoader
usa_dataset = CustomDataset(
    features=pfi_bundle_usa['test_feats'],
    labels=pfi_bundle_usa['test_labels'],
    cls_labels=pfi_bundle_usa['test_labels_class'],
    LWI=pfi_bundle_usa['test_LWI']
)
usa_loader = DataLoader(
    usa_dataset, 
    batch_size=pfi_bundle_usa['BATCH_SIZE'], 
    shuffle=False
)

# Amazon DataLoader
amazon_dataset = CustomDataset(
    features=pfi_bundle_amazon['test_feats'],
    labels=pfi_bundle_amazon['test_labels'],
    cls_labels=pfi_bundle_amazon['test_labels_class'],
    LWI=pfi_bundle_amazon['test_LWI']
)
amazon_loader = DataLoader(
    amazon_dataset, 
    batch_size=pfi_bundle_amazon['BATCH_SIZE'], 
    shuffle=False
)

print(f"USA DataLoader ready: {len(usa_loader)} batches ({len(usa_dataset)} samples)")
print(f"Amazon DataLoader ready: {len(amazon_loader)} batches ({len(amazon_dataset)} samples)")

LAND_THRESHOLD = 0.55  # LWI >= 0.55 is Land; LWI < 0.55 is Water

def get_permuted_features(features_np, feature_idx):
    """Shuffles a single feature channel across all samples."""
    permuted = features_np.copy()
    shuffled_channel = permuted[:, feature_idx, :, :].copy()
    np.random.shuffle(shuffled_channel)
    permuted[:, feature_idx, :, :] = shuffled_channel
    return permuted


def compute_masked_mae(model, dataloader, device, land_threshold=0.55):
    """
    Computes overall MAE, Land MAE (LWI >= 0.55), and Water MAE (LWI < 0.55).
    """
    model.eval()
    
    total_abs_err_overall = 0.0
    total_count_overall = 0
    
    total_abs_err_land = 0.0
    total_count_land = 0
    
    total_abs_err_water = 0.0
    total_count_water = 0
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch["features"].to(device)
            y = batch["labels"].to(device)
            lwi = batch.get("LWI", None)
            
            # Forward pass (regression only)
            yhat = model(x, return_reg_only=True)
            abs_err = torch.abs(yhat - y)
            
            # Overall MAE
            total_abs_err_overall += abs_err.sum().item()
            total_count_overall += abs_err.numel()
            
            # Land vs. Water MAE
            if lwi is not None:
                lwi = lwi.to(device)
                if lwi.ndim == 3:
                    lwi = lwi.unsqueeze(1)
                
                land_mask = (lwi >= land_threshold)
                water_mask = (lwi < land_threshold)
                
                total_abs_err_land += abs_err[land_mask].sum().item()
                total_count_land += land_mask.sum().item()
                
                total_abs_err_water += abs_err[water_mask].sum().item()
                total_count_water += water_mask.sum().item()
                
    mae_overall = total_abs_err_overall / max(total_count_overall, 1)
    mae_land = (total_abs_err_land / total_count_land) if total_count_land > 0 else np.nan
    mae_water = (total_abs_err_water / total_count_water) if total_count_water > 0 else np.nan
    
    return {
        "mae_overall": mae_overall,
        "mae_land": mae_land,
        "mae_water": mae_water
    }

#Calculate the baseline MAE
print("Calculating baseline MAE for USA...")
base_usa = compute_masked_mae(model, usa_loader, device)

print("Calculating baseline MAE for Amazon...")
base_amazon = compute_masked_mae(model, amazon_loader, device)

print("\n*** BASELINE RESULTS ***")
print(f"USA    -> Overall: {base_usa['mae_overall']:.6f} | Land: {base_usa['mae_land']:.6f} | Water: {base_usa['mae_water']:.6f}")
print(f"Amazon -> Overall: {base_amazon['mae_overall']:.6f} | Land: {base_amazon['mae_land']:.6f} | Water: {base_amazon['mae_water']:.6f}\n")



#Running the PFI for USA

NUM_REPEATS = 20  # Number of permutation passes per feature (increase later if desired)

usa_pfi_results = []
base_off = base_usa['mae_overall']
base_lnd = base_usa['mae_land']
base_wtr = base_usa['mae_water']

raw_feats_usa = pfi_bundle_usa['test_feats']

print(f"Starting USA PFI ({len(feature_names)} features, {NUM_REPEATS} repeats each)...")
print("-----------------------------------------------------------------------")

for i, name in enumerate(feature_names):
    overall_deltas = []
    land_deltas = []
    water_deltas = []

    for r in range(NUM_REPEATS):
        # 1. Permute (shuffle) single feature channel
        perm_feats = get_permuted_features(raw_feats_usa, i)

        # 2. Re-wrap into DataLoader
        perm_ds = CustomDataset(
            features=perm_feats,
            labels=pfi_bundle_usa['test_labels'],
            cls_labels=pfi_bundle_usa['test_labels_class'],
            LWI=pfi_bundle_usa['test_LWI']
        )
        perm_loader = DataLoader(
            perm_ds, 
            batch_size=pfi_bundle_usa['BATCH_SIZE'], 
            shuffle=False
        )

        # 3. Compute degraded MAE
        m = compute_masked_mae(model, perm_loader, device)

        # 4. Calculate performance drop (Delta MAE = Degraded - Baseline)
        overall_deltas.append(m['mae_overall'] - base_off)
        land_deltas.append(m['mae_land'] - base_lnd)
        water_deltas.append(m['mae_water'] - base_wtr)

    # Save mean importance scores
    usa_pfi_results.append({
        "Feature": name,
        "USA_Imp_Overall": np.mean(overall_deltas),
        "USA_Imp_Land": np.mean(land_deltas),
        "USA_Imp_Water": np.mean(water_deltas),
        "USA_Std_Overall": np.std(overall_deltas),
        "USA_Std_Land": np.std(land_deltas),
        "USA_Std_Water": np.std(water_deltas)
    })

    print(f"Feature: {name:12s} | Overall Imp: +{np.mean(overall_deltas):.5f} | Land Imp: +{np.mean(land_deltas):.5f} | Water Imp: +{np.mean(water_deltas):.5f}")

df_usa_pfi = pd.DataFrame(usa_pfi_results)
df_usa_pfi.to_csv("USA_PFI_Land_Water.csv", index=False)
print("\nSaved USA PFI results to USA_PFI_Land_Water.csv")

#PFI for Amazon
NUM_REPEATS = 20  # Matches USA repeats

amazon_pfi_results = []
base_off_amz = base_amazon['mae_overall']
base_lnd_amz = base_amazon['mae_land']
base_wtr_amz = base_amazon['mae_water']

raw_feats_amazon = pfi_bundle_amazon['test_feats']

print(f"Starting Amazon PFI ({len(feature_names)} features, {NUM_REPEATS} repeats each)...")
print("-----------------------------------------------------------------------")

for i, name in enumerate(feature_names):
    overall_deltas = []
    land_deltas = []
    water_deltas = []

    for r in range(NUM_REPEATS):
        # 1. Permute (shuffle) single feature channel
        perm_feats = get_permuted_features(raw_feats_amazon, i)

        # 2. Re-wrap into DataLoader
        perm_ds = CustomDataset(
            features=perm_feats,
            labels=pfi_bundle_amazon['test_labels'],
            cls_labels=pfi_bundle_amazon['test_labels_class'],
            LWI=pfi_bundle_amazon['test_LWI']
        )
        perm_loader = DataLoader(
            perm_ds, 
            batch_size=pfi_bundle_amazon['BATCH_SIZE'], 
            shuffle=False
        )

        # 3. Compute degraded MAE
        m = compute_masked_mae(model, perm_loader, device)

        # 4. Calculate performance drop (Delta MAE = Degraded - Baseline)
        overall_deltas.append(m['mae_overall'] - base_off_amz)
        land_deltas.append(m['mae_land'] - base_lnd_amz)
        water_deltas.append(m['mae_water'] - base_wtr_amz)

    # Save mean importance scores
    amazon_pfi_results.append({
        "Feature": name,
        "Amazon_Imp_Overall": np.mean(overall_deltas),
        "Amazon_Imp_Land": np.mean(land_deltas),
        "Amazon_Imp_Water": np.mean(water_deltas),
        "Amazon_Std_Overall": np.std(overall_deltas),
        "Amazon_Std_Land": np.std(land_deltas),
        "Amazon_Std_Water": np.std(water_deltas)
    })

    print(f"Feature: {name:12s} | Overall Imp: +{np.mean(overall_deltas):.5f} | Land Imp: +{np.mean(land_deltas):.5f} | Water Imp: +{np.mean(water_deltas):.5f}")

df_amazon_pfi = pd.DataFrame(amazon_pfi_results)
df_amazon_pfi.to_csv("Amazon_PFI_Land_Water.csv", index=False)
print("\nSaved Amazon PFI results to Amazon_PFI_Land_Water.csv")


# # =====================================================================
# # 2. DEFINE THE PLOTTING FUNCTION (UPDATED)
# # =====================================================================
# def plot_mae_importance_only(baseline_mae, mae_results, save_path=None):
#    # 1. Define your custom, clean labels
#    name_mapping = {
#        'cape': 'CAPE',
#        'cldfrac': 'Convective Cloud Fraction @ 440hPa',
#        'cldht': 'Convective Cloud Height',
#        'crh': 'Column Relative Humidity',
#        'iceflux': 'Max. Convective Ice Flux',
#        'iwc440': 'Ice Water Content 440',
#        'lcl': 'LCL',
#        'massflux': 'Cummulative Mass Flux @ 440hPa',
#        'maxcloudfrac': 'Max. Convective Cloud Fraction',
#        'mseratio': 'Moist Static Energy Ratio',
#        'precon': 'Deep Convective Precip.',
#        'tlapse': 'Temperature Lapse Rate'
#    }
    
#     # 2. Prepare Data (Now extracting Std Dev as well)
#    plot_data = []
#    for feat in mae_results['Feature'].unique():
#        # Extract mean and std values
#        r_val = mae_results.loc[mae_results['Feature'] == feat, 'Mean Importance'].values[0]
#        std_val = mae_results.loc[mae_results['Feature'] == feat, 'Std Dev'].values[0]

#        # Calculate percentage change for the mean impact
#        r_pct = -(r_val / baseline_mae) * 100
       
#        # Calculate the standard deviation as a percentage to scale with the bars
#        std_pct = (std_val / baseline_mae) * 100

#        clean_name = name_mapping.get(feat, feat)

#        plot_data.append({
#            "Feature": clean_name, 
#            "Performance Impact %": r_pct,
#            "Error %": std_pct
#        })

#    # Sort values so the most important features (most negative) are at the bottom
#    df = pd.DataFrame(plot_data).sort_values(by="Performance Impact %", ascending=True)

#     3. Plotting
#    fig, ax = plt.subplots(figsize=(14, 8))
#    y = np.arange(len(df))
#    width = 0.6 

# # Plot the bar chart with horizontal error bars (xerr)
#    ax.barh(y, df["Performance Impact %"], width, 
#            color='skyblue', edgecolor='black',
#            xerr=df["Error %"], 
#            error_kw={'ecolor': 'darkblue', 'capsize': 4, 'elinewidth': 1.5})

#     # Formatting
#    ax.set_xlabel('Change in Model Performance When Variable is Shuffled (%)', fontsize=14)
#    ax.set_title('SEILER-Net Feature Importance: MAE Performance Impact', fontsize=16, fontweight='bold')
#     ax.set_yticks(y)
#     ax.set_yticklabels(df["Feature"], fontsize=14) 

#     # Adjust x limits based on the furthest error bars, not just the bars themselves
#     x_min = (df["Performance Impact %"] - df["Error %"]).min()
#     x_max = (df["Performance Impact %"] + df["Error %"]).max()
#     ax.set_xlim(x_min - 5, x_max + 5) 

#     # Add a strong center line
#     ax.axvline(0, color='black', linewidth=1.5, alpha=0.8)
#     ax.grid(axis='x', linestyle='--', alpha=0.4)

#     # 4. Add data labels to the ends of the error bars
#     for i, (r, err) in enumerate(zip(df["Performance Impact %"], df["Error %"])):
#         # Shift the text past the end of the error bar cap (err + 0.5)
#         if r < 0:
#             offset = err + 0.5 
#             align = 'right'
#             text_x = r - offset
#         else:
#             offset = err + 0.5
#             align = 'left'
#             text_x = r + offset
            
#         ax.text(text_x, i, f'{r:.1f}%', va='center', ha=align, fontsize=11)

#     plt.tight_layout()

#     # Save the figure BEFORE plt.show()
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"Figure successfully saved to: {save_path}")

#     plt.show()

# =====================================================================
# 3. GENERATE THE PLOT (The Function Call)
# =====================================================================
# plot_mae_importance_only(
#     baseline_mae=baseline_mae,
#     mae_results=df_pfi_final,
#     save_path="SEILER_Net_Feature_Importance_MAE.png"
# )
