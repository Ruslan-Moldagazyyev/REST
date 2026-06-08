import os
import sys
import json
import random
import warnings
import copy
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from torchvision import transforms
from PIL import Image

from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score
from sklearn.cluster import KMeans

import medmnist
from medmnist import BreastMNIST

warnings.filterwarnings("ignore")

device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")

torch.cuda.set_device(device)

print("Using device:", device)
print("GPU name:", torch.cuda.get_device_name(device))


class DualLogger:
    """Logger that writes to both stdout and a file."""
    def __init__(self):
        self.terminal = sys.stdout
        self.log_file = None
    
    def start(self, log_path):
        self.log_file = open(log_path, 'w')
    
    def write(self, message):
        self.terminal.write(message)
        if self.log_file:
            self.log_file.write(message)
            self.log_file.flush()
    
    def flush(self):
        self.terminal.flush()
        if self.log_file:
            self.log_file.flush()
    
    def stop(self):
        if self.log_file:
            self.log_file.close()
            self.log_file = None


LABELED_FRACTION = 0.10

BATCH_SIZE_TRAIN = 16
BATCH_SIZE_EVAL = 32
NUM_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 7
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-1
NUM_CLASSES = 2  # Binary classification for BreastMNIST
IMG_SIZE = 28


CONFIDENCE_THRESHOLD = 0.75

# Alpha weight for pseudo-label loss (0-1)
# Loss = CE(true_labels) + alpha * CE(pseudo_labels)
# Lower alpha = less trust in pseudo-labels
PSEUDO_LABEL_ALPHA = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class BasicBlock(nn.Module):
    """Basic residual block for ResNet."""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet18(nn.Module):
    """
    ResNet-18 architecture for 28x28 grayscale images.
    Modified channel configuration [22, 44, 88, 176] for ~1.32M params.
    """
    def __init__(self, num_classes=2, in_channels=1):
        super().__init__()
        self.in_planes = 22
        
        self.conv1 = nn.Conv2d(in_channels, 22, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(22)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(22, 2, stride=1)    # 28x28
        self.layer2 = self._make_layer(44, 2, stride=2)    # 14x14
        self.layer3 = self._make_layer(88, 2, stride=2)    # 7x7
        self.layer4 = self._make_layer(176, 2, stride=2)   # 4x4
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(176, num_classes)
        
        self._initialize_weights()
        
    def _make_layer(self, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_planes != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        layers.append(BasicBlock(self.in_planes, out_channels, stride, downsample))
        self.in_planes = out_channels
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
    
    def get_penultimate_features(self, x):
        """Extract features from the penultimate layer (before fc)."""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # (batch, 176)
        return x


class AddGaussianNoise:
    """Add Gaussian noise to tensor."""
    def __init__(self, mean=0.0, std=0.05):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)


class AddSaltPepperNoise:
    """Add salt-and-pepper noise to tensor."""
    def __init__(self, prob=0.02):
        self.prob = prob
    
    def __call__(self, tensor):
        noise_mask = torch.rand_like(tensor)
        salt = (noise_mask < self.prob / 2).float()
        pepper = (noise_mask > 1 - self.prob / 2).float()
        noisy = tensor * (1 - salt - pepper) + salt
        return torch.clamp(noisy, 0.0, 1.0)


class AddSpeckleNoise:
    """Add speckle noise (multiplicative) - common in ultrasound images."""
    def __init__(self, std=0.1):
        self.std = std
    
    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + tensor * noise, 0.0, 1.0)


def get_augmentation_transforms():
    """
    Define three distinct augmentation strategies for BreastMNIST.
    BreastMNIST is grayscale breast ultrasound images.
    
    Key insight from research: For medical images, NOISE-BASED augmentations
    are more effective than aggressive intensity changes.
    
    For ultrasound specifically:
    - Speckle noise is characteristic of ultrasound imaging
    - Gaussian noise simulates sensor noise
    - Mild blur simulates probe coupling variations
    """
    
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    
    transform_pure = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize
    ])

  
    transform_noise = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        # Noise-based augmentations (applied to tensor)
        transforms.RandomApply([AddSpeckleNoise(std=0.08)], p=0.4),  # Ultrasound-specific
        transforms.RandomApply([AddGaussianNoise(mean=0.0, std=0.03)], p=0.3),
        transforms.RandomApply([AddSaltPepperNoise(prob=0.01)], p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))], p=0.25),
        # Very mild brightness/contrast only
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.1, contrast=0.1)
        ], p=0.3),
        normalize
    ])

  
    transform_geometric = transforms.Compose([
        transforms.Resize((IMG_SIZE + 4, IMG_SIZE + 4)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=5
        ),
        transforms.ToTensor(),
        normalize
    ])
    
    return {
        "minimal": transform_pure,       # Clean anchor view
        "color": transform_noise,        # Noise-based (key name kept for compatibility)
        "geometric": transform_geometric # Spatial variations
    }


def get_eval_transform():
    """Standard evaluation transform (no augmentation) - grayscale."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])



class MultiAugmentationDataset(Dataset):
    """Dataset wrapper that returns multiple augmented views of each image."""
    
    def __init__(self, base_dataset, transforms_dict):
        self.base_dataset = base_dataset
        self.transforms = transforms_dict
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img_array = self.base_dataset.imgs[idx]
        label = self.base_dataset.labels[idx]
        
        img_pil = Image.fromarray(img_array)
        
        img_minimal = self.transforms["minimal"](img_pil)
        img_color = self.transforms["color"](img_pil)
        img_geometric = self.transforms["geometric"](img_pil)
        
        return img_minimal, img_color, img_geometric, label


class MultiAugmentationSubset(Dataset):
    """Subset wrapper for MultiAugmentationDataset. Returns is_pseudo=0 (true labels)."""
    
    def __init__(self, multi_aug_dataset, indices):
        self.dataset = multi_aug_dataset
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        img_min, img_color, img_geo, label = self.dataset[self.indices[idx]]
        # Return is_pseudo = 0 (this is a true-labeled sample)
        return img_min, img_color, img_geo, label, 0


class PseudoLabelMultiAugDataset(Dataset):
    """Dataset that applies multiple augmentations with pseudo-labels. Returns is_pseudo=1."""
    
    def __init__(self, base_dataset, indices, pseudo_labels, transforms_dict):
        self.base_dataset = base_dataset
        self.indices = indices
        self.pseudo_labels = pseudo_labels
        self.transforms = transforms_dict
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        
        img_array = self.base_dataset.imgs[real_idx]
        img_pil = Image.fromarray(img_array)
        
        img_minimal = self.transforms["minimal"](img_pil)
        img_color = self.transforms["color"](img_pil)
        img_geometric = self.transforms["geometric"](img_pil)
        
        label = np.array([self.pseudo_labels[idx]], dtype=np.int64)
        
        # Return is_pseudo = 1 (this is a pseudo-labeled sample)
        return img_minimal, img_color, img_geometric, label, 1


class JointAugmentationEnsemble(nn.Module):
    """
    Joint Ensemble with SAME architecture but DIFFERENT augmentation views.
    """
    
    def __init__(self, num_classes=2, in_channels=1):
        super().__init__()
        
        self.model_minimal = ResNet18(num_classes=num_classes, in_channels=in_channels)
        self.model_color = ResNet18(num_classes=num_classes, in_channels=in_channels)
        self.model_geometric = ResNet18(num_classes=num_classes, in_channels=in_channels)
        
        self.models = [self.model_minimal, self.model_color, self.model_geometric]
        self.model_names = ["ResNet18_Pure", "ResNet18_Intensity", "ResNet18_Geometric"]
        self.augmentation_types = ["pure", "intensity", "geometric"]
    
    def forward(self, img_minimal, img_color, img_geometric, return_individual=False):
        logits_minimal = self.model_minimal(img_minimal)
        logits_color = self.model_color(img_color)
        logits_geometric = self.model_geometric(img_geometric)
        
        all_logits = torch.stack([logits_minimal, logits_color, logits_geometric], dim=0)
        ensemble_logits = all_logits.mean(dim=0)
        
        if return_individual:
            return ensemble_logits, [logits_minimal, logits_color, logits_geometric]
        return ensemble_logits
    
    def forward_single_image(self, img, return_individual=False):
        logits_minimal = self.model_minimal(img)
        logits_color = self.model_color(img)
        logits_geometric = self.model_geometric(img)
        
        all_logits = torch.stack([logits_minimal, logits_color, logits_geometric], dim=0)
        ensemble_logits = all_logits.mean(dim=0)
        
        if return_individual:
            return ensemble_logits, [logits_minimal, logits_color, logits_geometric]
        return ensemble_logits
    
    def get_ensemble_probabilities(self, img):
        """Get soft-voting probabilities for evaluation (single image input)."""
        self.eval()
        with torch.no_grad():
            logits_minimal = self.model_minimal(img)
            logits_color = self.model_color(img)
            logits_geometric = self.model_geometric(img)
            
            prob_minimal = F.softmax(logits_minimal, dim=1)
            prob_color = F.softmax(logits_color, dim=1)
            prob_geometric = F.softmax(logits_geometric, dim=1)
            
            ensemble_probs = (prob_minimal + prob_color + prob_geometric) / 3.0
            
        return ensemble_probs
    
    def get_concatenated_features(self, img):
        """
        Extract concatenated penultimate features from all 3 models.
        Returns (batch, 176*3=528) dimensional features for TypiClust.
        """
        self.eval()
        with torch.no_grad():
            feat1 = self.model_minimal.get_penultimate_features(img)
            feat2 = self.model_color.get_penultimate_features(img)
            feat3 = self.model_geometric.get_penultimate_features(img)
            return torch.cat([feat1, feat2, feat3], dim=1)  # (batch, 528)
    
    def print_param_counts(self):
        print("\n" + "="*70)
        print("AUGMENTATION-BASED JOINT ENSEMBLE ARCHITECTURE")
        print("="*70)
        print("Architecture: ResNet18 [22, 44, 88, 176] (identical for all three models)")
        print("Diversity Source: Different augmentation strategies")
        print("-"*70)
        total_params = 0
        for name, model, aug_type in zip(self.model_names, self.models, self.augmentation_types):
            params = sum(p.numel() for p in model.parameters())
            total_params += params
            print(f"  {name} [{aug_type}]: {params:,} params")
        print(f"  {'Total Ensemble'}: {total_params:,} params")
        print("="*70)


def build_joint_ensemble():
    """Build the joint augmentation ensemble model."""
    ensemble = JointAugmentationEnsemble(num_classes=NUM_CLASSES, in_channels=1).to(DEVICE)
    ensemble.print_param_counts()
    return ensemble


def compute_class_weights(dataset_raw, indices):
    """
    Compute inverse frequency class weights for handling class imbalance.
    Returns a tensor of weights suitable for nn.CrossEntropyLoss.
    """
    labels = [int(dataset_raw.labels[idx][0]) for idx in indices]
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    
    total = sum(class_counts)
    class_weights = total / (NUM_CLASSES * class_counts.astype(float))
    
    print(f"  Class distribution: {dict(enumerate(class_counts.tolist()))}, "
          f"weights: {[f'{w:.4f}' for w in class_weights]}")
    
    return torch.FloatTensor(class_weights).to(DEVICE)


def prepare_data(seed, num_batches):
    """Prepare BreastMNIST data with multi-augmentation support."""
    print("\n" + "="*60)
    print("DATA PREPARATION")
    print("="*60)
    
    aug_transforms = get_augmentation_transforms()
    eval_transform = get_eval_transform()
    
    train_dataset_raw = BreastMNIST(split='train', download=True, transform=None)
    val_dataset = BreastMNIST(split='val', download=True, transform=eval_transform)
    test_dataset = BreastMNIST(split='test', download=True, transform=eval_transform)
    
    train_multi_aug = MultiAugmentationDataset(train_dataset_raw, aug_transforms)
    
    n_train = len(train_dataset_raw)
    n_labeled = int(n_train * LABELED_FRACTION)
    
    rng = random.Random(seed)
    indices = list(range(n_train))
    rng.shuffle(indices)
    
    labeled_indices = indices[:n_labeled]
    unlabeled_indices = indices[n_labeled:]
    
    # Compute class weights from labeled set
    class_weights = compute_class_weights(train_dataset_raw, labeled_indices)
    
    # We no longer pre-split into batches - TypiClust selects dynamically
    batch_size_M = len(unlabeled_indices) // num_batches
    
    all_true_labels = {}
    for idx in range(n_train):
        label = train_dataset_raw.labels[idx]
        all_true_labels[idx] = int(label[0])
    
    print(f"Dataset: BreastMNIST (Binary, {NUM_CLASSES} classes)")
    print(f"Total training: {n_train}")
    print(f"Labeled ({LABELED_FRACTION*100:.0f}%): {n_labeled}")
    print(f"Unlabeled: {len(unlabeled_indices)}")
    print(f"Mini-batches: {num_batches} (~{batch_size_M} samples each)")
    print(f"Validation: {len(val_dataset)}")
    print(f"Test: {len(test_dataset)}")
    print("\nAugmentation Strategies:")
    print("  Model 1: Pure (no augmentation)")
    print("  Model 2: Intensity (brightness, contrast, blur)")
    print("  Model 3: Geometric (rotation, translation, scale, shear, flips)")
    
    return {
        "train_dataset_raw": train_dataset_raw,
        "train_multi_aug": train_multi_aug,
        "aug_transforms": aug_transforms,
        "eval_transform": eval_transform,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "labeled_indices": labeled_indices,
        "unlabeled_indices": unlabeled_indices,
        "num_batches": num_batches,
        "batch_size_M": batch_size_M,
        "all_true_labels": all_true_labels,
        "class_weights": class_weights,
        "n_train": n_train
    }



def train_joint_ensemble(ensemble, train_loader, val_loader, num_epochs, lr,
                         class_weights, verbose=True, pseudo_label_alpha=1.0):
    """
    Train the joint augmentation ensemble with class-weighted loss.
    
    Loss function:
    - For true labels: CE(y_true, y_pred) with weight 1.0
    - For pseudo-labels: CE(y_pseudo, y_pred) with weight pseudo_label_alpha
    
    This allows down-weighting pseudo-labels which are less reliable.
    """
    # CLASS-WEIGHTED LOSS to handle imbalance (reduction='none' for per-sample weighting)
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
    
    optimizer = optim.AdamW(ensemble.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_auc = 0.0
    best_epoch = 0
    best_state = None
    patience_counter = 0
    
    for epoch in range(1, num_epochs + 1):
        ensemble.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            # Check if batch includes is_pseudo flag (5 elements) or not (4 elements)
            if len(batch) == 5:
                img_minimal, img_color, img_geometric, labels, is_pseudo = batch
                is_pseudo = is_pseudo.float().to(DEVICE)
            else:
                # Fallback for datasets without is_pseudo flag (initial training, upper bound, etc.)
                img_minimal, img_color, img_geometric, labels = batch
                is_pseudo = None
            
            img_minimal = img_minimal.to(DEVICE)
            img_color = img_color.to(DEVICE)
            img_geometric = img_geometric.to(DEVICE)
            labels = labels.view(-1).long().to(DEVICE)
            
            optimizer.zero_grad()
            
            ensemble_logits = ensemble(img_minimal, img_color, img_geometric)
            
            # Compute per-sample loss
            per_sample_loss = criterion(ensemble_logits, labels)
            
            if is_pseudo is not None and pseudo_label_alpha < 1.0:
                sample_weights = 1.0 - is_pseudo * (1.0 - pseudo_label_alpha)
                loss = (per_sample_loss * sample_weights).mean()
            else:
                # No pseudo-labels or alpha=1.0, use standard mean
                loss = per_sample_loss.mean()
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        
        val_metrics = evaluate_joint_ensemble(ensemble, val_loader)
        val_auc = val_metrics["ensemble"]["auc"]
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_state = copy.deepcopy(ensemble.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            if verbose:
                print(f"    Early stop at epoch {epoch}, best: {best_epoch}, Val AUC: {best_val_auc:.4f}")
            break
    
    if best_state is not None:
        ensemble.load_state_dict(best_state)
    
    return ensemble, best_val_auc, best_epoch


def evaluate_joint_ensemble(ensemble, loader):
    ensemble.eval()
    
    all_ensemble_preds = []
    all_ensemble_probs = []
    all_individual_probs = [[], [], []]
    all_individual_preds = [[], [], []]
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            labels = labels.view(-1).long()
            
            ensemble_logits, individual_logits = ensemble.forward_single_image(imgs, return_individual=True)
            
            ensemble_probs = F.softmax(ensemble_logits, dim=1)
            ensemble_preds = ensemble_logits.argmax(1).cpu()
            
            all_ensemble_preds.extend(ensemble_preds.numpy())
            all_ensemble_probs.extend(ensemble_probs.cpu().numpy())
            
            for i, logits in enumerate(individual_logits):
                probs = F.softmax(logits, dim=1)
                preds = logits.argmax(1).cpu()
                all_individual_preds[i].extend(preds.numpy())
                all_individual_probs[i].extend(probs.cpu().numpy())
            
            all_labels.extend(labels.numpy())
    
    all_labels = np.array(all_labels)
    all_ensemble_probs = np.array(all_ensemble_probs)
    
    results = {}
    
    # Binary classification
    ensemble_probs_positive = all_ensemble_probs[:, 1]
    results["ensemble"] = {
        "balanced_acc": balanced_accuracy_score(all_labels, np.array(all_ensemble_preds)),
        "auc": roc_auc_score(all_labels, ensemble_probs_positive),
        "f1": f1_score(all_labels, np.array(all_ensemble_preds), average='binary')
    }
    
    results["Ensemble"] = results["ensemble"]
    
    for i, name in enumerate(["ResNet18_Pure", "ResNet18_Intensity", "ResNet18_Geometric"]):
        individual_probs = np.array(all_individual_probs[i])
        individual_probs_positive = individual_probs[:, 1]
        results[name] = {
            "balanced_acc": balanced_accuracy_score(all_labels, np.array(all_individual_preds[i])),
            "auc": roc_auc_score(all_labels, individual_probs_positive),
            "f1": f1_score(all_labels, np.array(all_individual_preds[i]), average='binary')
        }
    
    return results


def extract_ensemble_features(ensemble, dataset_raw, indices, eval_transform):
    """
    Extract concatenated penultimate features from all 3 models for given indices.
    Returns (N, 528) numpy array (176 features * 3 models).
    """
    ensemble.eval()
    
    class SimpleEvalDataset(Dataset):
        def __init__(self, base_dataset, indices, transform):
            self.base_dataset = base_dataset
            self.indices = indices
            self.transform = transform
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            real_idx = self.indices[idx]
            img_array = self.base_dataset.imgs[real_idx]
            img_pil = Image.fromarray(img_array)
            img = self.transform(img_pil)
            return img
    
    eval_dataset = SimpleEvalDataset(dataset_raw, indices, eval_transform)
    loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE_EVAL, shuffle=False)
    
    all_features = []
    with torch.no_grad():
        for imgs in loader:
            imgs = imgs.to(DEVICE)
            features = ensemble.get_concatenated_features(imgs)
            all_features.append(features.cpu().numpy())
    
    return np.concatenate(all_features, axis=0)


def typiclust_select(ensemble, dataset_raw, unlabeled_indices, eval_transform, batch_size):
    print(f"    [TypiClust] Extracting features for {len(unlabeled_indices)} unlabeled samples...")
    features = extract_ensemble_features(ensemble, dataset_raw, unlabeled_indices, eval_transform)
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    print(f"    [TypiClust] Running k-means with k={batch_size}...")
    kmeans = KMeans(n_clusters=batch_size, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_scaled)
    
    selected_local_indices = []
    for c in range(batch_size):
        cluster_mask = cluster_labels == c
        cluster_indices_local = np.where(cluster_mask)[0]
        
        if len(cluster_indices_local) == 0:
            continue
        
        cluster_features = features_scaled[cluster_indices_local]
        centroid = kmeans.cluster_centers_[c]
        distances = np.linalg.norm(cluster_features - centroid, axis=1)
        nearest = cluster_indices_local[np.argmin(distances)]
        selected_local_indices.append(nearest)
    
    selected_global = [unlabeled_indices[i] for i in selected_local_indices]
    
    selected_set = set(selected_local_indices)
    remaining_local = [i for i in range(len(unlabeled_indices)) if i not in selected_set]
    remaining_global = [unlabeled_indices[i] for i in remaining_local]
    
    print(f"    [TypiClust] Selected {len(selected_global)} representative samples")
    
    return selected_global, remaining_global


def ensemble_confidence_pseudo_labeling(ensemble, dataset_raw, batch_indices, true_labels,
                                        eval_transform, confidence_threshold=0.75):
    """
    Generate pseudo-labels using joint ensemble confidence.
    """
    n_samples = len(batch_indices)
    
    class SimpleEvalDataset(Dataset):
        def __init__(self, base_dataset, indices, transform):
            self.base_dataset = base_dataset
            self.indices = indices
            self.transform = transform
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            real_idx = self.indices[idx]
            img_array = self.base_dataset.imgs[real_idx]
            label = self.base_dataset.labels[real_idx]
            img_pil = Image.fromarray(img_array)
            img = self.transform(img_pil)
            return img, label
    
    eval_dataset = SimpleEvalDataset(dataset_raw, batch_indices, eval_transform)
    loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE_EVAL, shuffle=False)
    
    ensemble.eval()
    all_probs = []
    
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(DEVICE)
            probs = ensemble.get_ensemble_probabilities(imgs)
            all_probs.extend(probs.cpu().numpy())
    
    all_probs = np.array(all_probs)
    ensemble_preds = all_probs.argmax(axis=1)
    ensemble_confidences = all_probs.max(axis=1)
    
    accepted_indices = []
    accepted_labels = []
    
    class_distribution = {i: 0 for i in range(NUM_CLASSES)}
    
    stats = {
        "total": n_samples,
        "accepted": 0,
        "rejected": 0,
        "correct_pseudo": 0,
        "class_distribution": class_distribution,
        "confidences_accepted": [],
        "confidences_rejected": []
    }
    
    for i in range(n_samples):
        conf = ensemble_confidences[i]
        pred = ensemble_preds[i]
        
        if conf >= confidence_threshold:
            accepted_indices.append(batch_indices[i])
            accepted_labels.append(int(pred))
            stats["accepted"] += 1
            stats["confidences_accepted"].append(conf)
            stats["class_distribution"][int(pred)] += 1
            
            if pred == true_labels[batch_indices[i]]:
                stats["correct_pseudo"] += 1
        else:
            stats["rejected"] += 1
            stats["confidences_rejected"].append(conf)
    
    stats["acceptance_rate"] = stats["accepted"] / n_samples if n_samples > 0 else 0
    stats["pseudo_accuracy"] = stats["correct_pseudo"] / stats["accepted"] if stats["accepted"] > 0 else 0
    stats["avg_confidence_accepted"] = np.mean(stats["confidences_accepted"]) if stats["confidences_accepted"] else 0
    stats["avg_confidence_rejected"] = np.mean(stats["confidences_rejected"]) if stats["confidences_rejected"] else 0
    stats["avg_confidence_all"] = np.mean(ensemble_confidences)
    
    return accepted_indices, accepted_labels, stats



def run_upper_bound_experiment(data_dict):
    """Train joint ensemble with 100% labeled data to establish upper bound."""
    print("\n" + "="*70)
    print("UPPER BOUND EXPERIMENT: 100% LABELED DATA (AUGMENTATION ENSEMBLE)")
    print("="*70)
    
    ensemble = build_joint_ensemble()
    
    train_dataset_raw = data_dict["train_dataset_raw"]
    train_multi_aug = data_dict["train_multi_aug"]
    val_dataset = data_dict["val_dataset"]
    test_dataset = data_dict["test_dataset"]
    n_train = data_dict["n_train"]
    
    # Compute class weights from full training set
    all_indices = list(range(n_train))
    class_weights = compute_class_weights(train_dataset_raw, all_indices)
    
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_EVAL, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_EVAL, shuffle=False)
    
    full_train_subset = MultiAugmentationSubset(train_multi_aug, all_indices)
    train_loader = DataLoader(full_train_subset, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    
    print(f"\nTraining AUGMENTATION ENSEMBLE with 100% labeled data ({n_train} samples)...")
    
    ensemble, best_auc, best_epoch = train_joint_ensemble(
        ensemble, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, class_weights, verbose=True
    )
    print(f"  -> Best epoch: {best_epoch}, Best Val AUC: {best_auc:.4f}")
    
    print("\n  Upper Bound Evaluation:")
    val_metrics = evaluate_joint_ensemble(ensemble, val_loader)
    test_metrics = evaluate_joint_ensemble(ensemble, test_loader)
    
    upper_bound_metrics = {}
    for name in ["ResNet18_Pure", "ResNet18_Intensity", "ResNet18_Geometric", "Ensemble"]:
        upper_bound_metrics[name] = {
            "val": val_metrics[name],
            "test": test_metrics[name]
        }
        print(f"    {name} - Val AUC: {val_metrics[name]['auc']:.4f}, "
              f"Test AUC: {test_metrics[name]['auc']:.4f}")
    
    return upper_bound_metrics


def run_full_batch_st(data_dict, confidence_threshold, experiment_name="Aug_Full_Batch_ST"):
    """Full-Batch Self-Training (F-ST) baseline with augmentation ensemble."""
    
    print("\n" + "="*70)
    print(f"EXPERIMENT: {experiment_name}")
    print(f"Full-Batch ST (F-ST) with AUGMENTATION ENSEMBLE")
    print(f"Confidence Threshold: >= {confidence_threshold}")
    print("="*70)
    
    ensemble = build_joint_ensemble()
    
    train_dataset_raw = data_dict["train_dataset_raw"]
    train_multi_aug = data_dict["train_multi_aug"]
    aug_transforms = data_dict["aug_transforms"]
    eval_transform = data_dict["eval_transform"]
    val_dataset = data_dict["val_dataset"]
    test_dataset = data_dict["test_dataset"]
    labeled_indices = data_dict["labeled_indices"].copy()
    unlabeled_indices = data_dict["unlabeled_indices"]
    true_labels = data_dict["all_true_labels"]
    class_weights = data_dict["class_weights"]
    
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_EVAL, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_EVAL, shuffle=False)
    
    # Phase 1: Initial Training
    print("\n--- Phase 1: Initial Augmentation Ensemble Training on Labeled Data ---")
    
    labeled_subset = MultiAugmentationSubset(train_multi_aug, labeled_indices)
    train_loader = DataLoader(labeled_subset, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    
    ensemble, best_auc, best_epoch = train_joint_ensemble(
        ensemble, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, class_weights, verbose=True
    )
    print(f"  -> Best epoch: {best_epoch}, Best Val AUC: {best_auc:.4f}")
    
    print("\n  Initial Evaluation (Labeled Only):")
    init_val = evaluate_joint_ensemble(ensemble, val_loader)
    init_test = evaluate_joint_ensemble(ensemble, test_loader)
    init_metrics = {}
    for name in ["ResNet18_Pure", "ResNet18_Intensity", "ResNet18_Geometric", "Ensemble"]:
        init_metrics[name] = {"val": init_val[name], "test": init_test[name]}
        print(f"    {name} - Val AUC: {init_val[name]['auc']:.4f}, Test AUC: {init_test[name]['auc']:.4f}")
    
    # Phase 2: Pseudo-label all unlabeled
    print("\n--- Phase 2: Pseudo-label ALL Unlabeled Data (Full-Batch) ---")
    print(f"    Total unlabeled samples: {len(unlabeled_indices)}")
    
    accepted_indices, accepted_labels, stats = ensemble_confidence_pseudo_labeling(
        ensemble, train_dataset_raw, unlabeled_indices, true_labels,
        eval_transform, confidence_threshold
    )
    
    print(f"    Accepted: {stats['accepted']}/{stats['total']} ({stats['acceptance_rate']*100:.1f}%)")
    print(f"    Pseudo-label accuracy: {stats['pseudo_accuracy']*100:.1f}%")
    print(f"    Avg confidence - accepted: {stats['avg_confidence_accepted']:.4f}, "
          f"rejected: {stats['avg_confidence_rejected']:.4f}")
    print(f"    Accepted per class: {stats['class_distribution']}")

    # Phase 3: Retrain from scratch
    print("\n--- Phase 3: Retrain Augmentation Ensemble from Scratch ---")
    
    ensemble = build_joint_ensemble()
    
    labeled_subset = MultiAugmentationSubset(train_multi_aug, labeled_indices)
    
    # Recompute class weights including pseudo-labels
    all_training_labels = [int(train_dataset_raw.labels[idx][0]) for idx in labeled_indices]
    all_training_labels += accepted_labels
    training_counts = np.bincount(all_training_labels, minlength=NUM_CLASSES)
    total = sum(training_counts)
    updated_weights = total / (NUM_CLASSES * training_counts.astype(float))
    updated_class_weights = torch.FloatTensor(updated_weights).to(DEVICE)
    
    if accepted_indices:
        pseudo_subset = PseudoLabelMultiAugDataset(
            train_dataset_raw, accepted_indices, accepted_labels, aug_transforms
        )
        combined_dataset = ConcatDataset([labeled_subset, pseudo_subset])
    else:
        combined_dataset = labeled_subset
    
    train_loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    
    print(f"    Training samples: {len(labeled_indices)} labeled + {len(accepted_indices)} pseudo-labeled")
    print(f"    Pseudo-label weight (alpha): {PSEUDO_LABEL_ALPHA}")
    
    ensemble, best_auc, best_epoch = train_joint_ensemble(
        ensemble, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, 
        updated_class_weights, verbose=True, pseudo_label_alpha=PSEUDO_LABEL_ALPHA
    )
    print(f"  -> Best epoch: {best_epoch}, Best Val AUC: {best_auc:.4f}")
    
    print("\n  F-ST Final Evaluation:")
    final_val = evaluate_joint_ensemble(ensemble, val_loader)
    final_test = evaluate_joint_ensemble(ensemble, test_loader)
    final_metrics = {}
    for name in ["ResNet18_Pure", "ResNet18_Intensity", "ResNet18_Geometric", "Ensemble"]:
        final_metrics[name] = {"val": final_val[name], "test": final_test[name]}
        print(f"    {name} - Val AUC: {final_val[name]['auc']:.4f}, Test AUC: {final_test[name]['auc']:.4f}")
    
    return {
        "experiment_name": experiment_name,
        "initial_metrics": init_metrics,
        "final_metrics": final_metrics,
        "pseudo_stats": stats
    }


def run_joint_ensemble_srpm_st(data_dict, confidence_threshold, experiment_name="Aug_Ensemble_SRPM_ST"):
    
    print("\n" + "="*70)
    print(f"EXPERIMENT: {experiment_name}")
    print(f"AUGMENTATION-BASED JOINT ENSEMBLE SRPM-ST WITH TYPICLUST")
    print(f"Confidence Threshold: >= {confidence_threshold}")
    print("="*70)
    
    ensemble = build_joint_ensemble()
    
    train_dataset_raw = data_dict["train_dataset_raw"]
    train_multi_aug = data_dict["train_multi_aug"]
    aug_transforms = data_dict["aug_transforms"]
    eval_transform = data_dict["eval_transform"]
    val_dataset = data_dict["val_dataset"]
    test_dataset = data_dict["test_dataset"]
    labeled_indices = data_dict["labeled_indices"].copy()
    unlabeled_indices = data_dict["unlabeled_indices"].copy()
    true_labels = data_dict["all_true_labels"]
    class_weights = data_dict["class_weights"]
    num_batches = data_dict["num_batches"]
    batch_size_M = data_dict["batch_size_M"]
    
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_EVAL, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_EVAL, shuffle=False)
    
    iteration_results = []
    pseudo_label_stats = []
    
    # Phase 1: Initial Training
    print("\n--- Phase 1: Initial Augmentation Ensemble Training on Labeled Data ---")
    
    labeled_subset = MultiAugmentationSubset(train_multi_aug, labeled_indices)
    train_loader = DataLoader(labeled_subset, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    
    ensemble, best_auc, best_epoch = train_joint_ensemble(
        ensemble, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, class_weights, verbose=True
    )
    print(f"  -> Best epoch: {best_epoch}, Best Val AUC: {best_auc:.4f}")
    
    print("\n  Initial Evaluation (Labeled Only):")
    init_val = evaluate_joint_ensemble(ensemble, val_loader)
    init_test = evaluate_joint_ensemble(ensemble, test_loader)
    init_metrics = {}
    for name in ["ResNet18_Pure", "ResNet18_Intensity", "ResNet18_Geometric", "Ensemble"]:
        init_metrics[name] = {"val": init_val[name], "test": init_test[name]}
        print(f"    {name} - Val AUC: {init_val[name]['auc']:.4f}, Test AUC: {init_test[name]['auc']:.4f}")
    
    iteration_results.append({
        "iteration": 0,
        "phase": "initial",
        "training_samples": len(labeled_indices),
        "metrics": init_metrics
    })
    
    initial_training_metrics = copy.deepcopy(init_metrics)
    
    # Phase 2: Sequential TypiClust-based Mini-Batch Processing
    print("\n--- Phase 2: Sequential Mini-Batch Processing (TYPICLUST + AUGMENTATION ENSEMBLE) ---")
    print(f"    Using: TypiClust selection + Ensemble Confidence (>= {confidence_threshold})")
    print(f"    RETRAINING FROM SCRATCH each iteration (per SRPM-ST Algorithm 1)")
    
    pseudo_labels_dict = {}
    remaining_unlabeled = unlabeled_indices.copy()
    total_accepted = 0
    total_rejected = 0
    
    for batch_idx in range(num_batches):
        if len(remaining_unlabeled) == 0:
            print(f"\n  No more unlabeled samples. Stopping at iteration {batch_idx+1}.")
            break
        
        current_M = min(batch_size_M, len(remaining_unlabeled))
        if current_M < 2:
            print(f"\n  Too few remaining samples ({len(remaining_unlabeled)}). Stopping.")
            break
        
        print(f"\n  Iteration {batch_idx + 1}/{num_batches} "
              f"(selecting {current_M} from {len(remaining_unlabeled)} remaining)")
        
        # TypiClust: select representative samples
        selected_indices, remaining_unlabeled = typiclust_select(
            ensemble, train_dataset_raw, remaining_unlabeled, eval_transform, current_M
        )
        
        # Get pseudo-labels using ensemble confidence
        accepted_indices, accepted_labels, stats = ensemble_confidence_pseudo_labeling(
            ensemble, train_dataset_raw, selected_indices, true_labels,
            eval_transform, confidence_threshold
        )
        
        print(f"    Accepted: {stats['accepted']}/{stats['total']} ({stats['acceptance_rate']*100:.1f}%)")
        print(f"    Pseudo-label accuracy: {stats['pseudo_accuracy']*100:.1f}%")
        print(f"    Avg confidence - accepted: {stats['avg_confidence_accepted']:.4f}, "
              f"rejected: {stats['avg_confidence_rejected']:.4f}")
        print(f"    Accepted per class: {stats['class_distribution']}")
        
        total_accepted += stats['accepted']
        total_rejected += stats['rejected']
        
        pseudo_label_stats.append({"iteration": batch_idx + 1, **stats})
        
        # Add to pseudo-label dict
        for idx, label in zip(accepted_indices, accepted_labels):
            pseudo_labels_dict[idx] = label
        
        # Create combined dataset
        labeled_subset = MultiAugmentationSubset(train_multi_aug, labeled_indices)
        
        if pseudo_labels_dict:
            pseudo_indices = list(pseudo_labels_dict.keys())
            pseudo_labels_list = [pseudo_labels_dict[idx] for idx in pseudo_indices]
            pseudo_subset = PseudoLabelMultiAugDataset(
                train_dataset_raw, pseudo_indices, pseudo_labels_list, aug_transforms
            )
            combined_dataset = ConcatDataset([labeled_subset, pseudo_subset])
        else:
            combined_dataset = labeled_subset
        
        train_loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
        
        print(f"    Training samples: {len(labeled_indices) + len(pseudo_labels_dict)}")
        
        # Recompute class weights including pseudo-labels
        all_training_labels = [int(train_dataset_raw.labels[idx][0]) for idx in labeled_indices]
        if pseudo_labels_dict:
            all_training_labels += [pseudo_labels_dict[idx] for idx in pseudo_labels_dict]
        training_counts = np.bincount(all_training_labels, minlength=NUM_CLASSES)
        total = sum(training_counts)
        updated_weights = total / (NUM_CLASSES * training_counts.astype(float))
        updated_class_weights = torch.FloatTensor(updated_weights).to(DEVICE)
        
        # RETRAIN FROM SCRATCH per SRPM-ST Algorithm 1
        ensemble = JointAugmentationEnsemble(num_classes=NUM_CLASSES, in_channels=1).to(DEVICE)
        
        ensemble, best_auc, best_epoch = train_joint_ensemble(
            ensemble, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE,
            updated_class_weights, verbose=False, pseudo_label_alpha=PSEUDO_LABEL_ALPHA
        )
        print(f"    -> Best epoch: {best_epoch}, Val AUC: {best_auc:.4f}")
        
        # Evaluate
        iter_val = evaluate_joint_ensemble(ensemble, val_loader)
        iter_test = evaluate_joint_ensemble(ensemble, test_loader)
        iter_metrics = {}
        for name in ["ResNet18_Pure", "ResNet18_Intensity", "ResNet18_Geometric", "Ensemble"]:
            iter_metrics[name] = {"val": iter_val[name], "test": iter_test[name]}
        
        print(f"    Results:")
        for name in ["ResNet18_Pure", "ResNet18_Intensity", "ResNet18_Geometric", "Ensemble"]:
            print(f"      {name}: Val AUC {iter_metrics[name]['val']['auc']:.4f}, "
                  f"Test AUC {iter_metrics[name]['test']['auc']:.4f}")
        
        iteration_results.append({
            "iteration": batch_idx + 1,
            "phase": "progressive",
            "training_samples": len(labeled_indices) + len(pseudo_labels_dict),
            "pseudo_labeled_samples": len(pseudo_labels_dict),
            "batch_accepted": stats['accepted'],
            "metrics": iter_metrics
        })
    
    total_processed = total_accepted + total_rejected
    if total_processed > 0:
        print(f"\n  Overall Statistics:")
        print(f"    Total processed: {total_processed}")
        print(f"    Total accepted: {total_accepted} ({total_accepted/total_processed*100:.1f}%)")
        print(f"    Total rejected: {total_rejected} ({total_rejected/total_processed*100:.1f}%)")
    
    return {
        "experiment_name": experiment_name,
        "confidence_threshold": confidence_threshold,
        "iteration_results": iteration_results,
        "pseudo_label_stats": pseudo_label_stats,
        "initial_training_metrics": initial_training_metrics,
        "final_ensemble_state": ensemble.state_dict()
    }, ensemble


# ============================================================
# 12. REPORTING AND SAVING
# ============================================================

def print_final_results(results, upper_bound_metrics=None):
    """Print final results."""
    print("\n" + "="*100)
    print("FINAL RESULTS (AUGMENTATION-BASED JOINT ENSEMBLE SRPM-ST WITH TYPICLUST)")
    print("="*100)
    
    final_metrics = results["iteration_results"][-1]["metrics"]
    initial_metrics = results["initial_training_metrics"]
    
    print(f"\n{'Model':<25} {'Val_Acc':<10} {'Val_AUC':<10} {'Val_F1':<10} "
          f"{'Test_Acc':<10} {'Test_AUC':<10} {'Test_F1':<10}")
    print("-"*95)
    
    for model_name in ["ResNet18_Pure", "ResNet18_Intensity", "ResNet18_Geometric", "Ensemble"]:
        if model_name in final_metrics:
            v = final_metrics[model_name]['val']
            t = final_metrics[model_name]['test']
            print(f"{model_name:<25} {v['balanced_acc']:.4f}    {v['auc']:.4f}    {v['f1']:.4f}    "
                  f"{t['balanced_acc']:.4f}    {t['auc']:.4f}    {t['f1']:.4f}")
    
    print("\n" + "="*100)
    print("INITIAL TRAINING RESULTS (10% Labeled Only, Augmentation Ensemble)")
    print("="*100)
    
    print(f"\n{'Model':<25} {'Val_Acc':<10} {'Val_AUC':<10} {'Val_F1':<10} "
          f"{'Test_Acc':<10} {'Test_AUC':<10} {'Test_F1':<10}")
    print("-"*95)
    
    for model_name in ["ResNet18_Pure", "ResNet18_Intensity", "ResNet18_Geometric", "Ensemble"]:
        if model_name in initial_metrics:
            v = initial_metrics[model_name]['val']
            t = initial_metrics[model_name]['test']
            print(f"{model_name:<25} {v['balanced_acc']:.4f}    {v['auc']:.4f}    {v['f1']:.4f}    "
                  f"{t['balanced_acc']:.4f}    {t['auc']:.4f}    {t['f1']:.4f}")
    
    if upper_bound_metrics:
        print("\n" + "="*100)
        print("UPPER BOUND RESULTS (100% Labeled, Augmentation Ensemble)")
        print("="*100)
        
        print(f"\n{'Model':<25} {'Val_Acc':<10} {'Val_AUC':<10} {'Val_F1':<10} "
              f"{'Test_Acc':<10} {'Test_AUC':<10} {'Test_F1':<10}")
        print("-"*95)
        
        for model_name in ["ResNet18_Pure", "ResNet18_Intensity", "ResNet18_Geometric", "Ensemble"]:
            if model_name in upper_bound_metrics:
                v = upper_bound_metrics[model_name]['val']
                t = upper_bound_metrics[model_name]['test']
                print(f"{model_name:<25} {v['balanced_acc']:.4f}    {v['auc']:.4f}    {v['f1']:.4f}    "
                      f"{t['balanced_acc']:.4f}    {t['auc']:.4f}    {t['f1']:.4f}")


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def save_results(results, upper_bound_metrics, fst_results, output_dir, seed):
    """Save all results to files."""
    
    json_data = convert_to_serializable({
        "seed": seed,
        "experiment": results["experiment_name"],
        "confidence_threshold": results["confidence_threshold"],
        "iteration_results": results["iteration_results"],
        "pseudo_label_stats": results["pseudo_label_stats"],
        "upper_bound_metrics": upper_bound_metrics,
        "fst_results": fst_results
    })
    
    json_path = os.path.join(output_dir, f"results_seed_{seed}.json")
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"\nJSON saved to: {json_path}")
    
    summary_path = os.path.join(output_dir, f"summary_seed_{seed}.txt")
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"AUGMENTATION-BASED JOINT ENSEMBLE SRPM-ST WITH TYPICLUST - Seed {seed}\n")
        f.write("="*80 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: BreastMNIST (Binary, {NUM_CLASSES} classes)\n")
        f.write(f"Confidence Threshold: {results['confidence_threshold']}\n\n")
        
        f.write("KEY FEATURES:\n")
        f.write("1. Class-Weighted Loss: Inverse frequency weighting for imbalance\n")
        f.write("2. TypiClust: Feature-based representative sample selection\n")
        f.write("3. Pure Model 1: No augmentation at all (anchor view)\n")
        f.write("All 3 models are identical ResNet18 [22, 44, 88, 176] architectures.\n")
        f.write("Diversity comes from different augmentation strategies:\n")
        f.write("  - Model 1: Pure (no augmentation)\n")
        f.write("  - Model 2: Intensity (brightness, contrast, blur)\n")
        f.write("  - Model 3: Geometric (rotation, translation, scale, shear, flips)\n\n")
        
        final_metrics = results["iteration_results"][-1]["metrics"]
        f.write("FINAL SRPM-ST RESULTS:\n")
        for name in ["ResNet18_Pure", "ResNet18_Intensity", "ResNet18_Geometric", "Ensemble"]:
            if name in final_metrics:
                t = final_metrics[name]['test']
                f.write(f"  {name}: Test AUC={t['auc']:.4f}, Test BalAcc={t['balanced_acc']:.4f}\n")
    
    print(f"Summary saved to: {summary_path}")


# ============================================================
# 13. MAIN EXECUTION
# ============================================================

def run_single_seed(seed, num_batches, base_output_dir):
    """Run experiment for a single seed."""
    
    output_dir = os.path.join(base_output_dir, f"seed_{seed}")
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print(f"RUNNING AUGMENTATION ENSEMBLE EXPERIMENT - SEED: {seed}, NUM_BATCHES: {num_batches}")
    print("="*80)
    print(f"Output directory: {output_dir}")
    
    set_seed(seed)
    data_dict = prepare_data(seed, num_batches)
    
    # Run upper bound experiment
    set_seed(seed)
    upper_bound_metrics = run_upper_bound_experiment(data_dict)
    
    # Run Full-Batch ST baseline
    set_seed(seed)
    fst_results = run_full_batch_st(data_dict, confidence_threshold=CONFIDENCE_THRESHOLD)
    
    # Run SRPM-ST experiment with TypiClust
    set_seed(seed)
    results, ensemble = run_joint_ensemble_srpm_st(
        data_dict,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        experiment_name="Aug_Ensemble_SRPM_TypiClust_75"
    )
    
    # Print and save
    print_final_results(results, upper_bound_metrics)
    save_results(results, upper_bound_metrics, fst_results, output_dir, seed)
    
    # Save ensemble model
    torch.save(ensemble.state_dict(), os.path.join(output_dir, "aug_ensemble_final.pth"))
    
    return results, upper_bound_metrics, fst_results


def main():
    """Main function to run experiments for multiple seeds."""
    
    # Configuration - update these based on your tuning results
    SEEDS = [42, 43, 44]
    NUM_BATCHES = [32, 20, 24]  # Update with optimal values from tuning
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = f"results_aug_ensemble_breastmnist/{timestamp}"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Setup logging
    log_path = os.path.join(base_output_dir, "full_experiment_log.txt")
    logger = DualLogger()
    logger.start(log_path)
    sys.stdout = logger
    
    try:
        print("="*80)
        print("AUGMENTATION-BASED JOINT ENSEMBLE SRPM-ST WITH TYPICLUST")
        print(f"Dataset: BreastMNIST (Binary, {NUM_CLASSES} classes) | Confidence >= 0.75")
        print("Architecture: ResNet18 [22, 44, 88, 176] (identical for all 3 models)")
        print("Diversity: Different augmentation strategies (Pure, Intensity, Geometric)")
        print("Features: Class-Weighted Loss + TypiClust Selection")
        print("="*80)
        print(f"Seeds: {SEEDS}")
        print(f"Num Batches: {NUM_BATCHES}")
        print(f"Device: {DEVICE}")
        print(f"Output directory: {base_output_dir}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        all_results = {}
        for seed, num_batches in zip(SEEDS, NUM_BATCHES):
            results, upper_bound, fst = run_single_seed(seed, num_batches, base_output_dir)
            all_results[seed] = (results, upper_bound, fst)
        
        # Print aggregated summary
        print("\n" + "="*100)
        print("AGGREGATED RESULTS ACROSS ALL SEEDS")
        print("="*100)
        
        for seed, (results, upper_bound, fst) in all_results.items():
            final = results["iteration_results"][-1]["metrics"]["Ensemble"]["test"]
            init = results["initial_training_metrics"]["Ensemble"]["test"]
            upper = upper_bound["Ensemble"]["test"]
            fst_final = fst["final_metrics"]["Ensemble"]["test"]
            
            print(f"\nSeed {seed}:")
            print(f"  Initial (10% labeled): Test AUC = {init['auc']:.4f}")
            print(f"  F-ST:                  Test AUC = {fst_final['auc']:.4f}")
            print(f"  SRPM-ST (TypiClust):   Test AUC = {final['auc']:.4f}")
            print(f"  Upper Bound (100%):    Test AUC = {upper['auc']:.4f}")
        
        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"All results saved to: {base_output_dir}/")
        
        return all_results
    
    finally:
        sys.stdout = logger.terminal
        logger.stop()
        print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    all_results = main()
