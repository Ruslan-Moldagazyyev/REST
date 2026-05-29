import os
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

from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score
from sklearn.cluster import KMeans

import medmnist
from medmnist import BreastMNIST

from torchvision.transforms import ElasticTransform

warnings.filterwarnings("ignore")

device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")

torch.cuda.set_device(device)

print("Using device:", device)
print("GPU name:", torch.cuda.get_device_name(device))


SEED = 42
LABELED_FRACTION = 0.10
BATCH_SIZE_TRAIN = 16
BATCH_SIZE_EVAL = 32
NUM_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 7
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-1
NUM_CLASSES = 2
IMG_SIZE = 28

NUM_BATCHES_TO_TEST = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]

CONFIDENCE_THRESHOLD = 0.75

PSEUDO_LABEL_ALPHA = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = f"aug_ensemble_srpm_st_results_breast_{SEED}_{CONFIDENCE_THRESHOLD}_{LEARNING_RATE}_{WEIGHT_DECAY}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Device: {DEVICE}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Seed: {SEED}")
print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class BasicBlock(nn.Module):
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
    def __init__(self, mean=0.0, std=0.05):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)


class AddSaltPepperNoise:
    def __init__(self, prob=0.02):
        self.prob = prob
    
    def __call__(self, tensor):
        noise_mask = torch.rand_like(tensor)
        salt = (noise_mask < self.prob / 2).float()
        pepper = (noise_mask > 1 - self.prob / 2).float()
        noisy = tensor * (1 - salt - pepper) + salt
        return torch.clamp(noisy, 0.0, 1.0)


class AddSpeckleNoise:
    def __init__(self, std=0.1):
        self.std = std
    
    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + tensor * noise, 0.0, 1.0)


def get_augmentation_transforms():
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])  # GRAYSCALE: 1 channel
    
    transform_minimal = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize
    ])
    
    transform_noise = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.RandomApply([AddSpeckleNoise(std=0.08)], p=0.4),  # Ultrasound-specific
        transforms.RandomApply([AddGaussianNoise(mean=0.0, std=0.03)], p=0.3),
        transforms.RandomApply([AddSaltPepperNoise(prob=0.01)], p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))], p=0.25),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.1, contrast=0.1)
        ], p=0.3),
        normalize
    ])
    
    transform_spatial = transforms.Compose([
        transforms.Resize((IMG_SIZE + 4, IMG_SIZE + 4)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
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
        "minimal": transform_minimal,
        "color": transform_noise,  
        "geometric": transform_spatial
    }


def get_eval_transform():
    """Standard evaluation transform (no augmentation) - grayscale."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  
    ])



class MultiAugmentationDataset(Dataset):
    
    def __init__(self, base_dataset, transforms_dict):
        self.base_dataset = base_dataset
        self.transforms = transforms_dict
    
    def _setup_raw_access(self):
        pass
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img_array = self.base_dataset.imgs[idx]
        label = self.base_dataset.labels[idx]
        
        from PIL import Image
        img_pil = Image.fromarray(img_array)
        
        img_minimal = self.transforms["minimal"](img_pil)
        img_color = self.transforms["color"](img_pil)
        img_geometric = self.transforms["geometric"](img_pil)
        
        return img_minimal, img_color, img_geometric, label


class MultiAugmentationSubset(Dataset):
    
    def __init__(self, multi_aug_dataset, indices):
        self.dataset = multi_aug_dataset
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        img_min, img_color, img_geo, label = self.dataset[self.indices[idx]]
        return img_min, img_color, img_geo, label, 0


class PseudoLabelMultiAugDataset(Dataset):
    
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
        
        from PIL import Image
        img_pil = Image.fromarray(img_array)
        
        img_minimal = self.transforms["minimal"](img_pil)
        img_color = self.transforms["color"](img_pil)
        img_geometric = self.transforms["geometric"](img_pil)
        
        label = np.array([self.pseudo_labels[idx]], dtype=np.int64)
        
        return img_minimal, img_color, img_geometric, label, 1


class JointAugmentationEnsemble(nn.Module):

    
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



def compute_class_weights(dataset_raw, labeled_indices):

    labels = [int(dataset_raw.labels[idx][0]) for idx in labeled_indices]
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    
    total = sum(class_counts)
    class_weights = total / (NUM_CLASSES * class_counts.astype(float))
    
    print(f"\n  Class distribution in labeled set:")
    for c in range(NUM_CLASSES):
        print(f"    Class {c}: {class_counts[c]} samples, weight: {class_weights[c]:.4f}")
    
    return torch.FloatTensor(class_weights).to(DEVICE)


def prepare_data():
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
    
    indices = list(range(n_train))
    random.shuffle(indices)
    
    labeled_indices = indices[:n_labeled]
    unlabeled_indices = indices[n_labeled:]
    
    class_weights = compute_class_weights(train_dataset_raw, labeled_indices)
    
    all_true_labels = {}
    for idx in range(n_train):
        label = train_dataset_raw.labels[idx]
        all_true_labels[idx] = int(label[0])
    
    print(f"\nDataset: BreastMNIST (Binary, {NUM_CLASSES} classes)")
    print(f"Total training: {n_train}")
    print(f"Labeled ({LABELED_FRACTION*100:.0f}%): {n_labeled}")
    print(f"Unlabeled: {len(unlabeled_indices)}")
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
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "labeled_indices": labeled_indices,
        "unlabeled_indices": unlabeled_indices,
        "all_true_labels": all_true_labels,
        "class_weights": class_weights
    }



def train_joint_ensemble(
    ensemble: JointAugmentationEnsemble,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    lr: float,
    class_weights: torch.Tensor,
    verbose: bool = False,
    pseudo_label_alpha: float = 1.0
) -> Tuple[JointAugmentationEnsemble, float, int]:

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
            if len(batch) == 5:
                img_minimal, img_color, img_geometric, labels, is_pseudo = batch
                is_pseudo = is_pseudo.float().to(DEVICE)
            else:
                img_minimal, img_color, img_geometric, labels = batch
                is_pseudo = None
            
            img_minimal = img_minimal.to(DEVICE)
            img_color = img_color.to(DEVICE)
            img_geometric = img_geometric.to(DEVICE)
            labels = labels.view(-1).long().to(DEVICE)
            
            optimizer.zero_grad()
            
            ensemble_logits = ensemble(img_minimal, img_color, img_geometric)
            
            per_sample_loss = criterion(ensemble_logits, labels)
            
            if is_pseudo is not None and pseudo_label_alpha < 1.0:
                sample_weights = 1.0 - is_pseudo * (1.0 - pseudo_label_alpha)
                loss = (per_sample_loss * sample_weights).mean()
            else:
                loss = per_sample_loss.mean()
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        
        val_metrics = evaluate_joint_ensemble(ensemble, val_loader)
        val_auc = val_metrics["ensemble_auc"]
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_state = copy.deepcopy(ensemble.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        
        if verbose and epoch % 5 == 0:
            print(f"    Epoch {epoch}: Loss={epoch_loss/num_batches:.4f}, "
                  f"Val AUC={val_auc:.4f}")
        
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            if verbose:
                print(f"    Early stop at epoch {epoch}, best: {best_epoch}, AUC: {best_val_auc:.4f}")
            break
    
    if best_state is not None:
        ensemble.load_state_dict(best_state)
    
    return ensemble, best_val_auc, best_epoch

def evaluate_joint_ensemble(ensemble: JointAugmentationEnsemble, loader: DataLoader) -> Dict:
    ensemble.eval()
    
    all_ensemble_preds = []
    all_ensemble_probs = []
    all_individual_preds = [[], [], []]
    all_individual_probs = [[], [], []]
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
    
    ensemble_probs_positive = all_ensemble_probs[:, 1]
    results = {
        "ensemble_balanced_acc": balanced_accuracy_score(all_labels, np.array(all_ensemble_preds)),
        "ensemble_auc": roc_auc_score(all_labels, ensemble_probs_positive),
        "ensemble_f1": f1_score(all_labels, np.array(all_ensemble_preds), average='binary')
    }
    
    for i, name in enumerate(["ResNet18_Pure", "ResNet18_Intensity", "ResNet18_Geometric"]):
        individual_probs = np.array(all_individual_probs[i])
        individual_probs_positive = individual_probs[:, 1]
        results[f"{name}_auc"] = roc_auc_score(all_labels, individual_probs_positive)
        results[f"{name}_balanced_acc"] = balanced_accuracy_score(all_labels, np.array(all_individual_preds[i]))
    
    return results


def get_ensemble_predictions(
    ensemble: JointAugmentationEnsemble,
    dataset_raw,
    indices: List[int],
    eval_transform,
    confidence_threshold: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, List[int]]:

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
            label = self.base_dataset.labels[real_idx]
            
            from PIL import Image
            img_pil = Image.fromarray(img_array)
            img = self.transform(img_pil)
            
            return img, label
    
    eval_dataset = SimpleEvalDataset(dataset_raw, indices, eval_transform)
    loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE_EVAL, shuffle=False)
    
    all_preds = []
    all_confidences = []
    
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(DEVICE)
            
            ensemble_probs = ensemble.get_ensemble_probabilities(imgs)
            
            preds = ensemble_probs.argmax(1).cpu().numpy()
            max_probs = ensemble_probs.max(1)[0].cpu().numpy()
            
            all_preds.extend(preds)
            all_confidences.extend(max_probs)
    
    predictions = np.array(all_preds)
    confidences = np.array(all_confidences)
    
    if confidence_threshold > 0:
        accepted_mask = confidences >= confidence_threshold
        accepted_indices = [indices[i] for i in range(len(indices)) if accepted_mask[i]]
    else:
        accepted_indices = indices
    
    return predictions, confidences, accepted_indices



def extract_ensemble_features(ensemble, dataset_raw, indices, eval_transform):
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
            from PIL import Image
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


def run_joint_ensemble_srpm_st(
    data_dict: Dict,
    num_batches: int,
    verbose: bool = False
) -> Dict:

    ensemble = JointAugmentationEnsemble(num_classes=NUM_CLASSES, in_channels=1).to(DEVICE)
    
    train_dataset_raw = data_dict["train_dataset_raw"]
    train_multi_aug = data_dict["train_multi_aug"]
    aug_transforms = data_dict["aug_transforms"]
    val_dataset = data_dict["val_dataset"]
    test_dataset = data_dict["test_dataset"]
    labeled_indices = data_dict["labeled_indices"].copy()
    unlabeled_indices = data_dict["unlabeled_indices"].copy()
    true_labels = data_dict["all_true_labels"]
    class_weights = data_dict["class_weights"]
    
    eval_transform = get_eval_transform()
    
    batch_size_M = len(unlabeled_indices) // num_batches
    
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_EVAL, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_EVAL, shuffle=False)
    
    if verbose:
        print(f"  Phase 1: Initial training on {len(labeled_indices)} labeled samples...")
    
    labeled_subset = MultiAugmentationSubset(train_multi_aug, labeled_indices)
    train_loader = DataLoader(labeled_subset, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    
    ensemble, initial_val_auc, _ = train_joint_ensemble(
        ensemble, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, class_weights, verbose=verbose
    )
    
    if verbose:
        print(f"    Initial ensemble Val AUC: {initial_val_auc:.4f}")
    
    pseudo_labels_dict = {}
    remaining_unlabeled = unlabeled_indices.copy()
    
    for batch_idx in range(num_batches):
        if len(remaining_unlabeled) == 0:
            if verbose:
                print(f"  No more unlabeled samples. Stopping at iteration {batch_idx+1}.")
            break
        
        current_M = min(batch_size_M, len(remaining_unlabeled))
        if current_M < 2:
            if verbose:
                print(f"  Too few remaining samples ({len(remaining_unlabeled)}). Stopping.")
            break
        
        if verbose:
            print(f"  Phase 2: TypiClust iteration {batch_idx+1}/{num_batches} "
                  f"(selecting {current_M} from {len(remaining_unlabeled)} remaining)...")
        
        selected_indices, remaining_unlabeled = typiclust_select(
            ensemble, train_dataset_raw, remaining_unlabeled, eval_transform, current_M
        )

        preds, confidences, accepted = get_ensemble_predictions(
            ensemble, train_dataset_raw, selected_indices, eval_transform, CONFIDENCE_THRESHOLD
        )
        
        for i, idx in enumerate(selected_indices):
            if CONFIDENCE_THRESHOLD == 0 or confidences[i] >= CONFIDENCE_THRESHOLD:
                pseudo_labels_dict[idx] = int(preds[i])
        
        labeled_subset = MultiAugmentationSubset(train_multi_aug, labeled_indices)
        
        pseudo_indices = list(pseudo_labels_dict.keys())
        pseudo_labels_list = [pseudo_labels_dict[idx] for idx in pseudo_indices]
        pseudo_subset = PseudoLabelMultiAugDataset(
            train_dataset_raw, pseudo_indices, pseudo_labels_list, aug_transforms
        )
        
        combined_dataset = ConcatDataset([labeled_subset, pseudo_subset])
        
        if verbose:
            print(f"    Training set size: {len(combined_dataset)} "
                  f"(labeled: {len(labeled_indices)}, pseudo: {len(pseudo_indices)})")
        
        train_loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
        
        all_training_labels = [int(train_dataset_raw.labels[idx][0]) for idx in labeled_indices]
        all_training_labels += pseudo_labels_list
        training_counts = np.bincount(all_training_labels, minlength=NUM_CLASSES)
        total = sum(training_counts)
        updated_weights = total / (NUM_CLASSES * training_counts.astype(float))
        updated_class_weights = torch.FloatTensor(updated_weights).to(DEVICE)
        
        ensemble = JointAugmentationEnsemble(num_classes=NUM_CLASSES, in_channels=1).to(DEVICE)
        
        ensemble, batch_val_auc, _ = train_joint_ensemble(
        ensemble, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, updated_class_weights, verbose=False, pseudo_label_alpha=PSEUDO_LABEL_ALPHA
    )
        
        if verbose:
            print(f"    Val AUC after iteration {batch_idx+1}: {batch_val_auc:.4f}")
    
    val_metrics = evaluate_joint_ensemble(ensemble, val_loader)
    test_metrics = evaluate_joint_ensemble(ensemble, test_loader)
    
    correct = sum(1 for idx, pl in pseudo_labels_dict.items() if pl == true_labels[idx])
    pseudo_accuracy = correct / len(pseudo_labels_dict) if pseudo_labels_dict else 0
    
    return {
        "initial_val_auc": initial_val_auc,
        "val_auc": val_metrics["ensemble_auc"],
        "val_balanced_acc": val_metrics["ensemble_balanced_acc"],
        "val_f1": val_metrics["ensemble_f1"],
        "test_auc": test_metrics["ensemble_auc"],
        "test_balanced_acc": test_metrics["ensemble_balanced_acc"],
        "test_f1": test_metrics["ensemble_f1"],
        "pseudo_accuracy": pseudo_accuracy,
        "num_pseudo_labels": len(pseudo_labels_dict),
        "resnet_pure_test_auc": test_metrics["ResNet18_Pure_auc"],
        "resnet_intensity_test_auc": test_metrics["ResNet18_Intensity_auc"],
        "resnet_geometric_test_auc": test_metrics["ResNet18_Geometric_auc"],
    }


def tune_num_batches(data_dict: Dict) -> Dict:
    
    print("\n" + "="*70)
    print("TUNING NUMBER OF MINI-BATCHES FOR AUGMENTATION-BASED ENSEMBLE SRPM-ST")
    print("With TypiClust selection + Class-Weighted Loss")
    print("="*70)
    
    all_results = []
    
    for num_batches in NUM_BATCHES_TO_TEST:
        set_seed(SEED)
        
        print(f"\nTesting num_batches = {num_batches}...", end=" ")
        
        try:
            results = run_joint_ensemble_srpm_st(data_dict, num_batches, verbose=False)
            
            print(f"Val AUC: {results['val_auc']:.4f}, "
                  f"Test AUC: {results['test_auc']:.4f}, "
                  f"PL Acc: {results['pseudo_accuracy']*100:.1f}%")
            
            all_results.append({"num_batches": num_batches, **results})
            
        except Exception as e:
            print(f"ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            all_results.append({"num_batches": num_batches, "error": str(e)})
    
    return all_results


def save_results(all_results: List[Dict]):
    import csv
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    csv_path = os.path.join(OUTPUT_DIR, f"aug_ensemble_results_{timestamp}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'num_batches', 'initial_val_auc', 'val_auc', 'val_balanced_acc', 'val_f1',
            'test_auc', 'test_balanced_acc', 'test_f1', 'pseudo_accuracy', 'num_pseudo_labels',
            'resnet_pure_auc', 'resnet_intensity_auc', 'resnet_geometric_auc'
        ])
        for r in all_results:
            if "error" not in r:
                writer.writerow([
                    r['num_batches'], f"{r['initial_val_auc']:.4f}",
                    f"{r['val_auc']:.4f}", f"{r['val_balanced_acc']:.4f}", f"{r['val_f1']:.4f}",
                    f"{r['test_auc']:.4f}", f"{r['test_balanced_acc']:.4f}", f"{r['test_f1']:.4f}",
                    f"{r['pseudo_accuracy']:.4f}", r['num_pseudo_labels'],
                    f"{r['resnet_pure_test_auc']:.4f}", f"{r['resnet_intensity_test_auc']:.4f}",
                    f"{r['resnet_geometric_test_auc']:.4f}"
                ])
    print(f"\nCSV saved to: {csv_path}")
    
    json_path = os.path.join(OUTPUT_DIR, f"aug_ensemble_results_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"JSON saved to: {json_path}")
    
    summary_path = os.path.join(OUTPUT_DIR, f"summary_{timestamp}.txt")
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("AUGMENTATION-BASED JOINT ENSEMBLE SRPM-ST RESULTS\n")
        f.write("With TypiClust + Class-Weighted Loss\n")
        f.write("="*80 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: BreastMNIST (Binary, {NUM_CLASSES} classes)\n")
        f.write(f"Labeled Fraction: {LABELED_FRACTION*100:.0f}%\n")
        f.write(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("KEY FEATURES:\n")
        f.write("-"*80 + "\n")
        f.write("1. Class-Weighted Loss: Inverse frequency weighting for imbalance\n")
        f.write("2. TypiClust: Feature-based representative sample selection\n")
        f.write("3. Pure Model 1: No augmentation at all (anchor view)\n")
        f.write("All 3 models are identical ResNet18 architectures.\n")
        f.write("Diversity comes from different augmentation strategies:\n")
        f.write("  - Model 1: Pure (no augmentation)\n")
        f.write("  - Model 2: Intensity (brightness, contrast, blur)\n")
        f.write("  - Model 3: Geometric (rotation, translation, scale, shear, flips)\n")
        f.write("Joint training: All models forward → Soft voting → Single loss → Backprop to ALL\n\n")
        
        f.write("-"*80 + "\n")
        f.write("RESULTS BY NUMBER OF MINI-BATCHES\n")
        f.write("-"*80 + "\n\n")
        f.write(f"{'#Batches':<10} {'Init AUC':<12} {'Val AUC':<12} {'Test AUC':<12} {'PL Acc':<10}\n")
        f.write("-"*60 + "\n")
        
        valid_results = [r for r in all_results if "error" not in r]
        for r in valid_results:
            f.write(f"{r['num_batches']:<10} {r['initial_val_auc']:.4f}       "
                   f"{r['val_auc']:.4f}       {r['test_auc']:.4f}       "
                   f"{r['pseudo_accuracy']*100:.1f}%\n")
        
        if valid_results:
            best = max(valid_results, key=lambda x: x["val_auc"])
            f.write(f"\nBEST: num_batches={best['num_batches']}, "
                   f"Val AUC={best['val_auc']:.4f}, Test AUC={best['test_auc']:.4f}\n")
        
        f.write("\n" + "-"*80 + "\n")
        f.write("INDIVIDUAL MODEL PERFORMANCE (within ensemble)\n")
        f.write("-"*80 + "\n\n")
        
        if valid_results:
            best = max(valid_results, key=lambda x: x["val_auc"])
            f.write(f"At optimal num_batches={best['num_batches']}:\n")
            f.write(f"  Ensemble Test AUC:           {best['test_auc']:.4f}\n")
            f.write(f"  ResNet18_Pure AUC:           {best['resnet_pure_test_auc']:.4f}\n")
            f.write(f"  ResNet18_Intensity AUC:      {best['resnet_intensity_test_auc']:.4f}\n")
            f.write(f"  ResNet18_Geometric AUC:      {best['resnet_geometric_test_auc']:.4f}\n")
    
    print(f"Summary saved to: {summary_path}")


def print_final_summary(all_results: List[Dict]):
    """Print final summary to console."""
    print("\n" + "="*80)
    print("FINAL SUMMARY: AUGMENTATION-BASED JOINT ENSEMBLE SRPM-ST")
    print("With TypiClust + Class-Weighted Loss")
    print("="*80)
    
    valid_results = [r for r in all_results if "error" not in r]
    
    print(f"\n{'#Batches':<10} {'Init AUC':<12} {'Val AUC':<12} {'Test AUC':<12} {'PL Acc':<10}")
    print("-"*60)
    
    for r in valid_results:
        print(f"{r['num_batches']:<10} {r['initial_val_auc']:.4f}       "
              f"{r['val_auc']:.4f}       {r['test_auc']:.4f}       "
              f"{r['pseudo_accuracy']*100:.1f}%")
    
    if valid_results:
        best = max(valid_results, key=lambda x: x["val_auc"])
        print(f"\n✓ BEST: num_batches={best['num_batches']}")
        print(f"  Val AUC:  {best['val_auc']:.4f}")
        print(f"  Test AUC: {best['test_auc']:.4f}")
        print(f"  Test Balanced Acc: {best['test_balanced_acc']:.4f}")
        
        print("\n  Individual model performance (jointly trained, same architecture):")
        print(f"    ResNet18_Pure (no augmentation):  {best['resnet_pure_test_auc']:.4f}")
        print(f"    ResNet18_Intensity (photometric): {best['resnet_intensity_test_auc']:.4f}")
        print(f"    ResNet18_Geometric (spatial):     {best['resnet_geometric_test_auc']:.4f}")


def main():
    print("\n" + "="*80)
    print("AUGMENTATION-BASED JOINT ENSEMBLE SRPM-ST")
    print("Same Architecture (ResNet18), Different Augmentations")
    print("With TypiClust + Class-Weighted Loss")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    set_seed(SEED)
    
    ensemble = JointAugmentationEnsemble(num_classes=NUM_CLASSES, in_channels=1)
    ensemble.print_param_counts()
    
    data_dict = prepare_data()
    
    all_results = tune_num_batches(data_dict)
    
    print_final_summary(all_results)
    save_results(all_results)
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"All results saved to: {OUTPUT_DIR}/")
    
    return all_results


if __name__ == "__main__":
    all_results = main()
