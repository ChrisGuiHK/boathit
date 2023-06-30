import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm

def feature_extract(feature_extractor: nn.Module, dataloader: DataLoader, device: int, max_num_features=None) -> torch.Tensor:
    torch.device(f'cuda:{device}')
    feature_extractor.eval()
    features = []
    labels = []
    with torch.no_grad():
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            if max_num_features is not None and i >= max_num_features:
                break
            inputs = data[0].to(device)
            labels.append(data[1])
            feature = feature_extractor(inputs)
            feature = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())(feature).to(device)
            features.append(feature)
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)