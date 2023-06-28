import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(1)

def feature_extract(feature_extractor: nn.Module, dataloader: DataLoader, max_num_features=None) -> torch.Tensor:
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
            feature = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())(feature).cpu()
            features.append(feature)
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)