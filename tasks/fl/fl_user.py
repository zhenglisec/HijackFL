from dataclasses import dataclass
from torch.utils.data.dataloader import DataLoader
from typing import List, Dict, Any
@dataclass
class FLUser:
    user_id: int = 0
    compromised: bool = False
    A_train_loader: DataLoader = None
    H_train_loaders: DataLoader = None
    attack_num: int = 0
    mapping_func: List = None
    transformer: List = None
    relabeled: bool = False
    