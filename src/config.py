import json
from typing import Dict, Any
import os

class HParams:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                setattr(self, key, HParams(**value))
            else:
                setattr(self, key, value)

    def __repr__(self):
        return str(self.__dict__)

def load_config(config_path: str) -> HParams:
    """Load configuration from JSON file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    return HParams(**config_dict)

def get_hparams(init_from="config", config_path="configs/config_zhongli.json"):
    """
    Get hyperparameters from config file
    """
    hparams = load_config(config_path)
    hparams.model_dir = "models/zhongli_tts"
    hparams.data.training_files = os.path.join("data/preprocessed_v1", "zhongli_train.txt")
    hparams.data.validation_files = os.path.join("data/preprocessed_v1", "zhongli_val.txt")
    
    return hparams