from typing import Tuple
from tst_data_utils import (TextStyleTransferDataset, 
                            load_text_style_transfer_dataset)
from omegaconf import DictConfig

def make_text_style_transfer_datasets(
        config: "DictConfig") -> Tuple[TextStyleTransferDataset]: 
    assert config.direction in ['0_to_1', '1_to_0']
    label = int(config.direction[0])
    data_dict = {}
    for split in ['train', 'dev', 'test']: 
        # Hack - Only use 16 examples for Yelp validation to save time
        if config.dataset == "yelp" and split == 'dev': 
            max_size = 10
        else: 
            max_size = config.max_size

        source_texts, target_labels = load_text_style_transfer_dataset(
            config.dataset, label, split,
            config.dataset_seed, config.base_path, max_size,
            config.max_length, config.max_length_tokenizer)
        tst_dataset = TextStyleTransferDataset(source_texts, target_labels)
        data_dict[split] = tst_dataset

    return data_dict['train'], data_dict['dev'], data_dict['test']


def load_text_style_transfer_test_data(config: "DictConfig"): 
    label = int(config.direction[0])
    source_texts, target_labels = load_text_style_transfer_dataset(
        config.dataset, label, 'test',
        config.dataset_seed, config.base_path, config.max_size,
        config.max_length, config.max_length_tokenizer)
    
    if config.dataset == 'yelp': # Separate human-written reference
        ref_texts, _ = load_text_style_transfer_dataset(
            config.dataset, label, 'ref', 
            config.dataset_seed, config.base_path, None, 
            None, None)
    elif config.dataset == 'shakespeare': # Opposite test data are references
        target_label = int(config.direction[1])
        ref_texts, _ = load_text_style_transfer_dataset(
            config.dataset, target_label, 'test',
            config.dataset_seed, config.base_path, None,
            None, None)

    return source_texts, target_labels, ref_texts

# Key: (dataset, dataset_seed, split)
style_classifier_dict = {
    ('yelp', None, 'train'): './style_classifiers/yelp-bert-base-uncased-train/',
    ('yelp', None, 'test'): './style_classifiers/yelp-bert-base-uncased-test/',
    ('shakespeare', 0, 'train'): './style_classifiers/shakespeare-bert-base-uncased-train-100-0/',
    ('shakespeare', 1, 'train'): './style_classifiers/shakespeare-bert-base-uncased-train-100-1/',
    ('shakespeare', 2, 'train'): './style_classifiers/shakespeare-bert-base-uncased-train-100-2/',
    ('shakespeare', None, 'test'): './style_classifiers/shakespeare-bert-base-uncased-test-all/'}
def get_style_classifier(
    split: str, 
    config: "DictConfig"
) -> str: 
    dataset_seed = config.dataset_seed if split != 'test' else None
    return style_classifier_dict[(config.dataset, dataset_seed, split)]
