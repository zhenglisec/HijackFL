from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import List, Dict
import logging
import torch
logger = logging.getLogger('logger')

ALL_TASKS =  ['hijack', 'normal', 'sentinet_evasion', #'spectral_evasion',
                           'neural_cleanse', 'mask_norm', 'sums', 'neural_cleanse_part1']

@dataclass
class Params:

    # Corresponds to the class module: tasks.mnist_task.MNISTTask
    # See other tasks in the task folder.
    task: str = 'MNIST'
    build_task: str = None
    current_time: str = None
    name: str = None
    commit: float = None
    random_seed: int = None
    device: str = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # training params
    start_epoch: int = 1
    epochs: int = None
    log_interval: int = 200

    hijack_lr: float = 0.05
    hijack_same_subdata: bool = False
    hijack_same_task: bool = False
    hijack_local_epochs: int = 2
    hijack_train_setting: bool = str
    hijack_fake_benign: int = 1
    distributed_epochs: int = 10
    distributed_every_epochs: int = 1
    A_dataset: str = None
    H_dataset: List[str] = None
    H_dataset_size: List[int] = None
    # K_dataset: str = None
    # K_dataset_size: int = None
    # model arch is usually defined by the task
    model_arch: str = None
    pretrained: bool = False
    resume_model: str = None
    lr: float = None
    decay: float = None
    momentum: float = None
    optimizer: str = None
    scheduler: bool = False
    scheduler_milestones: List[int] = None
    # data
    data_path: str = 'XXX'
    batch_size: int = 64
    test_batch_size: int = 100
    transform_train: bool = True
    "Do not apply transformations to the training images."
    max_batch_id: int = None
    "For large datasets stop training earlier."
    input_shape = None
    "No need to set, updated by the Task class."

    # gradient shaping/DP params
    dp: bool = None
    dp_clip: float = None
    dp_sigma: float = None

    # attack params
    backdoor: bool = False
    backdoor_label: int = 8
    poisoning_proportion: float = 1.0  # backdoors proportion in backdoor loss
    synthesizer: str = 'pattern'
    backdoor_dynamic_position: bool = False

    # losses to balance: `normal`, `backdoor`, `neural_cleanse`, `sentinet`,
    # `backdoor_multi`.
    loss_tasks: List[str] = None

    loss_balance: str = 'MGDA'
    "loss_balancing: `fixed` or `MGDA`"

    loss_threshold: float = None

    # approaches to balance losses with MGDA: `none`, `loss`,
    # `loss+`, `l2`
    mgda_normalize: str = None
    fixed_scales: Dict[str, float] = None
    fixed_scales_hijack: float = 0.33
    # relabel images with poison_number
    poison_images: List[int] = None
    poison_images_test: List[int] = None
    # optimizations:
    alternating_attack: float = None
    clip_batch: float = None
    # Disable BatchNorm and Dropout
    switch_to_eval: float = None

    # nc evasion
    nc_p_norm: int = 1
    # spectral evasion
    spectral_similarity: 'str' = 'norm'

    # logging
    report_train_loss: bool = True
    log: bool = False
    tb: bool = False
    save_model: bool = None
    save_on_epochs: List[int] = None
    save_scale_values: bool = False
    print_memory_consumption: bool = False
    save_timing: bool = False
    timing_data = None

    # Temporary storage for running values
    running_losses = None
    running_scales = None

    # FL params
    fl: bool = False
    fl_no_models: int = 100
    fl_local_epochs: int = 2
    fl_total_participants: int = 80000
    fl_eta: int = 1
    fl_sample_dirichlet: bool = False
    fl_dirichlet_alpha: float = None
    fl_diff_privacy: bool = False
    fl_dp_clip: float = None
    fl_dp_noise: float = None
    # FL attack details. Set no adversaries to perform the attack:
    
    fl_number_of_adversaries: int = 0
    fl_single_epoch_attack: int = None
    fl_multi_epoch_attack: int = None
    attacking_start_epoch: int = None
    fl_multi_user_attack: bool = False
    fl_weight_scale: int = 1
    fake_benign_fl_weight_scale: int = 1
    h_test_data_relabeled_or_converted: bool = None
    converted_data: str = None
    converter_layer: str = None
    converter_epochs: int = 100
    greedy_brute: str = None
    mapping_func_reset: bool = False
    converter_multi_updating: bool = True
    # delay_attack: bool = False
    def __post_init__(self):
        # enable logging anyways when saving statistics
        if self.save_model or self.tb or self.save_timing or \
                self.print_memory_consumption:
            self.log = True

        if self.log:
            self.folder_path = f'saved_models/' \
                               f'{self.current_time}_{self.model_arch}_{self.A_dataset}_{self.H_dataset}_{self.fl_number_of_adversaries}_{self.fl_single_epoch_attack}_{self.fl_multi_epoch_attack}_{self.attacking_start_epoch}_{self.fl_multi_user_attack}_{self.hijack_train_setting}_{self.converted_data}_{self.converter_layer}_{self.converter_epochs}_{self.greedy_brute}_{self.mapping_func_reset}'
        self.running_losses = defaultdict(list)
        self.running_scales = defaultdict(list)
        self.timing_data = defaultdict(list)

        for t in self.loss_tasks:
            if t not in ALL_TASKS:
                raise ValueError(f'Task {t} is not part of the supported '
                                 f'tasks: {ALL_TASKS}.')

    def to_dict(self):
        return asdict(self)