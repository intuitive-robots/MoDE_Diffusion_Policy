from copy import deepcopy
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from mode.datasets.robomimic_dataset import SequenceDataset

"""
    Helper function from Robomimic to read hdf5 demonstrations into sequence dataset

    ISSUE: robomimic's SequenceDataset has two properties: seq_len and frame_stack,
    we should in principle use seq_len, but the paddings of the two are different.
    So that's why we currently use frame_stack instead of seq_len.
"""


def get_dataset(
    dataset_path,
    obs_modality,
    initialize_obs_utils=True,
    seq_len=1,
    frame_stack=1,
    filter_key=None,
    hdf5_cache_mode="low_dim",
    *args,
    **kwargs
):

    if initialize_obs_utils:
        ObsUtils.initialize_obs_utils_with_obs_specs({"obs": obs_modality})

    all_obs_keys = []
    for modality_name, modality_list in obs_modality.items():
        all_obs_keys += modality_list
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=dataset_path, all_obs_keys=all_obs_keys, verbose=False
    )

    seq_len = seq_len
    filter_key = filter_key
    dataset = SequenceDataset(
        hdf5_path=dataset_path,
        obs_keys=shape_meta["all_obs_keys"],
        dataset_keys=["actions"],
        load_next_obs=False,
        frame_stack=frame_stack,
        seq_length=seq_len,  # length-10 temporal sequences
        pad_frame_stack=True,
        pad_seq_length=True,  # pad last obs per trajectory to ensure all sequences are sampled
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode=hdf5_cache_mode,  # cache dataset in memory to avoid repeated file i/o
        hdf5_use_swmr=False,
        hdf5_normalize_obs=None,
        filter_by_attribute=filter_key,  # can optionally provide a filter key here
    )
    return dataset, shape_meta


def get_split_dataset(
    dataset_path,
    obs_modality,
    initialize_obs_utils=True,
    seq_len=1,
    frame_stack=1,
    filter_key=None,
    hdf5_cache_mode="low_dim",
    num_demos_per_task: int = 50,
    split_ratio: float = 0.5,  # New parameter to control the split ratio
    *args,
    **kwargs
    ):

    if initialize_obs_utils:
        ObsUtils.initialize_obs_utils_with_obs_specs({"obs": obs_modality})

    all_obs_keys = []
    for modality_name, modality_list in obs_modality.items():
        all_obs_keys += modality_list
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=dataset_path, all_obs_keys=all_obs_keys, verbose=False
    )

    seq_len = seq_len
    filter_key = filter_key
    full_dataset = SequenceDataset(
        hdf5_path=dataset_path,
        obs_keys=shape_meta["all_obs_keys"],
        dataset_keys=["actions"],
        load_next_obs=False,
        frame_stack=frame_stack,
        seq_length=seq_len,  # length-10 temporal sequences
        pad_frame_stack=True,
        pad_seq_length=True,  # pad last obs per trajectory to ensure all sequences are sampled
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode=hdf5_cache_mode,  # cache dataset in memory to avoid repeated file i/o
        hdf5_use_swmr=False,
        hdf5_normalize_obs=None,
        filter_by_attribute=filter_key,  # can optionally provide a filter key here
    )

    # Split the demos into two parts
    #if num_demos_per_task < 50:
    #    demos_per_task = num_demos_per_task

    n_demos = full_dataset.n_demos

    reduced_demos = full_dataset.demos[:num_demos_per_task]

    # mid_index = int(len(full_dataset.demos) * split_ratio)
    mid_index = int(len(reduced_demos)* split_ratio)
    demos_first_half = reduced_demos[:mid_index]
    demos_second_half = reduced_demos[mid_index:]

    # Create two subdatasets
    dataset1 = deepcopy(full_dataset)
    dataset1.load_demo_info(demos=demos_first_half)

    dataset2 = deepcopy(full_dataset)
    dataset2.load_demo_info(demos=demos_second_half)

    return (dataset1, dataset2), shape_meta