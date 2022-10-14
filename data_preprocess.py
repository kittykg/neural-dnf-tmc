import argparse
from itertools import chain, combinations
import pickle
from typing import List, Optional, Tuple

import numpy as np
import torch

from common import MultiLabelRawSample, MultiLabelDatasetSample


def attribute_reduction_mask_count_based(
    dataset: List[MultiLabelRawSample],
    total_number_attr: int,
    total_number_labels: int,
    comb_appearance_threshold: int,
    label_appearance_threshold: int,
    min_comb_threshold: int,
    min_label_threshold: int,
) -> np.ndarray:
    combination_attr_counter = dict()
    label_attr_matrix = np.zeros((total_number_labels, total_number_attr))

    for d in dataset:
        # Update label_attr_matrix
        for l in d.labels:
            for a in d.present_attributes:
                label_attr_matrix[l][a] += 1
        # Update combination_attr_counter
        comb = ",".join([str(l) for l in d.labels])
        if comb not in combination_attr_counter:
            combination_attr_counter[comb] = np.zeros(500)
        for a in d.present_attributes:
            combination_attr_counter[comb][a] += 1

    # Stack the combination_attr_counter to form a matrix
    combination_attr_matrix = np.stack(
        list(combination_attr_counter.values()), axis=0
    )

    # Compute the frequency of each label reaching appearance threshold in
    # both counter matrices.
    comb_attr_app_freq = np.count_nonzero(
        combination_attr_matrix >= comb_appearance_threshold, axis=0
    )
    label_attr_app_freq = np.count_nonzero(
        label_attr_matrix >= label_appearance_threshold, axis=0
    )

    # Check if the frequency reaches a threshold
    comb_attr_app_check = comb_attr_app_freq >= min_comb_threshold
    label_attr_app_check = label_attr_app_freq >= min_label_threshold

    # Attribute mask is idx of two checks' conjunction
    attribute_mask = np.asarray(
        np.logical_and(comb_attr_app_check, label_attr_app_check)
    ).nonzero()[0]

    return attribute_mask


def attribute_reduction_mask_mi_based(
    dataset: List[MultiLabelRawSample],
    total_number_attr: int,
    total_number_labels: int,
    mi_threshold: float,
) -> np.ndarray:
    # $I(X_i;Y) = H(X_i) - H(X_i|Y)$
    # $X_i$: Attribute $Y$: Label combination
    # To calculate MI, we need:
    # $H(X_i)$: this needs to know $p(x_i)$
    # $H(X_i|Y)$: this needs to know $p(y)$, $p(x_i|y)$

    # Get all combinations
    labels = list(range(total_number_labels))
    comb_keys_chain = chain.from_iterable(
        combinations(labels, r) for r in range(1, total_number_labels + 1)
    )
    comb_keys_str_list = [",".join(str(s) for s in k) for k in comb_keys_chain]

    # Counting
    combination_dict = {k: 0 for k in comb_keys_str_list}
    comb_attr_true_matrix = np.zeros(
        (len(comb_keys_str_list), total_number_attr)
    )
    comb_attr_false_matrix = np.zeros(
        (len(comb_keys_str_list), total_number_attr)
    )
    attr_true_matrix = np.zeros(total_number_attr)
    attr_false_matrix = np.zeros(total_number_attr)

    for d in dataset:
        for a in range(total_number_attr):
            if a in d.present_attributes:
                attr_true_matrix[a] += 1
            else:
                attr_false_matrix[a] += 1

        comb_key = ",".join([str(l) for l in d.labels])
        combination_dict[comb_key] += 1
        i = comb_keys_str_list.index(comb_key)

        for j in range(total_number_attr):
            if j in d.present_attributes:
                comb_attr_true_matrix[i][j] += 1
            else:
                comb_attr_false_matrix[i][j] += 1

    combination_count_matrix = np.array(
        [combination_dict[k] for k in comb_keys_str_list]
    )

    # Count checks
    assert np.all((attr_true_matrix + attr_false_matrix) == len(dataset))
    for i in range(len(comb_keys_str_list)):
        elem = (comb_attr_true_matrix + comb_attr_false_matrix)[i, 0]
        assert np.all(
            (comb_attr_true_matrix + comb_attr_false_matrix)[i, :] == elem
        )
    assert np.all(
        (comb_attr_true_matrix + comb_attr_false_matrix)[:, 0]
        == combination_count_matrix
    )
    assert np.sum(combination_count_matrix) == len(dataset)

    # Epsilon for log calculation
    epsilon = np.finfo(np.float32).eps

    # p(x_i)
    p_attr_true = attr_true_matrix / len(dataset)  # p(x_i=true)
    p_attr_true = np.where(p_attr_true == 0, p_attr_true + epsilon, p_attr_true)
    p_attr_true = np.where(p_attr_true == 1, p_attr_true - epsilon, p_attr_true)
    p_attr_false = 1 - p_attr_true  # p(x_i=false)

    # H(Xi)
    attr_entropy = -p_attr_true * np.log(p_attr_true) - p_attr_false * np.log(
        p_attr_false
    )

    # p(y)
    p_comb = combination_count_matrix / len(dataset)

    # p(x_i|y)
    p_attr_true_comb = comb_attr_true_matrix.T / combination_count_matrix
    p_attr_true_comb = np.where(
        p_attr_true_comb == 0, p_attr_true_comb + epsilon, p_attr_true_comb
    )
    p_attr_true_comb = np.where(
        p_attr_true_comb == 1, p_attr_true_comb - epsilon, p_attr_true_comb
    )
    p_attr_false_comb = 1 - p_attr_true_comb

    attr_comb_entropy_parts = p_comb * (
        p_attr_true_comb * np.log(p_attr_true_comb)
        + p_attr_false_comb * np.log(p_attr_false_comb)
    )
    attr_comb_entropy_parts = attr_comb_entropy_parts[
        :, ~np.isnan(attr_comb_entropy_parts).any(axis=0)
    ]  # ignore any nan column
    # H(X_i|Y)
    attr_comb_entropy = -np.sum(attr_comb_entropy_parts, axis=1)

    # I(X_i;Y)
    mi_attr_comb = attr_entropy - attr_comb_entropy

    attribute_mask = np.asarray(mi_attr_comb >= mi_threshold).nonzero()[0]

    return attribute_mask


def attribute_reduction(
    dataset: List[MultiLabelRawSample],
    raw_total_number_attr: int,
    total_number_labels: int,
    attribute_mask: np.ndarray,
) -> List[MultiLabelDatasetSample]:
    # Convert raw sample to dataset sample with attribute_mask
    def _get_new_data(
        d: MultiLabelRawSample,
    ) -> Optional[MultiLabelDatasetSample]:
        temp_d = d.to_dataset_sample(raw_total_number_attr, total_number_labels)
        # Adjust the attribute encoding
        new_attr_encoding = temp_d.attribute_encoding[attribute_mask]  # type: ignore
        if torch.count_nonzero(new_attr_encoding) == 0:
            return None
        return MultiLabelDatasetSample(
            temp_d.sample_id,
            temp_d.label_encoding,
            new_attr_encoding,
        )

    return [_get_new_data(d) for d in dataset if _get_new_data(d)]  # type: ignore


def data_subset_creation(
    dataset: List[MultiLabelRawSample], num_labels: int
) -> List[MultiLabelRawSample]:
    def _get_new_data(d: MultiLabelRawSample) -> Optional[MultiLabelRawSample]:
        # New data has only the first NUM_LABELS labels
        new_labels = [l for l in d.labels if l in range(num_labels)]
        if not new_labels:
            return None
        return MultiLabelRawSample(
            d.sample_id, new_labels, d.present_attributes
        )

    return [_get_new_data(d) for d in dataset if _get_new_data(d)]  # type: ignore


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Base arguments
    parser.add_argument("-rtrain", type=str, help="Raw train pkl path")
    parser.add_argument("-rval", type=str, help="Raw val pkl path")
    parser.add_argument("-rtest", type=str, help="Raw test pkl path")
    parser.add_argument(
        "-rna",
        type=int,
        default=500,
        help="Number of attributes in the raw dataset",
    )
    parser.add_argument(
        "-nl", type=int, help="Number of labels after filtering"
    )
    parser.add_argument(
        "-od", type=str, help="Output directory (with trailing '/')"
    )
    parser.add_argument(
        "-m",
        type=str,
        choices=["count", "mi"],
        required=True,
        help="The method used to compute attribute reduction mask",
    )

    # Arguments for count-based attribute reduction
    parser.add_argument(
        "-cat",
        type=int,
        nargs="?",
        default=None,
        help="Count-based: Combination-attribute Appearance Threshold",
    )
    parser.add_argument(
        "-lat",
        type=int,
        nargs="?",
        default=None,
        help="Count-based: Label-attribute Appearance Threshold",
    )
    parser.add_argument(
        "-cft",
        type=int,
        nargs="?",
        default=None,
        help="Count-based: Combination-attribute Frequency Threshold",
    )
    parser.add_argument(
        "-lft",
        type=int,
        nargs="?",
        default=None,
        help="Count-based: Label-attribute Frequency Threshold",
    )

    # Arguments for mutual-information-based attribute reduction
    parser.add_argument(
        "-mit",
        type=float,
        nargs="?",
        default=None,
        help="Mutual-information-based: MI threshold",
    )

    # Argument check
    args = parser.parse_args()
    if args.m == "count":
        # Check cat, lat, cft, lft are provided with a value
        for a in ["cat", "lat", "cft", "lft"]:
            assert vars(args)[a], f"No value provided under -{a} when -m=count"
    else:
        # Check et is provided
        assert args.mit, f"No value provided under -mit when -m=mi"

    print("Arguments accepted, start pre-processing")

    # Generate subsets filtered by labels only
    with open(args.rtrain, "rb") as f:
        train_raw = pickle.load(f)
    sub_train = data_subset_creation(train_raw, args.nl)

    with open(args.rval, "rb") as f:
        val_raw = pickle.load(f)
    sub_val = data_subset_creation(val_raw, args.nl)

    with open(args.rtest, "rb") as f:
        test_raw = pickle.load(f)
    sub_test = data_subset_creation(test_raw, args.nl)

    # Compute the attribute mask for attribute reduction
    if args.m == "count":
        # Count-based attribute reduction
        attribute_mask = attribute_reduction_mask_count_based(
            dataset=sub_train,
            total_number_attr=args.rna,
            total_number_labels=args.nl,
            comb_appearance_threshold=args.cat,
            label_appearance_threshold=args.lat,
            min_comb_threshold=args.cft,
            min_label_threshold=args.lft,
        )
    else:
        # Entropy-based attribute reduction
        attribute_mask = attribute_reduction_mask_mi_based(
            dataset=sub_train,
            total_number_attr=args.rna,
            total_number_labels=args.nl,
            mi_threshold=args.mit,
        )

    # Filter subsets' attributes and save
    train_dataset = attribute_reduction(
        dataset=sub_train,
        raw_total_number_attr=args.rna,
        total_number_labels=args.nl,
        attribute_mask=attribute_mask,
    )
    with open(f"{args.od}/train.pkl", "wb") as f:
        pickle.dump(train_dataset, f)
    print("Processed train dataset stored")

    val_dataset = attribute_reduction(
        dataset=sub_val,
        raw_total_number_attr=args.rna,
        total_number_labels=args.nl,
        attribute_mask=attribute_mask,
    )
    with open(f"{args.od}/val.pkl", "wb") as f:
        pickle.dump(val_dataset, f)
    print("Processed val dataset stored")

    test_dataset = attribute_reduction(
        dataset=sub_test,
        raw_total_number_attr=args.rna,
        total_number_labels=args.nl,
        attribute_mask=attribute_mask,
    )
    with open(f"{args.od}/test.pkl", "wb") as f:
        pickle.dump(test_dataset, f)
    print("Processed test dataset stored")

    # Store mask and median
    with open(f"{args.od}/mask.pkl", "wb") as f:
        pickle.dump(attribute_mask, f)
    print("Attr mask stored")

    print()
    print("---------------Summary---------------")
    print(f"Number of labels:          {args.nl}")
    if args.m == "count":
        print(f"Attribute reduction:       Count-based")
        print(f"Comb-attr appearance t:    {args.cat}")
        print(f"Label-attr appearance t:   {args.lat}")
        print(f"Comb-attr frequence t:     {args.cft}")
        print(f"Label-attr frequence t:    {args.lft}")
    else:
        print(f"Attribute reduction:       Entropy-based")
        print(f"MI threshold:              {args.mit}")
    print(f"Number of attributes used: {len(attribute_mask)}")
    print(f"Num train samples:         {len(train_dataset)}")
    print(f"Num val samples:           {len(val_dataset)}")
    print(f"Num test samples:          {len(test_dataset)}")
    print(f"Num train skipped:         {len(sub_train) - len(train_dataset)}")
    print(f"Num val skipped:           {len(sub_val) - len(val_dataset)}")
    print(f"Num test skipped:          {len(sub_test) - len(test_dataset)}")
    print("-------------------------------------")
