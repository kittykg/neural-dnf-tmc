import argparse
import pickle
from typing import List, Optional, Tuple

import numpy as np
import torch

from common import MultiLabelRawSample, MultiLabelDatasetSample


def attribute_reduction(
    dataset: List[MultiLabelRawSample],
    total_number_attr: int,
    total_number_labels: int,
    comb_appearance_threshold: int,
    label_appearance_threshold: int,
    min_comb_threshold: int,
    min_label_threshold: int,
    precomputed_mask: Optional[np.ndarray] = None,
) -> Tuple[List[MultiLabelDatasetSample], np.ndarray]:
    if precomputed_mask is not None:
        attribute_mask = precomputed_mask
    else:
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

    # Convert raw sample to dataset sample
    def _get_new_data(
        d: MultiLabelRawSample,
    ) -> Optional[MultiLabelDatasetSample]:
        temp_d = d.to_dataset_sample(total_number_attr, total_number_labels)
        # Adjust the attribute encoding
        new_attr_encoding = temp_d.attribute_encoding[attribute_mask]  # type: ignore
        if torch.count_nonzero(new_attr_encoding) == 0:
            return None
        return MultiLabelDatasetSample(
            temp_d.sample_id,
            temp_d.label_encoding,
            new_attr_encoding,
        )

    return [
        _get_new_data(d) for d in dataset if _get_new_data(d)
    ], attribute_mask  # type: ignore


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
        "-cat", type=int, help="Combination-attribute appearance threshold"
    )
    parser.add_argument(
        "-lat", type=int, help="Label-attribute appearance threshold"
    )
    parser.add_argument(
        "-cft", type=int, help="Combination-attribute frequency threshold"
    )
    parser.add_argument(
        "-lft", type=int, help="Label-attribute frequency threshold"
    )
    parser.add_argument(
        "-od", type=str, help="Output directory (no trailing '/')"
    )
    args = parser.parse_args()

    print("Start pre-process")

    # Train dataset
    with open(args.rtrain, "rb") as f:
        train_raw = pickle.load(f)
    train_dataset, mask = attribute_reduction(
        dataset=data_subset_creation(train_raw, args.nl),
        total_number_attr=args.rna,
        total_number_labels=args.nl,
        comb_appearance_threshold=args.cat,
        label_appearance_threshold=args.lat,
        min_comb_threshold=args.cft,
        min_label_threshold=args.lft,
    )
    with open(f"{args.od}/train.pkl", "wb") as f:
        pickle.dump(train_dataset, f)
    print("Processed train dataset stored")

    # Val dataset
    with open(args.rval, "rb") as f:
        val_raw = pickle.load(f)
    val_dataset, _ = attribute_reduction(
        dataset=data_subset_creation(val_raw, args.nl),
        total_number_attr=args.rna,
        total_number_labels=args.nl,
        comb_appearance_threshold=args.cat,
        label_appearance_threshold=args.lat,
        min_comb_threshold=args.cft,
        min_label_threshold=args.lft,
        precomputed_mask=mask,
    )
    with open(f"{args.od}/val.pkl", "wb") as f:
        pickle.dump(val_dataset, f)
    print("Processed val dataset stored")

    # Test dataset
    with open(args.rtest, "rb") as f:
        test_raw = pickle.load(f)
    test_dataset, _ = attribute_reduction(
        dataset=data_subset_creation(test_raw, args.nl),
        total_number_attr=args.rna,
        total_number_labels=args.nl,
        comb_appearance_threshold=args.cat,
        label_appearance_threshold=args.lat,
        min_comb_threshold=args.cft,
        min_label_threshold=args.lft,
        precomputed_mask=mask,
    )
    with open(f"{args.od}/test.pkl", "wb") as f:
        pickle.dump(test_dataset, f)
    print("Processed test dataset stored")

    # Store mask and median
    with open(f"{args.od}/mask.pkl", "wb") as f:
        pickle.dump(mask, f)
    print("Attr mask stored")

    print()
    print("---------------Summary---------------")
    print(f"Number of labels:          {args.nl}")
    print(f"Comb-attr appearance t:    {args.cat}")
    print(f"Label-attr appearance t:   {args.lat}")
    print(f"Comb-attr frequence t:     {args.cft}")
    print(f"Label-attr frequence t:    {args.lft}")
    print(f"Number of attributes used: {len(mask)}")
    print(f"Num train samples:         {len(train_dataset)}")
    print(f"Num val samples:           {len(val_dataset)}")
    print(f"Num test samples:          {len(test_dataset)}")
    print("-------------------------------------")
