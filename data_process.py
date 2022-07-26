import argparse
import pickle
from typing import List, Tuple

from sklearn.model_selection import train_test_split

from common import TmcRawSample

# 500 attributes, attribute id 0-499 are actual attribute labels
# 22 classes, attribute id 500-521 are actual class labels
CLASS_LABEL_ID_START = 500
CLASS_LABEL_ID_END = 521


def line_parse_sample(line_str: str, line_id: int) -> TmcRawSample:
    # The line starts and ends with a pair of {}
    tuple_list = line_str[1:-1].split(",")
    all_id_list = [int(t.split(" ")[0]) for t in tuple_list]

    def in_class_label_id_range(i: int) -> bool:
        return i >= CLASS_LABEL_ID_START and i <= CLASS_LABEL_ID_END

    att_list = [i for i in all_id_list if not in_class_label_id_range(i)]
    label_list = [i - 500 for i in all_id_list if in_class_label_id_range(i)]

    return TmcRawSample(
        sample_id=line_id, labels=label_list, present_attributes=att_list
    )


def parse_samples_file(file_path: str) -> List[TmcRawSample]:
    i = 1
    tmc_samples_list = []

    with open(file_path, "r") as f:
        for l in f:
            s = line_parse_sample(l, i)
            tmc_samples_list.append(s)
            i += 1

    return tmc_samples_list


def split_train_val(
    tmc_samples: List[TmcRawSample],
    random_seed: int,
) -> Tuple[List[TmcRawSample], List[TmcRawSample]]:
    return train_test_split(
        tmc_samples, test_size=0.2, random_state=random_seed
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-train", type=str, help="Train arff path")
    parser.add_argument("-test", type=str, help="Test arff path")
    parser.add_argument("-od", type=str, help="Output directory")
    parser.add_argument(
        "-rs", type=int, default="73", help="Random seed, deafult"
    )

    args = parser.parse_args()

    print("Start data processing")
    all_train = parse_samples_file(args.train)
    train_set, val_set = split_train_val(all_train, args.rs)
    test_set = parse_samples_file(args.test)

    for s, f_n in zip(
        [train_set, val_set, test_set], ["train.pkl", "val.pkl", "test.pkl"]
    ):
        with open(args.od + f_n, "wb") as f:
            pickle.dump(s, f)

    print("Data processing done")
