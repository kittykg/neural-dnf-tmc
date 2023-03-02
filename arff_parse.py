import argparse
import pickle
from typing import List, Tuple

from sklearn.model_selection import train_test_split

from common import MultiLabelRawSample

# M attributes, attribute id 0-(M-1) are actual attributes
# N label, attribute id M-(M+N-1) are actual labels
M = 500
N = 22
CLASS_LABEL_ID_START = M
CLASS_LABEL_ID_END = M + N - 1


def line_parse_sample(line_str: str, line_id: int) -> MultiLabelRawSample:
    # The line starts and ends with a pair of {}
    tuple_list = line_str[1:-1].split(",")
    all_id_list = [int(t.split(" ")[0]) for t in tuple_list]

    def in_class_label_id_range(i: int) -> bool:
        return i >= CLASS_LABEL_ID_START and i <= CLASS_LABEL_ID_END

    att_list = [i for i in all_id_list if not in_class_label_id_range(i)]
    label_list = [i - 500 for i in all_id_list if in_class_label_id_range(i)]

    return MultiLabelRawSample(
        sample_id=line_id, labels=label_list, present_attributes=att_list
    )


def parse_samples_file(file_path: str) -> List[MultiLabelRawSample]:
    i = 1
    samples_list = []

    with open(file_path, "r") as f:
        for l in f:
            s = line_parse_sample(l, i)
            samples_list.append(s)
            i += 1

    return samples_list


def split_train_val(
    samples: List[MultiLabelRawSample], random_seed: int
) -> List[List[MultiLabelRawSample]]:
    return train_test_split(samples, test_size=0.2, random_state=random_seed)


# Unused oversampling code
# def random_oversampling(data: List[MultiLabelRawSample], ratio: float = 1.0):
#     cnt = Counter()
#     for d in data:
#         for l in d.labels:
#             cnt[l] += 1
#     max_count = max(cnt.values())
#     additional_data = []
#     for k in cnt.keys():
#         gap = int(max_count * ratio - cnt[k])
#         if gap <= 0:
#             # No gap, don't oversample
#             continue
#         population = [d for d in data if k in d.labels]
#         additional_data += random.choices(population, k=gap)
#     return data + additional_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-train", type=str, help="Train arff path")
    parser.add_argument("-test", type=str, help="Test arff path")
    parser.add_argument("-od", type=str, help="Output directory")
    parser.add_argument(
        "-rs", type=int, default="73", help="Random seed, default 73"
    )

    args = parser.parse_args()

    print("Start data parsing")
    all_train = parse_samples_file(args.train)
    train_set, val_set = split_train_val(all_train, args.rs)
    test_set = parse_samples_file(args.test)

    for s, f_n in zip(
        [train_set, val_set, test_set], ["train.pkl", "val.pkl", "test.pkl"]
    ):
        with open(args.od + f_n, "wb") as f:
            pickle.dump(s, f)

    print("Data processing done")
