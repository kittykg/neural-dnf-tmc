import argparse
import pickle
import sys

from common import MultiLabelDatasetSample


def gen_las_example(
    pkl_data_path: str, save_file_path: str, num_classes: int
) -> None:
    def gen_example_from_data(sample: MultiLabelDatasetSample, file=sys.stdout):
        # Penalty
        print(
            f"#pos(eg_{sample.sample_id}@{1}, {{",
            file=file,
        )

        # Inclusion set
        inclusion_set = ",\n".join(
            [
                f"    class({i})"
                for i, c in enumerate(sample.label_encoding)
                if c == 1
            ]
        )
        print(inclusion_set, file=file)
        print("}, {", file=file)

        # Exclusion set
        exclusion_set = ",\n".join(
            [
                f"    class({i})"
                for i, c in enumerate(sample.label_encoding)
                if c == 0
            ]
        )
        print(exclusion_set, file=file)
        print("}, {", file=file)

        # Context
        for i, a in enumerate(sample.attribute_encoding):
            if a == 1:
                print(f"    has_attr_{i}.", file=file)
        print("}).\n", file=file)

    with open(pkl_data_path, "rb") as f:
        train_file = pickle.load(f)

    with open(save_file_path, "w") as f:
        for sample in train_file:
            gen_example_from_data(sample, f)


def gen_las_background_knowledge(
    save_file_path: str,
    num_classes: int,
    num_attributes: int,
    is_ilasp: bool = False,
) -> None:
    with open(save_file_path, "w") as f:
        print(f"class_id(0..{num_classes - 1}).", file=f)
        print("#modeh(class(const(class_id))).", file=f)
        for i in range(num_attributes):
            print(f"#modeb(has_attr_{i}).", file=f)
            if not is_ilasp:
                # FastLas requires explicit 'not' to include in hypothesis space
                print(f"#modeb(not has_attr_{i}).", file=f)
        if not is_ilasp:
            # FastLas scoring function
            print('#bias("penalty(1, head).").', file=f)
            print('#bias("penalty(1, body(X)) :- in_body(X).").', file=f)


################################################################################
#              Run below for LAS background + examples generation              #
################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-bk", type=str, help="Output file path for background")
    parser.add_argument("-e", type=str, help="Output file path for examples")
    parser.add_argument("-t", type=str, help="Input training file path")
    parser.add_argument("-nc", type=int, help="Number of classes")
    parser.add_argument("-na", type=int, help="Number of attributes")
    parser.add_argument("-ilasp", dest="is_ilasp", action="store_true")
    parser.add_argument("-fastlas", dest="is_ilasp", action="store_false")
    parser.set_defaults(is_ilasp=False)
    args = parser.parse_args()

    gen_las_background_knowledge(args.bk, args.nc, args.na, args.is_ilasp)
    gen_las_example(args.t, args.e, args.nc)
