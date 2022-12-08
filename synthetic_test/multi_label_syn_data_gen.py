from typing import List

import clingo
import numpy as np

RNG_SEED = 73
NUM_NULLARY = 15
NUM_CONJUNCTS = 5
NUM_LABELS = 3
FILE_PATH = f"synth_multi_label_data_in{NUM_NULLARY}_conj{NUM_CONJUNCTS}.npz"
GEN_SIZE = 10000


def get_rule_asp(and_kernel: np.ndarray, or_kernel: np.ndarray) -> List[str]:
    rule_asp = []

    for i, k in enumerate(and_kernel):
        conj_atoms = []
        for j, a in enumerate(k):
            if a == 1:
                conj_atoms.append(f"a{j}")
            elif a == -1:
                conj_atoms.append(f"not a{j}")
        rule_asp.append(f"c{i} :- " + ", ".join(conj_atoms) + ".")

    for i, k in enumerate(or_kernel.T):
        for j, a in enumerate(k):
            if a == 1:
                rule_asp.append(f"l{i} :- c{j}.")

    return rule_asp


def get_show_statements(num_labels: int) -> List[str]:
    show_statements = []
    # show_statements += [f'#show c{i}/0.' for i in range(num_conjuncts)]
    show_statements += [f"#show l{i}/0." for i in range(num_labels)]
    return show_statements


def example_tensor_to_asp(example: np.ndarray) -> List[str]:
    return [f"a{i}." for i in range(len(example)) if example[i] == 1]


def clingo_solve(
    example_asp: List[str], rule_asp: List[str], show_statements: List[str]
) -> str:
    ctl = clingo.Control(["--warn=none"])
    ctl.add("base", [], " ".join(rule_asp + example_asp + show_statements))
    ctl.ground([("base", [])])
    with ctl.solve(yield_=True) as handle:  # type: ignore
        model = handle.model()
    return str(model)


def generate_data() -> str:
    rng = np.random.default_rng(seed=RNG_SEED)

    in_size = NUM_NULLARY
    num_conjuncts = NUM_CONJUNCTS
    num_labels = NUM_LABELS
    gen_size = GEN_SIZE

    # For multi-label classification, multiple rules can fire at the same time.
    # So we want the data can trigger multiple conjunctions (used by different
    # rules) at the same time. A naive way to do it is to let each conjunction
    # only use a small subset of atoms.
    # Say we have 15 attributes and we have 5 conjunctions. If conjunction 1
    # used by label 1 is attribute 1 ^ 2 ^ 3, and conjunction 2 used by label 2
    # is attribute 4 ^ 5 ^ 6. Both conjunction 1 and 2 can fire at the same time
    # if a data point have all attribute 1-6. If both conjunction 1 and 2 are
    # fired, then label 1 and label 2 are both true, achieving multi-label.
    assert (
        in_size % num_conjuncts == 0
    ), "Expected full division of NUM_NULLARY / NUM_CONJUNCTS"
    atoms_to_use = int(in_size / num_conjuncts)

    # Create and_kernel such that each conjunction uses a subset of input.
    # We also make each sub-kernel different. This is not necessary, but make
    # the rules different and has variety.
    and_kernel = np.zeros((num_conjuncts, in_size)).astype(int)
    seen_sub_kernel = []
    for i in range(num_conjuncts):
        while True:
            sub_kernel = rng.choice([-1, 0, 1], size=atoms_to_use)
            unique = True
            for s in seen_sub_kernel:
                if (sub_kernel == s).all():
                    unique = False
            if unique:
                break
        and_kernel[i, i * atoms_to_use : (i + 1) * atoms_to_use] = sub_kernel

    or_kernel = rng.choice([0, 1], size=(num_conjuncts, num_labels))
    while not (or_kernel.any(axis=0)).all():  # We want at least one conjunction
        or_kernel = rng.choice([0, 1], size=(num_conjuncts, num_labels))

    rule_asp = get_rule_asp(and_kernel, or_kernel)
    show_statements = get_show_statements(num_labels)

    examples = []
    target = []
    i = gen_size

    while i > 0:
        kernel_choices = np.where(rng.choice([0, 1], size=num_conjuncts))[0]
        ckernel = and_kernel[kernel_choices]
        cmask_free = (ckernel == 0).all(axis=0)
        cmask_restrict = (ckernel == 1).any(axis=0)
        free_slot = cmask_free * rng.choice([-1, 1], size=in_size)
        restricted_slot = np.where(cmask_restrict, 1, -1)
        example = np.where(cmask_free, free_slot, restricted_slot)

        model = clingo_solve(
            example_tensor_to_asp(example), rule_asp, show_statements
        )
        if not model:
            # No label, ignore this example
            continue

        labels = [int(l[1]) for l in model.split(" ")]
        label_one_hot = np.zeros((num_labels)).astype(int)
        label_one_hot[labels] = 1

        examples.append(example)
        target.append(label_one_hot)
        i -= 1

    data = {
        "nullary": np.concatenate(examples).reshape((gen_size, in_size)),
        "target": np.concatenate(target).reshape((gen_size, num_labels)),
        "and_kernel": and_kernel,
        "or_kernel": or_kernel,
        "rule_str": rule_asp,
    }

    # Save the file
    print(f"Creating {str(FILE_PATH)} with keys: {str(data.keys())}")
    np.savez_compressed(FILE_PATH, **data)

    # Output rules to STDOUT
    for r in rule_asp:
        print(r)

    return str(FILE_PATH)


if __name__ == "__main__":
    # Generated npz should have keys:
    # ['nullary', 'target', 'and_kernel', 'or_kernel', 'rule_str']
    generate_data()
