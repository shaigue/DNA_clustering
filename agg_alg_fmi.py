# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Press the green button in the gutter to run the script.
# from clustering import *
import time

from clustering import *
from dna_data_structure import *
from clustering_accuracy import *
from fmi_metrics import *

"""Simple translation functions used to check article algorithm:"""

def acgt_to_string(s: list[list[str]]) -> list[list[str]]:
    """Translates from 'ACGT' form to '00011011' form, single string."""
    s_out = [[""] for i in range(len(s))]
    for i in range(len(s) - 1):
        h = ""
        for j in range(len(s[i])):
            if s[i][j] == 0:
                h += "00"
            if s[i][j] == 1:
                h += "01"
            if s[i][j] == 2:
                h += "10"
            if s[i][j] == 3:
                h += "11"
        s_out[i][0] = h
    return s_out


def single_string_to_actg(bin_str: str) -> str:
    """Translates from '00011011' form to 'ACGT' form, single string."""
    y = ""
    i = 1
    while (1):
        if i >= len(bin_str):
            break
        if bin_str[i - 1] == '0' and bin_str[i] == '0':
            y += "A"
        if bin_str[i - 1] == '0' and bin_str[i] == '1':
            y += "C"
        if bin_str[i - 1] == '1' and bin_str[i] == '0':
            y += "G"
        if bin_str[i - 1] == '1' and bin_str[i] == '1':
            y += "T"
        i = i + 2
    return y


def string_list_to_actg(b: list[list[str]]) -> list[list[str]]:
    """Translates from '00011011' form to 'ACGT' form, multiple string."""
    s_out = b
    for i in range(len(s_out)):
        for j in range(len(s_out[i])):
            s_out[i][j] = single_string_to_actg(b[i][j])
    return s_out


def find_str_in_dict(s: str, d: dict) -> int:
    """Finds the phase number of acgt string in dictionary."""
    for i in range(len(d['original_strands'])):
        if (d['original_strands'][i] == s):
            return i


def actg_to_set_list(s: list[list[str]], d: dict) -> list[set[int]]:
    """Translates from acgt form to int set, as in json files."""
    s_out = [set() for m in range(len(s))]
    for i in range(len(s)):
        for j in range(len(s[i])):
            t = find_str_in_dict(s[i][j], d)
            s_out[i].add(t)
    return s_out


def clear_none_sets(y: list[set[int]]) -> list[set[int]]:
    """Removes {None} sets."""
    for i in y:
        if (None in i):
            y.remove(i)
    return y


if __name__ == '__main__':
    orig_sample = DNASample([], 0)
    orig_sample_set = DNASampleSet([[]], [orig_sample])
    json_path = Path("data/complex-test-data.json")
    true_temp_clustering = orig_sample_set.get_cluster_from_json(json_path)  # array-like type
    true_clustering = convert_cluster_labels_to_partition(true_temp_clustering)
    data = orig_sample_set.from_json(json_path)  # DNASampleSet type
    data_dict = data.to_dict()
    print(true_clustering)
    #creating dna strands
    """start_dna = time.time()
    dna_samples = create_dna_samples()
    s = dna_samples.orig_strands
    end_dna = time.time()
    print(f"time: {end_dna - start_dna}")"""
    # clustering
    start_dna = time.time()
    s = acgt_to_string(data.orig_strands)
    r = 20
    q = 3
    w = 5
    l = 5
    t_low = 15
    t_high = 25
    local_steps = 10
    res = cluster(s, r, q, w, l, t_low, t_high, local_steps)
    # class_res = DNASampleSet(res, res)
    # class_res.to_json(Path("C:/Users/dalit/PycharmProjects/pythonProject/data/simple-test-res.json"))
    x = string_list_to_actg(res)
    estimated_clustering = clear_none_sets(actg_to_set_list(x, data_dict))
    print(estimated_clustering)
    print(len(estimated_clustering))
    print(fmi(true_clustering, estimated_clustering))
    end_dna = time.time()
    print(f"time: {end_dna - start_dna}")
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
