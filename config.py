from pathlib import Path
root_dir = Path(__file__).parent

n_clusters = 100
average_cluster_size = 1_000
strand_length = 100


complex_data_parameters = dict(
    n_orig_strands=n_clusters,
    length=strand_length,
    physical_redundancy=average_cluster_size // 4,
    p_sub_syn=0.02,
    p_del_syn=0.01,
    p_ins_syn=0.01,
    p_term_max_syn=0.1,
    p_decay=[0.002, 0.004, 0.006, 0.008],
    pcr_rounds=5,
    n_final_samples=average_cluster_size * n_clusters,
    p_sub_seq=0.02,
    ends_factor_seq=5
)


simple_data_parameters = dict(
    n_clusters=n_clusters,
    cluster_size=average_cluster_size,
    error_p=0.04,
    strand_length=strand_length
)
