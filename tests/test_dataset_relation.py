import json
import h5py
import numpy as np
from icsm_as.core.physics_data import PhysicsDataset

def test_materialized_dataset_exposes_x_rel(tmp_path):
    path = tmp_path / "prepared.h5"
    n, f, nt, k = 3, 2, 4, 2
    gram = np.zeros((n, f, nt, nt), dtype=np.float32)
    for i in range(n):
        for j in range(f):
            gram[i, j] = np.eye(nt, dtype=np.float32) * (1.0 + i + j)
    relation = np.zeros((n, nt, nt), dtype=np.float32)
    relation[:, np.arange(nt), np.arange(nt)] = 1.0
    with h5py.File(path, "w") as h:
        h.attrs["complete"] = True
        h.attrs["metadata_json"] = json.dumps({"n_tx": nt, "n_sel": k, "storage_mode": "materialized"})
        h.create_dataset("G_tx_real", data=gram)
        h.create_dataset("G_tx_imag", data=np.zeros_like(gram))
        h.create_dataset("X_rel", data=relation)
        h.create_dataset("effective_snr_db", data=np.array([0, 5, 10], np.float32))
        h.create_dataset("snr_reference_db", data=np.array([0, 5, 10], np.float32))
        h.create_dataset("snr_reference_linear", data=10 ** (np.array([0, 5, 10], np.float32) / 10))
        h.create_dataset("hq_capacity", data=np.ones(n, np.float32) * 2)
        h.create_dataset("greedy_capacity", data=np.ones(n, np.float32) * 2.1)
        h.create_dataset("lq_capacity", data=np.ones(n, np.float32) * 1.8)
        h.create_dataset("hq_labels", data=np.tile(np.array([0, 1]), (n, 1)))
        h.create_dataset("lq_labels", data=np.tile(np.array([1, 2]), (n, 1)))
        h.create_dataset("teacher_order", data=np.tile(np.array([0, 1]), (n, 1)))
        h.create_dataset("greedy_order", data=np.tile(np.array([0, 1]), (n, 1)))
        h.create_dataset("teacher_marginals", data=np.ones((n, k, nt), np.float32))
        h.create_dataset("sample_id", data=np.arange(n))
        h.create_dataset("base_sample_id", data=np.arange(n))
        h.create_dataset("site_id", data=np.arange(n))
        h.create_dataset("split_id", data=np.array([0, 1, 2]))
    ds = PhysicsDataset(path, split=0)
    sample = ds[0]
    assert sample["X_rel"].shape == (nt, nt)
    assert sample["relation_target"].shape == (nt, nt)
    ds.close()
