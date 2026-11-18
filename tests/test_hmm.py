import pytest
import numpy as np
import pandas as pd
from pathsnap.hmm.decoder import HMMViterbiDecoder
from pathsnap.models.evaluate import compute_route_continuity

def test_hmm_viterbi_decoding():
    # Synthetic road network: arc 0 (node 0->1), arc 1 (node 1->2), arc 2 (node 2->3)
    arcs = np.array([
        [0, 1],
        [1, 2],
        [2, 3]
    ])
    # 3 GPS observations
    timestamps = np.array([100.0, 101.0, 102.0])
    # Candidates DataFrame: obs 0 has [arc 0 (dist 2.0), arc 2 (dist 20.0)]
    # obs 1 has [arc 0 (dist 10.0), arc 1 (dist 1.5)]
    # obs 2 has [arc 1 (dist 15.0), arc 2 (dist 1.0)]
    cands_df = pd.DataFrame({
        "obs":  [0, 0, 1, 1, 2, 2],
        "arc":  [0, 2, 0, 1, 1, 2],
        "dist": [2.0, 20.0, 10.0, 1.5, 15.0, 1.0]
    })

    decoder = HMMViterbiDecoder(sigma=5.0, p_stay=0.7, p_conn=0.25, p_other=1e-4)
    pred_arcs = decoder.decode(cands_df, arcs, timestamps, num_obs=3)

    assert len(pred_arcs) == 3
    # Expected continuous sequence: 0 -> 1 -> 2
    assert list(pred_arcs) == [0, 1, 2]
    assert compute_route_continuity(pred_arcs, arcs) == 1.0

def test_hmm_gap_splitting():
    arcs = np.array([[0, 1], [1, 2]])
    # Observation 1 has a large time gap (> 30s)
    timestamps = np.array([100.0, 200.0])
    cands_df = pd.DataFrame({
        "obs":  [0, 1],
        "arc":  [0, 1],
        "dist": [2.0, 2.0]
    })
    decoder = HMMViterbiDecoder(gap_threshold_s=30.0)
    pred_arcs = decoder.decode(cands_df, arcs, timestamps, num_obs=2)
    assert len(pred_arcs) == 2
    assert pred_arcs[0] == 0 and pred_arcs[1] == 1
