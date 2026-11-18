import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

class HMMViterbiDecoder:
    """
    Hidden Markov Model sequence decoder for map-matching noisy GPS trajectories
    to continuous routes along a directed road graph using Viterbi decoding.
    
    Supports:
    - Pure distance-based Gaussian emissions or Hybrid (classifier probability + distance)
    - Topological transition probabilities respecting road connectivity
    - Automatic segment splitting on large GPS gaps (> gap_threshold_s)
    - Robust fallback on missing candidates and disconnected components
    """
    def __init__(
        self,
        sigma: float = 10.0,
        p_stay: float = 0.70,
        p_conn: float = 0.25,
        p_other: float = 1e-4,
        gap_threshold_s: float = 30.0,
        classifier_weight: float = 1.0,
        p_uturn: Optional[float] = None,
    ):
        self.sigma = sigma
        self.log_p_stay = np.log(p_stay)
        self.log_p_conn = np.log(p_conn)
        self.log_p_other = np.log(p_other)
        self.gap_threshold_s = gap_threshold_s
        self.classifier_weight = classifier_weight
        # Optional penalty for moving onto the reverse of the previous arc (a U-turn).
        # None keeps the original behaviour (U-turns treated as ordinary connections).
        self.log_p_uturn = None if p_uturn is None else np.log(p_uturn)

    def decode(
        self,
        candidates_df: pd.DataFrame,
        arcs: np.ndarray,
        timestamps: np.ndarray,
        num_obs: int,
        classifier_probs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Execute Viterbi sequence decoding over all observations.
        
        Args:
            candidates_df: DataFrame with ['obs', 'arc', 'dist']
            arcs: Directed road segments of shape (E, 2) [from_node, to_node]
            timestamps: Timestamps of each GPS observation in seconds (shape N,)
            num_obs: Total number of observations N
            classifier_probs: Optional predicted probabilities P(y=1) per candidate row
            
        Returns:
            predicted_arcs: np.ndarray of shape (num_obs,) with matched arc ID per step (-1 if missing)
        """
        pred_arcs = np.full(num_obs, -1, dtype=int)
        if len(candidates_df) == 0:
            return pred_arcs

        head, tail = arcs[:, 1], arcs[:, 0]

        # Group candidates by observation
        df = candidates_df.copy()
        if classifier_probs is not None:
            df["clf_p"] = np.clip(classifier_probs, 1e-6, 1.0 - 1e-6)
        else:
            df["clf_p"] = 0.5

        groups = {}
        for obs_id, grp in df.groupby("obs"):
            c_arcs = grp["arc"].values.astype(int)
            c_dists = grp["dist"].values.astype(float)
            c_probs = grp["clf_p"].values.astype(float)
            groups[int(obs_id)] = (c_arcs, c_dists, c_probs)

        # Split into contiguous segments based on observation gaps or time jumps
        obs_keys = sorted(groups.keys())
        if not obs_keys:
            return pred_arcs

        segments: List[List[int]] = []
        cur_seg = [obs_keys[0]]
        for i in range(1, len(obs_keys)):
            prev_i, curr_i = obs_keys[i - 1], obs_keys[i]
            # Gap if missing observation index or elapsed time > threshold
            dt = timestamps[curr_i] - timestamps[prev_i] if timestamps is not None else 1.0
            if (curr_i != prev_i + 1) or (dt > self.gap_threshold_s):
                segments.append(cur_seg)
                cur_seg = [curr_i]
            else:
                cur_seg.append(curr_i)
        segments.append(cur_seg)

        # Decode each segment independently with Viterbi
        for seg in segments:
            seg_len = len(seg)
            if seg_len == 0:
                continue

            # Compute emissions for segment observations
            # Log emission = -dist^2 / (2 * sigma^2) + clf_weight * log(P(clf))
            emissions = []
            cand_arcs = []
            for obs_id in seg:
                a_list, d_list, p_list = groups[obs_id]
                cand_arcs.append(a_list)
                em = -(d_list ** 2) / (2.0 * (self.sigma ** 2))
                if self.classifier_weight > 0:
                    em += self.classifier_weight * np.log(p_list)
                emissions.append(em)

            # Viterbi forward pass in log space
            V = [emissions[0]]
            B = []  # backpointers

            for t in range(1, seg_len):
                prev_arcs = cand_arcs[t - 1]
                curr_arcs = cand_arcs[t]
                prev_v = V[t - 1]

                # Transition matrix from prev_arcs to curr_arcs: shape (len(prev), len(curr))
                # Case 1: same arc -> log_p_stay
                # Case 2: head[prev] == tail[curr] -> log_p_conn
                # Case 3: disconnected -> log_p_other
                same_arc = prev_arcs[:, None] == curr_arcs[None, :]
                is_connected = head[prev_arcs, None] == tail[curr_arcs][None, :]

                trans = np.full((len(prev_arcs), len(curr_arcs)), self.log_p_other, dtype=float)
                trans[is_connected] = self.log_p_conn
                if self.log_p_uturn is not None:
                    is_reverse = (head[prev_arcs, None] == tail[curr_arcs][None, :]) & \
                                 (tail[prev_arcs, None] == head[curr_arcs][None, :])
                    trans[is_reverse] = self.log_p_uturn
                trans[same_arc] = self.log_p_stay

                # Total score matrix: shape (len(prev), len(curr))
                total_scores = prev_v[:, None] + trans
                best_prev = np.argmax(total_scores, axis=0)
                best_scores = np.max(total_scores, axis=0) + emissions[t]

                V.append(best_scores)
                B.append(best_prev)

            # Viterbi backward backtracking
            best_last = int(np.argmax(V[-1]))
            path = [best_last]
            for t in range(seg_len - 1, 0, -1):
                best_last = int(B[t - 1][best_last])
                path.append(best_last)
            path.reverse()

            for t, obs_id in enumerate(seg):
                chosen_cand_idx = path[t]
                pred_arcs[obs_id] = cand_arcs[t][chosen_cand_idx]

        return pred_arcs
