# PathSnap: Distributed Map-Matching and Route Reconstruction

PES University, UE24CS352A Machine Learning, PathSnap Hackathon.
Team 814_189_842_188: Juhi, Gagana, Himesh, Hitashree.

Technical report: [`reports/PathSnap_Technical_Report.pdf`](reports/PathSnap_Technical_Report.pdf)

## 1. What it does

GPS points are noisy and drift off the road. PathSnap assigns each GPS observation to the road segment (arc) the vehicle was on, then reconstructs a continuous route with an HMM and Viterbi decoding. A FastAPI backend (laptop 1) serves predictions to an evaluator UI (laptop 2).

## 2. Deliverables map

| Deliverable component | Where |
|---|---|
| Data processing (read, inspect, clean) | `pathsnap/data/loader.py`, `pathsnap/data/projection.py`, `notebooks/01_Data_Processing_and_EDA.ipynb` |
| Feature engineering | `pathsnap/data/candidates.py` (core 5 features), `pathsnap/data/features_v2.py` (extended features) |
| Labels (training and scoring only) | `pathsnap/data/alignment.py` (proxy nearest-route-arc labels and chronological DP-aligned labels) |
| Model training (every model) | `scripts/train_all_models.py`, `scripts/train_extended_models.py`, `notebooks/02_Model_Experiments_and_Bayesian_Optimisation.ipynb` |
| Model evaluation, metrics, plots | `pathsnap/models/evaluate.py`, `results/tables/model_comparison*.csv`, `results/figures/` |
| Bayesian hyperparameter optimisation | `scripts/run_bayesian_optimization.py`, `results/tables/optuna_trials.csv`, `results/tables/tuning_comparison.csv`, `results/figures/optuna_convergence.png` |
| HMM and Viterbi | `pathsnap/hmm/decoder.py`, `pathsnap/data/graph.py`, `scripts/run_hmm_benchmark.py`, `scripts/run_hmm_v2.py`, `notebooks/03_HMM_Viterbi_Route_Reconstruction.ipynb` |
| API (endpoints, model loading) | `pathsnap/api/app.py` |
| UI (evaluator interface) | `pathsnap/ui/index.html` |
| Deployment (config, dependencies, instructions) | `requirements.txt`, section 5 below |
| Visualisations (Folium and plots) | `scripts/generate_visualizations.py`, `results/maps/`, `results/figures/` |
| Supporting results and logs | `results/tables/`, `results/notebook_outputs/`, `results/logs/` |
| Saved models and configuration | `artifacts/models/`, `artifacts/configs/best.json` |
| API and UI screenshots (two laptops) | `results/screenshots/` |
| Tests | `tests/` |

The dataset is not in this repository. Place it as described in section 5.

## 3. Method

1. **Projection:** longitude and latitude are converted to metres with a local equirectangular projection.
2. **Candidates:** for each GPS point, the 40 nearest arc midpoints (KD-tree), kept if within 50 m. That gives about 11 candidates per point. Candidate miss rate is about 1.5% (student test).
3. **Core features:** `dist` (point-to-arc distance), `hd` (travel heading vs arc bearing), `alen` (arc length), `speed`, `dt`.
4. **Extended features (v2):** smoothed heading over ±5 points, mean and max distance of neighbouring GPS points to the same arc, distance rank and gap within the point's candidates, projection position on the arc, and node degree.
5. **Labels:** each candidate gets 1 if it is the reference label for that point. The route is used only for labels and scoring, never for prediction.
   - **Proxy labels:** the nearest reference-route arc within 50 m.
   - **Chronological labels:** a monotonic dynamic-programming alignment along the ordered route. It fixes out-and-back trips, where the proxy label picks the wrong direction (on train trip 00000011, the share of labels pointing against travel drops from 49.5% to 0%).
6. **Model selection:** 5-fold GroupKFold by trip on the 80 training trips. The 10 student-test trips are used only for reporting. Evaluator trips are never used in development.
7. **Bayesian optimisation:** Optuna TPE, 20 trials, 3-fold grouped CV, tuning HistGradientBoosting. Search space: learning rate, max_leaf_nodes, min_samples_leaf, l2_regularization, max_iter.
8. **HMM and Viterbi:**
   - Hidden states are the candidate arcs.
   - Emission is Gaussian on distance, optionally plus the classifier's log-probability.
   - Transitions: stay, connected (head of the previous arc = tail of the next), other, and an optional U-turn penalty.
   - Sequences are split at gaps over 30 s or at points without candidates.
   - Decoder parameters are chosen on training trips only.

## 4. Results (student-test partition)

### 4.1 Five baseline models, core features, proxy labels (`results/tables/model_comparison.csv`)

| Model | CV top-1 | Test top-1 | Test macro F1 | Log loss |
|---|---|---|---|---|
| Logistic Regression | 0.741 | 0.757 | 0.781 | 0.162 |
| Gaussian Naive Bayes | 0.711 | 0.715 | 0.767 | 0.311 |
| Decision Tree | 0.709 | 0.722 | 0.811 | 0.148 |
| Random Forest | 0.721 | 0.731 | 0.811 | 0.141 |
| HistGradientBoosting | 0.722 | 0.734 | 0.811 | 0.141 |

HistGradientBoosting was selected for the API (`hgb_tuned`) because it has the best probability quality (lowest log loss), which the HMM relies on.

### 4.2 Tuning (`results/tables/tuning_comparison.csv`)

| Variant | CV top-1 (3-fold) | Test top-1 | Test log loss |
|---|---|---|---|
| Untuned HGB | 0.6875 | 0.7292 | 0.1423 |
| Tuned HGB | 0.6893 | 0.7314 | 0.1453 |

### 4.3 HMM (`results/tables/hmm_vs_baseline.csv`)

| Method | Test top-1 | Continuity |
|---|---|---|
| Nearest arc | 0.553 | 0.806 |
| Classifier only | 0.731 | 0.882 |
| Pure HMM | 0.873 | 0.996 |
| Classifier + HMM | 0.797 | 0.993 |

### 4.4 Extended features (`results/tables/model_comparison_v2.csv`, `model_comparison_v2_dp.csv`)

| Model | Labels | CV top-1 | Test top-1 | Test top-1 vs proxy labels | Test log loss |
|---|---|---|---|---|---|
| LightGBM | proxy | 0.871 | 0.845 | 0.845 | 0.076 |
| Logistic Regression | proxy | 0.866 | 0.851 | 0.851 | 0.077 |
| Decision Tree | proxy | 0.865 | 0.850 | 0.850 | 0.082 |
| Gaussian NB | proxy | 0.839 | 0.816 | 0.816 | 0.970 |
| LightGBM | chronological | 0.885 | 0.888 | 0.853 | 0.052 |

Logistic Regression in the v2 table was fitted on a 400,000-row random sample per fold; the others used all rows.

### 4.5 HMM on extended LightGBM (`results/tables/hmm_v2_summary_lightgbm_dp.json`)

- 38 decoder settings were evaluated on out-of-fold training predictions (`results/tables/hmm_grid_train_v2_dp.csv`).
- The selected setting is σ = 10 m, p_other = 1e-6, U-turn penalty 1e-3, classifier weight 0.

| Student test | Top-1 (chronological labels) | Top-1 (proxy labels) | Continuity |
|---|---|---|---|
| LightGBM only | 0.888 | 0.853 | 0.955 |
| HMM | 0.915 | 0.877 | 0.998 |

The training-trip grid selected classifier weight 0, so on this data the classifier did not improve the HMM's decoding further.

## 5. How to run (Windows CMD)

### Data layout

```
data/
  train/<trip_id>/<trip_id>.track .nodes .arcs .route
  student_test/...
  evaluator_test/...
```

### Laptop 1: API

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
set "PATHSNAP_DATA=%CD%\data\student_test"
set PATHSNAP_ARTIFACTS=artifacts
uvicorn pathsnap.api.app:app --host 0.0.0.0 --port 8000
```

- Run `ipconfig` and note the Wi-Fi IPv4 address.
- If laptop 2 cannot connect, allow port 8000 through the firewall from an administrator CMD:
  ```
  netsh advfirewall firewall add rule name="PathSnap 8000" dir=in action=allow protocol=TCP localport=8000
  ```

Endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Service status and loaded model |
| `GET /trips?partition=student_test` | List trips in a partition |
| `POST /predict` | Body `{"trip_id": "...", "mode": "hybrid" or "classifier_only" or "hmm_only", "data_partition": "evaluator_test"}` |

Scoring happens on the server. The reference route is never returned to the client.

### Laptop 2: UI

```
cd <folder containing index.html>
python -m http.server 5500
```

- Open `http://localhost:5500/index.html`. Serve it over HTTP: OpenStreetMap blocks map tiles for pages opened as `file://`.
- Set the API URL to `http://<laptop-1-IP>:8000`, click Connect & Refresh, pick a trip, then Predict Route.

### Reproducing experiments

Set `PATHSNAP_DATA_ROOT` to the `data` folder, then run:

```
python scripts/train_all_models.py
python scripts/run_bayesian_optimization.py
python scripts/run_hmm_benchmark.py
python scripts/build_features_v2.py data cache
python scripts/build_labels_dp.py cache data
python scripts/train_extended_models.py cache data .
set PATHSNAP_LABELS=dp
python scripts/train_extended_models.py cache data . lightgbm
python scripts/run_hmm_v2.py cache data lightgbm_dp .
python scripts/generate_visualizations.py
pytest -q
```

## 6. Team contributions

| Member | Contribution | Branch |
|---|---|---|
| Juhi | Team lead; data audit and processing notebook; integration into `main`; FastAPI backend setup, two-laptop deployment and live demo; extended features, chronological labels and LightGBM experiments | `juhi/data-integration` |
| Gagana | Model experiments notebook (five models, grouped CV) and Optuna tuning | `gagana/model-experiments` |
| Himesh | HMM and Viterbi notebook, decoder parameter grid, HMM vs baseline comparison | `himesh/route-reconstruction` |
| Hitashree | Initial FastAPI and UI prototype | `hitashree/api-ui` |

The original notebook outputs of each member are kept in `results/notebook_outputs/`.

## 7. Limitations

- Labels are derived from the reference route, which is supplied reference information and not verified ground truth.
  - Proxy labels are ambiguous on out-and-back routes.
  - Some trips (for example train 00000033) have routes covering only part of the GPS track.
- About 1.5% of labelled points have no correct candidate within 50 m, which caps achievable top-1 accuracy at about 98.5%.
- The projection is a local equirectangular approximation.
- The API currently serves `hgb_tuned` with core features. The extended LightGBM pipeline (section 4.4 and 4.5) is evaluated offline and not yet wired into the API.
- The technical report PDF predates the extended results in sections 4.4 and 4.5.

## 8. AI assistance disclosure

AI assistants (Claude, among others) were used to draft code, scripts, notebooks and documentation, including the consolidated `pathsnap/` package, the extended-feature and chronological-label experiments, and this README. All experiments were run on the supplied data. Every reported figure comes from files in `results/`. Team members reviewed the work and can explain their parts.
