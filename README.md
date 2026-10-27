# Reservoir Computing & Chaos Prediction Framework

These codes cover different reservoir network setups and application conditions based on the methodologies outlined by Fleddermann, Herzog & Parlitz, *Chaos* 35, 053147 (2025). 

For now, the implementations are all Echo State Networks (ESNs). They were developed to analyze performance variations in chaotic time-series prediction through the benchmark of **valid times** across specific operational transitions:

*   **Prediction Mode:** Transitioning from open-loop, one-step-ahead prediction to closed-loop, recursive iterative prediction.
*   **System Knowledge:** Shifting from full state knowledge ($u^{(x)}, u^{(y)}, u^{(z)}$) to partial state knowledge ($u^{(y)}$-only) on the Lorenz-63 attractor.
*   **Memory Augmentation:** Implementing historical time-delayed values across both the reservoir states and the driving input variables to reconstruct hidden dynamics.

## Repository Structure

All simulation pipelines live inside the `src/` directory, organized by computational depth:
*   `src/level1_toy_sine.py`: Baseline validation of leaky integrator ESN dynamics using a regularized sine wave.
*   `src/level2_esn_onestep_prediction.py`: One-step-ahead prediction on the Lorenz attractor under full state knowledge.
*   `src/level3_esn_iterative_prediction.py`: Closed-loop iterative tracking comparing full vs. partial state accuracy.
*   `src/level4_esn_delay_coordinates.py`: Reconstructing partial state attractors using history delay coordinates ($N_I, N_R$).
*   `src/level5_global_optimization.py`: Global hyperparameter optimization using integrated grid searches and local Nelder-Mead simplex optimization.

## Experimental Scaling
To optimize execution times on local hardware while ensuring numerical convergence, the tracking parameters use a lightweight test matrix ($M=5$ random reservoirs, $K_{val}=10$, and $K_{test}=20$ tracking trajectories).
