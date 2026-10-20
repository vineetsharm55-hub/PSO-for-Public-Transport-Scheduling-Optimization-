# PSO for Public Transport Scheduling Optimization

A prototype that combines machine learning and **Particle Swarm Optimization (PSO)** to allocate buses across time slots and stops. A Random Forest predicts passenger demand, and PSO searches for a bus allocation that balances unmet demand against wasted capacity.

## Pipeline

1. **Data**: synthetic demand data (time, stop, demand, travel time, capacity) with a low / medium / high demand label.
2. **Demand model**: a scikit-learn pipeline (`StandardScaler` + `RandomForestClassifier`) tuned with `GridSearchCV`. Predicted classes are mapped to demand values of 20, 50 and 90 passengers.
3. **Optimization**: PSO over a 10 time slot × 5 stop grid of bus counts (30 particles, 50 iterations, w = 0.5, c1 = c2 = 1.5).
4. **Cost function**: unmet passengers cost 2 per passenger and unused seats cost 0.5 per seat, with 40 seats per bus.
5. **Dashboards**: early Streamlit and Gradio interface experiments at the end of the notebook.

## Status and limitations

This is a learning project, so read the numbers with care:

- The data is synthetic and tiny (50 rows in the model section), so the 1.0 test accuracy comes from only 10 test samples and says little about real-world performance.
- In the notebook's run, the PSO allocation scored a cost of 1375 against 1000 for the baseline of one bus per slot, so PSO did not beat the baseline there. Tuning the swarm or the cost function is the next thing to try.
- The Streamlit dashboard cells use placeholder values and are not yet connected to the real pipeline.

## Run it

Open `PSO_Public_Transport_Scheduling.ipynb` in Jupyter or Google Colab and run the cells in order.

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

The dashboard sections also need `streamlit`, `plotly` and `gradio`.

## Tech stack

Python, NumPy, pandas, scikit-learn, matplotlib, seaborn, Streamlit, Gradio

## Ideas for next steps

- Tune PSO (inertia, particle count, iterations) and compare against baselines fairly
- Replace synthetic data with real ridership data
- Connect the dashboard to the real prediction and optimization pipeline

## Author

[@vineetsharm55-hub](https://github.com/vineetsharm55-hub)
