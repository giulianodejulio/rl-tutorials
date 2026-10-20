# rl-tutorials

Percorso incrementale dal RL tabellare al controllo continuo. I primi esercizi
restano nei capitoli originali; il materiale successivo e' organizzato per tema.

## Struttura

```text
Chapter2/, Chapter3/, Chapter6/       # esercizi legati ai capitoli del libro
OriginalCodes/                       # codice originale
experiments/
  function_approximation/            # V e Q: feature lineari, quadratiche, one-hot
  continuous_control/                # ambiente 1D, policy manuale, random search
  policy_gradient/                   # REINFORCE, baseline temporale e training
solo12/                             # ambiente MuJoCo e strumenti specifici Solo12
plots/
  function_approximation/            # confronti con DP e TD tabellare
  policy_gradient/                   # rollout, gradienti, training e baseline
results/
  function_approximation/figures/    # immagini generate
  policy_gradient/figures/           # immagini generate
  policy_gradient/data/              # dati NumPy (.npz)
output_paths.py                      # percorsi di output indipendenti dalla cwd
```

I file precedentemente in `Chapter7/` mantengono i nomi, ma sono stati spostati
nelle cartelle tematiche. Le cache Python non sono versionate. Gli esperimenti
`quadratic_value_prediction` e `one_hot_value_prediction` producono anche figure,
sempre sotto `results/function_approximation/figures/`.

## Esecuzione

Dalla radice di questa repository, dentro il container `docker_solo12_rl_ws`,
usare `python3 -m` con il percorso del modulo (senza `.py`), non l'esecuzione
diretta del file. Gli import condivisi vengono cosi' risolti dalla radice.

```bash
python3 -m experiments.function_approximation.linear_value_prediction
python3 -m experiments.continuous_control.continuous_control_rollout
python3 -m experiments.policy_gradient.reinforce_training
python3 -m experiments.policy_gradient.reinforce_training_time_baseline
python3 -m solo12.inspect_solo12
python3 -m solo12.hold_solo12_pose --viewer --duration 20
python3 -m solo12.validate_solo12_env
python3 -m plots.policy_gradient.plot_reinforce_training
python3 -m plots.policy_gradient.plot_reinforce_training_time_baseline
```

Dal sistema host, ad esempio:

```bash
docker exec -w /home/gdj/workspaces/solo12_rl_ws/src/rl-tutorials \
  docker_solo12_rl_ws python3 -m plots.policy_gradient.plot_reinforce_training
```

I plot REINFORCE richiedono LaTeX, gia' configurato nel Dockerfile del workspace.
Gli script di plotting ricreano gli esperimenti e possono impiegare qualche minuto.
Immagini e dati vengono salvati in `results/`, mai accanto agli script.

Il codice specifico del quadrupede e' separato in `solo12/`.
Modello esterno, setup e comandi sono documentati in [solo12/README.md](solo12/README.md).
I relativi risultati restano in `results/solo12/`.

## Ordine Suggerito

1. `experiments/function_approximation/`: `linear_value_prediction`,
   `quadratic_value_prediction`, `one_hot_value_prediction`, `linear_q_learning`.
2. `experiments/continuous_control/`: `continuous_control_rollout`,
   `random_search_policy_learning`.
3. `experiments/policy_gradient/`: `reinforce_continuous_policy`,
   `reinforce_batch_gradient`, `reinforce_training`, `reinforce_time_baseline`,
   `reinforce_training_time_baseline`.

I corrispondenti script `plot_*` in `plots/` visualizzano ciascun passaggio.

## Ricreare il workspace su un altro PC

Dockerfile, bootstrap del modello Solo12 e script di build/avvio portabili sono
ora versionati in [workspace/README.md](workspace/README.md). Non servono gli
script personali del laptop. Il metodo del paper e i suoi risultati restano
separati in [paper/README.md](paper/README.md).
