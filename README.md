# XPINN Stokes–Darcy

Implémentation PyTorch de réseaux de neurones informés par la physique (XPINN) pour le problème couplé Stokes–Darcy avec conditions d'interface de Beavers–Joseph–Saffman (BJS).

Deux réseaux indépendants (`StokesNet`, `DarcyNet`) sont entraînés simultanément, couplés uniquement via les conditions d'interface (continuité de la vitesse normale, saut de contrainte, condition BJS).

## Cas de test

| Script | Description |
|---|---|
| `xpinn_case1.py` | Interface plate (y=0), solution exacte continue à l'interface |
| `xpinn_case2.py` | Interface plate (y=0), solution exacte discontinue (champs différents côté Stokes / côté Darcy) |
| `xpinn_case3.py` | Interface courbe `y = 0.0625·sin(4πx)`, solution exacte globale |

Chaque script est autonome et peut être lancé indépendamment :

```bash
pip install torch numpy matplotlib
python3 xpinn_case1.py
```

## Sorties

Chaque exécution crée un dossier `outputs_caseN/` contenant :
- `convergence_caseN.png` — courbes de perte (log) et d'erreur L2 relative (linéaire)
- `global_fields_caseN.png` — champs u, v, p prédits/exacts/erreur sur tout le domaine
- `fields_stokes_caseN.png`, `fields_darcy_caseN.png` — champs par sous-domaine
- `final_metrics_caseN.txt` — erreurs L2 relatives finales

## Paramètres physiques

- Viscosité `nu = 0.1`
- Perméabilité `K = 0.01`
- Coefficient BJS `alpha = 1.0`
- Réseaux : profondeur 5, largeur 100, activation Tanh
- Entraînement : 7000 époques Adam + 320 itérations L-BFGS (par réseau)

## Remarques

Le calcul est effectué en double précision (`float64`) avec dérivées d'ordre 2 par différentiation automatique imbriquée — coûteux en temps de calcul. Pour un test rapide, réduire `Nfs`, `Nfd`, `epochs_adam` et `epochs_lbfgs` dans la `Config` de chaque script.
