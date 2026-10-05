# Fantasy Premier League Team Management with Model-Based Reinforcement Learning

A decision-making policy that manages a Fantasy Premier League squad over a season. It decides each week whether to make a transfer so as to maximize points while respecting the budget, squad structure, and club limits.

*Texas A&M University, DAEN/ISEN 489 Reinforcement Learning, Final Project, Spring 2026. Three-person team project.*

## Formulation
- **Finite-horizon Markov Decision Process (MDP).** The state is the 15-player squad, remaining budget, per-player features, and gameweek. The action is one transfer or no transfer per week. The reward is starting-lineup points with captain points doubled.
- **Constraints enforced:** 2 GK / 5 DEF / 5 MID / 3 FWD, at most 3 players per club, budget, and a -4 point penalty for extra transfers.

## Approach: Two-Layer Model-Based RL
1. **Prediction layer.** A learned reward model predicts each player's points from leak-free features: player value, home/away, shifted rolling averages of points, minutes, goals, and assists, and expected metrics.
   - Compared Linear Regression, Random Forest, Gradient Boosting, and an MLP using a chronological 80/20 split.
   - **Random Forest** performed best (validation **MAE 0.647, RMSE 1.614**; 17,695 training and 4,571 validation samples).
2. **Control layer.** A one-step **greedy policy** evaluates top candidate transfers under all constraints, then re-optimizes the starting 11 and the captain each week.

## Data & Evaluation
- Public 2023-24 FPL dataset ([vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League)): 29,725 player-gameweek rows.
- Trained on gameweeks 1 to 30 and evaluated on gameweeks 31 to 38, starting from the same seeded squad.

## Results
| Policy | Total Points (GW 31-38) | Avg. per Gameweek |
|---|---|---|
| No-transfer baseline | 112 | 14.0 |
| **Greedy model-based policy** | **329** | **41.1** |

The model-based policy scored **+217 points (about 2.9x the baseline)** and beat the baseline every week. The biggest weeks were GW 33 (70 vs. 19) and GW 38 (68 vs. 21).

## Future Work
Multi-step lookahead with Q-learning or Monte Carlo tree search, to plan squad upgrades across several gameweeks.

## Tech Stack
Python, pandas, NumPy, scikit-learn (Random Forest, Gradient Boosting, Linear Regression), MLP reward model

> Source code (`policy.py` with the `StudentPolicy` class) will be added.
