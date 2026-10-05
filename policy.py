import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor


SQUAD_COMPOSITION = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
LINEUP_SIZE = 11
MIN_FORMATION = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
MAX_PER_CLUB = 3
TRANSFER_PENALTY = 4

POSITION_MAP = {
    "GKP": "GK",
    "GK": "GK",
    "DEF": "DEF",
    "MID": "MID",
    "FWD": "FWD",
}

FEATURE_COLS = [
    "value",
    "was_home",
    "avg_points_3",
    "avg_minutes_3",
    "avg_goals_3",
    "avg_assists_3",
    "expected_goals",
    "expected_assists",
    "xP",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build lag / rolling features without leakage.
    All rolling features use shift(1), so week t only uses info up to t-1.
    """
    df = df.copy()

    numeric = [
        "total_points",
        "minutes",
        "goals_scored",
        "assists",
        "value",
        "was_home",
        "expected_goals",
        "expected_assists",
        "xP",
        "GW",
    ]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "position" in df.columns:
        df["position"] = (
            df["position"]
            .astype(str)
            .str.upper()
            .str.strip()
            .map(lambda x: POSITION_MAP.get(x, "MID"))
        )

    player_col = "element" if "element" in df.columns else "name"

    df = df.sort_values([player_col, "GW"]).reset_index(drop=True)
    grp = df.groupby(player_col, sort=False)

    df["avg_points_3"] = grp["total_points"].shift(1).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    df["avg_minutes_3"] = grp["minutes"].shift(1).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    df["avg_goals_3"] = grp["goals_scored"].shift(1).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    df["avg_assists_3"] = grp["assists"].shift(1).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)

    return df


def _rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def compare_models(X_train, X_val, y_train, y_val):
    """
    Train LR, RF, GB; print validation comparison; return best model by MAE.
    """
    results = {}

    lr = LinearRegression().fit(X_train, y_train)
    lr_pred = lr.predict(X_val)
    results["LinearRegression"] = {
        "mae": mean_absolute_error(y_val, lr_pred),
        "rmse": _rmse(y_val, lr_pred),
        "model": lr,
    }

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    ).fit(X_train, y_train)
    rf_pred = rf.predict(X_val)
    results["RandomForest"] = {
        "mae": mean_absolute_error(y_val, rf_pred),
        "rmse": _rmse(y_val, rf_pred),
        "model": rf,
    }

    gb = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    ).fit(X_train, y_train)
    gb_pred = gb.predict(X_val)
    results["GradientBoosting"] = {
        "mae": mean_absolute_error(y_val, gb_pred),
        "rmse": _rmse(y_val, gb_pred),
        "model": gb,
    }

    sample_w = np.where(y_train > np.median(y_train), 1.35, 1.0)
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=300,
        random_state=42,
    ).fit(X_train, y_train, sample_weight=sample_w)
    mlp_pred = mlp.predict(X_val)
    results["ICQValueNet"] = {
        "mae": mean_absolute_error(y_val, mlp_pred),
        "rmse": _rmse(y_val, mlp_pred),
        "model": mlp,
    }

    print("\n--- Model Comparison (chronological validation) ---")
    for name, res in results.items():
        print(f"  {name:20s}  MAE={res['mae']:.4f}  RMSE={res['rmse']:.4f}")
    print("---------------------------------------------------")

    best_name = min(results, key=lambda k: results[k]["mae"])
    print(f"  Best model selected: {best_name}\n")
    return results[best_name]["model"]


class StudentPolicy:
    """
    Greedy model-based FPL policy:
      1. Prediction: estimate expected player reward from observable features
      2. Control: choose the best single transfer that improves predicted squad value
                 subject to position, club, and budget constraints
    """

    def __init__(self):
        self.model = None
        self.train_data = None
        self.player_col = None
        self._free_xfer = True

    def fit(self, train_data: pd.DataFrame):
        """
        Build features and train the reward model using a chronological split.
        """
        print("[fit] Engineering features ...")
        df = engineer_features(train_data)

        self.player_col = "element" if "element" in df.columns else "name"

        model_df = df.dropna(subset=FEATURE_COLS + ["total_points", "GW"]).copy()
        model_df = model_df.sort_values("GW").reset_index(drop=True)

        unique_gws = sorted(model_df["GW"].dropna().unique())
        if len(unique_gws) < 4:
            raise ValueError("Not enough gameweeks for chronological validation split.")

        split_idx = max(1, int(0.8 * len(unique_gws)))
        split_gw = unique_gws[split_idx - 1]

        train_part = model_df[model_df["GW"] <= split_gw].copy()
        val_part = model_df[model_df["GW"] > split_gw].copy()

        if len(val_part) == 0:
            train_part = model_df.iloc[:-max(1, len(model_df) // 5)].copy()
            val_part = model_df.iloc[-max(1, len(model_df) // 5):].copy()

        X_train = train_part[FEATURE_COLS].values.astype(float)
        y_train = train_part["total_points"].values.astype(float)

        X_val = val_part[FEATURE_COLS].values.astype(float)
        y_val = val_part["total_points"].values.astype(float)

        print("[fit] Comparing models ...")
        self.model = compare_models(X_train, X_val, y_train, y_val)

        self.train_data = model_df

        print(
            f"[fit] Done. Trained on {len(train_part):,} samples, "
            f"validated on {len(val_part):,} samples across "
            f"{model_df['GW'].nunique()} train gameweeks.\n"
        )

    def reset(self):
        self._free_xfer = True

    def predict_reward(self, player_dict: dict) -> float:
        if self.model is None:
            raise ValueError("Model has not been fit yet.")
        feats = np.array([float(player_dict.get(c, 0.0)) for c in FEATURE_COLS], dtype=float)
        feats = np.nan_to_num(feats, nan=0.0)
        return float(self.model.predict(feats.reshape(1, -1))[0])

    @staticmethod
    def _safe_float(x, default=0.0):
        try:
            val = float(x)
            if np.isnan(val):
                return default
            return val
        except Exception:
            return default

    @staticmethod
    def _safe_int(x, default=0):
        try:
            val = int(float(x))
            return val
        except Exception:
            return default

    @staticmethod
    def _pos_from_info(info: dict) -> str:
        raw = str(info.get("position", "MID")).upper().strip()
        return POSITION_MAP.get(raw, "MID")

    def _pos(self, pid, state) -> str:
        info = state.get("players", {}).get(pid, {})
        return self._pos_from_info(info)

    def _price(self, pid, state) -> int:
        info = state.get("players", {}).get(pid, {})
        return self._safe_int(info.get("value", 50), 50)

    def _club(self, pid, state) -> str:
        info = state.get("players", {}).get(pid, {})
        return str(info.get("team", "unknown"))

    def _pos_counts(self, squad, state) -> dict:
        counts = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
        for pid in squad:
            counts[self._pos(pid, state)] += 1
        return counts

    def _club_counts(self, squad, state) -> dict:
        counts = {}
        for pid in squad:
            c = self._club(pid, state)
            counts[c] = counts.get(c, 0) + 1
        return counts

    def _can_add(self, new_pid, squad_minus_out, state) -> bool:
        pos = self._pos(new_pid, state)

        pos_counts = self._pos_counts(squad_minus_out, state)
        if pos_counts.get(pos, 0) >= SQUAD_COMPOSITION[pos]:
            return False

        club = self._club(new_pid, state)
        club_counts = self._club_counts(squad_minus_out, state)
        if club_counts.get(club, 0) >= MAX_PER_CLUB:
            return False

        return True

    def _get_features_from_state_or_history(self, player_id, gw: int, state: dict):
        """
        Prefer current state's player feature row if available.
        Otherwise fall back to processed training history:
          - exact GW row if present
          - latest row before GW
          - latest known row
        """
        player_info = state.get("players", {}).get(player_id, {})

        if all(col in player_info for col in FEATURE_COLS):
            feats = np.array(
                [self._safe_float(player_info.get(col, 0.0), 0.0) for col in FEATURE_COLS],
                dtype=float,
            )
            return np.nan_to_num(feats, nan=0.0)

        if self.train_data is None:
            return None

        rows = self.train_data[self.train_data[self.player_col] == player_id]
        if rows.empty:
            return None

        exact = rows[rows["GW"] == gw]
        if not exact.empty:
            row = exact.iloc[-1]
        else:
            prev = rows[rows["GW"] < gw]
            row = prev.iloc[-1] if not prev.empty else rows.iloc[-1]

        feats = np.array([self._safe_float(row.get(c, 0.0), 0.0) for c in FEATURE_COLS], dtype=float)
        return np.nan_to_num(feats, nan=0.0)

    def _score(self, player_id, gw: int, state: dict) -> float:
        feats = self._get_features_from_state_or_history(player_id, gw, state)
        if feats is None:
            return 0.0
        return float(self.model.predict(feats.reshape(1, -1))[0])

    def select_lineup_and_captain(self, squad, state):
        """
        Select best valid 11 and captain based on predicted scores.
        Enforces:
          - 1 GK
          - at least 3 DEF
          - at least 2 MID
          - at least 1 FWD
        """
        gw = state.get("gw", state.get("gameweek", 1))
        scores = {pid: self._score(pid, gw, state) for pid in squad}
        ranked = sorted(squad, key=lambda p: scores[p], reverse=True)

        lineup = []
        pos_count = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}

        for pos, req in MIN_FORMATION.items():
            filled = 0
            for pid in ranked:
                if filled >= req:
                    break
                if pid in lineup:
                    continue
                if self._pos(pid, state) == pos:
                    if pos == "GK" and pos_count["GK"] >= 1:
                        continue
                    lineup.append(pid)
                    pos_count[pos] += 1
                    filled += 1

        for pid in ranked:
            if len(lineup) >= LINEUP_SIZE:
                break
            if pid in lineup:
                continue
            pos = self._pos(pid, state)
            if pos == "GK" and pos_count["GK"] >= 1:
                continue
            lineup.append(pid)
            pos_count[pos] += 1

        captain = max(lineup, key=lambda p: scores[p]) if lineup else None
        return lineup, captain

    def act(self, state: dict) -> list:
        """
        Greedy single-transfer decision.
        """
        if self.model is None:
            return []

        self._free_xfer = True

        squad = list(state.get("squad", []))
        budget = self._safe_int(state.get("budget", 0), 0)
        gw = state.get("gw", state.get("gameweek", 1))
        players = state.get("players", {})

        if not squad or not players:
            return []

        pool = [p for p in state.get("available_players", list(players.keys())) if p not in squad]

        squad_scores = {pid: self._score(pid, gw, state) for pid in squad}

        capped_pool = pool[:2000]
        pool_scores = {pid: self._score(pid, gw, state) for pid in capped_pool}
        sorted_pool = sorted(pool_scores, key=pool_scores.get, reverse=True)

        best_gain = 0.0
        best_transfer = None

        for out_pid in squad:
            out_score = squad_scores[out_pid]
            out_price = self._price(out_pid, state)
            available_funds = budget + out_price
            out_pos = self._pos(out_pid, state)
            squad_minus = [p for p in squad if p != out_pid]

            for in_pid in sorted_pool[:300]:
                if self._pos(in_pid, state) != out_pos:
                    continue
                if self._price(in_pid, state) > available_funds:
                    continue
                if not self._can_add(in_pid, squad_minus, state):
                    continue

                penalty = 0 if self._free_xfer else TRANSFER_PENALTY
                effective_gain = pool_scores[in_pid] - out_score - penalty

                if effective_gain > best_gain:
                    best_gain = effective_gain
                    best_transfer = (out_pid, in_pid)

        transfers = []
        if best_transfer is not None:
            transfers.append(best_transfer)
            self._free_xfer = False

        return transfers


def _build_player_meta(df: pd.DataFrame, gw: int = None) -> dict:
    """
    Build player metadata dictionary.
    If gw is given, prefer that GW's row for each player.
    Otherwise use the latest row available.
    """
    tmp = df.copy()
    player_col = "element" if "element" in tmp.columns else "name"

    if "position" in tmp.columns:
        tmp["position"] = (
            tmp["position"]
            .astype(str)
            .str.upper()
            .str.strip()
            .map(lambda x: POSITION_MAP.get(x, "MID"))
        )

    if gw is not None and "GW" in tmp.columns:
        tmp = tmp[tmp["GW"] <= gw].copy()

    tmp = tmp.sort_values([player_col, "GW"]).groupby(player_col, as_index=False).tail(1)

    player_meta = {}
    for _, row in tmp.iterrows():
        pid = row[player_col]
        player_meta[pid] = {
            "position": POSITION_MAP.get(str(row.get("position", "MID")).upper().strip(), "MID"),
            "value": int(pd.to_numeric(row.get("value", 50), errors="coerce") or 50),
            "team": str(row.get("team", "unknown")),
        }

        for col in FEATURE_COLS:
            if col in row.index:
                try:
                    player_meta[pid][col] = float(pd.to_numeric(row[col], errors="coerce"))
                except Exception:
                    player_meta[pid][col] = 0.0

    return player_meta


def build_random_squad(player_meta: dict, budget: int = 1000, seed: int = 42) -> tuple[list, int]:
    """
    Assemble a random valid 15-player squad respecting positional quotas, club limits, and budget.
    Assumes player values are in tenths (e.g. 50 == 5.0).
    Returns (squad, remaining_budget).
    """
    rng = np.random.default_rng(seed)

    pids = list(player_meta.keys())
    rng.shuffle(pids)

    squad = []
    spent = 0
    pos_counts = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
    club_counts = {}

    for pid in pids:
        info = player_meta[pid]
        pos = POSITION_MAP.get(str(info.get("position", "MID")).upper().strip(), "MID")
        price = int(info.get("value", 50))
        club = str(info.get("team", "unknown"))

        if pos_counts[pos] >= SQUAD_COMPOSITION[pos]:
            continue
        if club_counts.get(club, 0) >= MAX_PER_CLUB:
            continue
        if spent + price > budget:
            continue

        squad.append(pid)
        spent += price
        pos_counts[pos] += 1
        club_counts[club] = club_counts.get(club, 0) + 1

        if len(squad) == 15:
            break

    if len(squad) != 15:
        raise ValueError("Could not build a valid random squad under budget and constraints.")

    return squad, budget - spent


def _actual_points_map(df: pd.DataFrame, gw: int) -> dict:
    player_col = "element" if "element" in df.columns else "name"
    gw_df = df[df["GW"] == gw].copy()

    if gw_df.empty:
        return {}

    gw_df["total_points"] = pd.to_numeric(gw_df["total_points"], errors="coerce").fillna(0.0)
    return gw_df.set_index(player_col)["total_points"].to_dict()


def _score_lineup_actual(df: pd.DataFrame, lineup: list, captain, gw: int) -> float:
    """
    Score only the selected 11. Captain gets double points.
    """
    pts_map = _actual_points_map(df, gw)
    total = 0.0

    for pid in lineup:
        pts = float(pts_map.get(pid, 0.0))
        total += pts
        if captain == pid:
            total += pts

    return total


def _apply_transfers(squad: list, budget: int, transfers: list, state: dict) -> tuple[list, int]:
    """
    Apply transfers and update remaining budget.
    """
    new_squad = list(squad)
    new_budget = int(budget)

    for out_pid, in_pid in transfers:
        if out_pid not in new_squad:
            continue

        out_price = int(state["players"][out_pid]["value"])
        in_price = int(state["players"][in_pid]["value"])

        tentative_budget = new_budget + out_price - in_price
        if tentative_budget < 0:
            continue

        new_squad[new_squad.index(out_pid)] = in_pid
        new_budget = tentative_budget

    return new_squad, new_budget


def run_no_transfer_baseline(df: pd.DataFrame, train_gws: int = 30, seed: int = 42) -> float:
    """
    Hold initial squad fixed. Each week choose the best predicted lineup/captain from the same squad.
    """
    initial_meta = _build_player_meta(df, gw=train_gws)
    squad, budget = build_random_squad(initial_meta, budget=1000, seed=seed)

    dummy_policy = StudentPolicy()
    dummy_policy.model = None

    total_pts = 0.0
    print(f"[baseline] Squad size: {len(squad)} | Remaining budget: {budget}\n")

    for gw in range(train_gws + 1, int(df["GW"].max()) + 1):
        gw_meta = _build_player_meta(df, gw=gw)

        state = {
            "squad": squad,
            "budget": budget,
            "gw": gw,
            "players": gw_meta,
            "available_players": list(gw_meta.keys()),
        }

        def baseline_rank(pid):
            info = gw_meta.get(pid, {})
            return float(info.get("xP", info.get("value", 0)))

        ranked = sorted(squad, key=baseline_rank, reverse=True)

        lineup = []
        pos_count = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
        for pos, req in MIN_FORMATION.items():
            filled = 0
            for pid in ranked:
                if filled >= req:
                    break
                if pid in lineup:
                    continue
                p = POSITION_MAP.get(str(gw_meta[pid].get("position", "MID")).upper().strip(), "MID")
                if p == pos:
                    if pos == "GK" and pos_count["GK"] >= 1:
                        continue
                    lineup.append(pid)
                    pos_count[pos] += 1
                    filled += 1

        for pid in ranked:
            if len(lineup) >= LINEUP_SIZE:
                break
            if pid in lineup:
                continue
            p = POSITION_MAP.get(str(gw_meta[pid].get("position", "MID")).upper().strip(), "MID")
            if p == "GK" and pos_count["GK"] >= 1:
                continue
            lineup.append(pid)
            pos_count[p] += 1

        captain = lineup[0] if lineup else None

        week_pts = _score_lineup_actual(df, lineup, captain, gw)
        total_pts += week_pts

        print(
            f"  GW {gw:2d}  no transfer"
            f"{'':32s}  week={week_pts:6.1f}  total={total_pts:7.1f}"
        )

    return total_pts


def run_simulation(policy: StudentPolicy, df: pd.DataFrame, train_gws: int = 30, seed: int = 42) -> float:
    """
    Simulate policy on held-out gameweeks.
    Fixes:
      * proper budget updates
      * choose lineup + captain
      * score only lineup 11 with captain doubled
    """
    initial_meta = _build_player_meta(df, gw=train_gws)
    squad, budget = build_random_squad(initial_meta, budget=1000, seed=seed)

    print(f"[policy] Starting squad size: {len(squad)} | Remaining budget: {budget}\n")
    policy.reset()
    total_pts = 0.0

    for gw in range(train_gws + 1, int(df["GW"].max()) + 1):
        gw_meta = _build_player_meta(df, gw=gw)

        state = {
            "squad": squad,
            "budget": budget,
            "gw": gw,
            "players": gw_meta,
            "available_players": list(gw_meta.keys()),
        }

        transfers = policy.act(state)
        squad, budget = _apply_transfers(squad, budget, transfers, state)

        post_state = {
            "squad": squad,
            "budget": budget,
            "gw": gw,
            "players": gw_meta,
            "available_players": list(gw_meta.keys()),
        }

        lineup, captain = policy.select_lineup_and_captain(squad, post_state)
        week_pts = _score_lineup_actual(df, lineup, captain, gw)
        total_pts += week_pts

        xfer_str = str(transfers) if transfers else "no transfer"
        print(
            f"  GW {gw:2d}  {xfer_str:<40s}  "
            f"budget={budget:4d}  week={week_pts:6.1f}  total={total_pts:7.1f}"
        )

    return total_pts


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    URL = (
        "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League"
        "/master/data/2023-24/gws/merged_gw.csv"
    )

    print("=" * 60)
    print("DAEN/ISEN 489 — FPL Policy Evaluation")
    print("=" * 60)

    print("\nLoading data ...")
    df = pd.read_csv(URL)
    df["GW"] = pd.to_numeric(df["GW"], errors="coerce")
    print(f"Loaded {len(df):,} rows, GW range: {int(df['GW'].min())}–{int(df['GW'].max())}")

    TRAIN_GWS = 30
    train_df = df[df["GW"] <= TRAIN_GWS].copy()
    print(f"Training on GW 1–{TRAIN_GWS}, evaluating on GW {TRAIN_GWS+1}–{int(df['GW'].max())}\n")

    print("=" * 60)
    print("BASELINE — No-Transfer Policy")
    print("=" * 60)
    baseline_pts = run_no_transfer_baseline(df, train_gws=TRAIN_GWS, seed=42)

    print()
    print("=" * 60)
    print("POLICY — Greedy Model-Based Transfer Policy")
    print("=" * 60)
    policy = StudentPolicy()
    policy.fit(train_df)

    print("Running simulation ...\n")
    policy_pts = run_simulation(policy, df, train_gws=TRAIN_GWS, seed=42)

    lift = policy_pts - baseline_pts
    gws = int(df["GW"].max()) - TRAIN_GWS

    print(f"\n{'=' * 60}")
    print(f"RESULTS SUMMARY  (GW {TRAIN_GWS+1}–{int(df['GW'].max())}, {gws} gameweeks)")
    print(f"{'=' * 60}")
    print(f"  No-transfer baseline : {baseline_pts:7.1f} pts  ({baseline_pts/gws:.1f} pts/GW avg)")
    print(f"  Greedy policy        : {policy_pts:7.1f} pts  ({policy_pts/gws:.1f} pts/GW avg)")
    print(f"  Lift over baseline   : {lift:+.1f} pts  ({lift/gws:+.1f} pts/GW)")
    print(f"{'=' * 60}")
