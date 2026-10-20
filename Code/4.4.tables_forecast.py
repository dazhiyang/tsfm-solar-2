"""Build LaTeX tables of hold-out GHI forecast verification (MBE, RMSE, skill).

Reads {STATION}_{pers,chronos,chronos_ft_full,timesfm,timesfm_ft_full}_{L}.txt under Data/result.
Missing exports leave blank cells (e.g. full-pass TimesFM-BSRN at L15).
Updates three embedded tables in tex/manuscript.tex, preserving all text outside
their labelled markers. No separate table files or directories are created.

  python -B Code/4.4.tables_forecast.py
  python -B Code/4.4.tables_forecast.py --self-test
  python -B Code/4.4.tables_forecast.py --require-complete --stdout

--stdout previews the updated source without modifying the manuscript.
--require-complete refuses missing model/station/context combinations.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np

FORECAST_LENGTH = 15
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_DIR = PROJECT_ROOT / "Data" / "result"
DEFAULT_MANUSCRIPT = PROJECT_ROOT / "tex" / "manuscript.tex"

METHODS = (
    ("pers", "Persistence"),
    ("chronos", "Chronos-2"),
    ("chronos_ft_full", "Chronos-2-BSRN"),
    ("timesfm", "TimesFM"),
    ("timesfm_ft_full", "TimesFM-BSRN"),
)
OPTIONAL_PREFIXES = frozenset({"chronos_ft_full", "timesfm", "timesfm_ft_full"})


def finetune_index_module():
    path = Path(__file__).with_name("1.1.prepare_finetuning_input.py")
    spec = importlib.util.spec_from_file_location("finetune_index", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def input_module():
    path = Path(__file__).with_name("1.2.prepare_forecast_input.py")
    spec = importlib.util.spec_from_file_location("forecast_input", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_ghi_pairs(path: Path, *, return_times: bool = False):
    """Return flattened GHI pairs; optionally include their per-origin timestamps."""
    with path.open(encoding="utf-8") as stream:
        header = stream.readline().rstrip().split("\t")
    obs_idx = [header.index(f"Obs{j}") for j in range(1, FORECAST_LENGTH + 1)]
    fcst_idx = [header.index(f"Fcst{j}") for j in range(1, FORECAST_LENGTH + 1)]
    table = np.loadtxt(path, delimiter="\t", skiprows=1, dtype=np.float64, ndmin=2)
    if table.shape[0] == 0 or table.shape[1] != 31 or len(header) != 31:
        raise ValueError(f"Expected a nonempty 31-column forecast export: {path}")
    times = table[:, header.index("Time")]
    if (not np.isfinite(times).all() or not np.equal(times, np.rint(times)).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError(f"Invalid or unordered origin timestamps in {path}")
    obs = table[:, obs_idx].ravel()
    fcst = table[:, fcst_idx].ravel()
    if not (np.isfinite(obs).all() and np.isfinite(fcst).all()):
        raise ValueError(f"Non-finite GHI in {path}")
    return (obs, fcst, times) if return_times else (obs, fcst)


def ghi_metrics(obs: np.ndarray, fcst: np.ndarray, pers_fcst: np.ndarray) -> dict[str, float]:
    err = fcst - obs
    return metrics_from_sums(
        obs.size, float(obs.sum()), float(err.sum()), float(np.square(err).sum()),
        float(np.square(pers_fcst - obs).sum()),
    )


def metrics_from_sums(n, sum_obs, sum_error, sum_squared_error, sum_pers_squared_error):
    """Compute sample-pooled metrics without averaging station RMSEs or skills."""
    mbe = sum_error / n
    mse_fcst = sum_squared_error / n
    rmse = float(np.sqrt(mse_fcst))
    mean_obs = sum_obs / n
    mse_pers = sum_pers_squared_error / n
    skill = 1.0 - rmse / float(np.sqrt(mse_pers)) if mse_pers > 0 else float("nan")
    return {
        "n": n,
        "sum_obs": sum_obs,
        "sum_error": sum_error,
        "sum_squared_error": sum_squared_error,
        "sum_pers_squared_error": sum_pers_squared_error,
        "mbe": mbe,
        "rmse": rmse,
        "skill": skill,
        "mbe_pct": 100.0 * mbe / mean_obs if mean_obs > 0 else float("nan"),
        "rmse_pct": 100.0 * rmse / mean_obs if mean_obs > 0 else float("nan"),
    }


def fmt_number(value: float) -> str:
    """One decimal place, retaining the sign of small negative biases."""
    return f"{value:.1f}"


def fmt_w_m2_pct(value: float, pct: float) -> str:
    return f"${fmt_number(value)}$ $({fmt_number(pct)}\\,\\unit{{\\percent}})$"


def fmt_skill(skill: float) -> str:
    if not np.isfinite(skill):
        return "\\textemdash"
    return f"${fmt_number(100.0 * skill)}\\,\\unit{{\\percent}}$"


def collect_station_metrics(
    result_dir: Path,
    station: str,
    history_length: int,
) -> dict[str, dict[str, float]]:
    pers_path = result_dir / f"{station}_pers_{history_length}.txt"
    if not pers_path.is_file():
        raise FileNotFoundError(pers_path)
    obs, pers_fcst, times = load_ghi_pairs(pers_path, return_times=True)
    out: dict[str, dict[str, float]] = {}
    for prefix, _label in METHODS:
        path = result_dir / f"{station}_{prefix}_{history_length}.txt"
        if not path.is_file():
            if prefix in OPTIONAL_PREFIXES:
                out[prefix] = None
                continue
            raise FileNotFoundError(path)
        method_obs, fcst, method_times = load_ghi_pairs(path, return_times=True)
        if (fcst.shape != obs.shape or not np.array_equal(method_obs, obs)
                or not np.array_equal(method_times, times)):
            raise ValueError(f"Observation alignment mismatch for {path}")
        out[prefix] = ghi_metrics(obs, fcst, pers_fcst)
    return out


def build_grid(
    result_dir: Path,
    stations: tuple[str, ...],
    history_lengths: tuple[int, ...],
) -> dict[int, dict[str, dict[str, dict[str, float]]]]:
    """grid[L][station][method_prefix] -> metrics dict."""
    grid: dict[int, dict[str, dict[str, dict[str, float]]]] = {}
    for length in history_lengths:
        grid[length] = {}
        for station in stations:
            grid[length][station] = collect_station_metrics(result_dir, station, length)
        grid[length]["All"] = {}
        for prefix, _ in METHODS:
            station_values = [grid[length][station][prefix] for station in stations]
            if any(values is None for values in station_values):
                if not all(values is None for values in station_values):
                    raise ValueError(f"Incomplete station set for {prefix}, L={length}.")
                grid[length]["All"][prefix] = None
                continue
            keys = ("n", "sum_obs", "sum_error", "sum_squared_error", "sum_pers_squared_error")
            totals = [sum(values[key] for values in station_values) for key in keys]
            grid[length]["All"][prefix] = metrics_from_sums(*totals)
    return grid


def cell_for_metric(
    metric: str,
    prefix: str,
    values: dict[str, float],
) -> str:
    if values is None:
        return ""
    if metric == "mbe":
        return fmt_w_m2_pct(values["mbe"], values["mbe_pct"])
    if metric == "rmse":
        return fmt_w_m2_pct(values["rmse"], values["rmse_pct"])
    if metric == "skill":
        return "\\textemdash" if prefix == "pers" else fmt_skill(values["skill"])
    raise ValueError(metric)


def render_metric_table(
    metric: str,
    caption: str,
    label: str,
    stations: tuple[str, ...],
    history_lengths: tuple[int, ...],
    grid: dict[int, dict[str, dict[str, dict[str, float]]]],
) -> str:
    columns = (*stations, "All")
    col_spec = "@{\\extracolsep{\\fill}}ll" + "c" * len(columns) + "@{}"
    header = " & ".join(["$L$", "Method"] + list(columns))
    rows: list[str] = []
    methods = tuple(item for item in METHODS if metric != "skill" or item[0] != "pers")
    n_methods = len(methods)
    float_env = "table*"
    column_padding = 6
    for index, length in enumerate(history_lengths):
        if index > 0:
            rows.append("\\hline")
        for method_index, (prefix, label_method) in enumerate(methods):
            cells = [
                cell_for_metric(metric, prefix, grid[length][station][prefix])
                for station in columns
            ]
            horizon_cell = (
                f"\\multirow{{{n_methods}}}{{*}}{{{length}}}"
                if method_index == 0
                else ""
            )
            rows.append(
                f"  {horizon_cell} & {label_method} & "
                + " & ".join(cells)
                + " \\\\"
            )

    body = "\n".join(rows)
    return f"""\\begin{{{float_env}}}[!ht]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\setlength{{\\tabcolsep}}{{{column_padding}pt}}
\\footnotesize
\\renewcommand{{\\arraystretch}}{{1.5}}
\\begin{{tabular*}}{{\\textwidth}}{{{col_spec}}}
\\hline\\hline
{header} \\\\
\\hline
{body}
\\hline
\\end{{tabular*}}
\\end{{{float_env}}}
"""


def replace_table(source: str, metric: str, table: str) -> str:
    """Replace one marked table without altering the surrounding manuscript."""
    start = f"% BEGIN GENERATED TABLE: {metric}"
    end = f"% END GENERATED TABLE: {metric}"
    if source.count(start) != 1 or source.count(end) != 1:
        raise ValueError(f"Expected exactly one marker pair for {metric}.")
    before, remainder = source.split(start)
    _, after = remainder.split(end)
    if end in before:
        raise ValueError(f"Reversed markers for {metric}.")
    return before + start + "\n" + table.rstrip() + "\n" + end + after


def render_tables(result_dir: Path, manuscript: Path, stations: tuple[str, ...],
                  require_complete: bool = False) -> str:
    original = manuscript.read_text(encoding="utf-8")
    for metric in ("mbe", "rmse", "skill"):
        replace_table(original, metric, "")  # Validate markers before computing metrics.
    history_lengths = tuple(input_module().HISTORY_LENGTHS)
    grid = build_grid(result_dir, stations, history_lengths)
    if require_complete:
        for length in history_lengths:
            for station in stations:
                for prefix, _ in METHODS:
                    if grid[length][station][prefix] is None:
                        raise ValueError(f"Missing forecast: {station}, {prefix}, L={length}")

    specs = (
        (
            "mbe",
            "tab:forecast-ghi-mbe",
            "MBEs of GHI forecasts at five hold-out stations. "
            "Values are in \\unit{\\watt\\per\\metre\\squared}; percentage values are shown in parentheses "
            "normalized by mean observed GHI. The ``All'' column pools all forecast--observation pairs "
            "across stations and forecast horizons.",
        ),
        (
            "rmse",
            "tab:forecast-ghi-rmse",
            "Same as Table~\\ref{tab:forecast-ghi-mbe}, but for RMSE.",
        ),
        (
            "skill",
            "tab:forecast-ghi-skill",
            "Same as Table~\\ref{tab:forecast-ghi-mbe}, but for forecast skill (\\unit{\\percent}).",
        ),
    )
    updated = original
    for metric, label, caption in specs:
        text = render_metric_table(metric, caption, label, stations, history_lengths, grid)
        updated = replace_table(updated, metric, text)
    if manuscript.read_text(encoding="utf-8") != original:
        raise RuntimeError("Manuscript changed during computation; no update written.")
    return updated


def write_tables(result_dir: Path, manuscript: Path, stations: tuple[str, ...],
                 require_complete: bool = False) -> Path:
    updated = render_tables(result_dir, manuscript, stations, require_complete)
    manuscript.write_text(updated, encoding="utf-8")
    return manuscript


def self_test() -> None:
    assert fmt_w_m2_pct(-0.1, -0.01) == "$-0.1$ $(-0.0\\,\\unit{\\percent})$"
    assert fmt_skill(-0.013) == "$-1.3\\,\\unit{\\percent}$"
    rng = np.random.default_rng(0)
    n = 40
    obs = rng.uniform(100, 500, size=(n, FORECAST_LENGTH))
    pers = obs + rng.normal(0, 20, size=obs.shape)
    fcst = obs + rng.normal(0, 15, size=obs.shape)
    flat_obs = obs.ravel()
    m = ghi_metrics(flat_obs, fcst.ravel(), pers.ravel())
    assert m["rmse"] < ghi_metrics(flat_obs, pers.ravel(), pers.ravel())["rmse"]
    assert -100 < m["mbe_pct"] < 100
    assert 0 < m["skill"] < 1
    # Unequal counts and error scales distinguish pooling from averaging RMSEs.
    obs_a, obs_b = np.array([100.0]), np.array([200.0, 300.0, 400.0])
    pred_a, pred_b = obs_a + 10, obs_b + 2
    pers_a, pers_b = obs_a + 20, obs_b + 4
    station_metrics = [ghi_metrics(obs_a, pred_a, pers_a), ghi_metrics(obs_b, pred_b, pers_b)]
    keys = ("n", "sum_obs", "sum_error", "sum_squared_error", "sum_pers_squared_error")
    pooled = metrics_from_sums(*(sum(item[key] for item in station_metrics) for key in keys))
    direct = ghi_metrics(np.r_[obs_a, obs_b], np.r_[pred_a, pred_b], np.r_[pers_a, pers_b])
    for key in direct:
        np.testing.assert_allclose(pooled[key], direct[key])
    np.testing.assert_allclose(pooled["skill"], 0.5)
    assert ghi_metrics(obs_a, pers_a, pers_a)["skill"] == 0.0
    assert np.isnan(ghi_metrics(obs_a, pred_a, obs_a)["skill"])
    assert not np.isclose(pooled["rmse"], np.mean([item["rmse"] for item in station_metrics]))
    grid = {15: {station: {prefix: m for prefix, _ in METHODS} for station in ("ASP", "All")}}
    for metric in ("mbe", "rmse", "skill"):
        table = render_metric_table(metric, "Caption", "tab:test", ("ASP",), (15,), grid)
        env = "table*"
        assert f"\\begin{{{env}}}[!ht]" in table
        assert "ASP & All" in table
        assert "\\caption{Caption}" in table
        assert "\\hline\\hline" in table
        assert not any(rule in table for rule in ("\\toprule", "\\midrule", "\\bottomrule"))
        assert f"\\multirow{{{len(METHODS) - (metric == 'skill')}}}" in table
        assert "\\begin{tabular*}{\\textwidth}" in table
        if metric == "skill":
            assert " & Persistence & " not in table
        source = f"Before\n% BEGIN GENERATED TABLE: {metric}\nold\n% END GENERATED TABLE: {metric}\nAfter"
        updated = replace_table(source, metric, table)
        assert updated.startswith("Before\n") and updated.endswith("\nAfter")
        assert table.rstrip() in updated and "\nold\n" not in updated
    assert cell_for_metric("rmse", "chronos_ft_full", None) == ""
    assert cell_for_metric("rmse", "timesfm_ft_full", None) == ""
    grid[15]["ASP"]["chronos_ft_full"] = None
    grid[15]["All"]["chronos_ft_full"] = None
    table = render_metric_table("rmse", "Caption", "tab:test", ("ASP",), (15,), grid)
    assert " & Chronos-2-BSRN &  &  \\\\" in table
    grid[15]["ASP"]["timesfm"] = m
    grid[15]["All"]["timesfm"] = m
    table = render_metric_table("rmse", "Caption", "tab:test", ("ASP",), (15,), grid)
    assert " & TimesFM & " in table
    grid[60] = {
        station: {prefix: (None if prefix == "timesfm" else m) for prefix, _ in METHODS}
        for station in ("ASP", "All")
    }
    table = render_metric_table("rmse", "Caption", "tab:test", ("ASP",), (60,), grid)
    assert " & TimesFM &  &  \\\\" in table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--manuscript-file", type=Path, default=DEFAULT_MANUSCRIPT)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--stdout", action="store_true", help="Preview source without saving")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("self-test ok")
        return
    stations = finetune_index_module().KOPPEN_HOLDOUT_STATIONS
    if args.stdout:
        print(render_tables(args.result_dir, args.manuscript_file, stations, args.require_complete), end="")
        return
    path = write_tables(args.result_dir, args.manuscript_file, stations, args.require_complete)
    print(f"Updated embedded tables in {path}")


if __name__ == "__main__":
    main()
