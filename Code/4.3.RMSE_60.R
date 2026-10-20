# Five Köppen hold-out stations: clear-sky index RMSE by forecast horizon (L = 60 context).
# Two panels (Chronos-2, TimesFM); color = station, linetype = method.
# Rscript Code/4.3.RMSE_60.R
suppressPackageStartupMessages({
  library(ggplot2)
  library(readr)
  library(dplyr)
})

# --- Editable publication parameters ---
text_size_pt <- 8
font_family <- "Times"
width_mm <- 85
height_mm <- 124
line_width_data <- 0.4
line_width_grid <- 0.15
point_size <- 0.3
legend_point_size <- 0.65
legend_axis_gap_mm <- 1.5
legend_row_gap_mm <- 0.75
legend_key_width_lines <- 1
legend_key_height_lines <- 0.5
panel_spacing_mm <- 0
legend_nrow_station <- 1L
legend_nrow_method <- 1L
plot_margin_mm <- c(3, 4, 0, 2)
axis_expand_fraction <- 0.02
grid_color <- "gray85"
viridis_family <- "viridis"
viridis_quantile_note <- "equal-count quantiles of finite data (unused here)"
wong_palette <- c("#E69F00", "#56B4E9", "#009E73", "#CC79A7",
                  "#D55E00", "#F0E442", "#0072B2", "#000000")
stations_order <- c("ASP", "QIQ", "PAL", "KWA", "SYO")
method_levels <- c("Persistence", "Pretrained", "Adapted")
method_linetypes <- c(
  "Persistence" = "solid",
  "Pretrained" = "dashed",
  "Adapted" = "dotted"
)
panel_levels <- c("Chronos-2", "TimesFM")
missing_note <- "TimesFM-BSRN L60 not yet run"

script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
root <- normalizePath(file.path(dirname(script), ".."))
python <- Sys.getenv("PYTHON", "python3")
out_pdf <- file.path(root, "tex", "rmse_holdout_koppen.pdf")

refresh_holdout_metrics <- function(root, python, out_csv) {
  code_dir <- file.path(root, "Code")
  root_py <- gsub("\\\\", "/", normalizePath(root, winslash = "/"))
  code_py <- gsub("\\\\", "/", normalizePath(code_dir, winslash = "/"))
  out_py <- gsub("\\\\", "/", normalizePath(out_csv, winslash = "/"))
  py <- tempfile(fileext = ".py")
  writeLines(
    c(
      "import csv, importlib.util",
      "from pathlib import Path",
      "import numpy as np",
      "import pyarrow.parquet as pq",
      paste0("ROOT = Path('", root_py, "')"),
      "RESULT = ROOT / 'Data' / 'result'",
      paste0("OUT = Path('", out_py, "')"),
      "HOLDOUTS = ('ASP', 'QIQ', 'PAL', 'KWA', 'SYO')",
      "YEARS = (2018, 2025, 2023, 2002, 2024)",
      "L = 15",
      "HISTORY_L = 60",
      "SCALE = 100.0",
      "FAMILIES = (",
      "    ('Chronos-2', 'chronos', 'chronos_ft_full'),",
      "    ('TimesFM', 'timesfm', 'timesfm_ft_full'),",
      ")",
      "def load(name, fname):",
      paste0("    p = Path('", code_py, "') / fname"),
      "    s = importlib.util.spec_from_file_location(name, p)",
      "    m = importlib.util.module_from_spec(s)",
      "    s.loader.exec_module(m)",
      "    return m",
      "helper = load('inp', '1.2.prepare_forecast_input.py')",
      "examples, meta = helper.load_eval_examples(stations=list(HOLDOUTS), years=list(YEARS))",
      "rows = []",
      "def read_fcst(path, n, origin_times):",
      "    fcst = np.empty((n, L))",
      "    with path.open() as f:",
      "        hdr = f.readline().rstrip().split('\\t')",
      "        cols = [hdr.index(f'Fcst{j}') for j in range(1, L + 1)]",
      "        time_col = hdr.index('Time')",
      "        count = 0",
      "        for r, line in enumerate(f):",
      "            parts = line.rstrip().split('\\t')",
      "            if r >= n or int(parts[time_col]) != int(origin_times[r]):",
      "                raise ValueError(f'Forecast origin mismatch in {path}, row {r}')",
      "            fcst[r, :] = [float(parts[c]) for c in cols]",
      "            count += 1",
      "    if count != n or not np.isfinite(fcst).all():",
      "        raise ValueError(f'Incomplete or invalid forecasts in {path}: {count}/{n}')",
      "    return fcst",
      "for station in HOLDOUTS:",
      "    mask = examples['station'] == station",
      "    src = int(np.unique(examples['source_index'][mask])[0])",
      "    centi = pq.read_table(meta['sources'][src]['path'], columns=['ghi_clear_centi'])['ghi_clear_centi'].to_numpy()",
      "    leads = np.arange(1, L + 1, dtype=np.int64)",
      "    idx = examples['origin_row'][mask][:, None] + leads",
      "    clear = centi[idx].astype(np.float64) / SCALE",
      "    target = examples['target'][mask]",
      "    n = int(mask.sum())",
      "    origin_times = examples['origin_utc_minute'][mask]",
      "    pers = np.repeat(examples['history'][mask, -1:], L, axis=1)",
      "    err_p = pers - target",
      "    for i, lead in enumerate(range(1, L + 1), start=0):",
      "        rmse_p = float(np.sqrt(np.mean(np.square(err_p[:, i]))))",
      "        for panel in ('Chronos-2', 'TimesFM'):",
      "            rows.append({'station': station, 'panel': panel, 'method': 'Persistence',",
      "                         'lead_min': lead, 'rmse_kappa': rmse_p})",
      "    for panel, pre, ft in FAMILIES:",
      "        for method, prefix in (('Pretrained', pre), ('Adapted', ft)):",
      "            path = RESULT / f'{station}_{prefix}_{HISTORY_L}.txt'",
      "            if not path.is_file():",
      "                if panel == 'TimesFM' and method == 'Adapted':",
      "                    continue",
      "                raise FileNotFoundError(path)",
      "            fcst = read_fcst(path, n, origin_times)",
      "            kappa = fcst / clear",
      "            err = kappa - target",
      "            for i, lead in enumerate(range(1, L + 1), start=0):",
      "                rows.append({'station': station, 'panel': panel, 'method': method,",
      "                             'lead_min': lead,",
      "                             'rmse_kappa': float(np.sqrt(np.mean(np.square(err[:, i]))))})",
      "with OUT.open('w', newline='') as f:",
      "    w = csv.DictWriter(f, fieldnames=['station', 'panel', 'method', 'lead_min', 'rmse_kappa'])",
      "    w.writeheader()",
      "    w.writerows(rows)"
    ),
    py
  )
  on.exit(unlink(py), add = TRUE)
  status <- system2(python, c("-B", py), stdout = TRUE, stderr = TRUE)
  if (length(attr(status, "status")) && attr(status, "status") != 0) {
    stop(paste(c("Hold-out RMSE export failed:", status), collapse = "\n"))
  }
}

metrics_csv <- tempfile(fileext = ".csv")
file.create(metrics_csv)
on.exit(unlink(metrics_csv), add = TRUE)
refresh_holdout_metrics(root, python, metrics_csv)

df <- readr::read_csv(metrics_csv, show_col_types = FALSE) %>%
  dplyr::filter(station %in% stations_order) %>%
  dplyr::mutate(
    station = factor(station, levels = stations_order),
    panel = factor(panel, levels = panel_levels),
    method = factor(method, levels = method_levels),
    lead_min = as.integer(lead_min)
  )

station_colors <- stats::setNames(wong_palette[seq_along(stations_order)], stations_order)

timesfm_adapted_missing <- !any(df$panel == "TimesFM" & df$method == "Adapted", na.rm = TRUE)
missing_data <- NULL
if (timesfm_adapted_missing) {
  tf <- df %>% dplyr::filter(panel == "TimesFM")
  missing_data <- tibble::tibble(
    panel = factor("TimesFM", levels = panel_levels),
    lead_min = stats::median(tf$lead_min),
    rmse_kappa = mean(range(tf$rmse_kappa)),
    note = missing_note
  )
  message("TimesFM Adapted forecasts missing; panel annotated like 4.2.")
}

p <- ggplot(df, aes(x = lead_min, y = rmse_kappa, colour = station, linetype = method)) +
  geom_line(linewidth = line_width_data) +
  geom_point(size = point_size) +
  facet_wrap(~panel, ncol = 1, scales = "free_y") +
  scale_colour_manual(values = station_colors, name = "Station") +
  scale_linetype_manual(values = method_linetypes, name = "Model",
                        labels = c("Persistence", "Pretrained", "Fine-tuned")) +
  scale_x_continuous(breaks = seq(1, 15, by = 2), expand = expansion(mult = axis_expand_fraction)) +
  scale_y_continuous(expand = expansion(mult = axis_expand_fraction),
                     breaks = scales::breaks_pretty(n = 3)) +
  labs(x = "Forecast horizon (min)", y = expression(RMSE~"(clear-sky index)")) +
  theme_bw(base_family = font_family, base_size = text_size_pt) +
  theme(
    text = element_text(family = font_family, size = text_size_pt),
    axis.text = element_text(size = text_size_pt),
    axis.title = element_text(size = text_size_pt),
    strip.text = element_text(size = text_size_pt),
    plot.title = element_blank(), plot.subtitle = element_blank(),
    plot.margin = margin(plot_margin_mm[1], plot_margin_mm[2],
                           plot_margin_mm[3], plot_margin_mm[4], "mm"),
    panel.spacing = unit(panel_spacing_mm, "mm"),
    panel.background = element_rect(fill = "transparent"),
    plot.background = element_rect(fill = "transparent", colour = NA),
    panel.grid.major = element_line(colour = grid_color, linewidth = line_width_grid),
    panel.grid.minor = element_blank(),
    legend.position = "bottom",
    legend.box = "vertical",
    legend.direction = "horizontal",
    legend.title = element_text(size = text_size_pt),
    legend.text = element_text(size = text_size_pt),
    legend.margin = margin(0, 0, 0, 0, "mm"),
    legend.box.margin = margin(0, 0, 0, 0, "mm"),
    legend.box.spacing = unit(legend_axis_gap_mm, "mm"),
    legend.spacing.y = unit(legend_row_gap_mm, "mm"),
    legend.key.width = unit(legend_key_width_lines, "lines"),
    legend.key.height = unit(legend_key_height_lines, "lines"),
    legend.background = element_rect(fill = "transparent", colour = NA)
  ) +
  guides(
    colour = guide_legend(
      nrow = legend_nrow_station,
      byrow = TRUE,
      override.aes = list(linetype = "solid", linewidth = line_width_data, size = legend_point_size)
    ),
    linetype = guide_legend(
      nrow = legend_nrow_method,
      byrow = TRUE,
      override.aes = list(size = legend_point_size)
    )
  )
if (!is.null(missing_data)) {
  p <- p + geom_text(
    data = missing_data,
    aes(x = lead_min, y = rmse_kappa, label = note),
    inherit.aes = FALSE,
    family = font_family,
    size = text_size_pt / ggplot2::.pt
  )
}

dir.create(dirname(out_pdf), recursive = TRUE, showWarnings = FALSE)
ggsave(
  filename = out_pdf,
  plot = p,
  width = width_mm,
  height = height_mm,
  units = "mm",
  device = grDevices::pdf,
  bg = "transparent"
)
message("Saved: ", out_pdf)
