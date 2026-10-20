# Rscript Code/4.5.training_sample_comparison.R
# Identical hold-out observations; missing exports leave empty bar positions.
suppressPackageStartupMessages(library(ggplot2))

# --- Editable publication parameters ---
text_size_pt <- 8
font_family <- "Times"
width_mm <- 85
height_mm <- 75
bar_width <- 0.84
line_width_data <- 0.2
line_width_grid <- 0.15
point_size <- 0.3
panel_spacing_mm <- 0
legend_axis_gap_mm <- 1.5
legend_key_width_mm <- 3
legend_key_height_mm <- 2.5
legend_row_gap_mm <- 0.75
plot_margin_mm <- c(2, 3, 1, 2)
viridis_family <- "viridis"
viridis_quantile_note <- "equal-count quantiles of finite data (unused here)"
wong_palette <- c("#E69F00", "#56B4E9", "#009E73", "#CC79A7",
                  "#D55E00", "#F0E442", "#0072B2", "#000000")
stations <- c("ASP", "QIQ", "PAL", "KWA", "SYO")
models <- c("Chronos-2", "TimesFM")
series_levels <- c("1000 updates, L15", "Full pass, L15",
                   "1000 updates, L60", "Full pass, L60")

script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
root <- normalizePath(file.path(dirname(script), ".."))
python <- Sys.getenv("PYTHON", "python3")
output <- file.path(root, "tex", "training_sample_comparison.pdf")
dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)

# Reuse the table reader; no new intermediate project files.
py <- tempfile(fileext = ".py")
writeLines(c(
  "import csv, importlib.util, sys",
  "from pathlib import Path",
  "import numpy as np",
  "root = Path(sys.argv[1])",
  "spec = importlib.util.spec_from_file_location('tables', root / 'Code/4.4.tables_forecast.py')",
  "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)",
  "w = csv.writer(sys.stdout); w.writerow(['station', 'model', 'series', 'rmse'])",
  "for station in ('ASP', 'QIQ', 'PAL', 'KWA', 'SYO'):",
  "    ref, _, times = m.load_ghi_pairs(root / f'Data/result/{station}_pers_60.txt', return_times=True)",
  "    for model, short, full in (('Chronos-2', 'chronos_ft', 'chronos_ft_full'),",
  "                                ('TimesFM', 'timesfm_ft', 'timesfm_ft_full')):",
  "        for L in (15, 60):",
  "            for label, prefix in (('1000 updates', short), ('Full pass', full)):",
  "                path = root / f'Data/result/{station}_{prefix}_{L}.txt'",
  "                value = ''",
  "                if path.is_file():",
  "                    obs, fcst, method_times = m.load_ghi_pairs(path, return_times=True)",
  "                    if not np.array_equal(obs, ref) or not np.array_equal(method_times, times):",
  "                        raise ValueError(f'Origin/observation mismatch: {path}')",
  "                    value = float(np.sqrt(np.mean((fcst - obs)**2)))",
  "                w.writerow([station, model, f'{label}, L{L}', value])"
), py)
csv <- tryCatch(system2(python, c("-B", shQuote(py), shQuote(root)), stdout = TRUE),
                finally = unlink(py))
if (!is.null(attr(csv, "status")) && attr(csv, "status") != 0) stop("RMSE calculation failed")
df <- read.csv(text = paste(csv, collapse = "\n"), na.strings = "")
df$station <- factor(df$station, levels = stations)
df$model <- factor(df$model, levels = models)
df$series <- factor(df$series, levels = series_levels)
print(df, row.names = FALSE)

p <- ggplot(df, aes(station, rmse, fill = series, group = series)) +
  geom_col(width = bar_width, position = position_dodge(width = bar_width), na.rm = TRUE) +
  facet_wrap(~model, ncol = 1) +
  scale_fill_manual(values = setNames(wong_palette[seq_along(series_levels)], series_levels),
                    drop = FALSE) +
  scale_y_continuous(limits = c(0, NA), expand = expansion(mult = c(0, 0.04))) +
  labs(x = NULL, y = expression(RMSE~(W~m^{-2})), fill = NULL) +
  theme_bw(base_family = font_family, base_size = text_size_pt) +
  theme(
    text = element_text(family = font_family, size = text_size_pt),
    axis.text = element_text(size = text_size_pt),
    axis.title = element_text(size = text_size_pt),
    strip.text = element_text(size = text_size_pt),
    legend.text = element_text(size = text_size_pt),
    plot.title = element_blank(), plot.subtitle = element_blank(),
    panel.spacing = unit(panel_spacing_mm, "mm"),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(colour = "gray85", linewidth = line_width_grid),
    panel.grid.minor = element_blank(),
    panel.background = element_rect(fill = "transparent"),
    plot.background = element_rect(fill = "transparent", colour = NA),
    legend.position = "bottom", legend.title = element_blank(),
    legend.margin = margin(0, 0, 0, 0, "mm"),
    legend.box.margin = margin(0, 0, 0, 0, "mm"),
    legend.background = element_rect(fill = "transparent", colour = NA),
    legend.box.spacing = unit(legend_axis_gap_mm, "mm"),
    legend.spacing.y = unit(legend_row_gap_mm, "mm"),
    legend.key.width = unit(legend_key_width_mm, "mm"),
    legend.key.height = unit(legend_key_height_mm, "mm"),
    plot.margin = margin(plot_margin_mm[1], plot_margin_mm[2],
                         plot_margin_mm[3], plot_margin_mm[4], "mm")
  ) + guides(fill = guide_legend(ncol = 2, byrow = FALSE))
ggsave(output, p, width = width_mm, height = height_mm, units = "mm",
       device = grDevices::pdf, bg = "transparent")
message("Saved: ", output)
