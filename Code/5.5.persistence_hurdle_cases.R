# Selected single-origin instances: 60 context minutes + 15 forecast minutes.
# Rscript Code/5.5.persistence_hurdle_cases.R
# 4.1 selects instances and streams CSV through stdout; no intermediate files.
# Examples include upward and downward changes, deliberately selected for
# improvement rather than as representative cases.

suppressPackageStartupMessages(library(ggplot2))

# --- Editable parameter block (aligned with AMT Validation15min.r) ---
text_size_pt <- 8
font_family <- "Times"
width_mm <- 85
height_mm <- 88
line_width_data <- 0.3
line_width_origin <- line_width_data * 2
line_width_grid <- line_width_data
point_size <- 0.05
plot_margin_lines <- c(0.1, 0.2, 0, 0)
panel_spacing_lines <- 0
panel_spacing_x_mm <- 3
origin_color <- "gray45"
grid_color <- "gray80"
legend_nrow <- 1L
legend_key_width_lines <- 1
legend_key_height_lines <- 0.35
legend_margin_mm <- 0
legend_axis_gap_mm <- 1.5
axis_expand_fraction <- 0.02
annotation_headroom_fraction <- 0.65
annotation_vjust <- 1.05
viridis_family <- "viridis"
viridis_quantile_note <- "equal-count quantiles of finite data (unused here)"
wong_palette <- c(
  "#E69F00", "#56B4E9", "#009E73", "#CC79A7",
  "#D55E00", "#F0E442", "#0072B2", "#000000"
)
methods <- c("Observed", "Persistence", "TimesFM", "TimesFM-BSRN")
method_colors <- c(
  Observed = "#000000",
  Persistence = wong_palette[[1]],
  `TimesFM` = wong_palette[[2]],
  `TimesFM-BSRN` = wong_palette[[3]]
)
context_minutes <- 60L
forecast_minutes <- 15L
expected_cases <- 6L
group_levels <- c("Upward ramps", "Downward ramps")

script_path <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
project_root <- if (length(script_path)) {
  normalizePath(file.path(dirname(script_path), ".."))
} else {
  normalizePath(getwd())
}
python <- Sys.getenv("PYTHON", "python3")
finder <- file.path(project_root, "Code", "4.1.persistence_hurdle.py")
output <- file.path(project_root, "tex", "persistence_hurdle_cases.pdf")
dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)

case_group_from_label <- function(label) {
  kind <- sub("\n.*", "", label)
  kind <- sub("^.*: ", "", kind)
  switch(
    kind,
    up = "Upward ramps",
    down = "Downward ramps",
    stop("Unrecognized case kind in label: ", label)
  )
}

message("Finding single-origin 75-minute GHI instances using 4.1...")
csv_lines <- system2(python, c("-B", shQuote(finder), "--plot-data"),
                     stdout = TRUE,
                     env = c("PYTHONDONTWRITEBYTECODE=1", "OPENBLAS_NUM_THREADS=1"))
if (!is.null(attr(csv_lines, "status")) && attr(csv_lines, "status") != 0) {
  stop("Case finder failed; existing figure was not replaced")
}
data <- read.csv(text = paste(csv_lines, collapse = "\n"), check.names = FALSE)
if (length(unique(data$case)) != expected_cases) stop("Unexpected number of selected cases")

case_levels <- unique(data$case)
case_meta <- data.frame(
  case = case_levels,
  group_facet = vapply(case_levels, case_group_from_label, character(1)),
  stringsAsFactors = FALSE
)
case_meta <- case_meta[order(
  match(case_meta$group_facet, group_levels),
  case_meta$case
), , drop = FALSE]
case_meta$case_idx <- ave(
  seq_len(nrow(case_meta)),
  case_meta$group_facet,
  FUN = seq_along
)
case_meta$case_tag <- sub("^\\(([^)]+)\\).*", "(\\1)", case_meta$case)
case_meta$case_tag <- factor(
  case_meta$case_tag,
  levels = unique(case_meta$case_tag)
)
case_meta$station <- sub("^\\([^)]+\\) ([^:]+):.*", "\\1", sub("\n.*", "", case_meta$case))
case_meta$origin <- sub("^[^\n]*\n", "", case_meta$case)
case_meta$panel_label <- paste0(case_meta$case_tag, " ", case_meta$station, "\n",
                               sub(" UTC$", "", case_meta$origin))
case_meta$group_facet <- factor(case_meta$group_facet, levels = group_levels)
case_meta$case_idx <- factor(case_meta$case_idx, levels = 1:3)
data <- merge(
  data,
  case_meta[, c("case", "group_facet", "case_idx", "case_tag")],
  by = "case",
  all.x = TRUE,
  sort = FALSE
)
data$case <- factor(data$case, levels = case_meta$case)
data$group_facet <- factor(data$group_facet, levels = group_levels)
data$case_idx <- factor(data$case_idx, levels = 1:3)
data$method <- factor(data$method, levels = methods)

for (label in levels(data$case)) {
  observed <- subset(data, case == label & method == "Observed")
  stopifnot(identical(as.integer(observed$minute), seq.int(1L - context_minutes, forecast_minutes)))
  for (method in methods[-1]) {
    prediction <- data[data$case == label & data$method == method, , drop = FALSE]
    stopifnot(identical(as.integer(prediction$minute), seq.int(0L, forecast_minutes)))
  }
}

pub_theme <- theme_bw(base_family = font_family, base_size = text_size_pt) +
  theme(
    plot.margin = unit(plot_margin_lines, "lines"),
    text = element_text(family = font_family, size = text_size_pt),
    axis.text = element_text(size = text_size_pt),
    axis.title = element_text(size = text_size_pt),
    strip.text = element_text(size = text_size_pt),
    strip.text.y = element_blank(),
    strip.background.y = element_blank(),
    panel.spacing = unit(panel_spacing_lines, "lines"),
    panel.spacing.x = unit(panel_spacing_x_mm, "mm"),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(colour = grid_color, linewidth = line_width_grid),
    panel.background = element_rect(fill = "transparent"),
    plot.background = element_rect(fill = "transparent", color = NA),
    plot.title = element_blank(),
    plot.subtitle = element_blank(),
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = text_size_pt),
    legend.background = element_rect(fill = "transparent"),
    legend.key = element_rect(fill = "transparent"),
    legend.margin = margin(legend_margin_mm, legend_margin_mm, legend_margin_mm, legend_margin_mm, unit = "mm"),
    legend.box.margin = margin(legend_margin_mm, legend_margin_mm, legend_margin_mm, legend_margin_mm, unit = "mm"),
    legend.box.spacing = unit(legend_axis_gap_mm, "mm"),
    legend.key.width = unit(legend_key_width_lines, "lines"),
    legend.key.height = unit(legend_key_height_lines, "lines")
  )

p <- ggplot(data, aes(minute, ghi, color = method)) +
  geom_vline(
    xintercept = 0,
    color = origin_color,
    linewidth = line_width_origin,
    linetype = "dashed"
  ) +
  geom_line(linewidth = line_width_data) +
  geom_label(
    data = case_meta,
    aes(x = 1 - context_minutes, y = Inf, label = panel_label),
    inherit.aes = FALSE,
    family = font_family,
    size = text_size_pt / ggplot2::.pt,
    hjust = 0,
    vjust = annotation_vjust,
    label.padding = unit(0, "lines"),
    linewidth = 0,
    fill = NA,
    color = "black"
  ) +
  facet_grid(
    case_idx ~ group_facet,
    scales = "free_y",
    labeller = labeller(group_facet = label_value, case_idx = label_value)
  ) +
  scale_color_manual(values = method_colors, breaks = methods[-1], name = NULL) +
  scale_x_continuous(
    breaks = c(-60, -30, 0, 15),
    limits = c(-60, forecast_minutes),
    labels = function(x) parse(text = as.character(x)),
    expand = expansion(mult = axis_expand_fraction)
  ) +
  scale_y_continuous(
    labels = function(x) parse(text = as.character(x)),
    expand = expansion(mult = c(axis_expand_fraction, annotation_headroom_fraction))
  ) +
  labs(x = "Time from forecast origin (min)", y = expression(GHI~(W~m^{-2}))) +
  guides(
    color = guide_legend(
      nrow = legend_nrow,
      byrow = TRUE,
      override.aes = list(linewidth = line_width_data * 2, linetype = "solid")
    )
  ) +
  pub_theme

ggsave(output, p, width = width_mm, height = height_mm, units = "mm", device = grDevices::pdf)
message("Wrote ", output, "; each panel is one 60+15-minute instance in GHI.")
