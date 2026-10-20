# Ramp cohorts (4.2): median(first five future kappa) - kappa(origin) >= +0.25 or <= -0.25.
# L60 hold-outs: per-origin persistence RMSE minus each TimesFM variant's RMSE.
# No performance filter; positive = better than persistence; retain negative gains.
# Rscript Code/5.6.ramp_forecast_errors.R
suppressPackageStartupMessages(library(ggplot2))

# --- Editable publication parameters ---
text_size_pt <- 8
font_family <- "Times"
width_mm <- 85
height_mm <- 65
line_width_data <- 0.4
line_width_grid <- 0.15
point_size <- 1.1
panel_spacing_mm <- 0
plot_margin_mm <- c(3, 4, 0, 2)
delta_axis_limits <- c(-250, 450)
delta_axis_breaks <- seq(-200, 400, 100)
legend_axis_gap_mm <- 1.5
legend_key_width_lines <- 1
legend_key_height_lines <- 0.5
legend_nrow <- 1L
zero_color <- "gray45"
grid_color <- "gray85"
viridis_family <- "viridis"
viridis_quantile_note <- "equal-count quantiles of finite data (unused here)"
wong_palette <- c("#E69F00", "#56B4E9", "#009E73", "#CC79A7",
                  "#D55E00", "#F0E442", "#0072B2", "#000000")
groups <- c("Upward ramps", "Downward ramps")
methods <- c("Persistence", "TimesFM", "TimesFM-BSRN")
method_colors <- setNames(wong_palette[seq_along(methods)], methods)

script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
root <- normalizePath(file.path(dirname(script), ".."))
python <- Sys.getenv("PYTHON", "python3")
output <- file.path(root, "tex", "ramp_forecast_errors.pdf")
dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
message("Building paired RMSE-difference distributions against persistence with 4.2...")
lines <- system2(python,
  c("-B", shQuote(file.path(root, "Code", "4.2.ramp_forecast_errors.py")), "--plot-data"),
  stdout = TRUE, env = c("PYTHONDONTWRITEBYTECODE=1", "OPENBLAS_NUM_THREADS=1"))
if (!is.null(attr(lines, "status")) && attr(lines, "status") != 0) {
  stop("Metric calculation failed; existing figure was not replaced")
}
data <- read.csv(text = paste(lines, collapse = "\n"))
data$group <- factor(data$group, levels = groups)
data$method <- factor(data$method, levels = methods)
# Within each cohort every model uses identical origins and 15 targets.
# Adjacent forecast windows overlap. Counts are not independent physical events.
# KDEs retain all scores in each observation-defined group, including failures.
# Group summaries are also printed by 4.2.
pub_theme <- theme_bw(base_family = font_family, base_size = text_size_pt) +
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
    legend.position = "bottom", legend.title = element_blank(),
    legend.text = element_text(size = text_size_pt),
    legend.margin = margin(0, 0, 0, 0, "mm"),
    legend.box.margin = margin(0, 0, 0, 0, "mm"),
    legend.box.spacing = unit(legend_axis_gap_mm, "mm"),
    legend.key.width = unit(legend_key_width_lines, "lines"),
    legend.key.height = unit(legend_key_height_lines, "lines"),
    legend.background = element_rect(fill = "transparent", colour = NA)
  )
make_plot <- function() {
  page_data <- data
  page_data$panel <- factor(page_data$group, levels = groups)
  x_label <- expression("RMSE difference from persistence"~(W~m^{-2}))
  # Crop the display only; retain extreme errors in densities and summaries.
  x_scale <- scale_x_continuous(breaks = delta_axis_breaks,
                 labels = function(x) parse(text = as.character(x)),
                 expand = expansion(mult = c(0.015, 0.015)))
  ggplot(page_data, aes(value, density, colour = method)) +
  geom_vline(xintercept = 0, colour = zero_color, linewidth = line_width_grid, linetype = "dashed") +
  geom_line(linewidth = line_width_data) +
  scale_colour_manual(values = method_colors, breaks = methods[-1], name = NULL) +
  facet_wrap(~panel, ncol = 1) +
  x_scale +
  coord_cartesian(xlim = delta_axis_limits) +
  scale_y_continuous(limits = c(0, NA), breaks = scales::breaks_pretty(n = 3),
                     labels = function(x) format(x, trim = TRUE, scientific = FALSE),
                     expand = expansion(mult = c(0, 0.04))) +
  labs(x = x_label, y = "Probability density") +
  pub_theme +
  guides(colour = guide_legend(nrow = legend_nrow, byrow = TRUE))
}
grDevices::pdf(output, width = width_mm/25.4, height = height_mm/25.4,
               family = font_family, bg = "transparent", onefile = TRUE)
print(make_plot())
grDevices::dev.off()
message("Saved: ", output)
