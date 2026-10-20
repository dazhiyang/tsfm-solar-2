# Layer-wise final-context-token PCA from 4.4; no models loaded or PCA refitted.
# Rscript Code/5.8.timesfm_embeddings.R [positive|negative]
# Each column uses one PCA fitted jointly to both models' random-sample origins.
# Coordinates/limits match between models within a column, not between layers.
# Color distinguishes upward ramps, downward ramps, and other contexts.
# Original-input branch is default. The native forecast combines BOTH branches;
# this figure is a descriptive view of one branch, not the final forecast itself.
suppressPackageStartupMessages({
  library(ggplot2)
  library(ggh4x)
  library(ggrepel)
  library(scattermore)
})

# --- Editable publication parameters ---
text_size_pt <- 8
font_family <- "Times"
width_mm <- 160
height_mm <- 58
point_size <- 0.54
example_point_size <- 0.75
point_alpha <- 0.6
background_alpha <- 0.13
background_pixels <- c(600L, 600L)
background_pointsize_px <- 4  # Raster point radius; sampling and PCA unchanged.
line_width_data <- 0.25
line_width_grid <- 0.12
panel_spacing_mm <- 1
legend_axis_gap_mm <- 0
legend_key_width_mm <- 3
legend_key_height_lines <- 0.35
legend_text_margin_pt <- 1
plot_margin_mm <- c(1, 1, 1, 1) / 2.845276
axis_expand_fraction <- 0.05
zoom_quantiles <- c(0.01, 0.99)  # Display only; PCA/data retain all origins.
label_padding_lines <- 0.15
show_illustrations <- TRUE  # Retain the six forecast examples, labelled (a)--(f).
legend_point_size <- 1.5
group_levels <- c("Upward ramps", "Downward ramps", "Other")
model_levels <- c("TimesFM", "TimesFM-BSRN")
stage_levels <- c(0L, 1L, 5L, 10L, 15L, 20L)
wong_palette <- c("#E69F00", "#56B4E9", "#009E73", "#CC79A7",
                  "#D55E00", "#F0E442", "#0072B2", "#000000")
group_colors <- setNames(c(wong_palette[1:2], "gray65"), group_levels)
grid_color <- grDevices::adjustcolor("black", alpha.f = 0.20)
viridis_family <- "viridis"
viridis_quantile_note <- "equal-count quantiles of finite values (unused here)"

script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
root <- normalizePath(file.path(dirname(script), ".."))
python <- Sys.getenv("PYTHON", "python3")
input <- file.path(root, "Data", "result", "timesfm_embeddings_L60.parquet")
output <- file.path(root, "tex", "timesfm_embeddings.pdf")
dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
args <- commandArgs(trailingOnly = TRUE)
branch <- if (length(args)) args[[1]] else "positive"
if (length(args) > 1L || !branch %in% c("positive", "negative")) {
  stop("Use at most one argument: positive or negative")
}
if (!file.exists(input)) stop("Run the approved 4.4 extraction first")

# Stream only plotting columns through stdout; no intermediate files.
python_code <- paste(
  "import sys, pandas as pd",
  "cols=['station','origin_utc_minute','random_sample','illustration','group','model','layer','branch','pc1','pc2','pc1_variance_ratio','pc2_variance_ratio']",
  "d=pd.read_parquet(sys.argv[1], columns=cols)",
  "d=d[d.branch.eq(sys.argv[2])]",
  "d.to_csv(sys.stdout,index=False)", sep = "; ")
lines <- system2(python, c("-B", "-c", shQuote(python_code), shQuote(input), branch),
                 stdout = TRUE, env = "PYTHONDONTWRITEBYTECODE=1")
if (!is.null(attr(lines, "status")) && attr(lines, "status") != 0) stop("Parquet read failed")
d <- read.csv(text = paste(lines, collapse = "\n"), stringsAsFactors = FALSE)
if (!nrow(d) || any(!is.finite(d$pc1)) || any(!is.finite(d$pc2))) stop("Invalid PCA coordinates")
d$illustration[is.na(d$illustration)] <- ""
d$group <- factor(d$group, levels = group_levels)
d$model <- factor(d$model, levels = model_levels)
if (anyNA(d$group) || anyNA(d$model)) stop("Unknown group/model labels")

boundaries <- lapply(stage_levels, function(stage) {
  x <- d[d$layer == stage, ]
  if (!nrow(x)) stop(paste("Missing stage", stage))
  counts <- table(x$model)
  if (any(counts == 0) || length(unique(as.integer(counts))) != 1) stop("Unpaired stage")
  a <- x[x$model == model_levels[1], c("station", "origin_utc_minute")]
  b <- x[x$model == model_levels[2], c("station", "origin_utc_minute")]
  rownames(a) <- rownames(b) <- NULL
  if (!identical(a, b)) stop("Origin alignment mismatch")
  # Central display range, shared across models, retaining all six examples.
  core <- c(stats::quantile(x$pc1, zoom_quantiles),
            stats::quantile(x$pc2, zoom_quantiles))
  illustrated <- x[x$illustration != "", ]
  limits <- range(c(core, illustrated$pc1, illustrated$pc2))
  padding <- diff(limits) * axis_expand_fraction
  limits <- limits + c(-padding, padding)
  label <- if (stage == 0L) "Input" else paste0("L", stage)
  message(label, ": PC1 ", round(100 * x$pc1_variance_ratio[1], 1),
          "%, PC2 ", round(100 * x$pc2_variance_ratio[1], 1), "%")
  expand.grid(pc1 = limits, pc2 = limits, model = model_levels, layer = stage)
})
bounds <- do.call(rbind, boundaries)
stage_names <- c("Input", paste0("L", stage_levels[-1]))
d$stage <- factor(d$layer, levels = stage_levels, labels = stage_names)
bounds$stage <- factor(bounds$layer, levels = stage_levels, labels = stage_names)
bounds$model <- factor(bounds$model, levels = model_levels)
sample_rows <- tolower(as.character(d$random_sample)) == "true"
sampled <- d[sample_rows, ]
other <- sampled[sampled$group == "Other", ]
ramps <- sampled[sampled$group != "Other", ]
examples <- d[d$illustration != "", ]
examples$label <- paste0("(", examples$illustration, ")")
p <- ggplot(d, aes(pc1, pc2, colour = group)) +
    geom_blank(data = bounds, aes(pc1, pc2), inherit.aes = FALSE) +
    scattermore::geom_scattermore(data = other, pixels = background_pixels,
                                  pointsize = background_pointsize_px,
                                  alpha = background_alpha) +
    geom_point(data = ramps, size = point_size, alpha = point_alpha, stroke = 0) +
    ggh4x::facet_grid2(rows = vars(model), cols = vars(stage),
                      scales = "free", independent = "all", drop = FALSE) +
    scale_colour_manual(values = group_colors, limits = group_levels, drop = FALSE) +
    scale_fill_manual(values = group_colors, limits = group_levels, guide = "none") +
    scale_x_continuous(expand = expansion(mult = 0)) +
    scale_y_continuous(expand = expansion(mult = 0)) +
    coord_cartesian() +
    labs(x = "PC1", y = "PC2", colour = NULL) +
    theme_bw(base_family = font_family, base_size = text_size_pt) +
    theme(
      text = element_text(family = font_family, size = text_size_pt),
      axis.text = element_text(size = text_size_pt),
      axis.title = element_text(size = text_size_pt),
      strip.text = element_text(size = text_size_pt),
      strip.background = element_rect(fill = "gray85", linewidth = line_width_data),
      strip.placement = "outside", panel.spacing = unit(panel_spacing_mm, "mm"),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(linewidth = line_width_data),
      panel.background = element_rect(fill = "white", colour = NA),
      plot.background = element_rect(fill = "white", colour = NA),
      plot.title = element_blank(), plot.subtitle = element_blank(),
      axis.ticks = element_blank(), axis.text.x = element_blank(), axis.text.y = element_blank(),
      legend.text = element_text(size = text_size_pt,
                                 margin = margin(l = legend_text_margin_pt, r = legend_text_margin_pt, unit = "pt")),
      legend.title = element_text(size = text_size_pt),
      legend.position = "bottom",
      legend.key.width = unit(legend_key_width_mm, "mm"),
      legend.key.height = unit(legend_key_height_lines, "lines"),
      legend.box.spacing = unit(legend_axis_gap_mm, "mm"),
      legend.margin = margin(0, 0, 0, 0, "mm"),
      plot.margin = margin(plot_margin_mm[1], plot_margin_mm[2],
                           plot_margin_mm[3], plot_margin_mm[4], "mm")
    ) + guides(colour = guide_legend(nrow = 1, override.aes = list(alpha = 1, size = legend_point_size)))
# Facet-specific view limits: boundary rows alone cannot override outliers.
# The same limits apply to both models; no PCA coordinates are recalculated.
view_scale <- function(layer, axis) {
  stage_name <- stage_names[match(layer, stage_levels)]
  view_limits <- range(bounds$pc1[bounds$layer == layer])
  if (axis == "x") {
    stage == stage_name ~ scale_x_continuous(limits = view_limits, expand = expansion(mult = 0))
  } else {
    stage == stage_name ~ scale_y_continuous(limits = view_limits, expand = expansion(mult = 0))
  }
}
p <- p + ggh4x::facetted_pos_scales(
  x = lapply(stage_levels, view_scale, axis = "x"),
  y = lapply(stage_levels, view_scale, axis = "y")
)
if (show_illustrations) {
  p <- p +
    geom_point(data = examples, aes(fill = group), size = example_point_size, shape = 21,
               colour = "black", stroke = line_width_data, show.legend = FALSE) +
    geom_text_repel(data = examples, aes(label = label), colour = "black",
                    family = font_family, size = text_size_pt / ggplot2::.pt,
                    seed = 42, box.padding = label_padding_lines, min.segment.length = 0,
                    segment.size = line_width_grid, max.overlaps = Inf, show.legend = FALSE)
}
ggsave(output, p, width = width_mm, height = height_mm, units = "mm",
       device = grDevices::pdf, bg = "transparent")
message("Saved ", branch, " branch: ", output)
