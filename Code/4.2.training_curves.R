# Full-pass fine-tuning histories only; excludes all 1000-step trials.
# Rscript Code/4.2.training_curves.R
# Manuscript caption: Training and validation losses during one pass through the pooled BSRN training origins.
# Native losses are not comparable between model families. No smoothing.
# X: fraction of the eligible full-pass training origins, not raw BSRN minutes.
suppressPackageStartupMessages(library(ggplot2))

# --- Editable publication parameters ---
text_size_pt <- 8
font_family <- "Times"
width_mm <- 85
height_mm <- 115
line_width_data <- 0.3
line_width_grid <- 0.15
point_size <- 1
panel_spacing_mm <- 3
legend_axis_gap_mm <- 1.5
legend_key_width_lines <- 1.5
plot_margin_mm <- c(2, 3, 1, 2)
grid_color <- "gray85"
viridis_family <- "viridis"
viridis_quantile_note <- "equal-count quantiles of finite data (unused here)"
wong_palette <- c("#E69F00", "#56B4E9", "#009E73", "#CC79A7",
                  "#D55E00", "#F0E442", "#0072B2", "#000000")
series_levels <- c("Training", "Validation")
series_linetypes <- c("solid", "solid")
contexts <- c(15L, 30L, 60L)
models <- c("Chronos-2", "TimesFM")
percentage_breaks <- c(0, 50, 100)

script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
root <- normalizePath(file.path(dirname(script), ".."))
output <- file.path(root, "tex", "training_curves.pdf")
dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
# Active runs have no final manifest yet; all full-pass configurations share
# the reference training-origin pool and batch size (including the last batch).
reference_manifest <- jsonlite::fromJSON(file.path(root, "Data", "models",
    "chronos-2-full-L60", "training_manifest.json"))
specs <- expand.grid(model = models, context = contexts, stringsAsFactors = FALSE)
histories <- list()
missing <- list()
labels <- character(nrow(specs))
for (i in seq_len(nrow(specs))) {
  model <- specs$model[i]
  context <- specs$context[i]
  prefix <- if (model == "Chronos-2") "chronos-2-full-L" else "timesfm-2.5-full-L"
  folder <- file.path(root, "Data", "models", paste0(prefix, context))
  path <- file.path(folder, "loss_history.jsonl")
  manifest_path <- file.path(folder, "training_manifest.json")
  manifest <- if (file.exists(manifest_path)) jsonlite::fromJSON(manifest_path) else reference_manifest
  complete <- file.exists(manifest_path) && isTRUE(manifest$completed)
  n_origins <- manifest$n_train_index_rows
  batch_size <- manifest$batch_size
  if (manifest$train_mode != "one_pass_without_replacement" ||
      length(n_origins) != 1L || n_origins < 1 ||
      length(batch_size) != 1L || batch_size < 1) stop("Invalid full-pass manifest: ", folder)
  rows <- list()
  if (file.exists(path)) {
    # The active run can be appending its final line while this snapshot is read.
    lines <- readLines(path, warn = FALSE)
    for (j in seq_along(lines)) {
      row <- tryCatch(jsonlite::fromJSON(lines[j]), error = function(e) NULL)
      if (is.null(row)) {
        if (j != length(lines)) stop("Invalid JSON inside ", path, " at line ", j)
        warning("Ignoring incomplete final log record: ", path)
        next
      }
      if (!row$event %in% c("training", "validation")) next
      field <- if (model == "Chronos-2") {
        if (row$event == "training") "loss" else "eval_loss"
      } else {
        if (row$event == "training") "train_mse" else "val_mse"
      }
      value <- row[[field]]
      if (length(value) != 1L || !is.finite(value) || value < 0 ||
          !is.finite(row$elapsed_seconds) || row$elapsed_seconds < 0 ||
          row$history_length != context) stop("Invalid loss record: ", path)
      rows[[length(rows) + 1L]] <- data.frame(
        hours = row$elapsed_seconds / 3600, loss = value,
        series = if (row$event == "training") "Training" else "Validation",
        step = row$step, panel_id = i)
    }
  }
  label <- paste0(model, "\nL = ", context)
  if (!length(rows)) {
    labels[i] <- label
    missing[[length(missing) + 1L]] <- data.frame(
      percentage = 50, loss = 0, panel_id = i, note = "Not yet run")
    next
  }
  labels[i] <- paste0(label, if (!complete) " (partial)" else "")
  run <- do.call(rbind, rows)
  if (any(diff(run$hours) < 0) || any(diff(run$step) < 0)) {
    stop("History is not chronological: ", path)
  }
  if (max(run$step) > ceiling(n_origins / batch_size)) stop("Step exceeds one pass: ", path)
  run$percentage <- 100 * pmin(run$step * batch_size, n_origins) / n_origins
  if (complete && tail(run$percentage, 1) != 100) stop("Completed run does not reach 100%: ", path)
  histories[[length(histories) + 1L]] <- run
  message(label, ": ", max(run$step), " updates; ",
          round(max(run$hours), 2), " h; ", if (complete) "complete" else "partial")
}
if (!length(histories)) stop("No full-pass training histories found")
data <- do.call(rbind, histories)
data$panel <- factor(labels[data$panel_id], levels = labels)
data$series <- factor(data$series, levels = series_levels)
missing_data <- if (length(missing)) do.call(rbind, missing) else NULL
if (!is.null(missing_data)) {
  missing_data$panel <- factor(labels[missing_data$panel_id], levels = labels)
}

p <- ggplot(data, aes(percentage, loss, colour = series, linetype = series)) +
  geom_line(linewidth = line_width_data) +
  facet_wrap(~panel, ncol = length(models), scales = "free_y", drop = FALSE) +
  scale_colour_manual(values = setNames(wong_palette[seq_along(series_levels)], series_levels)) +
  scale_linetype_manual(values = setNames(series_linetypes, series_levels)) +
  scale_x_continuous(limits = c(0, 100), breaks = percentage_breaks) +
  scale_y_continuous(breaks = scales::breaks_pretty(n = 3)) +
  labs(x = "BSRN training origins processed (%)", y = "Native loss", colour = NULL, linetype = NULL) +
  theme_bw(base_family = font_family, base_size = text_size_pt) +
  theme(
    text = element_text(family = font_family, size = text_size_pt),
    axis.text = element_text(size = text_size_pt),
    axis.title = element_text(size = text_size_pt),
    strip.text = element_text(size = text_size_pt),
    legend.text = element_text(size = text_size_pt),
    plot.title = element_blank(), plot.subtitle = element_blank(),
    panel.spacing = unit(panel_spacing_mm, "mm"),
    panel.grid.major = element_line(colour = grid_color, linewidth = line_width_grid),
    panel.grid.minor = element_blank(),
    panel.background = element_rect(fill = "transparent"),
    plot.background = element_rect(fill = "transparent", colour = NA),
    legend.position = "bottom", legend.title = element_blank(),
    legend.background = element_rect(fill = "transparent", colour = NA),
    legend.box.spacing = unit(legend_axis_gap_mm, "mm"),
    legend.key.width = unit(legend_key_width_lines, "lines"),
    plot.margin = margin(plot_margin_mm[1], plot_margin_mm[2],
                         plot_margin_mm[3], plot_margin_mm[4], "mm")
  )
if (!is.null(missing_data)) {
  p <- p + geom_text(data = missing_data, aes(x = percentage, y = loss, label = note),
                     inherit.aes = FALSE, family = font_family,
                     size = text_size_pt / ggplot2::.pt)
}
ggsave(output, p, width = width_mm, height = height_mm, units = "mm",
       device = grDevices::pdf, bg = "transparent")
message("Saved: ", output)
