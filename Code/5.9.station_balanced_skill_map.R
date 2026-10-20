# Rscript Code/5.9.station_balanced_skill_map.R
# Balanced minus full-data RMSE skill: positive favors balanced training.
suppressPackageStartupMessages({
  library(ggplot2)
  library(sf)
  library(rnaturalearth)
})

# --- Editable publication parameters ---
text_size_pt <- 8
font_family <- "Times"
width_mm <- 85
height_mm <- 55
point_size <- 1.8
line_width_data <- 0.2
line_width_grid <- 0.15
legend_axis_gap_mm <- 1.5
legend_width_mm <- 60
legend_height_mm <- 1.5
plot_margin_mm <- c(0, 1, 0, 1)
land_fill <- "gray85"
coast_color <- "gray65"
viridis_family <- "viridis"
viridis_quantile_note <- "equal-count quantiles of finite differences (overridden by existing map convention)"
# Match the user-approved map colors: blue negative, red positive, neutral zero.
difference_negative_color <- "#2166AC"
difference_zero_color <- "#F7F7F7"
difference_positive_color <- "#B2182B"
color_limits <- c(-4, 4)
color_breaks <- seq(-4, 4, 2)
wong_palette <- c("#E69F00", "#56B4E9", "#009E73", "#CC79A7",
                  "#D55E00", "#F0E442", "#0072B2", "#000000")

script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
root <- normalizePath(file.path(dirname(script), ".."))
python <- Sys.getenv("PYTHON", "python3")
output <- file.path(root, "tex", "station_balanced_skill_difference.pdf")
dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
raw <- system2(python, c("-B", shQuote(file.path(root, "Code", "4.5.station_balanced_skill.py"))), stdout = TRUE)
if (!is.null(attr(raw, "status")) && attr(raw, "status") != 0) stop("Paired skill calculation failed")
scores <- read.csv(text = paste(raw, collapse = "\n"))
stopifnot(nrow(scores) == 70L, !anyDuplicated(scores$station),
          all(is.finite(scores$skill_difference_pp)),
          all(scores$skill_difference_pp >= color_limits[1]),
          all(scores$skill_difference_pp <= color_limits[2]))
world <- rnaturalearth::ne_countries(scale = "small", returnclass = "sf")
p <- ggplot() +
  geom_sf(data = world, fill = land_fill, color = coast_color, linewidth = line_width_grid) +
  geom_point(data = scores, aes(lon, lat, fill = skill_difference_pp), shape = 21,
             color = "black", stroke = line_width_data, size = point_size) +
  scale_fill_gradient2(low = difference_negative_color, mid = difference_zero_color,
                        high = difference_positive_color, midpoint = 0,
                        limits = color_limits, breaks = color_breaks,
                        labels = function(x) parse(text = formatC(x, digits = 0, format = "f")),
                        name = "RMSE skill score difference from TimesFM-BSRN (%)",
                        guide = guide_colorbar(title.position = "top", title.hjust = 0.5,
                                              barwidth = unit(legend_width_mm, "mm"),
                                              barheight = unit(legend_height_mm, "mm"))) +
  coord_sf(xlim = c(-180, 180), ylim = c(-90, 90), expand = FALSE, clip = "off") +
  theme_bw(base_family = font_family, base_size = text_size_pt) +
  theme(
    text = element_text(family = font_family, size = text_size_pt),
    axis.text = element_blank(), axis.title = element_blank(), axis.ticks = element_blank(),
    panel.grid = element_blank(),
    panel.border = element_rect(fill = NA, color = "black", linewidth = line_width_data),
    panel.background = element_rect(fill = "transparent", color = NA),
    plot.background = element_rect(fill = "transparent", color = NA),
    plot.title = element_blank(), plot.subtitle = element_blank(),
    legend.position = "bottom", legend.direction = "horizontal",
    legend.title = element_text(size = text_size_pt), legend.text = element_text(size = text_size_pt),
    legend.margin = margin(0, 0, 0, 0, "mm"), legend.box.margin = margin(0, 0, 0, 0, "mm"),
    legend.background = element_rect(fill = "transparent", color = NA),
    legend.box.spacing = unit(legend_axis_gap_mm, "mm"),
    plot.margin = margin(plot_margin_mm[1], plot_margin_mm[2], plot_margin_mm[3], plot_margin_mm[4], "mm")
  )
ggsave(output, p, width = width_mm, height = height_mm, units = "mm",
       device = grDevices::pdf, bg = "transparent")
message("Saved: ", output)
