# Rscript Code/4.8.network_skill_map.R
# Shared zero-centered RMSE skill scale; no intermediate data files.
suppressPackageStartupMessages({
  library(ggplot2)
  library(sf)
  library(rnaturalearth)
})

# --- Editable publication parameters ---
text_size_pt <- 8
font_family <- "Times"
width_mm <- 160
height_mm <- 58
point_size <- 1.8
line_width_data <- 0.2
line_width_grid <- 0.15
legend_axis_gap_mm <- 1.5
legend_width_mm <- 75
legend_height_mm <- 1.5
panel_spacing_mm <- 0
plot_margin_mm <- c(0, 1, 0, 1)
land_fill <- "gray85"
coast_color <- "gray65"
viridis_family <- "viridis"
viridis_quantile_note <- "equal-count quantile bands computed jointly from finite skills in both panels"
# User override: linear diverging colors instead of viridis quantile bands.
skill_negative_color <- "#2166AC"
skill_zero_color <- "#F7F7F7"
skill_positive_color <- "#B2182B"
skill_lower_limit <- -5
wong_palette <- c("#E69F00", "#56B4E9", "#009E73", "#CC79A7",
                  "#D55E00", "#F0E442", "#0072B2", "#000000")
panel_levels <- c("TimesFM", "TimesFM-BSRN")

script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
root <- normalizePath(file.path(dirname(script), ".."))
python <- Sys.getenv("PYTHON", "python3")
output <- file.path(root, "tex", "network_skill_map.pdf")
dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
raw <- system2(python, c("-B", shQuote(file.path(root, "Code", "5.3.network_skill.py"))), stdout = TRUE)
if (!is.null(attr(raw, "status")) && attr(raw, "status") != 0) stop("Station skill calculation failed")
scores <- read.csv(text = paste(raw, collapse = "\n"))
scores$model <- factor(scores$model, levels = panel_levels)
stopifnot(!anyDuplicated(scores[c("station", "model")]),
          all(is.finite(scores$lat)), all(is.finite(scores$lon)))
finite <- scores$skill_percent[is.finite(scores$skill_percent)]
if (!length(finite)) stop("No finite station skills")
skill_extent <- ceiling(max(abs(finite))/5)*5
if (skill_extent == 0) skill_extent <- 5
skill_limits <- c(skill_lower_limit, skill_extent)
if (min(finite) < skill_lower_limit) stop("Skill below the configured color-scale minimum")
breaks <- sort(unique(c(skill_lower_limit, 0, pretty(c(0, skill_extent), n = 4))))
breaks <- breaks[breaks >= skill_limits[1] & breaks <= skill_limits[2]]
# Identical linear magnitudes and a neutral zero are used in both panels.
world <- rnaturalearth::ne_countries(scale = "small", returnclass = "sf")
p <- ggplot() +
  geom_sf(data = world, fill = land_fill, color = coast_color, linewidth = line_width_grid) +
  geom_point(data = scores, aes(lon, lat, fill = skill_percent), shape = 21,
             color = "black", stroke = line_width_data, size = point_size) +
  facet_wrap(~model, nrow = 1) +
  scale_fill_gradient2(low = skill_negative_color, mid = skill_zero_color,
                    high = skill_positive_color, midpoint = 0,
                    breaks = breaks, limits = skill_limits,
                    labels = function(x) parse(text = formatC(x, digits = 0, format = "f")),
                    name = "RMSE skill score difference from persistence (%)", na.value = "white",
                    guide = guide_colorbar(
                                             title.position = "top", title.hjust = 0.5,
                                             barwidth = unit(legend_width_mm, "mm"),
                                             barheight = unit(legend_height_mm, "mm"))) +
  coord_sf(xlim = c(-180, 180), ylim = c(-90, 90), expand = FALSE, clip = "off") +
  theme_bw(base_family = font_family, base_size = text_size_pt) +
  theme(
    text = element_text(family = font_family, size = text_size_pt),
    axis.text = element_blank(), axis.title = element_blank(), axis.ticks = element_blank(),
    panel.grid = element_blank(),
    panel.border = element_rect(fill = NA, colour = "black", linewidth = line_width_data),
    panel.spacing = unit(panel_spacing_mm, "mm"),
    strip.text = element_text(size = text_size_pt),
    panel.background = element_rect(fill = "transparent", colour = NA),
    plot.background = element_rect(fill = "transparent", colour = NA),
    plot.title = element_blank(), plot.subtitle = element_blank(),
    legend.position = "bottom", legend.direction = "horizontal",
    legend.title = element_text(size = text_size_pt),
    legend.text = element_text(size = text_size_pt),
    legend.margin = margin(0, 0, 0, 0, "mm"),
    legend.box.margin = margin(0, 0, 0, 0, "mm"),
    legend.background = element_rect(fill = "transparent", colour = NA),
    legend.box.spacing = unit(legend_axis_gap_mm, "mm"),
    plot.margin = margin(plot_margin_mm[1], plot_margin_mm[2],
                         plot_margin_mm[3], plot_margin_mm[4], "mm")
  )
ggsave(output, p, width = width_mm, height = height_mm, units = "mm",
       device = grDevices::pdf, bg = "transparent")
message("Saved: ", output)
