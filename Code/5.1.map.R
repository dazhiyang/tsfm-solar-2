# Project Data/bsrn_1min stations; lat/lon/kgc from bsrn.constants.BSRN_STATIONS
# (editable install: .../bsrn-qc/src/bsrn/constants.py). Legend colors = Beck RGB.
# Project map layout: lowercase station labels and a 160-mm canvas.

suppressPackageStartupMessages({
  library(ggplot2)
  library(ggrepel)
  library(sf)
  library(readr)
  library(dplyr)
  library(rnaturalearth)
})

# --- Editable publication parameters ---
plot.size <- 8
line.size <- 0.2
point.size <- 0.05
legend.size <- 0.4
text.size <- plot.size * 5 / 14
width_mm <- 160
height_mm <- 90
legend_nrow <- 2L
# Reserve a separate row below the map so the legend cannot cover stations.
legend_position <- "bottom"
legend_justification <- "center"
legend_gap_mm <- 2
ggsave_scale <- 1
ggsave_dpi <- 300
text_size_pt <- plot.size
viridis_family <- "viridis"
viridis_quantile_note <- "unused (discrete Köppen classes on map)"
wong_palette <- c(
  "#E69F00", "#56B4E9", "#009E73", "#CC79A7",
  "#D55E00", "#F0E442", "#0072B2", "#000000"
)

script_path <- sub("^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
if (length(script_path)) {
  project_root <- normalizePath(file.path(dirname(script_path), ".."))
} else {
  project_root <- normalizePath(getwd())
}

python_bin <- Sys.getenv("PYTHON", "python3")
# Parent of the bsrn package (contains bsrn/constants.py with BSRN_STATIONS).
bsrn_package_src <- Sys.getenv(
  "BSRN_PACKAGE_SRC",
  ""
)
bsrn_constants_py <- file.path(bsrn_package_src, "bsrn", "constants.py")
data_root <- file.path(project_root, "Data", "bsrn_1min")
out_dir <- file.path(project_root, "tex")
out_pdf <- file.path(out_dir, "map_bsrn_koppen.pdf")

load_map_stations <- function(data_root, package_src, constants_py, python) {
  if (!dir.exists(data_root)) {
    stop("Project station data directory not found: ", data_root)
  }
  if (nzchar(package_src) && !file.exists(constants_py)) {
    stop("BSRN constants not found: ", constants_py)
  }
  py <- tempfile(fileext = ".py")
  writeLines(
    c(
      "import csv, io, os, sys",
      "from pathlib import Path",
      "sys.path.insert(0, os.environ['BSRN_PACKAGE_SRC'])",
      "from bsrn.constants import BSRN_STATIONS",
      "root = Path(os.environ['BSRN_MAP_DATA_ROOT'])",
      "codes = sorted({path.parent.name.upper() for path in root.glob('*/*.parquet')})",
      "if not codes:",
      "    raise ValueError(f'No station-year parquet files found in {root}')",
      "buf = io.StringIO()",
      "writer = csv.writer(buf)",
      "writer.writerow(['stn', 'lat', 'lon', 'kgc'])",
      "for code in codes:",
      "    meta = BSRN_STATIONS[code]",
      "    writer.writerow([code.lower(), meta['lat'], meta['lon'], meta['kgc']])",
      "sys.stdout.write(buf.getvalue())"
    ),
    py
  )
  on.exit(unlink(py), add = TRUE)
  old_env <- Sys.getenv(c("BSRN_PACKAGE_SRC", "BSRN_MAP_DATA_ROOT"), unset = NA)
  Sys.setenv(
    BSRN_PACKAGE_SRC = package_src,
    BSRN_MAP_DATA_ROOT = data_root
  )
  on.exit(
    {
      if (is.na(old_env["BSRN_PACKAGE_SRC"])) {
        Sys.unsetenv("BSRN_PACKAGE_SRC")
      } else {
        Sys.setenv(BSRN_PACKAGE_SRC = old_env["BSRN_PACKAGE_SRC"])
      }
      if (is.na(old_env["BSRN_MAP_DATA_ROOT"])) {
        Sys.unsetenv("BSRN_MAP_DATA_ROOT")
      } else {
        Sys.setenv(BSRN_MAP_DATA_ROOT = old_env["BSRN_MAP_DATA_ROOT"])
      }
    },
    add = TRUE
  )
  raw <- system2(python, c("-B", py), stdout = TRUE, stderr = TRUE)
  if (length(attr(raw, "status")) && attr(raw, "status") != 0) {
    stop(paste(c("Failed to read bsrn.constants:", raw), collapse = "\n"))
  }
  readr::read_csv(I(paste(raw, collapse = "\n")), show_col_types = FALSE)
}

kgc_legend <- tibble::tribble(
  ~id, ~symbol, ~hex,
  1L, "Af", "#0000FF",
  2L, "Am", "#0078FF",
  3L, "Aw", "#46AAFA",
  4L, "BWh", "#FF0000",
  5L, "BWk", "#FF9696",
  6L, "BSh", "#F5A500",
  7L, "BSk", "#FFDC64",
  8L, "Csa", "#FFFF00",
  9L, "Csb", "#C8C800",
  10L, "Csc", "#969600",
  11L, "Cwa", "#96FF96",
  12L, "Cwb", "#64C864",
  13L, "Cwc", "#329632",
  14L, "Cfa", "#C8FF50",
  15L, "Cfb", "#64FF50",
  16L, "Cfc", "#32C800",
  17L, "Dsa", "#FF00FF",
  18L, "Dsb", "#C800C8",
  19L, "Dsc", "#963296",
  20L, "Dsd", "#966496",
  21L, "Dwa", "#AAF0FF",
  22L, "Dwb", "#5A78DC",
  23L, "Dwc", "#4B50B4",
  24L, "Dwd", "#320087",
  25L, "Dfa", "#00FFFF",
  26L, "Dfb", "#37C8FF",
  27L, "Dfc", "#007D7D",
  28L, "Dfd", "#00465F",
  29L, "ET", "#B2B2B2",
  30L, "EF", "#666666"
)

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

bsrn <- load_map_stations(data_root, bsrn_package_src, bsrn_constants_py, python_bin) %>%
  dplyr::rename(symbol = kgc)
stopifnot(
  !anyDuplicated(bsrn$stn),
  all(is.finite(bsrn$lat) & abs(bsrn$lat) <= 90),
  all(is.finite(bsrn$lon) & abs(bsrn$lon) <= 180),
  all(bsrn$symbol %in% kgc_legend$symbol)
)
message("Mapping ", nrow(bsrn), " project data stations from ", data_root)

coastlines_sim2_df <- rnaturalearth::ne_countries(scale = "medium", returnclass = "sf")

kgc_present <- sort(unique(bsrn$symbol))
kgc_colors <- stats::setNames(kgc_legend$hex, kgc_legend$symbol)

p <- ggplot() +
  geom_sf(data = coastlines_sim2_df, fill = "gray80", color = "gray60", linewidth = line.size) +
  geom_point(
    data = bsrn,
    aes(lon, lat, color = symbol),
    size = point.size * 50
  ) +
  ggrepel::geom_text_repel(
    data = bsrn,
    aes(lon, lat, label = stn),
    family = "Times",
    max.overlaps = 20,
    size = text.size,
    segment.size = line.size,
    point.padding = 0,
    force = 0.5
  ) +
  labs(color = "KGC") +
  scale_color_manual(values = kgc_colors, breaks = kgc_present) +
  coord_sf(xlim = c(-180, 180), ylim = c(-90, 90), expand = FALSE, clip = "off") +
  theme_bw() +
  theme(
    plot.margin = unit(c(0.1, 0.1, 0.1, 0), "lines"),
    panel.spacing = unit(0.02, "lines"),
    panel.grid = element_blank(),
    panel.border = element_blank(),
    plot.title = element_blank(),
    plot.subtitle = element_blank(),
    text = element_text(family = "Times", size = plot.size),
    axis.ticks = element_blank(),
    axis.text = element_blank(),
    axis.title = element_blank(),
    legend.direction = "horizontal",
    legend.text = element_text(family = "Times", size = plot.size),
    legend.key.height = unit(0.2, "lines"),
    legend.key.width = unit(0.4, "lines"),
    legend.box.background = element_rect(fill = "transparent", color = "transparent"),
    legend.background = element_rect(fill = "transparent", colour = "transparent"),
    legend.position = legend_position,
    legend.justification = legend_justification,
    legend.title = element_text(family = "Times", size = plot.size),
    legend.box.margin = margin(0, 0, 0, 0, unit = "mm"),
    legend.margin = margin(0, 0, 0, 0, unit = "mm"),
    legend.box.spacing = unit(legend_gap_mm, "mm"),
    legend.spacing.x = unit(0.15, "lines"),
    legend.spacing.y = unit(0.08, "lines"),
    plot.background = element_rect(fill = "transparent", colour = "transparent")
  ) +
  guides(
    color = guide_legend(
      nrow = legend_nrow,
      byrow = TRUE,
      override.aes = list(size = 3)
    )
  )

ggsave(
  filename = out_pdf,
  plot = p,
  scale = ggsave_scale,
  width = width_mm,
  height = height_mm,
  unit = "mm",
  dpi = ggsave_dpi
)
message("Wrote ", out_pdf)
