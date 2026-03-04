#!/usr/bin/env Rscript
# Plot scaling behavior of eigensubstitution accumulation backends.
# Reads benchmarks/results/*.json, writes benchmarks/figures/*.pdf.

library(jsonlite)
library(ggplot2)
library(dplyr)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

# Determine script directory
script_dir <- "."
args <- commandArgs(trailingOnly = FALSE)
script_arg <- grep("--file=", args, value = TRUE)
if (length(script_arg) > 0) {
  script_dir <- dirname(sub("--file=", "", script_arg[1]))
}
results_dir <- file.path(script_dir, "results")
figures_dir <- file.path(script_dir, "figures")

dir.create(figures_dir, showWarnings = FALSE, recursive = TRUE)

json_files <- list.files(results_dir, pattern = "\\.json$", full.names = TRUE)
if (length(json_files) == 0) {
  stop("No result files found in ", results_dir)
}

all_data <- do.call(rbind, lapply(json_files, function(f) {
  raw <- fromJSON(f)
  df <- as.data.frame(raw$results, stringsAsFactors = FALSE)
  df$hardware_id <- raw$hardware_id
  df
}))

# Ensure numeric types
all_data$A <- as.integer(all_data$A)
all_data$C <- as.integer(all_data$C)
all_data$R <- as.integer(all_data$R)
all_data$mean_seconds <- as.numeric(all_data$mean_seconds)
all_data$std_seconds <- as.numeric(all_data$std_seconds)

# Nicer backend labels
all_data$Backend <- factor(all_data$backend,
  levels = c("jax_cpu", "jax_gpu", "oracle", "rust_native"),
  labels = c("JAX (CPU)", "JAX (GPU)", "Oracle (NumPy)", "Rust (native)")
)

cat("Loaded", nrow(all_data), "records from", length(json_files), "file(s)\n")

# R may mangle "function" to "function." in data.frame
fn_col <- if ("function." %in% names(all_data)) "function." else "function"

# Common theme
theme_bench <- theme_minimal(base_size = 11) +
  theme(
    legend.position = "bottom",
    strip.text = element_text(face = "bold"),
    plot.title = element_text(size = 13, face = "bold")
  )

# ---------------------------------------------------------------------------
# Plot 1: Scaling by alignment width C
# ---------------------------------------------------------------------------

for (func in unique(all_data[[fn_col]])) {
  df <- all_data[all_data[[fn_col]] == func, ]
  if (nrow(df) == 0) next

  p <- ggplot(df, aes(x = C, y = mean_seconds, color = Backend, shape = Backend)) +
    geom_line(size = 0.7) +
    geom_point(size = 2.5) +
    geom_errorbar(aes(ymin = pmax(mean_seconds - std_seconds, 1e-6),
                      ymax = mean_seconds + std_seconds),
                  width = 0.05, size = 0.4) +
    facet_grid(R ~ A, labeller = label_both, scales = "free_y") +
    scale_x_log10() +
    scale_y_log10() +
    labs(
      title = paste0(func, "(): scaling by alignment width C"),
      x = "Alignment width C (columns)",
      y = "Time (seconds)"
    ) +
    theme_bench

  fname <- paste0("scaling_C_", func, ".pdf")
  ggsave(file.path(figures_dir, fname), p, width = 10, height = 8)
  cat("Wrote", fname, "\n")
}

# ---------------------------------------------------------------------------
# Plot 2: Scaling by alphabet size A
# ---------------------------------------------------------------------------

for (func in unique(all_data[[fn_col]])) {
  df <- all_data[all_data[[fn_col]] == func, ]
  if (nrow(df) == 0) next

  p <- ggplot(df, aes(x = A, y = mean_seconds, color = Backend, shape = Backend)) +
    geom_line(size = 0.7) +
    geom_point(size = 2.5) +
    geom_errorbar(aes(ymin = pmax(mean_seconds - std_seconds, 1e-6),
                      ymax = mean_seconds + std_seconds),
                  width = 0.05, size = 0.4) +
    facet_grid(R ~ C, labeller = label_both, scales = "free_y") +
    scale_x_log10() +
    scale_y_log10() +
    labs(
      title = paste0(func, "(): scaling by alphabet size A"),
      x = "Alphabet size A",
      y = "Time (seconds)"
    ) +
    theme_bench

  fname <- paste0("scaling_A_", func, ".pdf")
  ggsave(file.path(figures_dir, fname), p, width = 10, height = 8)
  cat("Wrote", fname, "\n")
}

# ---------------------------------------------------------------------------
# Plot 3: Scaling by tree size R
# ---------------------------------------------------------------------------

for (func in unique(all_data[[fn_col]])) {
  df <- all_data[all_data[[fn_col]] == func, ]
  if (nrow(df) == 0) next

  p <- ggplot(df, aes(x = R, y = mean_seconds, color = Backend, shape = Backend)) +
    geom_line(size = 0.7) +
    geom_point(size = 2.5) +
    geom_errorbar(aes(ymin = pmax(mean_seconds - std_seconds, 1e-6),
                      ymax = mean_seconds + std_seconds),
                  width = 0.05, size = 0.4) +
    facet_grid(A ~ C, labeller = label_both, scales = "free_y") +
    scale_x_log10() +
    scale_y_log10() +
    labs(
      title = paste0(func, "(): scaling by tree size R"),
      x = "Tree size R (nodes)",
      y = "Time (seconds)"
    ) +
    theme_bench

  fname <- paste0("scaling_R_", func, ".pdf")
  ggsave(file.path(figures_dir, fname), p, width = 10, height = 8)
  cat("Wrote", fname, "\n")
}

# ---------------------------------------------------------------------------
# Plot 4: Backend comparison (relative to JAX CPU)
# ---------------------------------------------------------------------------

for (func in unique(all_data[[fn_col]])) {
  df <- all_data[all_data[[fn_col]] == func, ]
  if (nrow(df) == 0) next

  # Pivot to get JAX CPU as baseline
  jax_cpu <- df[df$backend == "jax_cpu", c("A", "C", "R", "mean_seconds")]
  names(jax_cpu)[4] <- "jax_cpu_time"
  df_ratio <- merge(df, jax_cpu, by = c("A", "C", "R"))
  df_ratio$ratio <- df_ratio$mean_seconds / df_ratio$jax_cpu_time
  df_ratio$config <- paste0("A=", df_ratio$A, " C=", df_ratio$C)

  if (nrow(df_ratio) == 0) next

  p <- ggplot(df_ratio, aes(x = config, y = ratio, fill = Backend)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.7) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "grey40") +
    facet_wrap(~ R, labeller = label_both) +
    scale_y_log10() +
    labs(
      title = paste0(func, "(): time relative to JAX CPU"),
      x = "Configuration",
      y = "Ratio (vs JAX CPU)"
    ) +
    theme_bench +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 7))

  fname <- paste0("backend_comparison_", func, ".pdf")
  ggsave(file.path(figures_dir, fname), p, width = 12, height = 6)
  cat("Wrote", fname, "\n")
}

cat("All plots written to", figures_dir, "\n")
