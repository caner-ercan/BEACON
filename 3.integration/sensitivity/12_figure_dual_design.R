# =====================================================================
# 12 - Dual-design robustness figure: nuclear + immune-ecology metrics,
#      inter-slide vs intra-slide, faceted exactly as CE specified:
#
#        nuclear features:            immune ecological features:
#          - Inter-slide                - Inter-slide
#          - Intra-slide                 - Intra-slide
#
# No reference to Fig. 3D / Supp. Fig. 1 anywhere in the plot (title,
# subtitle, facet labels, caption) - CE's explicit instruction. Effect
# sign is uniform across both designs: positive = higher in DNA-content
# abnormal (pred==1 slides for inter; attn1 regions for intra) - see the
# sign-convention comment in 11_dual_design_associations.R.
# =====================================================================

here <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
if (is.na(here) || !nzchar(here)) here <- "."
source(file.path(here, "00_config.R")); source(file.path(here, "lib_sweep.R"))
suppressPackageStartupMessages({ library(ggplot2) })

res <- readRDS(file.path(OUT_DIR, "dual_design_associations.RDS")) %>% filter(is.finite(effect))

d <- res %>%
  mutate(sig = ifelse(!is.na(p) & p < 0.05, "p < 0.05", "n.s."),
         feature_panel = ifelse(feature_type == "nuclear", "Nuclear features", "Immune ecological features"),
         design_panel  = ifelse(design == "inter", "Inter-slide", "Intra-slide"))
d$feature_panel <- factor(d$feature_panel, levels = c("Nuclear features", "Immune ecological features"))
d$design_panel  <- factor(d$design_panel,  levels = c("Inter-slide", "Intra-slide"))

# Colour by ordinal rank within each row's own grid (nuclear rows have a
# single "published" setting only, so they get one colour throughout -
# consistent with the design used for the plasma/progression figure).
d <- d %>%
  group_by(label, design) %>%
  # rank() on a single-row group just returns 1, so this works uniformly
  # for both the 1-point nuclear rows and the 3-point ecology rows.
  mutate(n_set = n(),
         setting_rank = factor(rank(ifelse(is.na(param_value), -Inf, param_value), ties.method = "first"),
                               levels = 1:3, labels = c("Setting 1 (narrowest/smallest)", "Setting 2",
                                                        "Setting 3 (widest/largest)"))) %>%
  ungroup()

# Order rows: nuclear metrics first (by median effect), then ecology,
# so each feature-type facet reads top-to-bottom by effect strength.
ord <- d %>% group_by(label, feature_type) %>% summarise(m = median(effect), .groups = "drop") %>%
  arrange(feature_type == "ecology", m)
d$label <- factor(d$label, levels = ord$label)

p <- ggplot(d, aes(x = effect, y = label)) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey40") +
  geom_line(aes(group = label), colour = "grey70") +
  # Halo ring marks the PUBLISHED/default setting. Drawn first, underneath.
  # (Previously size was mapped to `n_set > 1`, which encodes "this row has a
  # grid" - not "this is the published point" - so every ecology point looked
  # identical and the published setting was unreadable. That is what made the
  # off-published n.s. triangles look like they were the published values.)
  geom_point(data = d %>% filter(is_published), shape = 21, size = 5.6,
             fill = NA, colour = "grey25", stroke = 0.45) +
  geom_point(aes(shape = sig, fill = setting_rank), size = 3.1,
             colour = "black", stroke = 0.3) +
  scale_shape_manual(values = c("p < 0.05" = 21, "n.s." = 24)) +
  scale_fill_manual(values = c("Setting 1 (narrowest/smallest)" = "#4C72B0",
                               "Setting 2" = "#DD8452",
                               "Setting 3 (widest/largest)" = "#C44E52")) +
  # fill/shape are mapped to different aesthetics, so the fill legend would
  # otherwise draw with the unfillable default shape 19 and render all three
  # setting keys solid black.
  guides(fill  = guide_legend(override.aes = list(shape = 21, size = 3.4)),
         shape = guide_legend(override.aes = list(fill = "grey70", size = 3.4))) +
  facet_grid(feature_panel ~ design_panel, scales = "free_y", space = "free_y") +
  labs(x = "Effect size (rank-biserial r; positive = higher in DNA-content-abnormal)",
       y = NULL, fill = NULL, shape = NULL,
       title = "Robustness of nuclear and immune-ecological findings: inter-slide vs. intra-slide",
       caption = paste(
         "Inter-slide: DNA-content abnormal vs. normal slides (all slides, unpaired). Intra-slide:",
         "abnormal vs. normal regions within the\nsame slide, restricted to DNA-content-abnormal",
         "slides (paired). The published setting is ringed in grey; nuclear features have no\nspatial",
         "parameter and are shown as a single (published) point. Colour = rank within each row's own",
         "grid\n(narrowest/smallest to widest/largest), not a shared scale across rows.",
         "Point shape indicates nominal significance\n(p < 0.05, two-sided Wilcoxon). Read for",
         "stability of sign and magnitude, not significance at every setting.\n",
         "Ecology grids - Moran's I / Getis-Ord Gi* neighbourhood: 1/2/3 tile rings (200/350/500 px).",
         "Ripley's L integration range:\nper-biopsy default / 100 um / 200 um.")) +
  theme_bw(base_size = 9) +
  theme(legend.position = "bottom", plot.caption = element_text(hjust = 0, size = 6.8),
        strip.text = element_text(size = 8, face = "bold"))

nrows <- n_distinct(d$label)
out <- file.path(OUT_DIR, "fig", "SuppFig_dual_design_robustness.pdf")
ggsave(out, p, width = 10, height = max(4, 0.4 * nrows + 2.5), limitsize = FALSE)
message(sprintf("[done] %s (%d metrics x 2 designs)", out, nrows))

# ---- cross-design consistency check: does sign hold BETWEEN inter and
# intra for the same metric? (the within-design "sign_consistent" from
# 11's stability output only covers the parameter grid, not this.) -----
cross <- res %>% filter(is_published | feature_type == "nuclear") %>%
  select(label, feature_type, design, effect, p) %>%
  tidyr::pivot_wider(names_from = design, values_from = c(effect, p)) %>%
  mutate(cross_design_sign_consistent = sign(effect_inter) == sign(effect_intra))
write.csv(cross, file.path(OUT_DIR, "dual_design_cross_check.csv"), row.names = FALSE)
message("\n=== cross-design (inter vs intra) sign check, at published setting ===")
print(as.data.frame(cross %>% select(label, feature_type, effect_inter, effect_intra,
                                     cross_design_sign_consistent)), row.names = FALSE)

tab <- d %>% select(label, feature_type, design, param_label, param_value, n1, n0, effect, p) %>%
  arrange(feature_type, label, design, param_value)
write.csv(tab, file.path(OUT_DIR, "SuppTable_dual_design_robustness.csv"), row.names = FALSE)
message("[done] SuppTable_dual_design_robustness.csv")
