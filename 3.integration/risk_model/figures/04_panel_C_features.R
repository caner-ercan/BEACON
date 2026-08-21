# =====================================================================
# Panel C - the LASSO model's selected features, ordered by |coefficient|
# (matches the published feature_plot.Rmd's style/palette exactly:
# steelblue/coral fill by sign), with two additions:
#   - bar outline: solid black if the feature is ALSO retained by the
#     alpha=0.5 elastic net, none if LASSO-only
#   - a dagger (dagger) suffix on stroma-related feature labels
# This is the figure behind "25 of the 28 LASSO features were also
# retained by the elastic net" and the R1#11 stroma answer.
# =====================================================================
here <- tryCatch(dirname(normalizePath(sys.frame(1)$ofile)), error = function(e) ".")
if (!file.exists(file.path(here, "00_config.R"))) here <- "BEACON/4.plotting/code/rev1_elastic_net_fig"
source(file.path(here, "00_config.R"))
suppressPackageStartupMessages({library(ggplot2); library(dplyr); library(forcats)})

fits <- readRDS(file.path(OUT_FIG, "fits_lasso_en.RDS"))
L <- fits$lasso$coefficients
E <- fits$en$coefficients

df <- data.frame(Variable = names(L), Coefficient = as.numeric(L), stringsAsFactors = FALSE) %>%
  mutate(
    Importance = abs(Coefficient),
    Sign = ifelse(Coefficient > 0, "Positive", "Negative"),
    in_EN = Variable %in% names(E),
    is_stroma = grepl("Stroma", Variable, ignore.case = TRUE),
    # ASCII marker, not a unicode dagger: R's default pdf() device silently
    # replaces unmappable glyphs (dagger included) with "...", which
    # corrupted exactly the stroma-flagged rows when this used "†".
    Label = ifelse(is_stroma, paste0(Variable, " *"), Variable),
    Label = fct_reorder(Label, Importance)
  )

n_shared <- sum(df$in_EN)
cat(sprintf("LASSO features: %d | also in EN(alpha=0.5): %d | stroma-related: %d\n",
            nrow(df), n_shared, sum(df$is_stroma)))

p <- ggplot(df, aes(x = Importance, y = Label, fill = Sign, color = in_EN)) +
  geom_col(linewidth = 0.9) +
  scale_fill_manual(values = PAL$coef_sign) +
  scale_color_manual(values = c(`TRUE` = "black", `FALSE` = NA),
                      labels = c(`TRUE` = "retained by Elastic Net (alpha=0.5)",
                                 `FALSE` = "LASSO only"),
                      name = NULL) +
  labs(
    title = "LASSO-selected features for risk stratification",
    subtitle = sprintf("%d of %d features (%.0f%%) also retained by Elastic Net (alpha=0.5); * stroma-related",
                       n_shared, nrow(df), 100 * n_shared / nrow(df)),
    x = "Absolute coefficient value", y = NULL
  ) +
  guides(fill = "none") +
  theme_minimal() +
  theme(panel.grid = element_blank(), panel.border = element_blank(),
        axis.line = element_line(size = 0.5),
        axis.text.y = element_text(size = 8),
        plot.subtitle = element_text(size = 10),
        legend.position = "bottom")

ggsave(file.path(OUT_FIG, "figS_C_feature_comparison.pdf"), p, height = 9, width = 10)
write.csv(df %>% select(Variable, Coefficient, Sign, in_EN, is_stroma),
          file.path(OUT_FIG, "figS_C_feature_comparison_data.csv"), row.names = FALSE)
cat("wrote figS_C_feature_comparison.pdf\n")
