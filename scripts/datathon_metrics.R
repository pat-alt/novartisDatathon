cwd <- "../datathon2020/"

#### Metrics to be computed ####

# Custom uncertainty metric (forecast intervals metric)
UncertaintyMetric <- function(actuals, upper, lower, avgVolume) {

  avgVolume <- mean(avgVolume)

  uncertainty_first6 <- (0.85 * sum(abs(upper[1:6] - lower[1:6])) + # Wide intervals are penalized
                           0.15 * 2 / 0.05 * (
                             # If it's outside, it adds error
                             sum((lower[1:6] - actuals[1:6]) * (actuals[1:6] < lower[1:6])) + 
                               sum((actuals[1:6] - upper[1:6]) * (actuals[1:6] > upper[1:6])))
                         ) / (6 * avgVolume) * 100

  uncertainty_last18 <- (0.85 * sum(abs(upper[7:24] - lower[7:24])) +
                           0.15 * 2 / 0.05 * (
                             sum((lower[7:24] - actuals[7:24]) * (actuals[7:24] < lower[7:24])) +
                               sum((actuals[7:24] - upper[7:24]) * (actuals[7:24] > upper[7:24])))
                         ) / (18 * avgVolume) * 100

  return(0.6 * uncertainty_first6 + 0.4 * uncertainty_last18)
}

AccuracyMetric <- function(actuals, forecast, avgVolume) {

  if ((length(actuals) != 24 | length(forecast) != 24)) stop("actuals and forecast should have the same length (24).")

  avgVolume <- mean(avgVolume)

  custom.mape <- sum(abs(actuals - forecast)) / (24 * avgVolume) * 100
  six.month.mape <- abs(sum(actuals[1:6]) - sum(forecast[1:6])) / (6 * avgVolume) * 100
  twelve.month.mape <- abs(sum(actuals[7:12]) - sum(forecast[7:12])) / (6 * avgVolume) * 100
  twelve.last.month.mape <- abs(sum(actuals[13:24]) - sum(forecast[13:24])) / (12 * avgVolume) * 100

  # Compute the custom metric
  custom.metric <- 0.5 * custom.mape + 0.3 * six.month.mape + 0.1 * (twelve.month.mape + twelve.last.month.mape)

  return(custom.metric)
}

### How to apply: mock dataset
df.mock <- data.table::fread(paste0(cwd, "/data/gx_volume.csv"))

# We will make up forecasts and the average_last_12_mo for country brands
df.mock[, ":="(forecast = volume,
               lower_bound = volume * 0.9,
               upper_bound = volume * 1.1,
               avg_last = max(volume)),
        by = c("country", "brand")]

# Take 2 examples to display. Remember that they must have 24 months (0 to 23) post generic
df.metric <- df.mock[
  (
    (country == "country_8" & brand == "brand_117") | (country == "country_7" & brand == "brand_5")
    ) & month_num >= 0 & month_num < 24
  , ]

## Compute metric per country-brand pair (it will be the same for all country-brand values)
data.table::setDT(df.metric)[, ":="(accuracy = AccuracyMetric(volume, forecast, avg_last),
                                    uncertainty = UncertaintyMetric(volume, upper_bound, lower_bound, avg_last)), c("country", "brand")]

# Reduce the metric to one row per country-brand
metrics.result <- df.metric[, .(accuracy_metric = mean(accuracy),
                                 uncertainty_metric = mean(uncertainty)),
                          .(country, brand)]

# Get global metric, that's the mean by the whole dataset
metrics.global <- metrics.result[, .(accuracy_metric = mean(accuracy_metric),
                              uncertainty_metric = mean(uncertainty_metric))]

