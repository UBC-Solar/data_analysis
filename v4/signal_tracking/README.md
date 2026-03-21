# Signal Tracking

This project scrapes InfluxDB to find which days there are data available for, and finds the earliest and latest day that each telemetry signal (`TotalPackVoltage`, `InputCurrentA`) is available for.
This was used to develop the `localization.toml` files in `data_tools/localization` for tracking when telemetry names become available and are deprecated. 

## Results

The `collect_signals.ipynb` notebook is where all the analysis was performed. The notebook was compiled to a PDF, `results/collect_signals.pdf` and the day-by-day coverage plot was exported to `results/coverage.png`.

## Future Use

You'll need to update the maximum day to search for, which was hardcoded to February 28th, 2026. Other than that, it should work fine!
