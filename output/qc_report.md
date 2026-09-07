# UKBMS data cleaning / QC report

## Assumptions made by this pipeline

1. **Day fields are "days after 1 April", not day-of-year 1-366** (per `ukbms_phenology_2024.docx`). Negatives (before 1 April) and values > 182 (after 30 Sept) are legitimate; plausibility checks use this scale.
2. **Species join on scientific name** (SPECIES_NAME == SPECIES), not common name, since phenology has no species code. Verified 1:1 and lossless across all 60 species.
3. **SITE_INDEX == -2 is a documented sentinel** (insufficient monitoring), not a real abundance. Kept as -2 and flagged via `SITE_INDEX_INSUFFICIENT_MONITORING`; never imputed.
4. **Exact duplicate rows are collapsed** (logged, not silent). Conflicting KEYS (different SITE_INDEX for the same site/species/year) are never auto-resolved: one row is kept, SITE_INDEX set NaN, and `SITE_INDEX_AMBIGUOUS_CONFLICT=True` so it stays visible.
5. **GB grid bounds checked only for England/Scotland/Wales.** NI (Irish Grid) and Channel Islands (local grid, negative Northing) are valid systems, reported as "out of GB-grid scope" rather than invalid.
6. **species_trends / collated_indices attach as species- / national-level columns only**, left-joined on COMMON_NAME and (SPECIES_NAME, YEAR, COUNTRY), never multiplying the grain. Channel Islands/Isle of Man rows get NaN national columns by design (no matching national entry).
7. **No destructive filtering by default** beyond #4: all other CONFIG['filters'] are False, so the saved table is the full flagged table unless a toggle is flipped.

## 1. File profiles



### Profile: phenology

- Shape: 648,886 rows x 13 columns
- Columns: SITENO, SITENAME, GRIDREF, SPECIES_NAME, COMMON_NAME, YEAR, FIRSTDAY, LASTDAY, PEAKDAY, PEAKCOUNT, MEAN_FLIGHT_DATE, FLIGHTPERIOD_SD, FLIGHTPERIOD_RANGE

#### Dtypes

| column             | dtype   |
|:-------------------|:--------|
| SITENO             | int64   |
| SITENAME           | str     |
| GRIDREF            | str     |
| SPECIES_NAME       | str     |
| COMMON_NAME        | str     |
| YEAR               | int64   |
| FIRSTDAY           | int64   |
| LASTDAY            | int64   |
| PEAKDAY            | int64   |
| PEAKCOUNT          | int64   |
| MEAN_FLIGHT_DATE   | float64 |
| FLIGHTPERIOD_SD    | float64 |
| FLIGHTPERIOD_RANGE | int64   |

#### Null counts (columns with >0 nulls)

| column   |   n_null |   pct_null |
|:---------|---------:|-----------:|
| GRIDREF  |      130 |       0.02 |

#### N unique per column

| column             |   n_unique |
|:-------------------|-----------:|
| SITENO             |       3974 |
| SITENAME           |       3934 |
| GRIDREF            |       3868 |
| SPECIES_NAME       |         60 |
| COMMON_NAME        |         60 |
| YEAR               |         52 |
| FIRSTDAY           |        238 |
| LASTDAY            |        234 |
| PEAKDAY            |        237 |
| PEAKCOUNT          |        922 |
| MEAN_FLIGHT_DATE   |      15599 |
| FLIGHTPERIOD_SD    |       7445 |
| FLIGHTPERIOD_RANGE |        227 |

#### Head (first 3 rows)

|   SITENO | SITENAME        | GRIDREF   | SPECIES_NAME          | COMMON_NAME         |   YEAR |   FIRSTDAY |   LASTDAY |   PEAKDAY |   PEAKCOUNT |   MEAN_FLIGHT_DATE |   FLIGHTPERIOD_SD |   FLIGHTPERIOD_RANGE |
|---------:|:----------------|:----------|:----------------------|:--------------------|-------:|-----------:|----------:|----------:|------------:|-------------------:|------------------:|---------------------:|
|        1 | Woodwalton Farm | TL214817  | Aglais io             | Peacock             |   1976 |         20 |       180 |       104 |           7 |              85.67 |             42.89 |                  160 |
|        1 | Woodwalton Farm | TL214817  | Aglais urticae        | Small Tortoiseshell |   1976 |         20 |       145 |        93 |           5 |              79.69 |             37.52 |                  125 |
|        1 | Woodwalton Farm | TL214817  | Aphantopus hyperantus | Ringlet             |   1976 |         93 |       104 |        93 |          15 |              96.14 |              4.97 |                   11 |

#### Describe (numeric columns)

|                    |   count |      mean |       std |   min |     25% |     50% |     75% |    max |
|:-------------------|--------:|----------:|----------:|------:|--------:|--------:|--------:|-------:|
| SITENO             |  648886 | 2353.05   | 1473.78   |     1 | 1322    | 2254    | 3341    | 9010   |
| YEAR               |  648886 | 2012.37   |   10.2098 |  1973 | 2007    | 2015    | 2020    | 2024   |
| FIRSTDAY           |  648886 |   68.4934 |   39.5374 |   -30 |   36    |   72    |   98    |  215   |
| LASTDAY            |  648886 |  129.98   |   38.5512 |   -24 |  111    |  136    |  159    |  215   |
| PEAKDAY            |  648886 |   97.3029 |   40.1738 |   -26 |   73    |  105    |  124    |  215   |
| PEAKCOUNT          |  648886 |   18.1588 |   48.1575 |     1 |    2    |    5    |   15    | 5583   |
| MEAN_FLIGHT_DATE   |  648886 |  100.287  |   31.7323 |   -24 |   85.4  |  105.5  |  121    |  215   |
| FLIGHTPERIOD_SD    |  648886 |   18.6377 |   16.249  |     0 |    6.18 |   13.11 |   31.24 |  104.5 |
| FLIGHTPERIOD_RANGE |  648886 |   61.4869 |   51.5234 |     0 |   17    |   48    |  105    |  235   |

### Profile: site_indices

- Shape: 821,300 rows x 7 columns
- Columns: SITE_CODE, COUNTRY, SPECIES_CODE, SPECIES, COMMON_NAME, YEAR, SITE_INDEX

#### Dtypes

| column       | dtype   |
|:-------------|:--------|
| SITE_CODE    | int64   |
| COUNTRY      | str     |
| SPECIES_CODE | int64   |
| SPECIES      | str     |
| COMMON_NAME  | str     |
| YEAR         | int64   |
| SITE_INDEX   | int64   |

#### Null counts (columns with >0 nulls)

_none found_

#### N unique per column

| column       |   n_unique |
|:-------------|-----------:|
| SITE_CODE    |       4910 |
| COUNTRY      |          6 |
| SPECIES_CODE |         60 |
| SPECIES      |         60 |
| COMMON_NAME  |         60 |
| YEAR         |         52 |
| SITE_INDEX   |       2891 |

#### Head (first 3 rows)

|   SITE_CODE | COUNTRY   |   SPECIES_CODE | SPECIES   | COMMON_NAME   |   YEAR |   SITE_INDEX |
|------------:|:----------|---------------:|:----------|:--------------|-------:|-------------:|
|           1 | England   |             84 | Aglais io | Peacock       |   1976 |           21 |
|           1 | England   |             84 | Aglais io | Peacock       |   1977 |            7 |
|           1 | England   |             84 | Aglais io | Peacock       |   1978 |            7 |

#### Describe (numeric columns)

|              |   count |      mean |       std |   min |   25% |   50% |   75% |    max |
|:-------------|--------:|----------:|----------:|------:|------:|------:|------:|-------:|
| SITE_CODE    |  821300 | 2315.44   | 1702.26   |     1 |  1208 |  2224 |  3230 |  30008 |
| SPECIES_CODE |  821300 |   72.0524 |   39.2355 |     2 |    29 |    84 |   104 |    123 |
| YEAR         |  821300 | 2010.62   |   11.0786 |  1973 |  2003 |  2014 |  2020 |   2024 |
| SITE_INDEX   |  821300 |   58.7147 | 1027.59   |    -2 |     0 |     4 |    32 | 257446 |

### Profile: site_location

- Shape: 6,082 rows x 12 columns
- Columns: Site_Number, Site_Name, Gridreference, Easting, Northing, Length, Country, N_sections, N_yrs_surveyed, First_year_surveyed, Last_year_surveyed, Survey_type

#### Dtypes

| column              | dtype   |
|:--------------------|:--------|
| Site_Number         | int64   |
| Site_Name           | str     |
| Gridreference       | str     |
| Easting             | int64   |
| Northing            | int64   |
| Length              | float64 |
| Country             | str     |
| N_sections          | float64 |
| N_yrs_surveyed      | int64   |
| First_year_surveyed | int64   |
| Last_year_surveyed  | int64   |
| Survey_type         | str     |

#### Null counts (columns with >0 nulls)

| column     |   n_null |   pct_null |
|:-----------|---------:|-----------:|
| Site_Name  |     2325 |      38.23 |
| Length     |     1174 |      19.3  |
| N_sections |      490 |       8.06 |

#### N unique per column

| column              |   n_unique |
|:--------------------|-----------:|
| Site_Number         |       6082 |
| Site_Name           |       3714 |
| Gridreference       |       5992 |
| Easting             |       2651 |
| Northing            |       3025 |
| Length              |       1633 |
| Country             |          6 |
| N_sections          |         17 |
| N_yrs_surveyed      |         51 |
| First_year_surveyed |         51 |
| Last_year_surveyed  |         45 |
| Survey_type         |          2 |

#### Head (first 3 rows)

|   Site_Number | Site_Name       | Gridreference   |   Easting |   Northing |   Length | Country   |   N_sections |   N_yrs_surveyed |   First_year_surveyed |   Last_year_surveyed | Survey_type   |
|--------------:|:----------------|:----------------|----------:|-----------:|---------:|:----------|-------------:|-----------------:|----------------------:|---------------------:|:--------------|
|             1 | Woodwalton Farm | TL214817        |    521400 |     281700 |     2038 | England   |           12 |               49 |                  1976 |                 2024 | UKBMS         |
|             2 | Bevill's Wood   | TL203794        |    520300 |     279400 |     1532 | England   |           10 |               44 |                  1974 |                 2017 | UKBMS         |
|             3 | Holkham         | TF873453        |    587300 |     345300 |     3454 | England   |            6 |               47 |                  1976 |                 2024 | UKBMS         |

#### Describe (numeric columns)

|                     |   count |         mean |          std |    min |       25% |      50% |      75% |            max |
|:--------------------|--------:|-------------:|-------------:|-------:|----------:|---------:|---------:|---------------:|
| Site_Number         |    6082 |  21500.7     |  23395.1     |      1 |   2620.25 |   4715.5 |  50815.8 |  52377         |
| Easting             |    6082 | 414615       | 115164       |  20059 | 345000    | 423000   | 500775   | 653000         |
| Northing            |    6082 | 285002       | 189485       | -81059 | 145900    | 231000   | 370000   |      1.015e+06 |
| Length              |    4908 |   1989.81    |    927.35    |     60 |   1478.5  |   2000   |   2229   |  12455         |
| N_sections          |    5592 |      8.72425 |      2.92697 |      1 |      7    |     10   |     10   |     20         |
| N_yrs_surveyed      |    6082 |      8.16047 |      8.2546  |      1 |      2    |      5   |     11   |     52         |
| First_year_surveyed |    6082 |   2010.37    |      9.60278 |   1973 |   2007    |   2011   |   2017   |   2024         |
| Last_year_surveyed  |    6082 |   2018.64    |      7.4907  |   1976 |   2014    |   2023   |   2024   |   2024         |

### Profile: species_trends

- Shape: 58 rows x 18 columns
- Columns: COMMON_NAME, SCI_NAME, NYEARS, F_LIN_B, F_LIN_SE, F_LIN_P, F_TRENDDETAIL, F_FULL_R, T20_LIN_B, T20_LIN_SE, T20_LIN_P, T20_TRENDDETAIL, T20_20_R, T10_LIN_B, T10_LIN_SE, T10_LIN_P, T10_TRENDDETAIL, T10_10_R

#### Dtypes

| column          | dtype   |
|:----------------|:--------|
| COMMON_NAME     | str     |
| SCI_NAME        | str     |
| NYEARS          | int64   |
| F_LIN_B         | float64 |
| F_LIN_SE        | float64 |
| F_LIN_P         | float64 |
| F_TRENDDETAIL   | str     |
| F_FULL_R        | float64 |
| T20_LIN_B       | float64 |
| T20_LIN_SE      | float64 |
| T20_LIN_P       | float64 |
| T20_TRENDDETAIL | str     |
| T20_20_R        | float64 |
| T10_LIN_B       | float64 |
| T10_LIN_SE      | float64 |
| T10_LIN_P       | float64 |
| T10_TRENDDETAIL | str     |
| T10_10_R        | float64 |

#### Null counts (columns with >0 nulls)

_none found_

#### N unique per column

| column          |   n_unique |
|:----------------|-----------:|
| COMMON_NAME     |         58 |
| SCI_NAME        |         58 |
| NYEARS          |         12 |
| F_LIN_B         |         58 |
| F_LIN_SE        |         51 |
| F_LIN_P         |         56 |
| F_TRENDDETAIL   |          3 |
| F_FULL_R        |         58 |
| T20_LIN_B       |         58 |
| T20_LIN_SE      |         54 |
| T20_LIN_P       |         58 |
| T20_TRENDDETAIL |          3 |
| T20_20_R        |         57 |
| T10_LIN_B       |         54 |
| T10_LIN_SE      |         50 |
| T10_LIN_P       |         57 |
| T10_TRENDDETAIL |          3 |
| T10_10_R        |         54 |

#### Head (first 3 rows)

| COMMON_NAME      | SCI_NAME        |   NYEARS |   F_LIN_B |   F_LIN_SE |   F_LIN_P | F_TRENDDETAIL   |   F_FULL_R |   T20_LIN_B |   T20_LIN_SE |   T20_LIN_P | T20_TRENDDETAIL   |   T20_20_R |   T10_LIN_B |   T10_LIN_SE |   T10_LIN_P | T10_TRENDDETAIL   |   T10_10_R |
|:-----------------|:----------------|---------:|----------:|-----------:|----------:|:----------------|-----------:|------------:|-------------:|------------:|:------------------|-----------:|------------:|-------------:|------------:|:------------------|-----------:|
| Swallowtail      | Papilio machaon |       48 | -0.000416 |    0.00229 |  0.857    | Stable          |      -4.5  |    -0.0194  |      0.00646 |     0.00766 | Rapid decline     |     -57.22 |     0.00515 |       0.0207 |       0.81  | Stable            |      11.23 |
| Dingy Skipper    | Erynnis tages   |       49 | -0.000904 |    0.00125 |  0.471    | Stable          |      -9.51 |     0.00402 |      0.00439 |     0.371   | Stable            |      19.23 |    -0.0148  |       0.0125 |       0.269 | Stable            |     -26.41 |
| Grizzled Skipper | Pyrgus malvae   |       49 | -0.00608  |    0.00145 |  0.000117 | Rapid decline   |     -48.93 |    -0.00667 |      0.00432 |     0.14    | Stable            |     -25.25 |    -0.00242 |       0.0104 |       0.822 | Stable            |      -4.89 |

#### Describe (numeric columns)

|            |   count |         mean |          std |        min |         25% |        50% |        75% |       max |
|:-----------|--------:|-------------:|-------------:|-----------:|------------:|-----------:|-----------:|----------:|
| NYEARS     |      58 | 45.9138      |   6.60485    |  16        |  46         | 49         | 49         |   49      |
| F_LIN_B    |      58 |  0.000328534 |   0.0110302  |  -0.0219   |  -0.0058575 | -0.00094   |  0.005615  |    0.0339 |
| F_LIN_SE   |      58 |  0.00277309  |   0.00193222 |   0.000901 |   0.0016125 |  0.00208   |  0.0029975 |    0.0091 |
| F_LIN_P    |      58 |  0.158525    |   0.279152   |   1.91e-11 |   2.715e-05 |  0.00487   |  0.202     |    0.928  |
| F_FULL_R   |      58 | 85.5203      | 335.983      | -88.58     | -43.64      | -9.165     | 85.915     | 2355.54   |
| T20_LIN_B  |      58 |  0.000679652 |   0.0128113  |  -0.027    |  -0.0064975 | -0.0005626 |  0.0072875 |    0.0409 |
| T20_LIN_SE |      58 |  0.00694172  |   0.00405113 |   0.00306  |   0.004595  |  0.00592   |  0.008115  |    0.0235 |
| T20_LIN_P  |      58 |  0.330579    |   0.298724   |   6.44e-05 |   0.047275  |  0.275     |  0.5015    |    0.99   |
| T20_20_R   |      58 | 22.7372      |  91.0653     | -69.32     | -24.71      | -2.415     | 37.635     |  499.04   |
| T10_LIN_B  |      58 | -0.00321495  |   0.0259124  |  -0.0759   |  -0.016675  | -0.0008485 |  0.0099075 |    0.065  |
| T10_LIN_SE |      58 |  0.0165881   |   0.00844442 |   0.00788  |   0.011     |  0.0138    |  0.020275  |    0.0536 |
| T10_LIN_P  |      58 |  0.437002    |   0.309609   |   0.00813  |   0.16275   |  0.377     |  0.773     |    0.97   |
| T10_10_R   |      58 |  7.74983     |  64.5301     | -79.18     | -29.2225    | -1.74      | 22.765     |  283.26   |

### Profile: collated_indices

- Shape: 7,750 rows x 9 columns
- Columns: SPECIES_CODE, SPECIES, COMMON_NAME, YEAR, N_SITES, COLLATED_INDEX, YEAR_RANK, TIME_PERIOD, COUNTRY

#### Dtypes

| column         | dtype   |
|:---------------|:--------|
| SPECIES_CODE   | int64   |
| SPECIES        | str     |
| COMMON_NAME    | str     |
| YEAR           | int64   |
| N_SITES        | int64   |
| COLLATED_INDEX | float64 |
| YEAR_RANK      | int64   |
| TIME_PERIOD    | str     |
| COUNTRY        | str     |

#### Null counts (columns with >0 nulls)

_none found_

#### N unique per column

| column         |   n_unique |
|:---------------|-----------:|
| SPECIES_CODE   |         58 |
| SPECIES        |         58 |
| COMMON_NAME    |         58 |
| YEAR           |         49 |
| N_SITES        |       1406 |
| COLLATED_INDEX |        236 |
| YEAR_RANK      |         49 |
| TIME_PERIOD    |          3 |
| COUNTRY        |          5 |

#### Head (first 3 rows)

|   SPECIES_CODE | SPECIES   | COMMON_NAME   |   YEAR |   N_SITES |   COLLATED_INDEX |   YEAR_RANK | TIME_PERIOD   | COUNTRY   |
|---------------:|:----------|:--------------|-------:|----------:|-----------------:|------------:|:--------------|:----------|
|             84 | Aglais io | Peacock       |   1976 |        31 |             1.77 |          45 | 1976-2024     | England   |
|             84 | Aglais io | Peacock       |   1977 |        47 |             1.65 |          49 | 1976-2024     | England   |
|             84 | Aglais io | Peacock       |   1978 |        50 |             1.97 |          26 | 1976-2024     | England   |

#### Describe (numeric columns)

|                |   count |       mean |        std |    min |     25% |     50% |     75% |     max |
|:---------------|--------:|-----------:|-----------:|-------:|--------:|--------:|--------:|--------:|
| SPECIES_CODE   |    7750 |   64.6436  |  38.4998   |    2   |   27    |   70    |   99    |  123    |
| YEAR           |    7750 | 2002.39    |  13.6503   | 1976   | 1991    | 2004    | 2014    | 2024    |
| N_SITES        |    7750 |  304.019   | 544.22     |    1   |   21    |   71    |  289    | 3130    |
| COLLATED_INDEX |    7750 |    1.99997 |   0.286117 |    0.1 |    1.85 |    2.01 |    2.15 |    3.82 |
| YEAR_RANK      |    7750 |   22.3041  |  13.626    |    1   |   11    |   21    |   34    |   49    |

## 2. Join key harmonisation



### Site key harmonisation

- No dtype/formatting issues found; all three site keys are clean int64.

#### Site ID overlap across files

| comparison                              |   n_sites |
|:----------------------------------------|----------:|
| phenology sites NOT in site_location    |       368 |
| site_indices sites NOT in site_location |      1158 |
| site_location sites NOT in phenology    |      2476 |
| site_location sites NOT in site_indices |      2330 |

#### Interpretation

Sites in phenology/site_indices but not site_location carry NaN coordinates/country/survey metadata after the merge (kept, but unusable in the climate join until resolved). Sites only in site_location were never linked to any phenology/abundance record here and are correctly dropped by a phenology-anchored merge.

### Species lookup (joined on scientific name)

- Distinct scientific names in phenology: 60
- Distinct scientific names in site_indices: 60
- Scientific names in phenology with NO match in site_indices: 0
- Scientific names in site_indices with NO match in phenology: 0
- Scientific names with inconsistent common-name spelling within phenology: 0
- Scientific names with inconsistent common-name spelling within site_indices: 0
- Scientific names mapping to >1 SPECIES_CODE in site_indices (lookup ambiguity): 0


## 3. Dtype coercion



### Dtype coercion report: phenology

| column             |   n_values |   n_failed_coercion | example_failed_values   |
|:-------------------|-----------:|--------------------:|:------------------------|
| YEAR               |     648886 |                   0 | []                      |
| FIRSTDAY           |     648886 |                   0 | []                      |
| LASTDAY            |     648886 |                   0 | []                      |
| PEAKDAY            |     648886 |                   0 | []                      |
| PEAKCOUNT          |     648886 |                   0 | []                      |
| MEAN_FLIGHT_DATE   |     648886 |                   0 | []                      |
| FLIGHTPERIOD_SD    |     648886 |                   0 | []                      |
| FLIGHTPERIOD_RANGE |     648886 |                   0 | []                      |

### Dtype coercion report: site_indices

| column     |   n_values |   n_failed_coercion | example_failed_values   |
|:-----------|-----------:|--------------------:|:------------------------|
| YEAR       |     821300 |                   0 | []                      |
| SITE_INDEX |     821300 |                   0 | []                      |

### Dtype coercion report: site_location

| column   |   n_values |   n_failed_coercion | example_failed_values   |
|:---------|-----------:|--------------------:|:------------------------|
| Easting  |       6082 |                   0 | []                      |
| Northing |       6082 |                   0 | []                      |

## 4. Phenology day-field plausibility



### Day-field definition (IMPORTANT -- corrects the day-of-year assumption)

`ukbms_phenology_2024.docx` defines these fields as **"the day number after 1 April"** (e.g. `20` = 20th April), not a 1-366 day-of-year -- hence the legitimate negatives (min FIRSTDAY = -30). Checks below use that scale: hard bounds -90 to 274, core season 0 to 182.

#### Distribution of day fields (days after 1 April)

|                  |   count |     mean |     std |   min |   25% |   50% |   75% |   max |
|:-----------------|--------:|---------:|--------:|------:|------:|------:|------:|------:|
| FIRSTDAY         |  648886 |  68.4934 | 39.5374 |   -30 |  36   |  72   |    98 |   215 |
| LASTDAY          |  648886 | 129.98   | 38.5512 |   -24 | 111   | 136   |   159 |   215 |
| PEAKDAY          |  648886 |  97.3029 | 40.1738 |   -26 |  73   | 105   |   124 |   215 |
| MEAN_FLIGHT_DATE |  648886 | 100.287  | 31.7323 |   -24 |  85.4 | 105.5 |   121 |   215 |

#### Plausibility flag counts

- Rows with any day field outside hard bounds [-90, 274]: 0
- Rows with any day field outside core UKBMS season [0, 182] (informational -- legitimate for early/late records): 14,262

#### Example out-of-core-season rows

|   SITE_ID | SITENAME      | GRIDREF   | SPECIES_NAME      | COMMON_NAME         |   YEAR |   FIRSTDAY |   LASTDAY |   PEAKDAY |   PEAKCOUNT |   MEAN_FLIGHT_DATE |   FLIGHTPERIOD_SD |   FLIGHTPERIOD_RANGE |
|----------:|:--------------|:----------|:------------------|:--------------------|-------:|-----------:|----------:|----------:|------------:|-------------------:|------------------:|---------------------:|
|         2 | Bevill's Wood | TL203794  | Pararge aegeria   | Speckled Wood       |   2008 |         35 |       191 |       150 |          32 |             127.76 |             36.18 |                  156 |
|         3 | Holkham       | TF873453  | Vanessa atalanta  | Red Admiral         |   1977 |        167 |       183 |       183 |           2 |             177.25 |              6.57 |                   16 |
|         3 | Holkham       | TF873453  | Aglais urticae    | Small Tortoiseshell |   2020 |         -6 |       158 |         1 |           6 |              52.18 |             55.54 |                  164 |
|         3 | Holkham       | TF873453  | Gonepteryx rhamni | Brimstone           |   2020 |         -6 |        37 |        37 |           2 |              22.67 |             20.27 |                   43 |
|         3 | Holkham       | TF873453  | Pieris rapae      | Small White         |   2020 |         -6 |       128 |       128 |          45 |             104.48 |             40.16 |                  134 |

## 5. Logical consistency



### Logical consistency checks



#### Summary

| check                                         |   n_rows_flagged |
|:----------------------------------------------|-----------------:|
| NOT (FIRSTDAY <= PEAKDAY <= LASTDAY)          |                0 |
| |FLIGHTPERIOD_RANGE - (LASTDAY-FIRSTDAY)| > 0 |                0 |
| PEAKCOUNT <= 0                                |                0 |
| FLIGHTPERIOD_SD < 0                           |                0 |

## 6. Duplicates / hidden generations



### Duplicate (SITE_ID, SPECIES_NAME, YEAR) check: phenology

- Duplicate-key groups (>1 row for same SITE_ID, SPECIES_NAME, YEAR): 98
- Rows involved in duplicate-key groups: 196
- ...of which byte-identical (exact) duplicate rows: 196
- ...of which CONFLICTING (same key, different data -- e.g. different SITE_INDEX/PEAKDAY): 0

Exact duplicates collapse losslessly. Conflicting rows are genuine disagreements with no brood/GENERATION column to explain them, so they are NOT deduplicated automatically -- resolve them before treating the table as one row per site-species-year.

#### Example exact-duplicate rows

|   SITE_ID | SITENAME         | GRIDREF   | SPECIES_NAME   | COMMON_NAME   |   YEAR |   FIRSTDAY |   LASTDAY |   PEAKDAY |   PEAKCOUNT |   MEAN_FLIGHT_DATE |   FLIGHTPERIOD_SD |   FLIGHTPERIOD_RANGE | FLAG_DOY_IMPLAUSIBLE   | FLAG_DOY_OUTSIDE_CORE_SEASON   | FLAG_PEAK_ORDER_VIOLATION   | FLAG_RANGE_MISMATCH   | FLAG_PEAKCOUNT_NONPOSITIVE   | FLAG_SD_NEGATIVE   |
|----------:|:-----------------|:----------|:---------------|:--------------|-------:|-----------:|----------:|----------:|------------:|-------------------:|------------------:|---------------------:|:-----------------------|:-------------------------------|:----------------------------|:----------------------|:-----------------------------|:-------------------|
|        85 | Derbyshire Dales | SK180659  | Aglais io      | Peacock       |   2003 |          7 |       156 |        17 |          18 |              54.05 |             51.75 |                  149 | False                  | False                          | False                       | False                 | False                        | False              |
|        85 | Derbyshire Dales | SK180659  | Aglais io      | Peacock       |   2003 |          7 |       156 |        17 |          18 |              54.05 |             51.75 |                  149 | False                  | False                          | False                       | False                 | False                        | False              |
|        85 | Derbyshire Dales | SK180659  | Aglais io      | Peacock       |   2004 |         13 |       162 |        20 |           9 |              36.84 |             35.2  |                  149 | False                  | False                          | False                       | False                 | False                        | False              |
|        85 | Derbyshire Dales | SK180659  | Aglais io      | Peacock       |   2004 |         13 |       162 |        20 |           9 |              36.84 |             35.2  |                  149 | False                  | False                          | False                       | False                 | False                        | False              |
|        85 | Derbyshire Dales | SK180659  | Aricia agestis | Brown Argus   |   2003 |         59 |       167 |       107 |           2 |              97.88 |             30.3  |                  108 | False                  | False                          | False                       | False                 | False                        | False              |

### Duplicate (SITE_ID, SPECIES_CODE, YEAR) check: site_indices

- Duplicate-key groups (>1 row for same SITE_ID, SPECIES_CODE, YEAR): 1,393
- Rows involved in duplicate-key groups: 2,786
- ...of which byte-identical (exact) duplicate rows: 2,686
- ...of which CONFLICTING (same key, different data -- e.g. different SITE_INDEX/PEAKDAY): 100

Exact duplicates collapse losslessly. Conflicting rows are genuine disagreements with no brood/GENERATION column to explain them, so they are NOT deduplicated automatically -- resolve them before treating the table as one row per site-species-year.

#### Example CONFLICTING duplicate-key rows

|   SITE_ID | COUNTRY   |   SPECIES_CODE | SPECIES         | COMMON_NAME      |   YEAR |   SITE_INDEX |
|----------:|:----------|---------------:|:----------------|:-----------------|-------:|-------------:|
|        10 | Wales     |            123 | Vanessa cardui  | Painted Lady     |   2023 |            0 |
|        10 | Wales     |            123 | Vanessa cardui  | Painted Lady     |   2023 |           -2 |
|        14 | England   |            123 | Vanessa cardui  | Painted Lady     |   2023 |            0 |
|        14 | England   |            123 | Vanessa cardui  | Painted Lady     |   2023 |            1 |
|       182 | Scotland  |             43 | Erebia epiphron | Mountain Ringlet |   2018 |            0 |
|       182 | Scotland  |             43 | Erebia epiphron | Mountain Ringlet |   2018 |           -2 |
|      1018 | England   |             93 | Pararge aegeria | Speckled Wood    |   2008 |            0 |
|      1018 | England   |             93 | Pararge aegeria | Speckled Wood    |   2008 |           83 |
|      1019 | England   |            123 | Vanessa cardui  | Painted Lady     |   2023 |            4 |
|      1019 | England   |            123 | Vanessa cardui  | Painted Lady     |   2023 |            0 |

#### Example exact-duplicate rows

|   SITE_ID | COUNTRY   |   SPECIES_CODE | SPECIES        | COMMON_NAME   |   YEAR |   SITE_INDEX |
|----------:|:----------|---------------:|:---------------|:--------------|-------:|-------------:|
|       135 | England   |            123 | Vanessa cardui | Painted Lady  |   2023 |            0 |
|       135 | England   |            123 | Vanessa cardui | Painted Lady  |   2023 |            0 |
|      1039 | England   |            123 | Vanessa cardui | Painted Lady  |   2023 |            0 |
|      1039 | England   |            123 | Vanessa cardui | Painted Lady  |   2023 |            0 |
|      1055 | England   |            123 | Vanessa cardui | Painted Lady  |   2023 |            0 |

## 8. Coverage / reliability



### Coverage / reliability thresholds

- Sites with N_yrs_surveyed < 5: 2,701 of 6,082 (51,248 phenology rows would be affected if filtered)
- (Site, species) series with fewer than 3 phenology records: 30,432 (42,051 phenology rows would be affected if filtered)

These are reported, not applied -- toggle `filters.exclude_below_min_years_surveyed` / `filters.exclude_below_min_site_species_records` in CONFIG to apply.

#### Example short-surveyed sites (< 5 yrs)

|   SITE_ID | Site_Name          | Gridreference   |   Easting |   Northing |   Length | Country   |   N_sections |   N_yrs_surveyed |   First_year_surveyed |   Last_year_surveyed | Survey_type   |
|----------:|:-------------------|:----------------|----------:|-----------:|---------:|:----------|-------------:|-----------------:|----------------------:|---------------------:|:--------------|
|         5 | Monks Wood Fields  | TL200800        |    520000 |     280000 |      nan | England   |           15 |                3 |                  1974 |                 1976 | UKBMS         |
|        33 | Roudsea Wood (old) | SD330820        |    333000 |     482000 |      nan | England   |           15 |                2 |                  1977 |                 1978 | UKBMS         |
|        35 | Stow Longa         | TL117701        |    511700 |     270100 |     1506 | England   |            5 |                2 |                  1976 |                 1977 | UKBMS         |
|        38 | Avon Gorge (old)   | ST550730        |    355000 |     173000 |      nan | England   |           11 |                1 |                  1976 |                 1976 | UKBMS         |
|        40 | Tregarron Bog      | SN687625        |    268700 |     262500 |     3649 | Wales     |            3 |                1 |                  1977 |                 1977 | UKBMS         |

#### Example short (site,species) series (< 3 records)

|   SITE_ID | SPECIES_NAME     |   n_records |
|----------:|:-----------------|------------:|
|         1 | Pyrgus malvae    |           2 |
|         2 | Colias croceus   |           1 |
|         2 | Satyrium w-album |           1 |
|         3 | Satyrium w-album |           2 |
|         4 | Colias croceus   |           2 |

## 9. Survey type



### Survey type



#### Survey_type counts

| Survey_type   |   n_sites |
|:--------------|----------:|
| UKBMS         |      3756 |
| WCBS          |      2326 |

#### Non-standard transect impact

- Non-UKBMS survey types found: ['WCBS']
- Sites on non-standard survey types: 2,326
- Phenology rows that would be affected if excluded: 0

WCBS sites are visited 2-3 times/year (vs up to 26 for standard transects), so their phenology metrics rest on much sparser data. Toggle `filters.exclude_nonstandard_survey_type` to exclude.

## 10. Coordinates



### Coordinate validity (OSGB National Grid)

- Sites missing Easting/Northing: 0
- Sites with Easting or Northing == 0: 0
- GB-country sites (England/Scotland/Wales) outside GB grid bounds (E [0,700000], N [0,1300000]): 0
- Sites outside GB-grid scope by country (Northern Ireland/Channel Islands/Isle of Man): 183 -- these use the Irish Grid or a local Channel Islands grid (negative Northing is a convention there, not an error), so GB bounds do not apply and they are not flagged invalid.
- Grid-reference prefix length inconsistent with the site's expected grid system: 0

Sites without valid GB coordinates (missing/zero/out-of-bounds) break the GB climate join and must be resolved or excluded first; NI/CI/IoM sites additionally need their own grid-to-lat/lon conversion (not a straight OSGB one).

#### Example GB sites out of grid bounds

_none found_

#### Example non-GB-grid sites (by country)

| Country          |   n_sites |
|:-----------------|----------:|
| Channel Islands  |        51 |
| Isle of Man      |         6 |
| Northern Ireland |       126 |

#### Example grid-reference prefix mismatches

_none found_

## 11. Species scope



### Species scope

- Distinct species: 60
- Species with fewer than 200 phenology records (rare/restricted-range candidates): 5
- Migrant/highly-mobile candidate species present: ['Painted Lady', 'Clouded Yellow', 'Red Admiral']

None excluded by default. `Essex/Small Skipper` is a deliberate combined-ID category (the two are hard to separate in the field), distinct from the separate `Essex Skipper` and `Small Skipper` -- a real taxon, not an error. It and `Mountain Ringlet` are the two species with no matching species_trends row.

#### Rarest species (record count)

| COMMON_NAME          |   n_records |
|:---------------------|------------:|
| Large Blue           |          48 |
| Mountain Ringlet     |          82 |
| Cryptic Wood White   |          82 |
| Glanville Fritillary |         151 |
| Chequered Skipper    |         166 |

## 12. Temporal alignment



### Temporal alignment



#### YEAR range per file

| file                                |   min_year |   max_year |   n_distinct_years |
|:------------------------------------|-----------:|-----------:|-------------------:|
| phenology                           |       1973 |       2024 |                 52 |
| site_indices                        |       1973 |       2024 |                 52 |
| collated_indices                    |       1976 |       2024 |                 49 |
| site_location (First_year_surveyed) |       1973 |       2024 |                 51 |
| site_location (Last_year_surveyed)  |       1976 |       2024 |                 45 |

#### 2024 coverage check

All files extend to 2024.

## Duplicate resolution applied before merge



### Phenology exact-duplicate collapse

Collapsed 648,886 -> 648,788 rows by dropping byte-identical repeats (all 98 removed rows were confirmed full-row duplicates, see section 6). Distinct (site, species, year) combinations in raw phenology: 648,788.

### site_indices exact-duplicate collapse + conflict isolation

Collapsed 821,300 -> 819,957 rows (byte-identical repeats removed). 100 rows across 50 keys still have genuinely different SITE_INDEX values for the same (site, species, year). These rows are retained (first occurrence per key) but SITE_INDEX is set to NaN and `SITE_INDEX_AMBIGUOUS_CONFLICT=True` so the disagreement is visible rather than guessed at.

## COUNTRY consistency: site_indices.COUNTRY vs site_location.Country

0 merged rows have a different COUNTRY value between the two source files for the same site.

## species_trends attachment

Left-joined on COMMON_NAME (species-level table, 58 distinct species, 0 duplicate COMMON_NAME keys). Species in the merged table with no matching trend row: ['Essex/Small Skipper', 'Mountain Ringlet'].

## collated_indices attachment

Left-joined on (scientific name, YEAR, site's COUNTRY). collated_indices has no Channel Islands/Isle of Man entries, so those sites get NaN NATIONAL_* columns by design.

## 7. Missingness in merged table



### Null counts per column (merged table)

| column                             |   n_null |   pct_null |
|:-----------------------------------|---------:|-----------:|
| GRIDREF                            |      130 |       0.02 |
| SPECIES_CODE                       |     2914 |       0.45 |
| COUNTRY                            |     2914 |       0.45 |
| SITE_INDEX                         |     2961 |       0.46 |
| SITE_INDEX_AMBIGUOUS_CONFLICT      |     2914 |       0.45 |
| SITE_INDEX_INSUFFICIENT_MONITORING |     2914 |       0.45 |
| Site_Name_siteloc                  |    18009 |       2.78 |
| Gridreference_siteloc              |    18009 |       2.78 |
| Easting_siteloc                    |    18009 |       2.78 |
| Northing_siteloc                   |    18009 |       2.78 |
| Length_siteloc                     |    26531 |       4.09 |
| Country_siteloc                    |    18009 |       2.78 |
| N_sections_siteloc                 |    18046 |       2.78 |
| N_yrs_surveyed_siteloc             |    18009 |       2.78 |
| First_year_surveyed_siteloc        |    18009 |       2.78 |
| Last_year_surveyed_siteloc         |    18009 |       2.78 |
| Survey_type_siteloc                |    18009 |       2.78 |
| TREND_SCI_NAME                     |    14776 |       2.28 |
| TREND_NYEARS                       |    14776 |       2.28 |
| TREND_F_LIN_B                      |    14776 |       2.28 |
| TREND_F_LIN_SE                     |    14776 |       2.28 |
| TREND_F_LIN_P                      |    14776 |       2.28 |
| TREND_F_TRENDDETAIL                |    14776 |       2.28 |
| TREND_F_FULL_R                     |    14776 |       2.28 |
| TREND_T20_LIN_B                    |    14776 |       2.28 |
| TREND_T20_LIN_SE                   |    14776 |       2.28 |
| TREND_T20_LIN_P                    |    14776 |       2.28 |
| TREND_T20_TRENDDETAIL              |    14776 |       2.28 |
| TREND_T20_20_R                     |    14776 |       2.28 |
| TREND_T10_LIN_B                    |    14776 |       2.28 |
| TREND_T10_LIN_SE                   |    14776 |       2.28 |
| TREND_T10_LIN_P                    |    14776 |       2.28 |
| TREND_T10_TRENDDETAIL              |    14776 |       2.28 |
| TREND_T10_10_R                     |    14776 |       2.28 |
| NATIONAL_N_SITES                   |    25181 |       3.88 |
| NATIONAL_COLLATED_INDEX            |    25181 |       3.88 |
| NATIONAL_YEAR_RANK                 |    25181 |       3.88 |
| NATIONAL_TIME_PERIOD               |    25181 |       3.88 |

### SITE_INDEX missingness interpretation

- NaN SITE_INDEX from an ambiguous source conflict (flagged above): 47
- NaN SITE_INDEX because no site_indices row exists for that (site, species, year) -- phenology recorded the species but no index was published: 2,961

The phenology target (FIRSTDAY/LASTDAY/PEAKDAY/etc.) is never imputed.

## Filtering stage (only filters toggled True in CONFIG are applied)



### Filter impact summary

| filter                                                              | toggle                                 | applied   |   rows_before |   rows_would_remove |   rows_after |
|:--------------------------------------------------------------------|:---------------------------------------|:----------|--------------:|--------------------:|-------------:|
| Exclude non-standard survey type (WCBS)                             | exclude_nonstandard_survey_type        | False     |        648788 |                   0 |       648788 |
| Exclude species in CONFIG species_exclude_list                      | exclude_species_list                   | False     |        648788 |                   0 |       648788 |
| Exclude sites with N_yrs_surveyed < 5                               | exclude_below_min_years_surveyed       | False     |        648788 |               51248 |       648788 |
| Exclude (site,species) series with < 3 phenology records            | exclude_below_min_site_species_records | False     |        648788 |               42051 |       648788 |
| Null invalid GB coordinates (Easting/Northing) for GB-country sites | null_invalid_gb_coordinates            | False     |        648788 |                   0 |       648788 |

### How to change filtering behaviour

Edit `CONFIG['filters']` at the top of `scripts/clean_pipeline.py` and re-run. Every filter above is computed and reported regardless of toggle state; only toggles set to `True` change the saved output.

## Row-count log (all stages)

| stage                                                       |   n_rows |
|:------------------------------------------------------------|---------:|
| raw: phenology                                              |   648886 |
| raw: site_indices                                           |   821300 |
| raw: site_location                                          |     6082 |
| raw: species_trends                                         |       58 |
| raw: collated_indices                                       |     7750 |
| after exact-dup collapse: phenology                         |   648788 |
| after phenology conflict resolution (1 row/key retained)    |   648788 |
| after exact-dup collapse: site_indices                      |   819957 |
| after site_indices conflict resolution (1 row/key retained) |   819907 |
| after phenology-siteindices merge                           |   648788 |
| after + site_location merge                                 |   648788 |
| after + species_trends merge                                |   648788 |
| after + collated_indices merge                              |   648788 |
| FINAL (after filtering stage)                               |   648788 |

## Output files

- `output/ukbms_site_species_year_clean.parquet`
- `output/ukbms_site_species_year_clean.csv`
- `output/data_dictionary.md`

