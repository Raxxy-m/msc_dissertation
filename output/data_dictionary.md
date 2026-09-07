# Data dictionary: ukbms_site_species_year_clean

Grain: one row per (SITE_ID, SPECIES_NAME, YEAR). 648,788 rows, 57 columns.

| column                             | dtype   | description                                                                                                                                  |
|:-----------------------------------|:--------|:---------------------------------------------------------------------------------------------------------------------------------------------|
| SITE_ID                            | int64   | UKBMS site identifier (harmonised from SITENO/SITE_CODE/Site_Number)                                                                         |
| SITENAME                           | str     | Site name as recorded in the phenology download                                                                                              |
| GRIDREF                            | str     | Grid reference, phenology download (start of transect)                                                                                       |
| SPECIES_NAME                       | str     | Scientific (binomial) name; primary species join key                                                                                         |
| COMMON_NAME                        | str     | Vernacular species name                                                                                                                      |
| YEAR                               | Int64   | Survey year                                                                                                                                  |
| FIRSTDAY                           | int64   | Days after 1 April on which species first recorded at site/year (can be negative)                                                            |
| LASTDAY                            | int64   | Days after 1 April on which species last recorded at site/year                                                                               |
| PEAKDAY                            | int64   | Days after 1 April of largest count at site/year                                                                                             |
| PEAKCOUNT                          | int64   | Largest single count recorded at site/year                                                                                                   |
| MEAN_FLIGHT_DATE                   | float64 | Weighted mean flight date, days after 1 April                                                                                                |
| FLIGHTPERIOD_SD                    | float64 | SD (days) around MEAN_FLIGHT_DATE -- synchronisation/length of flight period                                                                 |
| FLIGHTPERIOD_RANGE                 | int64   | LASTDAY - FIRSTDAY, days                                                                                                                     |
| FLAG_DOY_IMPLAUSIBLE               | bool    | Any day field outside hard plausibility bounds (CONFIG day_value_min/max)                                                                    |
| FLAG_DOY_OUTSIDE_CORE_SEASON       | bool    | Any day field outside core UKBMS season (1 Apr-30 Sep); informational                                                                        |
| FLAG_PEAK_ORDER_VIOLATION          | bool    | NOT (FIRSTDAY <= PEAKDAY <= LASTDAY)                                                                                                         |
| FLAG_RANGE_MISMATCH                | bool    | |FLIGHTPERIOD_RANGE - (LASTDAY-FIRSTDAY)| exceeds tolerance                                                                                  |
| FLAG_PEAKCOUNT_NONPOSITIVE         | bool    | PEAKCOUNT <= 0                                                                                                                               |
| FLAG_SD_NEGATIVE                   | bool    | FLIGHTPERIOD_SD < 0                                                                                                                          |
| FLAG_PHENOLOGY_KEY_CONFLICT        | bool    | Source phenology had conflicting duplicate rows for this (site,species,year); first row retained                                             |
| SPECIES_CODE                       | float64 | UKBMS numeric species code (from site_indices)                                                                                               |
| COUNTRY                            | str     | Country, from site_indices                                                                                                                   |
| SITE_INDEX                         | float64 | Site abundance index for species/site/year; -2 = insufficient monitoring (see flag)                                                          |
| SITE_INDEX_AMBIGUOUS_CONFLICT      | object  | True if source site_indices had conflicting duplicate values for this key (SITE_INDEX set NaN)                                               |
| SITE_INDEX_INSUFFICIENT_MONITORING | object  | True if raw SITE_INDEX was the -2 sentinel                                                                                                   |
| SITE_NAME_SITELOC                  | str     | Site name as recorded in the site_location download                                                                                          |
| GRIDREF_SITELOC                    | str     | Grid reference, site_location download (transect centre)                                                                                     |
| EASTING                            | float64 | OSGB Easting (m) for GB sites; Irish Grid/local grid for NI/Channel Islands -- see QC report                                                 |
| NORTHING                           | float64 | OSGB Northing (m) for GB sites; Irish Grid/local grid for NI/Channel Islands -- see QC report                                                |
| LENGTH_M                           | float64 | Transect length, metres                                                                                                                      |
| COUNTRY_SITELOC                    | str     | Country, from site_location (cross-checked against COUNTRY)                                                                                  |
| N_SECTIONS                         | float64 | Number of transect sections                                                                                                                  |
| N_YRS_SURVEYED                     | float64 | Number of years site surveyed under UKBMS up to 2024                                                                                         |
| FIRST_YEAR_SURVEYED                | float64 | First year site surveyed                                                                                                                     |
| LAST_YEAR_SURVEYED                 | float64 | Most recent year site surveyed                                                                                                               |
| SURVEY_TYPE                        | str     | UKBMS (standard transect) or WCBS (Wider Countryside Butterfly Scheme)                                                                       |
| TREND_SCI_NAME                     | str     | TREND_* = species-level trend stat (speciesTrends, see source docx); NATIONAL_* = national collated index (collatedindices, see source docx) |
| TREND_NYEARS                       | float64 | TREND_* = species-level trend stat (speciesTrends, see source docx); NATIONAL_* = national collated index (collatedindices, see source docx) |
| TREND_F_LIN_B                      | float64 | TREND_* = species-level trend stat (speciesTrends, see source docx); NATIONAL_* = national collated index (collatedindices, see source docx) |
| TREND_F_LIN_SE                     | float64 | TREND_* = species-level trend stat (speciesTrends, see source docx); NATIONAL_* = national collated index (collatedindices, see source docx) |
| TREND_F_LIN_P                      | float64 | TREND_* = species-level trend stat (speciesTrends, see source docx); NATIONAL_* = national collated index (collatedindices, see source docx) |
| TREND_F_TRENDDETAIL                | str     | TREND_* = species-level trend stat (speciesTrends, see source docx); NATIONAL_* = national collated index (collatedindices, see source docx) |
| TREND_F_FULL_R                     | float64 | TREND_* = species-level trend stat (speciesTrends, see source docx); NATIONAL_* = national collated index (collatedindices, see source docx) |
| TREND_T20_LIN_B                    | float64 | TREND_* = species-level trend stat (speciesTrends, see source docx); NATIONAL_* = national collated index (collatedindices, see source docx) |
| TREND_T20_LIN_SE                   | float64 | TREND_* = species-level trend stat (speciesTrends, see source docx); NATIONAL_* = national collated index (collatedindices, see source docx) |
| TREND_T20_LIN_P                    | float64 | TREND_* = species-level trend stat (speciesTrends, see source docx); NATIONAL_* = national collated index (collatedindices, see source docx) |
| TREND_T20_TRENDDETAIL              | str     | TREND_* = species-level trend stat (speciesTrends, see source docx); NATIONAL_* = national collated index (collatedindices, see source docx) |
| TREND_T20_20_R                     | float64 | TREND_* = species-level trend stat (speciesTrends, see source docx); NATIONAL_* = national collated index (collatedindices, see source docx) |
| TREND_T10_LIN_B                    | float64 | TREND_* = species-level trend stat (speciesTrends, see source docx); NATIONAL_* = national collated index (collatedindices, see source docx) |
| TREND_T10_LIN_SE                   | float64 | TREND_* = species-level trend stat (speciesTrends, see source docx); NATIONAL_* = national collated index (collatedindices, see source docx) |
| TREND_T10_LIN_P                    | float64 | TREND_* = species-level trend stat (speciesTrends, see source docx); NATIONAL_* = national collated index (collatedindices, see source docx) |
| TREND_T10_TRENDDETAIL              | str     | TREND_* = species-level trend stat (speciesTrends, see source docx); NATIONAL_* = national collated index (collatedindices, see source docx) |
| TREND_T10_10_R                     | float64 | TREND_* = species-level trend stat (speciesTrends, see source docx); NATIONAL_* = national collated index (collatedindices, see source docx) |
| NATIONAL_N_SITES                   | float64 | TREND_* = species-level trend stat (speciesTrends, see source docx); NATIONAL_* = national collated index (collatedindices, see source docx) |
| NATIONAL_COLLATED_INDEX            | float64 | TREND_* = species-level trend stat (speciesTrends, see source docx); NATIONAL_* = national collated index (collatedindices, see source docx) |
| NATIONAL_YEAR_RANK                 | float64 | TREND_* = species-level trend stat (speciesTrends, see source docx); NATIONAL_* = national collated index (collatedindices, see source docx) |
| NATIONAL_TIME_PERIOD               | str     | TREND_* = species-level trend stat (speciesTrends, see source docx); NATIONAL_* = national collated index (collatedindices, see source docx) |