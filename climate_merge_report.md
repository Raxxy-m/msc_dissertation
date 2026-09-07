# Climate merge report

## Sites
- total distinct sites: 3974
- usable BNG (England/Scotland/Wales, valid EPSG:27700): 3496
- non-BNG coords (NI Irish Grid / Channel Islands / local, negative northings): 110 — EXCLUDED from join (would mis-locate)
- sites with no coordinates: 368 — cannot join

## Join QC
- max site-to-cell distance: 3536 m (5km grid => expect <= ~3535 m)
- all-NaN (sea/edge) sites: 66 (snap-to-land OFF)

## Features (site x year)
- rows: 171304; columns: 33
- monthly mean Tmax/Tmin/Tmean + monthly total rainfall (prev Dec = *_M12P, Jan..May = *_M01..M05)
- seasonal DJF & MAM mean Tmax/Tmin/Tmean + total rainfall
- GDD: sum of max(0, Tmean - 5.0) from DOY 1 through 31 May

## Leakage window
- Predictors use ONLY climate from 1 Dec (year-1) through 31 May (year): the winter/spring
  that PRECEDE the flight period. No post-flight climate is used.
- Caveat: very early species (flying in Apr/early May) may have flight dates inside the
  spring window; a per-event dynamic cut-off is a documented future refinement.

## Assumptions
- coordinate names: x=projection_x_coordinate y=projection_y_coordinate time=time
- data variable name == folder name (tasmax/tasmin/rainfall); Tmean=(tasmax+tasmin)/2
- nearest-cell selection via xarray method='nearest'; T_BASE=5.0, GDD_START_DOY=1
- merge join keys: SITE_ID + YEAR; base rows preserved (648788).
