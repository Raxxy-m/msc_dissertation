#!/usr/bin/env python
"""
Download HadUK-Grid v1.3.1 5km DAILY data (tasmax, tasmin, rainfall) from the
CEDA archive, filtered to Jan 1976 - Dec 2024, then verify each file.

Auth: OAuth2 Bearer token read ONLY from env CEDA_TOKEN (see config.get_token).
No username/password is ever used. On 401/403 or zero-length files the script
stops and tells you to regenerate the token.

Behaviour:
  * Discovers the dated version sub-folder (vYYYYMMDD/) per variable via the
    directory listing (JSON then HTML), with a filename-pattern fallback.
  * Streams each monthly .nc to data/haduk/<var>/, idempotent (skips valid files).
  * Retries transient failures with exponential backoff.
  * Verifies every file with xarray (valid netCDF, expected data var, BNG coords,
    non-empty) and writes a manifest CSV.

Run:  .venv/bin/python -m scripts.climate.download_haduk
  or  .venv/bin/python scripts/climate/download_haduk.py
"""
from __future__ import annotations

import re
import sys
import time
import json
from pathlib import Path

import requests

# Allow running as a plain script (python scripts/climate/download_haduk.py)
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.climate import config as C
else:
    from . import config as C


# --------------------------------------------------------------------------
# HTTP helpers
# --------------------------------------------------------------------------
# Transient HTTP statuses worth retrying (CEDA occasionally 500s on a good file).
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class DownloadError(Exception):
    """Non-fatal per-file failure: caller logs it and moves on."""


def _get(url: str, session: requests.Session, *, stream: bool = False) -> requests.Response:
    """GET with bearer auth, retries + backoff.

    - 401/403 -> fatal SystemExit (token problem).
    - 404 -> returned to caller (used to skip non-existent months).
    - 429/5xx and connection/timeout errors -> retried with exponential backoff.
    - Still failing after MAX_RETRIES, or a non-retryable 4xx -> DownloadError
      (caller decides; for file downloads this means "skip and continue").
    """
    last_err: object = None
    for attempt in range(1, C.MAX_RETRIES + 1):
        try:
            r = session.get(url, headers=C.auth_headers(), timeout=C.REQUEST_TIMEOUT,
                            stream=stream)
        except (requests.ConnectionError, requests.Timeout) as e:
            last_err = f"{e.__class__.__name__}"
            _backoff(attempt, url, last_err)
            continue

        if r.status_code in (401, 403):
            raise SystemExit(
                f"ERROR: HTTP {r.status_code} for {url}\n"
                "  Your CEDA_TOKEN is missing, invalid, or expired.\n"
                "  Regenerate it at https://services.ceda.ac.uk/ and re-export CEDA_TOKEN."
            )
        if r.status_code == 404:
            return r  # let caller decide (used to skip non-existent months)
        if r.status_code in RETRYABLE_STATUS:
            last_err = f"HTTP {r.status_code}"
            _backoff(attempt, url, last_err)
            continue
        if not r.ok:
            raise DownloadError(f"HTTP {r.status_code} for {url}")
        return r

    raise DownloadError(f"giving up on {url} after {C.MAX_RETRIES} retries ({last_err})")


def _backoff(attempt: int, url: str, why: str) -> None:
    wait = C.BACKOFF_BASE ** attempt
    print(f"  transient error ({why}) on {url}; "
          f"retry {attempt}/{C.MAX_RETRIES} in {wait:.0f}s")
    time.sleep(wait)


# --------------------------------------------------------------------------
# Directory discovery
# --------------------------------------------------------------------------
_VDIR_RE = re.compile(r"v(\d{8})")


def discover_version_dir(base_url: str, var: str, session: requests.Session) -> str:
    """Find the dated version sub-folder name (e.g. 'v20240514') under day/.

    Priority: explicit override -> env -> JSON listing -> HTML listing.
    """
    # 1) explicit override / env
    override = C.VERSION_DIR_OVERRIDE.get(var) or _env_vdir(var)
    if override:
        print(f"[{var}] using override version dir: {override}")
        return override.strip("/")

    # 2) JSON listing (CEDA dap supports ?json on directories)
    for suffix in ("?json", ""):
        try:
            r = _get(base_url + suffix, session)
        except SystemExit:
            raise
        except Exception:
            continue
        if r.status_code == 404:
            continue
        names = _extract_names(r, base_url)
        vdirs = sorted({m.group(0) for n in names if (m := _VDIR_RE.search(n))})
        if vdirs:
            chosen = vdirs[-1]  # latest dated folder
            print(f"[{var}] discovered version dir(s) {vdirs} -> using {chosen}")
            return chosen

    raise SystemExit(
        f"ERROR: could not discover the dated version sub-folder for '{var}' at\n"
        f"  {base_url}\n"
        "  The listing may be blocked. Open the URL in your browser, read the\n"
        "  vYYYYMMDD/ folder name, and set it in config.VERSION_DIR_OVERRIDE\n"
        f"  (or export HADUK_VDIR_{var.upper()}=vYYYYMMDD), then re-run."
    )


def _env_vdir(var: str) -> str | None:
    import os
    return os.environ.get(f"HADUK_VDIR_{var.upper()}")


def _extract_names(r: requests.Response, base_url: str) -> list[str]:
    """Extract child names from a JSON or HTML directory listing."""
    ctype = r.headers.get("Content-Type", "")
    text = r.text
    names: list[str] = []
    if "json" in ctype or text.lstrip().startswith(("{", "[")):
        try:
            data = json.loads(text)
            names = _walk_json_for_names(data)
        except Exception:
            names = []
    if not names:
        # HTML: pull href targets
        names = re.findall(r'href="([^"?]+)"', text)
    # normalise to leaf names
    out = []
    for n in names:
        n = n.rstrip("/")
        leaf = n.rsplit("/", 1)[-1]
        if leaf:
            out.append(leaf)
    return out


def _walk_json_for_names(data) -> list[str]:
    """Best-effort pull of file/dir names from an unknown JSON listing shape."""
    found: list[str] = []
    if isinstance(data, dict):
        for key in ("items", "children", "contents", "files", "entries"):
            if key in data and isinstance(data[key], (list, dict)):
                data = data[key]
                break
    if isinstance(data, dict):
        found.extend(str(k) for k in data.keys())
        for v in data.values():
            if isinstance(v, dict):
                for nk in ("name", "path", "title"):
                    if nk in v:
                        found.append(str(v[nk]))
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                found.append(item)
            elif isinstance(item, dict):
                for nk in ("name", "path", "title", "href"):
                    if nk in item:
                        found.append(str(item[nk]))
                        break
    return found


def list_month_files(listing_url: str, var: str, session: requests.Session) -> list[str]:
    """Return .nc filenames in the version dir matching the month pattern."""
    pat = re.compile(C.FILENAME_RE.format(var=var, res=C.RESOLUTION))
    for suffix in ("?json", ""):
        r = _get(listing_url + suffix, session)
        if r.status_code == 404:
            continue
        names = _extract_names(r, listing_url)
        matches = sorted({n for n in names if pat.fullmatch(n)})
        if matches:
            return matches
    return []


# --------------------------------------------------------------------------
# Filtering / verification
# --------------------------------------------------------------------------
def in_range(fname: str, var: str) -> bool:
    """True if the file's start date falls within YEAR_MIN..YEAR_MAX."""
    m = re.match(C.FILENAME_RE.format(var=var, res=C.RESOLUTION), fname)
    if not m:
        return False
    start_year = int(m.group(1)[:4])
    return C.YEAR_MIN <= start_year <= C.YEAR_MAX


def month_of(fname: str, var: str) -> str:
    m = re.match(C.FILENAME_RE.format(var=var, res=C.RESOLUTION), fname)
    return m.group(1)[:6] if m else ""


def verify_netcdf(path: Path, var: str) -> tuple[bool, str]:
    """Open with xarray; confirm valid, has data var + BNG coords, non-empty."""
    import xarray as xr
    if not path.exists() or path.stat().st_size == 0:
        return False, "missing or zero-length"
    try:
        with xr.open_dataset(path) as ds:
            if var not in ds.variables:
                return False, f"data var '{var}' absent (vars: {list(ds.data_vars)})"
            for c in (C.X_COORD, C.Y_COORD):
                if c not in ds.variables:
                    return False, f"coordinate '{c}' absent"
            if ds[var].size == 0:
                return False, "data variable is empty"
            # at least some finite data
            import numpy as np
            sample = ds[var].isel({d: 0 for d in ds[var].dims if d not in (C.X_COORD, C.Y_COORD)})
            if not np.isfinite(sample.values).any():
                return False, "first timestep all-NaN"
        return True, "ok"
    except Exception as e:
        return False, f"open failed: {e.__class__.__name__}: {e}"


def download_file(url: str, dest: Path, session: requests.Session) -> tuple[bool, int, str]:
    """Stream one file to disk. Returns (ok, bytes, status)."""
    r = _get(url, session, stream=True)
    if r.status_code == 404:
        return False, 0, "404 not found"
    tmp = dest.with_suffix(dest.suffix + ".part")
    n = 0
    with open(tmp, "wb") as fh:
        for chunk in r.iter_content(chunk_size=1 << 20):
            if chunk:
                fh.write(chunk)
                n += len(chunk)
    if n == 0:
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"ERROR: zero-length download for {url}\n"
            "  This usually means the CEDA_TOKEN is missing/expired. Regenerate it."
        )
    tmp.replace(dest)
    return True, n, "downloaded"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    C.get_token()  # fail fast if unset
    C.HADUK_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    manifest_rows = []
    expected = downloaded = skipped = failed = 0
    total_bytes = 0

    for var in C.VARIABLES:
        base = C.BASE_URLS[var]
        print(f"\n=== {var} ===\n  base: {base}")
        vdir = discover_version_dir(base, var, session)
        listing_url = f"{base}{vdir}/"
        var_out = C.HADUK_DIR / var
        var_out.mkdir(parents=True, exist_ok=True)

        files = list_month_files(listing_url, var, session)
        if files:
            files = [f for f in files if in_range(f, var)]
            print(f"  {len(files)} monthly files in {C.YEAR_MIN}-{C.YEAR_MAX} (from listing)")
        else:
            # Fallback: construct filenames month by month, skip 404s.
            print("  listing empty/blocked -> constructing filenames and skipping 404s")
            files = _construct_month_names(var)

        for fname in files:
            expected += 1
            url = listing_url + fname
            dest = var_out / fname
            month = month_of(fname, var)

            if dest.exists() and dest.stat().st_size > 0:
                ok, msg = verify_netcdf(dest, var)
                if ok:
                    skipped += 1
                    total_bytes += dest.stat().st_size
                    manifest_rows.append((var, fname, month, dest.stat().st_size, "skip-valid"))
                    continue
                else:
                    print(f"  re-downloading invalid existing {fname}: {msg}")
                    dest.unlink(missing_ok=True)

            try:
                ok, nbytes, status = download_file(url, dest, session)
            except SystemExit:
                raise
            except DownloadError as e:
                failed += 1
                dest.with_suffix(dest.suffix + ".part").unlink(missing_ok=True)
                manifest_rows.append((var, fname, month, 0, f"failed: {e}"))
                print(f"  ! {fname}: {e}  (skipped — re-run later to retry)")
                continue
            if not ok:
                failed += 1
                manifest_rows.append((var, fname, month, 0, status))
                print(f"  ! {fname}: {status}")
                continue

            vok, vmsg = verify_netcdf(dest, var)
            if vok:
                downloaded += 1
                total_bytes += nbytes
                manifest_rows.append((var, fname, month, nbytes, "downloaded-verified"))
                print(f"  + {fname} ({nbytes/1e6:.1f} MB)")
            else:
                failed += 1
                manifest_rows.append((var, fname, month, nbytes, f"verify-failed: {vmsg}"))
                print(f"  ! {fname} verify failed: {vmsg}")
            time.sleep(C.POLITE_DELAY)

    _write_manifest(manifest_rows)

    print("\n===== DOWNLOAD SUMMARY =====")
    print(f"  expected files : {expected}")
    print(f"  downloaded     : {downloaded}")
    print(f"  skipped(valid) : {skipped}")
    print(f"  failed         : {failed}")
    print(f"  total size     : {total_bytes/1e9:.2f} GB")
    print(f"  manifest       : {C.MANIFEST_CSV}")
    if failed:
        print("  WARNING: some files failed — see manifest 'status' column.")
    return 1 if failed else 0


def _construct_month_names(var: str) -> list[str]:
    names = []
    import calendar
    for y in range(C.YEAR_MIN, C.YEAR_MAX + 1):
        for mo in range(1, 13):
            last = calendar.monthrange(y, mo)[1]
            start = f"{y}{mo:02d}01"
            end = f"{y}{mo:02d}{last:02d}"
            names.append(f"{var}_hadukgrid_uk_{C.RESOLUTION}_day_{start}-{end}.nc")
    return names


def _write_manifest(rows) -> None:
    import csv
    C.MANIFEST_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(C.MANIFEST_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variable", "filename", "month", "bytes", "status"])
        w.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
