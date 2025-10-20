# token=eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6ImVsbGFiYWdjaGkxMjMiLCJleHAiOjE3NjQ1NDcxOTksImlhdCI6MTc1OTMzNTU2NCwiaXNzIjoiaHR0cHM6Ly91cnMuZWFydGhkYXRhLm5hc2EuZ292IiwiaWRlbnRpdHlfcHJvdmlkZXIiOiJlZGxfb3BzIiwiYWNyIjoiZWRsIiwiYXNzdXJhbmNlX2xldmVsIjozfQ.zN_m8n0-u0KmJv2-JOAQ9BJ4MMHHQNCpuGwt5hmUXkOzdtJ10ivD85Ze0mpTSNaK4_imm-20hujcTkxoKV1f4Mges3la1ku6yNrq7C5-9InxPt-lwAI-PQn8cQv7wahYo2XO_4P8gOWcp0o9H04-i9m6uoqOvNNXjLaGj4g9-73RMh5Sb9RNx4LcNaNpNG7PkBFvcZqEfD9Kxjoj9aeHhP-F-JgkbIoVbo5TwBxaZ1K1Lgjcbxbe3Q-NQPfxHdBxYhqCoFhzS6b-D0ieIZfQrtIDFd-OpM4Kk-aJBP2Kv75d6jS4J9P8Sb8yCXSxaoBsPjH2WlGczg_7Kgpc_v5rsQ

#!/bin/bash
# MAIAC AOD (MCD19A2CMG v061) downloader
# Minimal changes from your OMI script:
# - Uses LAADS DAAC with Bearer token auth
# - Traverses YYYY/DDD folders
# - Parses AYYYYDDD in filenames and converts Julian day -> YYYYMMDD
# - Keeps tmp staging + nullglob; renames to {out_stub}_YYYYMMDD.hdf
#
# REQUIREMENT: Set an Earthdata LAADS token in env before running:
#   export LAADS_TOKEN="paste-your-token-here"
# Get a token from the LAADS site (Login ▸ Token) and copy it. (Token is used as an Authorization: Bearer header.)

#!/bin/bash
# MAIAC AOD (MCD19A2CMG v061) downloader — robust move/rename
# Handles nested directories from wget, converts AYYYYDDD -> YYYYMMDD,
# and moves everything into new_aod/<year>.

#!/usr/bin/env bash
# MAIAC AOD (MCD19A2CMG v061) downloader — resilient, with built-in sweep/rename.
# - Recurses into YYYY/DDD on LAADS
# - Accepts only .hdf
# - Converts AYYYYDDD -> YYYYMMDD
# - Moves to /new_aod/<year>/ as maiac_aod_YYYYMMDD.hdf
# - Handles partial failures per year without aborting the whole loop

set -uo pipefail   # (intentionally NOT using -e so one failed year/day won't stop the loop)

: "${LAADS_TOKEN:?Set LAADS_TOKEN env var with your Earthdata LAADS token, e.g. export LAADS_TOKEN='...'}"

# ---- change these as needed ----
YEAR_START=2014
YEAR_END=2024            # e.g., 2024
BASE_SAVE="/home/ellab/air_pollution/src/data/new_aod"
OUT_STUB="maiac_aod"
# --------------------------------

download_year() {
  local year="$1"
  local base_url="https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MCD19A2CMG/${year}"
  local prefix="MCD19A2CMG.A${year}"
  local tmp_dir="./maiac_tmp_${year}"
  local target_dir="${BASE_SAVE}/${year}"

  mkdir -p "$tmp_dir" "$target_dir"

  echo "== Year $year =="
  echo "--> Downloading from $base_url"
  # We only accept .hdf; recurse 2 levels (YYYY/DDD).
  # If LAADS has transient errors for some days, wget may return non-zero; we trap and continue.
  if ! wget --header "Authorization: Bearer ${LAADS_TOKEN}" \
            --directory-prefix="$tmp_dir" \
            --accept "${prefix}*.hdf" \
            --no-parent \
            --recursive --level=2 --continue --tries=3 \
            --reject ".html,.tmp,.xml,.json,.met" \
            "$base_url/"; then
    echo "!! Warning: wget reported errors for year ${year}, proceeding to sweep whatever was fetched."
  fi

  sweep_tmp "${year}" "${tmp_dir}" "${target_dir}" "${prefix}"
  # Clean temp only after attempting sweep
  rm -rf "$tmp_dir"
}

sweep_tmp() {
  local year="$1" tmp_dir="$2" target_dir="$3" prefix="$4"

  echo "--> Sweeping and renaming files from ${tmp_dir} -> ${target_dir}"
  local found=0 moved=0 skipped=0 failed=0

  # Find every .hdf anywhere under tmp_dir
  while IFS= read -r -d '' file; do
    ((found++)) || true
    local filename ajul y j ymd new_filename
    filename=$(basename "$file")

    # Extract 'AYYYYDDD'
    ajul=$(echo "$filename" | grep -oE 'A[0-9]{7}' | head -n1 | tr -d 'A' || true)
    if [[ -z "${ajul:-}" || ${#ajul} -ne 7 ]]; then
      echo "   skip (no AYYYYDDD): $filename"
      ((skipped++)) || true
      continue
    fi

    y=${ajul:0:4}
    j=${ajul:4:3}

    # Convert Julian -> calendar; GNU date handles leap years
    if ! ymd=$(date -u -d "${y}-01-01 +$((10#$j - 1)) days" +%Y%m%d 2>/dev/null); then
      echo "   fail (date conversion): $filename"
      ((failed++)) || true
      continue
    fi

    new_filename="${OUT_STUB}_${ymd}.hdf"

    # If a file for the same YYYYMMDD already exists, keep the newest/last one (overwrite).
    if mv -f "$file" "$target_dir/$new_filename"; then
      echo "   saved: $target_dir/$new_filename"
      ((moved++)) || true
    else
      echo "   fail (move): $filename"
      ((failed++)) || true
    fi
  done < <(find "$tmp_dir" -type f -name "${prefix}*.hdf" -print0)

  echo "--> Year ${year} sweep summary: found=${found} moved=${moved} skipped=${skipped} failed=${failed}"
}

main() {
  mkdir -p "$BASE_SAVE"
  for year in $(seq "$YEAR_START" "$YEAR_END"); do
    download_year "$year"
  done
  echo "All done."
}

main "$@"

