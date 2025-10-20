#!/bin/bash
# Minimal changes: year-agnostic date parsing + nullglob, works across many years.
# Uncomment ONE product block (NO2 or HCHO) before running.

# for year in the full span you need:
for year in {2016..2024}; do

    ##### --- CHOOSE ONE PRODUCT BLOCK --- #####

    # --- NO2 (OMNO2d, Level-3, daily) ---
    base_url="https://acdisc.gesdisc.eosdis.nasa.gov/data/Aura_OMI_Level3/OMNO2d.003/${year}"
    prefix="OMI-Aura_L3-OMNO2d_${year}"
    target_dir="/home/ellab/air_pollution/src/data/new_no2/${year}"
    out_stub="omi_no2"

    # --- HCHO (OMHCHOG, Level-2G, daily) ---
    # base_url="https://acdisc.gsfc.nasa.gov/data/Aura_OMI_Level2G/OMHCHOG.003/${year}"
    # prefix="OMI-Aura_L2G-OMHCHOG_${year}"
    # target_dir="/home/ellab/air_pollution/src/data/new_hcho/${year}"
    # out_stub="omi_hcho"

    ##### -------------------------------- #####

    tmp_dir="./omi_tmp_${year}"
    mkdir -p "$tmp_dir" "$target_dir"

    wget --load-cookies ~/.urs_cookies \
         --save-cookies ~/.urs_cookies \
         --keep-session-cookies \
         --directory-prefix="$tmp_dir" \
         --accept "${prefix}*.he5" \
         --no-parent --no-directories --cut-dirs=6 \
         --recursive --level=1 --continue --tries=3 \
         "$base_url/"

    # Avoid iterating a literal glob if no files; parse YYYYmMMDD independent of $year
    shopt -s nullglob
    for file in "$tmp_dir"/${prefix}*.he5; do
        [[ -f "$file" ]] || continue
        filename=$(basename "$file")
        datetag=$(echo "$filename" | grep -oE '[0-9]{4}m[0-9]{4}' | head -n1 | tr -d 'm')
        if [[ -n "$datetag" && ${#datetag} -eq 8 ]]; then
            y=${datetag:0:4}; m=${datetag:4:2}; d=${datetag:6:2}
            new_filename="${out_stub}_${y}${m}${d}.he5"
            mv -f "$file" "$target_dir/$new_filename"
            echo "Saved: $target_dir/$new_filename"
        else
            echo "Failed to get date from $filename"
        fi
    done
    shopt -u nullglob

    rm -rf "$tmp_dir"
done
