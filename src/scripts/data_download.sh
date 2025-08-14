#!/bin/bash
# this code works for downloading daily data 1 year at a time-- change the year as needed, change variable (NO2, O3, HCHO) as needed
for year in 2013; do


    # base_url="https://acdisc.gesdisc.eosdis.nasa.gov/data/Aura_OMI_Level3/OMNO2d.003/${year}" # uncomment for NO2-- change all variables accordingly
    # prefix="OMI-Aura_L3-OMNO2d_${year}"

    # base_url="https://acdisc.gsfc.nasa.gov/data/Aura_OMI_Level3/OMDOAO3e.003/${year}" #  uncommment for O3
    # prefix="OMI-Aura_L3-OMDOAO3e_${year}"

    base_url="https://acdisc.gsfc.nasa.gov/data/Aura_OMI_Level2G/OMHCHOG.003/${year}" #  uncommment for HCHO
    prefix="OMI-Aura_L2G-OMHCHOG_${year}"

    tmp_dir="./omi_hcho_tmp_${year}"
    target_dir="/home/ellab/air_pollution/src/data/new_hcho/${year}"


    mkdir -p "$tmp_dir"
    mkdir -p "$target_dir"


    wget --load-cookies ~/.urs_cookies \
         --save-cookies ~/.urs_cookies \
         --keep-session-cookies \
         --directory-prefix="$tmp_dir" \
         --accept "${prefix}*.he5" \
         --no-parent --no-directories --cut-dirs=6 \
         --recursive --level=1 \
         "$base_url/"


    for file in "$tmp_dir"/${prefix}*.he5; do
        if [[ -f "$file" ]]; then
            filename=$(basename "$file")
            datetag=$(echo "$filename" | grep -oP "${year}m\d{4}" | sed 's/m//')
            if [[ -n "$datetag" ]]; then
                y=${datetag:0:4}
                m=${datetag:4:2}
                d=${datetag:6:2}
                new_filename="omi_hcho_${y}${m}${d}.he5"
                mv "$file" "$target_dir/$new_filename"
                echo "Saved: $target_dir/$new_filename"
            else
                echo "Failed to get date from $filename"
            fi
        fi
    done


    rm -rf "$tmp_dir"
done
