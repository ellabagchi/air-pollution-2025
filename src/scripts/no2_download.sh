#!/bin/bash
# downloads NO2 data 2005-2024, puts it in ubuntu full_data-- successful!
# Total number of years (used for basic progress)
# start_year=2005
# end_year=2024
# total_years=$((end_year - start_year + 1))
# year_count=0

# for year in $(seq $start_year $end_year); do
#     year_count=$((year_count + 1))

#     echo -e "\n📦 Downloading year $year ($year_count of $total_years)..."

#     base_url="https://acdisc.gesdisc.eosdis.nasa.gov/data/Aura_OMI_Level3/OMNO2d.003/${year}"
#     prefix="OMI-Aura_L3-OMNO2d_${year}"

#     tmp_dir="./omi_no2_tmp_${year}"
#     target_dir="/home/ellab/air_pollution/src/data/full_data/${year}"

#     mkdir -p "$tmp_dir"
#     mkdir -p "$target_dir"

#     wget --load-cookies ~/.urs_cookies \
#          --save-cookies ~/.urs_cookies \
#          --keep-session-cookies \
#          --directory-prefix="$tmp_dir" \
#          --accept "${prefix}*.he5" \
#          --no-parent --no-directories --cut-dirs=6 \
#          --recursive --level=1 \
#          "$base_url/"

#     total_files=$(ls "$tmp_dir"/*.he5 2>/dev/null | wc -l)
#     current_file=0

#     for file in "$tmp_dir"/${prefix}*.he5; do
#         if [[ -f "$file" ]]; then
#             filename=$(basename "$file")
#             datetag=$(echo "$filename" | grep -oP "${year}m\d{4}")
#             if [[ -n "$datetag" ]]; then
#                 y=${datetag:0:4}
#                 m=${datetag:6:2}
#                 d=${datetag:8:2}
#                 new_filename="omi_no2_${y}${m}${d}.he5"
#                 target_path="$target_dir/$new_filename"

#                 if [[ -f "$target_path" ]]; then
#                     echo "⚠️  Exists: $new_filename — skipping"
#                 else
#                     current_file=$((current_file + 1))
#                     echo -n "📄 Moving ($current_file/$total_files): $new_filename "
#                     pv -q "$file" > "$target_path" && rm "$file"
#                     echo "✅"
#                 fi
#             else
#                 echo "❌ Date not found in: $filename"
#             fi
#         fi
#     done

#     rm -rf "$tmp_dir"
#     echo "🧹 Cleaned up: $tmp_dir"

# done

# echo -e "\n🎉 Done downloading OMNO2d data from $start_year to $end_year!"


# this code works for downloading no2 daily data 1 year at a time-- change the year as needed
for year in 2024; do


    base_url="https://acdisc.gesdisc.eosdis.nasa.gov/data/Aura_OMI_Level3/OMNO2d.003/${year}"
    prefix="OMI-Aura_L3-OMNO2d_${year}"


    tmp_dir="./omi_no2_tmp_${year}"
    target_dir="/home/ellab/air_pollution/src/data/new_no2/${year}"


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
                new_filename="omi_no2_${y}${m}${d}.he5"
                mv "$file" "$target_dir/$new_filename"
                echo "Saved: $target_dir/$new_filename"
            else
                echo "Failed to get date from $filename"
            fi
        fi
    done


    rm -rf "$tmp_dir"
done
