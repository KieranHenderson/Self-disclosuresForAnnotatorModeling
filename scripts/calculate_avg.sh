#!/bin/bash

# Check if folder path was provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <folder_path> [output_file]"
    exit 1
fi

folder_path="$1"
output_file="${2:-last_lines_summary.txt}"
temp_file=$(mktemp)

# Verify the folder exists
if [ ! -d "$folder_path" ]; then
    echo "Error: Folder '$folder_path' does not exist"
    exit 1
fi

# Clear or create the output file
> "$output_file"

# Process each file in the folder and collect metrics
declare -A groups
declare -A counts

# Process each file in the folder
for file in "$folder_path"/*; do
    # Skip directories
    if [ -d "$file" ]; then
        continue
    fi
    
    # Get the filename without path
    filename=$(basename "$file")

    # Extract the group name (everything before last underscore and digit)
    group_name=$(echo "$filename" | sed -E 's/(.*)_[0-9]+(\..*)?$/\1/')
    
    # Get the last line of the file
    last_line=$(tail -n 1 "$file" 2>/dev/null)
    
    # Skip empty files
    if [ -z "$last_line" ]; then
        continue
    fi

    # Extract the metrics dictionary portion
    metrics_part="${last_line#*\[INFO\] }"

    # Extract each metric value
    accuracy=$(echo "$metrics_part" | grep -oP "'accuracy'\s*:\s*\K[0-9.]+")
    f1_weighted=$(echo "$metrics_part" | grep -oP "'f1_weighted'\s*:\s*\K[0-9.]+")
    macro=$(echo "$metrics_part" | grep -oP "'macro'\s*:\s*\K[0-9.]+")
    micro=$(echo "$metrics_part" | grep -oP "'micro'\s*:\s*\K[0-9.]+")
    binary=$(echo "$metrics_part" | grep -oP "'binary'\s*:\s*\K[0-9.]+")
    
    # Add to group's running totals
    groups["${group_name}_accuracy"]=$(bc <<< "${groups["${group_name}_accuracy"]:-0} + $accuracy")
    groups["${group_name}_f1_weighted"]=$(bc <<< "${groups["${group_name}_f1_weighted"]:-0} + $f1_weighted")
    groups["${group_name}_macro"]=$(bc <<< "${groups["${group_name}_macro"]:-0} + $macro")
    groups["${group_name}_micro"]=$(bc <<< "${groups["${group_name}_micro"]:-0} + $micro")
    groups["${group_name}_binary"]=$(bc <<< "${groups["${group_name}_binary"]:-0} + $binary")
    
    # Increment count for this group
    counts["$group_name"]=$(( ${counts["$group_name"]:-0} + 1 ))
done

# Calculate averages and write to output file
for group_name in "${!counts[@]}"; do
    count=${counts["$group_name"]}
    
    # Calculate averages and convert to percentages with 1 decimal place
    accuracy_avg=$(bc <<< "scale=6; ${groups["${group_name}_accuracy"]} / $count * 100" | awk '{printf "%.1f", $1}')
    f1_weighted_avg=$(bc <<< "scale=6; ${groups["${group_name}_f1_weighted"]} / $count * 100" | awk '{printf "%.1f", $1}')
    macro_avg=$(bc <<< "scale=6; ${groups["${group_name}_macro"]} / $count * 100" | awk '{printf "%.1f", $1}')
    micro_avg=$(bc <<< "scale=6; ${groups["${group_name}_micro"]} / $count * 100" | awk '{printf "%.1f", $1}')
    binary_avg=$(bc <<< "scale=6; ${groups["${group_name}_binary"]} / $count * 100" | awk '{printf "%.1f", $1}')
    
    # Write to output file
    printf "%s,%s,%s,%s,%s,%s,%d\n" "$group_name" "$accuracy_avg" "$f1_weighted_avg" "$macro_avg" "$micro_avg" "$binary_avg" "$count" >> "$temp_file"
done

# Sort the output by group name and write to final file
sort "$temp_file" >> "$output_file"
rm "$temp_file"

echo "Averaged metrics summary written to: $output_file"