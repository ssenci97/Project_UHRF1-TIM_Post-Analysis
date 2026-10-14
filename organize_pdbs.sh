#!/bin/bash

SOURCE_DIR="data/synced_pdbs"
DEST_BASE="data/models_by_group"

mkdir -p "$DEST_BASE"

if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory $SOURCE_DIR does not exist."
    exit 1
fi

copied_count=0
skipped_count=0

for filepath in "$SOURCE_DIR"/*.pdb; do
    [ -e "$filepath" ] || continue
    filename=$(basename "$filepath")

    # 1. Skip files matching 'paireval'
    if [[ "$filename" == *paireval* ]]; then
        ((skipped_count++))
        echo "Skipping (paireval): $filename"
        continue
    fi

    group_name=""
    new_filename="$filename"

    # 2. Handle AlphaFold format starting with 'fold_a_' (e.g., fold_a_tim_pab_x_b_uhrf1_ubl_model_0.pdb)
    if [[ "$filename" =~ ^fold_a_(.*)_x_b_(.*)_(model_[0-9]+\.pdb|abin_emref_[0-9]+\.pdb|pdbeval_.*)$ ]]; then
        left_part="${BASH_REMATCH[1]}"
        right_part="${BASH_REMATCH[2]}"
        suffix_part="${BASH_REMATCH[3]}"

        # Replace underscores with hyphens and convert to uppercase
        p1_formatted=$(echo "$left_part" | tr '_' '-' | tr '[:lower:]' '[:upper:]')
        p2_formatted=$(echo "$right_part" | tr '_' '-' | tr '[:lower:]' '[:upper:]')
        
        group_name="${p1_formatted}_x_${p2_formatted}"
        new_filename="${group_name}_${suffix_part}"

    # 3. Handle already structured format (e.g., TIM-DBD_x_UHRF1-PHD_abin_emref_130.pdb)
    elif [[ "$filename" =~ ^(.*_x_.*)_(abin_emref_[0-9]+\.pdb|pdbeval_.*|model_[0-9]+\.pdb)$ ]]; then
        group_name="${BASH_REMATCH[1]}"
        new_filename="$filename"
    fi

    # 4. Copy to group directory
    if [ -n "$group_name" ]; then
        group_dir="$DEST_BASE/$group_name"
        mkdir -p "$group_dir"
        cp "$filepath" "$group_dir/$new_filename"
        ((copied_count++))
    else
        echo "Warning: Could not parse pattern for file: $filename"
    fi
done

echo "--- Summary ---"
echo "Successfully copied: $copied_count files"
echo "Skipped (paireval): $skipped_count files"
echo "Target directory: $(realpath "$DEST_BASE")"

