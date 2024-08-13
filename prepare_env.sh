#!/bin/bash

folder_names=("grafana" "grafana/grafana_data")

# Loop through each folder name
for folder_name in "${folder_names[@]}"; do
    # Check if the folder does not exist
    if [ ! -d "$folder_name" ]; then
        # Create the folder
        mkdir "$folder_name"
        echo "Folder created: $folder_name"
    else
        echo "Folder already exists: $folder_name"
    fi

    sudo chmod 755 "$folder_name"
    echo "Permissions set to 755 for folder: $folder_name"
done


# # Define the directories where Dockerfiles are located
directories=("dashboard" "mlflow" "grafana")

# Loop through each directory and copy the .env file there
for dir in "${directories[@]}"; do
  # Check if .env exists in the destination, if so, remove it first to avoid copying into a directory
  if [ -d "./$dir/.env" ]; then
    rm -r "./$dir/.env"
  fi

  # Copy .env to the directory containing the Dockerfile
  cp .env ./$dir/
done

# echo "Copied .env to all specified directories."


chmod +x ./grafana/grafana-entrypoint.sh