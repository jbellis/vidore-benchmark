#!/bin/bash

# Initial values
train_files=200
val_files=50
max_train_files=6400
max_val_files=400

# Run loop until max files reached
while [ $train_files -le $max_train_files ]; do
    echo "Running training with $train_files train files and $val_files validation files"
    
    # Run the training command and time it
    time python finetune.py --train-files $train_files --val-files $val_files
    
    # Double the number of files for next iteration
    train_files=$((train_files * 2))
    
    # Double val files but cap at max_val_files
    val_files=$((val_files * 2))
    if [ $val_files -gt $max_val_files ]; then
        val_files=$max_val_files
    fi
    
    echo "----------------------------------------"
done

echo "Training complete!"
