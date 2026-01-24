#!/bin/bash

# Script to backup subdirectories from one S3 bucket to another
# Source and destination buckets
# SOURCE_BUCKET="s3://infini-gram-lite/index"
SOURCE_BUCKET="s3://infini-gram/index"
DEST_BUCKET="s3://infini-gram-backup/index"

echo "=========================================="
echo "S3 Backup Script"
echo "Source: $SOURCE_BUCKET"
echo "Destination: $DEST_BUCKET"
echo "=========================================="
echo ""

# List top-level subdirectories under the source bucket
echo "Listing subdirectories under $SOURCE_BUCKET..."
SUBDIRS=$(aws s3 ls "$SOURCE_BUCKET/" | grep "PRE" | awk '{print $2}')

# Check if any subdirectories were found
if [ -z "$SUBDIRS" ]; then
    echo "No subdirectories found under $SOURCE_BUCKET"
    exit 1
fi

echo "Found the following subdirectories:"
echo "$SUBDIRS"
echo ""

# Loop through each subdirectory and copy it to the destination
for SUBDIR in $SUBDIRS; do
    # Remove trailing slash
    SUBDIR_NAME=${SUBDIR%/}

    echo "=========================================="
    echo "Copying: $SUBDIR_NAME"
    echo "From: $SOURCE_BUCKET/$SUBDIR_NAME/"
    echo "To: $DEST_BUCKET/$SUBDIR_NAME/"
    echo "=========================================="

    # Use aws s3 sync to copy the directory
    # --no-progress reduces output noise
    # --storage-class DEEP_ARCHIVE puts files in Glacier Deep Archive (cheapest long-term storage)
    # You can add --dryrun flag to test without actually copying
    aws s3 cp "$SOURCE_BUCKET/$SUBDIR_NAME/" "$DEST_BUCKET/$SUBDIR_NAME/" --recursive --no-progress --storage-class DEEP_ARCHIVE

    if [ $? -eq 0 ]; then
        echo "✓ Successfully copied $SUBDIR_NAME"
    else
        echo "✗ Failed to copy $SUBDIR_NAME"
        exit 1
    fi
    echo ""
done

echo "=========================================="
echo "Backup completed successfully!"
echo "=========================================="

