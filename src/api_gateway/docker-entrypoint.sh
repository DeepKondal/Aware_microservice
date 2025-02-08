#!/bin/bash
set -e

# Validate P_VALUE
if [ -z "$P_VALUE" ]; then
    echo "Error: P_VALUE environment variable not set"
    exit 1
fi

echo $P_VALUE
# Set weights
export USER_SERVICE_V1_WEIGHT=$P_VALUE
export USER_SERVICE_V2_WEIGHT=$((100 - P_VALUE))

echo $USER_SERVICE_V1_WEIGHT
echo $USER_SERVICE_V2_WEIGHT

# Replace environment variables in kong.yml.template
envsubst < /etc/kong/kong.yml.template > /etc/kong/kong.yml

cat /etc/kong/kong.yml

# Prepare Kong prefix directory
kong prepare -p /usr/local/kong

# Start Kong
exec kong start --nginx-conf /usr/local/kong/nginx.conf --vv