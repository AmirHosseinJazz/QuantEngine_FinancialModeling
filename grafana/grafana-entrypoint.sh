#!/bin/bash
grafana-cli plugins install yesoreyeram-infinity-datasource

# Use envsubst to substitute environment variables
envsubst < /etc/grafana/provisioning/datasources/datasource.yml.tmpl > /etc/grafana/provisioning/datasources/datasource.yml

# Call the original Grafana entrypoint script
exec /run.sh