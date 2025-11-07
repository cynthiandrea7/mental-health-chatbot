
#!/bin/bash

#This script loads environment variables from a .env file and runs the data ingestion script.

#--- 1. Sourcing Environment Variables ---

ENV_FILE=".env"
if [ -f "$ENV_FILE" ]; then
echo "Sourcing environment variables from $ENV_FILE..."
# 'source' (or '.') is used to execute the script in the current shell environment,
# making the variables available to subsequent commands like 'python'.
source "$ENV_FILE"
else
echo "FATAL ERROR: $ENV_FILE not found in the server directory. Please create it and fill in your secrets."
exit 1
fi

#-- 2. Check for critical variables ---

#Check if the required MongoDB and Google keys are set (they should be exported by .env)

if [ -z "$MONGO_USER" ] || [ -z "$MONGO_PASS" ] || [ -z "$MONGO_HOST" ] || [ -z "$GOOGLE_API_KEY" ] || [ -z "$KAGGLE_USERNAME" ] || [ -z "$KAGGLE_KEY" ]; then
echo "FATAL ERROR: One or more critical environment variables are missing or empty."
echo "Please ensure MONGO_USER, MONGO_PASS, MONGO_HOST, GOOGLE_API_KEY, KAGGLE_USERNAME, and KAGGLE_KEY are all set in your .env file."
exit 1
fi

#--- 3. Execute the Data Ingestion Script ---

echo "--- Starting data ingestion into MongoDB Atlas ---"

# We navigate up a directory (..) and into the notebooks folder to find the script.

python ../notebooks/ingest_data.py

echo "--- Script finished ---"