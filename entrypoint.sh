#!/bin/bash
cd /data || exit

# Create and start virtual env
python3 -m venv venv
source venv/bin/activate

# Install / check requirements on startup
echo "Installing requirements..."
pip3 install --upgrade pip
pip3 install --upgrade -r requirements.txt

# Main loop
while true
do
  # Fetch repo info
  git fetch
  changed=0
  git_branch="$(git rev-parse --abbrev-ref HEAD)"
  local_commit="$(git rev-parse "$git_branch")"
  origin_commit="$(git rev-parse origin/"$git_branch")"
  changed_files="$(git diff --name-only "$local_commit" "$origin_commit")"

  # For manually running everything
  if [ ! -e /data/.updated ];
  then
    echo "No updated-file found."
    changed=1
  fi

  # Check for changes in relevant files of the new commit
  if [ $changed = 0 ] && [ ! "$origin_commit" = "$local_commit" ];
  then
    # Pull changes
    echo "New git commit found on branch $git_branch: $local_commit (remote: $origin_commit). Pulling repo."
    git pull
    if [[ $changed_files =~ config.json ]] || [[ $changed_files =~ main.py ]] || [[ $changed_files =~ cluster_experiments.py ]] || [[ $changed_files =~ generate_dataset.py ]] || [[ $changed_files =~ feature_engineering.py ]] || [[ $changed_files =~ create_model.py ]] || [[ $changed_files =~ train.py ]];
    then
      echo "Changes in relevant files found."
      changed=1
      rm /data/.updated
    fi
  fi

  if [ $changed = 1 ];
  then
    # Run main pipeline
    echo "Running crystalML pipeline..."
    python3 /data/main.py
    # Prevent loop
    touch /data/.updated
  fi
  sleep 30s
done