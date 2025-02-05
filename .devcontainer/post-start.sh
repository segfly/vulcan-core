#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Latchfield Technologies http://latchfield.com

set -o errexit
set -o nounset

sudo service cron start

# Configure automatic git commit signing using SSH agent key if available
if [ -z "$(git config gpg.format)" ]; then
    key_count=$(ssh-add -L | wc -l || true)
    if [ "$key_count" -eq 1 ]; then
        set -o xtrace
        git config --global gpg.format ssh
        git config --global user.signingkey "$(ssh-add -L)"
        git config --global commit.gpgsign true
        set +o xtrace
        echo "INFO: Git commit signing configured with the SSH key from SSH agent."
    elif [ "$key_count" -gt 1 ]; then
        echo "WARNING: More than one SSH key presented by the SSH agent. Skipping git commit signing configuration."
    else
        echo "WARNING: SSH agent or key not available. Skipping git commit signing configuration."
    fi
else
    echo "INFO: Git signing already configured, skipping SSH signing configuration."
fi