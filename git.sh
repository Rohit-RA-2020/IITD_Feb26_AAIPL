#!/bin/bash
set -e

# ====== USER CONFIGURATION ======
REPO_NAME="IITD_Feb26_AAIPL"
VISIBILITY="public"                    # "public" or "private"
COMMIT_MESSAGE="Initial commit"
GITHUB_USERNAME="Rohit-ra-2020"
GITHUB_EMAIL="ranjanrohit812@gmail.com"
# =================================

# Step 1: Install GitHub CLI if missing
if ! command -v gh &> /dev/null; then
    echo "GitHub CLI (gh) not found. Installing..."
    sudo apt update && sudo apt install -y curl git
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | \
      gpg --dearmor -o /usr/share/keyrings/githubcli-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | \
      sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
    sudo apt update && sudo apt install -y gh
fi

# Step 2: Authenticate GitHub CLI
if ! gh auth status &>/dev/null; then
    echo "Please authenticate with GitHub CLI:"
    gh auth login
fi

# Step 3: Use gh as git credential helper
gh auth setup-git

# Step 4: Git setup (local config only)
if [ ! -d .git ]; then
    git init
fi
git config user.name "$GITHUB_USERNAME"
git config user.email "$GITHUB_EMAIL"

# Step 5: Warn if no .gitignore
if [ ! -f .gitignore ]; then
    echo "⚠️  Warning: No .gitignore found. All files will be committed."
fi

git add .
git commit -m "$COMMIT_MESSAGE"
git branch -M main

# Step 6: Create GitHub repo (if it doesn't already exist)
REMOTE_URL="https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"

if gh repo view "$GITHUB_USERNAME/$REPO_NAME" &>/dev/null; then
    echo "Repo already exists on GitHub. Skipping creation."
else
    echo "Creating remote repo on GitHub..."
    if [ "$VISIBILITY" = "private" ]; then
        gh repo create "$REPO_NAME" --private
    else
        gh repo create "$REPO_NAME" --public
    fi
fi

# Step 7: Set remote & push
git remote set-url origin "$REMOTE_URL" 2>/dev/null || git remote add origin "$REMOTE_URL"

echo "Pushing code to GitHub..."
git push -u origin main

echo "Done! Repo pushed to: $REMOTE_URL"
