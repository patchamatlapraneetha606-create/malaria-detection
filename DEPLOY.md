# Deploy to Streamlit Community Cloud

Follow these steps to get a public URL for your Malaria Detection app.

## Step 1: Initialize Git and push to GitHub

1. Open Terminal in the project folder.

2. Initialize git (if not already done):
   ```bash
   cd /Users/vallipatchamatla/Downloads/praneeproject
   git init
   ```

3. Set your Git identity (first time only):
   ```bash
   git config user.email "your-email@example.com"
   git config user.name "Your Name"
   ```

4. Add all files:
   ```bash
   git add .
   ```

5. Commit:
   ```bash
   git commit -m "Malaria Detection System - Hackathon submission"
   ```

6. Create a new repo on GitHub:
   - Go to [github.com](https://github.com) → New repository
   - Name it `malaria-detection` (or any name)
   - Do NOT add README, .gitignore, or license (we already have them)
   - Click Create repository

7. Connect and push:
   ```bash
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/malaria-detection.git
   git push -u origin main
   ```
   Replace `YOUR_USERNAME` with your GitHub username.

## Step 2: Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)

2. Sign in with your GitHub account.

3. Click **New app**.

4. Fill in:
   - **Repository:** your-username/malaria-detection
   - **Branch:** main
   - **Main file path:** app.py

5. Click **Deploy**.

6. Wait 2–5 minutes. Your app will be live at:
   `https://YOUR_APP_NAME.streamlit.app`

## Share your link

Use the Streamlit URL in your hackathon submission so judges can try the app directly.
