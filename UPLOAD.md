# Upload checklist

This directory intentionally does not contain a `.git` directory. To update the existing GitHub
repository while preserving its history:

```powershell
cd "<path-to-your-clone>\eNRRCrew"
git init
git branch -M main
git remote add origin https://github.com/nkuhuxu/eNRRCrew.git
git fetch origin main
git reset origin/main
git add -A
git status
git diff --cached --stat
git commit -m "Replace legacy app with the uv-based Streamlit application"
git push origin main
```

`git reset origin/main` above is a mixed reset: it connects the staging directory to the remote
history without discarding these prepared files. Review `git status` before committing.
Because `git add -A` stages removals too, the commit replaces the tracked legacy application
instead of leaving `appUI.py` or `eNRRCrew_chainlit/` in the repository.

Security checks before pushing:

```powershell
Get-ChildItem -Recurse -Force -Filter .env
git diff --cached -- .env
git grep -n "GRAPHRAG_API_KEY=" -- ':!*.example'
```

The prepared tree contains no populated `.env`. The current public repository previously tracked
an `.env` file, so rotate any credential that may ever have been stored there and consider purging
it from Git history separately if it contained a real secret.
