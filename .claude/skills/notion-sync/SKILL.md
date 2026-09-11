---
name: notion-sync
description: Reconcile the private Notion "Pelagos-py Tasks" tracker with GitHub issues/PRs and the current branch. Use when the user asks to sync, check, or update the Notion tracker, or after a batch of fixes.
---

# Notion tracker sync

Tracker: https://app.notion.com/p/c48ec90236e24fedbfbb537ef967251d
Data source: `collection://b79d601d-cfe6-42af-8bd0-9c4e1cde6fd8`
Repo for links: https://github.com/NOC-OBG-Autonomy/pelagos-py (never the Orlando-PB fork).

Never push, comment on, close, or create anything on GitHub. Read-only `gh` calls only.

## Schema
- Status (select, in order): Not started / Started / In branch / PR open / On main
- Type, Area (multi), Priority (High/Medium/Low), Milestone (V1.0.0/Later)
- Branch (text), GitHub (issue url), PR (url), Notes
- Issue open (checkbox): linked issue still open on GitHub
- Action (formula): "🔴 Close GitHub issue" when Status = On main and Issue open

## Steps
1. `gh issue list --state open --limit 100 --json number,title,labels,author` and `gh pr list --state all --limit 30 --json number,title,state,headRefName`.
2. Query all rows (SQL mode: url, Name, Status, GitHub, PR, Issue open, Notes).
3. For every open issue with no row: create one (Type/Area/Priority from labels and title, Milestone from #168 body, Notes with reporter, Issue open ticked).
4. For every row with a GitHub link: set Issue open to match GitHub. If an issue was closed, untick.
5. For rows with a PR link: if the PR merged, set Status = On main. If open, Status = PR open.
6. For rows In branch: check `git branch --contains` / `git log main..HEAD` so anything now on main becomes On main.
7. Report what changed, and list rows showing "🔴 Close GitHub issue" so the user can close them by hand.

## Row conventions
- Keep Notes short and factual: what exists, where (file, commit, PR), what is still missing, "not run-tested" where you only read code.
- Don't mark On main unless the change is actually in `main` (check with `git show main:path` or the merged PR).
