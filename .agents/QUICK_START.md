# Agent System Quick Start

**TL;DR (run from repo root):**

```bash
# Ensure you're in the repository root directory
cat .agents/handoffs/CURRENT_STATE.md
cat .agents/contexts/<your_agent>_context.md
# do work...
./scripts/agent_handoff.sh <agent> "What you did"
```

## Starting a Work Session

### Step 0: Go to repo root

```bash
# Navigate to your local repository root directory
# All commands in this guide should be run from the repo root
```

### Step 1: Choose Your Agent

- **Agent 1 (Frontend):** Working on React UI
- **Agent 2 (Audio):** Working on Rust/C++ backend
- **Agent 3 (Music Brain):** Working on Python music generation
- **Agent 4 (DevOps):** Working on docs/build/tests

### Step 2: Read Current State

```bash
cat .agents/handoffs/CURRENT_STATE.md
```

### Step 3: Read Your Context

```bash
# For Agent 1:
cat .agents/contexts/frontend_context.md

# For Agent 2:
cat .agents/contexts/audio_context.md

# For Agent 3:
cat .agents/contexts/musicbrain_context.md

# For Agent 4:
cat .agents/contexts/devops_context.md
```

### Step 4: Start Working

Follow priorities listed in your context file; only touch files relevant to your agent.

## Ending a Work Session

### Step 1: Commit Your Work

```bash
git add .
git commit -m "[Agent Name] What you did"
git push origin main
```

### Step 2: Log Handoff

```bash
./scripts/agent_handoff.sh [agent_name] "Summary of work"
```

Creates a timestamped log in `.agents/logs/`.

### Step 3: Update Current State

Edit `.agents/handoffs/CURRENT_STATE.md` with:

- What you accomplished
- What's blocking you
- What's next

## Example Full Session

```bash
# START SESSION
# From the repository root directory:
cat .agents/handoffs/CURRENT_STATE.md
cat .agents/contexts/musicbrain_context.md

# DO WORK
# ... coding happens here ...

# END SESSION
git add .
git commit -m "[Music Brain] Built FastAPI server with /generate endpoint"
git push origin main
./scripts/agent_handoff.sh musicbrain "Created API server, tested with Kelly song"
vim .agents/handoffs/CURRENT_STATE.md  # update status
```

## Tools & Commands Reference

### View all agent logs

```bash
ls -lt .agents/logs/
```

### View specific agent's work

```bash
cat .agents/logs/frontend_*.md
```

### Check what files each agent should touch

```bash
grep "Key Files" .agents/contexts/*.md
```

### See agent priorities

```bash
grep -A 10 "Current Priorities" .agents/contexts/*.md
```
