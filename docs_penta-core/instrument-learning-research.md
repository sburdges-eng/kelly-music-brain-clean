# Instrument Learning Research Tool

An automated GitHub repository research tool for discovering instrument learning and music education applications.

---

## Overview

This tool automates the discovery of open-source repositories related to instrument learning, music education, and practice applications. It uses the GitHub Search API with AI-generated query variations to maximize coverage.

---

## Requirements

```
openai>=1.0.0
requests>=2.31.0
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GITHUB_TOKEN` | Yes | GitHub Personal Access Token for API access |
| `OPENAI_API_KEY` | Yes | OpenAI API key for query generation |

---

## Configuration

```python
# ----------------------------
# Config (tune if you must)
# ----------------------------
RUNS = 100                          # Number of research runs

# Per run: GitHub search provides up to 1,000 results per search
PER_PAGE = 100                      # Results per API page
MAX_PAGES = 10                      # 10 * 100 = 1,000 max results
SORT = "stars"                      # Sort by star count
ORDER = "desc"                      # Descending order (most stars first)

# GitHub Search rate limit: 30 requests/min authenticated
MIN_SECONDS_BETWEEN_GH_CALLS = 2.2  # Respect rate limits

# Timeout/reset behavior
REQUEST_TIMEOUT_SEC = 20            # API request timeout
RUN_DEADLINE_SEC = 240              # Per-run wall-clock deadline
COOLDOWN_ON_RESET_SEC = 3.0         # Pause on run timeout/errors
MAX_SLEEP_FOR_RATE_RESET_SEC = 120  # Max wait for rate limit reset

# Output paths
OUT_DIR = "research/instrument-learning"
STATE_PATH = os.path.join(OUT_DIR, "state.json")
AGG_TXT = os.path.join(OUT_DIR, "all_repos_deduped.txt")
AGG_JSONL = os.path.join(OUT_DIR, "all_repos_deduped.jsonl")

# AI Model
MODEL_LOW_COST = "gpt-4o-mini"      # Fast/affordable for query generation
```

---

## Search Strategy

### Base Qualifiers

```python
BASE_QUALIFIERS = "fork:false archived:false"
```

Excludes forks and archived repositories to focus on original, active projects.

### Search Hints

The tool uses these seed topics for AI-powered query expansion:

```python
BASE_HINTS = [
    "ear training",
    "sight reading", 
    "fretboard",
    "guitar trainer",
    "piano learning",
    "solfege",
    "interval trainer",
    "rhythm trainer",
    "metronome",
    "chord trainer",
    "scales practice",
    "music theory",
    "drum practice",
    "ukulele",
    "violin practice",
    "MIDI trainer",
    "sheet music",
    "tab",
    "practice app"
]
```

---

## Data Model

### RepoRow

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class RepoRow:
    full_name: str      # e.g., "username/repo-name"
    html_url: str       # GitHub URL
    stars: int          # Star count
    language: str       # Primary programming language
    description: str    # Repository description
    run_id: int         # Which research run found this
    query: str          # The search query that found it
```

---

## Core Components

### 1. Query Generator

Uses OpenAI's GPT-4o-mini to generate diverse search queries:

```python
def generate_queries(hint: str, count: int = 5) -> list[str]:
    """
    Generate search query variations for a given topic hint.
    
    Args:
        hint: Base topic (e.g., "ear training")
        count: Number of query variations to generate
        
    Returns:
        List of GitHub search queries
    """
    client = OpenAI()
    response = client.chat.completions.create(
        model=MODEL_LOW_COST,
        messages=[
            {"role": "system", "content": "Generate GitHub search queries for finding music education repositories."},
            {"role": "user", "content": f"Generate {count} search queries related to: {hint}"}
        ]
    )
    return parse_queries(response.choices[0].message.content)
```

### 2. GitHub Search Client

Handles paginated search with rate limit awareness:

```python
def search_github(query: str, page: int = 1) -> dict:
    """
    Search GitHub repositories with rate limit handling.
    
    Args:
        query: Search query with qualifiers
        page: Page number (1-indexed)
        
    Returns:
        GitHub API response JSON
    """
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }
    params = {
        "q": f"{query} {BASE_QUALIFIERS}",
        "sort": SORT,
        "order": ORDER,
        "per_page": PER_PAGE,
        "page": page
    }
    response = requests.get(
        "https://api.github.com/search/repositories",
        headers=headers,
        params=params,
        timeout=REQUEST_TIMEOUT_SEC
    )
    handle_rate_limit(response)
    return response.json()
```

### 3. Rate Limit Handler

Respects GitHub API limits:

```python
def handle_rate_limit(response: requests.Response) -> None:
    """
    Handle GitHub rate limiting with exponential backoff.
    
    Checks X-RateLimit-Remaining header and sleeps if needed.
    """
    remaining = int(response.headers.get("X-RateLimit-Remaining", 1))
    reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
    
    if remaining == 0:
        sleep_time = min(
            reset_time - time.time() + 1,
            MAX_SLEEP_FOR_RATE_RESET_SEC
        )
        if sleep_time > 0:
            print(f"Rate limited. Sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)
```

### 4. State Manager

Persists progress for resumable runs:

```python
def load_state() -> dict:
    """Load state from disk or return empty state."""
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"completed_runs": [], "seen_repos": set()}

def save_state(state: dict) -> None:
    """Persist current state to disk."""
    # Convert set to list for JSON serialization
    state_copy = state.copy()
    state_copy["seen_repos"] = list(state.get("seen_repos", []))
    with open(STATE_PATH, "w") as f:
        json.dump(state_copy, f, indent=2)
```

### 5. Deduplication

Ensures unique repositories across all runs:

```python
def deduplicate_results(repos: list[RepoRow], seen: set) -> list[RepoRow]:
    """
    Filter out already-seen repositories.
    
    Args:
        repos: New repositories from search
        seen: Set of already-seen full_names
        
    Returns:
        List of new, unique repositories
    """
    new_repos = []
    for repo in repos:
        if repo.full_name not in seen:
            seen.add(repo.full_name)
            new_repos.append(repo)
    return new_repos
```

---

## Output Formats

### Text Format (`all_repos_deduped.txt`)

Simple list for quick review:

```
username/repo-name (1234 ⭐) - A great ear training app
another/project (567 ⭐) - Piano learning with MIDI
```

### JSONL Format (`all_repos_deduped.jsonl`)

Structured data for analysis:

```jsonl
{"full_name": "username/repo-name", "html_url": "https://github.com/...", "stars": 1234, "language": "Python", "description": "A great ear training app", "run_id": 1, "query": "ear training app"}
{"full_name": "another/project", "html_url": "https://github.com/...", "stars": 567, "language": "JavaScript", "description": "Piano learning with MIDI", "run_id": 2, "query": "piano MIDI trainer"}
```

---

## Main Execution Flow

```python
def main():
    """Run the research automation."""
    os.makedirs(OUT_DIR, exist_ok=True)
    state = load_state()
    
    for run_id in range(1, RUNS + 1):
        if run_id in state["completed_runs"]:
            continue
            
        print(f"\n=== Run {run_id}/{RUNS} ===")
        run_start = time.time()
        
        try:
            # Pick a random hint and generate queries
            hint = random.choice(BASE_HINTS)
            queries = generate_queries(hint)
            
            # Search for each query
            for query in queries:
                if time.time() - run_start > RUN_DEADLINE_SEC:
                    print("Run deadline exceeded, moving on")
                    break
                    
                all_repos = []
                for page in range(1, MAX_PAGES + 1):
                    result = search_github(query, page)
                    repos = parse_repos(result, run_id, query)
                    all_repos.extend(repos)
                    
                    if len(repos) < PER_PAGE:
                        break  # No more results
                        
                    time.sleep(MIN_SECONDS_BETWEEN_GH_CALLS)
                
                # Deduplicate and save
                new_repos = deduplicate_results(all_repos, state["seen_repos"])
                append_to_outputs(new_repos)
                
            state["completed_runs"].append(run_id)
            save_state(state)
            
        except Exception as e:
            print(f"Run {run_id} failed: {e}")
            time.sleep(COOLDOWN_ON_RESET_SEC)
    
    print(f"\nComplete! Results in {OUT_DIR}/")
```

---

## Usage

### Basic Execution

```bash
# Set environment variables
export GITHUB_TOKEN="ghp_xxxx..."
export OPENAI_API_KEY="sk-xxxx..."

# Run the research
python research_tool.py
```

### Resuming Interrupted Runs

The tool automatically resumes from the last checkpoint:

```bash
# State is persisted in research/instrument-learning/state.json
# Just run again to continue
python research_tool.py
```

### Custom Configuration

```python
# Override defaults before running
RUNS = 50  # Fewer runs for testing
BASE_HINTS = ["custom", "topics", "here"]
```

---

## Rate Limits Reference

### GitHub Search API

| Limit Type | Authenticated | Unauthenticated |
|------------|---------------|-----------------|
| Requests/minute | 30 | 10 |
| Results/search | 1,000 | 1,000 |
| Results/page | 100 | 100 |

*Note: Code search has different limits (10 requests/min authenticated)*

### OpenAI API

| Model | Rate Limit |
|-------|------------|
| gpt-4o-mini | Varies by tier |

---

## Output Directory Structure

```
research/
└── instrument-learning/
    ├── state.json              # Checkpoint state
    ├── all_repos_deduped.txt   # Human-readable list
    └── all_repos_deduped.jsonl # Machine-readable data
```

---

## Analysis Examples

### Count by Language

```bash
cat research/instrument-learning/all_repos_deduped.jsonl | \
  jq -r '.language' | sort | uniq -c | sort -rn
```

### Top Starred Repos

```bash
cat research/instrument-learning/all_repos_deduped.jsonl | \
  jq -s 'sort_by(-.stars) | .[0:20]'
```

### Filter by Topic

```bash
cat research/instrument-learning/all_repos_deduped.jsonl | \
  jq 'select(.description | test("piano"; "i"))'
```

---

## Extending the Tool

### Adding New Hints

```python
BASE_HINTS.extend([
    "music game",
    "audio feedback",
    "tuner app",
    "pitch detection"
])
```

### Custom Query Generation

```python
def generate_queries(hint: str, count: int = 5) -> list[str]:
    """Override with custom query generation logic."""
    # Add language-specific queries
    languages = ["Python", "JavaScript", "Rust", "Swift"]
    queries = [f"{hint} language:{lang}" for lang in languages]
    return queries[:count]
```

### Alternative Output Formats

```python
def export_to_csv(repos: list[RepoRow], path: str) -> None:
    """Export results to CSV format."""
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RepoRow.__dataclass_fields__.keys())
        writer.writeheader()
        for repo in repos:
            writer.writerow(asdict(repo))
```

---

## Troubleshooting

### Rate Limit Errors

```
Error: API rate limit exceeded
```

**Solution**: Increase `MIN_SECONDS_BETWEEN_GH_CALLS` or wait for rate limit reset.

### Timeout Errors

```
Error: Request timed out
```

**Solution**: Increase `REQUEST_TIMEOUT_SEC` or check network connectivity.

### Empty Results

```
Warning: No new repositories found
```

**Possible causes**:
- All repositories already discovered (check `seen_repos` in state)
- Query too specific
- Rate limit preventing new queries

---

## Best Practices

1. **Start Small**: Test with `RUNS = 5` before running the full 100
2. **Monitor Progress**: Check `state.json` for completion status
3. **Backup Outputs**: Copy output files before re-running
4. **Rate Limit Awareness**: Don't run multiple instances simultaneously
5. **Review Results**: Manually review top results for quality

---

## References

- [GitHub Search API Documentation](https://docs.github.com/en/rest/search/search)
- [GitHub Rate Limits](https://docs.github.com/en/rest/rate-limit)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [gpt-4o-mini Model](https://platform.openai.com/docs/models/gpt-4o-mini)
