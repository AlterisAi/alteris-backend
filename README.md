# Loom Backend

Python backend for the Loom computational autobiography system. Ingests events from Mac-native databases, resolves identities, extracts claims, synthesizes beliefs, and generates blind spot briefings.

## Setup

```bash
./setup.sh
# or manually:
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

```bash
python -m loom.cli --help
python -m loom.cli ingest
python -m loom.cli brief
```

## Testing

```bash
pytest tests/ -v
```

## Architecture

Four-layer processing pipeline:

1. **Events** — Raw data from Mac-native sources (Mail, iMessage, WhatsApp, Calendar, Contacts, Granola, Slack)
2. **Claims** — Deterministic and LLM-extracted structured assertions about events
3. **Beliefs** — Synthesized, confidence-scored knowledge compiled from claims
4. **Briefings** — Actionable blind spot reports generated from beliefs

Every insight traces back through beliefs → claims → source events.
