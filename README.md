# Maintenance Log Triage System

An NLP pipeline that classifies ambiguous, non-technical maintenance
complaints into structured failure categories, scores severity, and
routes them to the right technician queue.

## Motivation

While working in the technical maintenance department at Calicut
International Airport, I managed fueling, vehicle maintenance
scheduling (fire & safety vehicles, ambulances, jeeps, tractors),
and upkeep of runway equipment. Maintenance complaints were logged
in a physical record book by staff from different departments who
had no technical background — entries like "noise from this side"
or "AC not working" were common, and manually figuring out what they
actually meant was a real bottleneck.

This project simulates that exact problem: turning vague, ambiguous
complaint text into a properly triaged, prioritized maintenance ticket.

## How it works

1. **Context enrichment** — attaches vehicle type, department,
   criticality tier, and history to the raw complaint text.
2. **Hybrid classification** — combines zero-shot LLM tagging
   (`facebook/bart-large-mnli`) with sentence embedding similarity
   (`all-MiniLM-L6-v2`) to assign a failure category, since no
   labeled training data exists for this kind of log.
3. **Confidence-gated human review** — low-confidence predictions
   are routed to a technician for a quick confirm/correct step,
   which also generates labeled data for future model improvement.
4. **Severity scoring & routing** — combines failure category with
   vehicle criticality (e.g. ambulance vs. grass cutter) to assign
   a priority tier and route to the correct technician specialization.

## Tech stack

Python, Hugging Face Transformers, Sentence-Transformers, pandas,
scikit-learn, matplotlib, ipywidgets — all run in a single Google
Colab notebook.

## Results

Classifier accuracy on confidently-classified entries vs. flagged
entries: see notebook output. [add your actual numbers here after
you run it]

## Notebook

See `maintenance_triage.ipynb` for the full pipeline.
