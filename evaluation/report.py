#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fasst alle Messläufe zu Tabellen und Abbildungen für die Arbeit zusammen.

Quellen sind die Tabelle eval_runs sowie die JSON-Artefakte der
Vektor-Benchmarks. Erzeugt CSV (für Anhang und Nachrechnen), Markdown
(zum direkten Übernehmen) und PNG-Abbildungen.

Beispiel:
    python -m evaluation.report --out-dir artifacts/report
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import psycopg2.extras as pgx  # noqa: E402

from etl.config import PgConfig  # noqa: E402
from etl.pg_db import pg_connect  # noqa: E402

# Sprechende Namen für die Läufe der Messreihe
LABELS = {
    "m_dense_minilm": "Dense (MiniLM, englisch)",
    "m_dense_e5": "Dense (E5, mehrsprachig)",
    "m_lexical": "Lexikalisch (BM25, deutsch)",
    "m_hybrid_minilm": "Hybrid (MiniLM + BM25)",
    "m_hybrid_e5": "Hybrid (E5 + BM25)",
}

RETRIEVAL_COLUMNS = ["hit@1", "hit@3", "hit@10", "case_hit@10", "mrr", "ndcg@10", "latency_p50_ms"]

# Sprechende Namen für die Serverkonfigurationen der Benchmarkläufe
BENCH_LABELS = {
    "baseline_default_config": "Werkseinstellung",
    "tuned_reload_only": "angepasst, ohne Neustart",
    "tuned_full_restart": "angepasst, mit Neustart",
}

# Farbpalette der FOM, entnommen den CSS-Variablen von fom.de
FOM = {
    "primaer_hell": "#00c6b2",   # --color-primary-500
    "primaer": "#009f8f",        # --color-primary-600
    "primaer_dunkel": "#00776b", # --color-primary-700
    "sekundaer": "#0071de",      # --color-secondary-500
    "gruen": "#77b502",          # --color-green
    "grau": "#9a9a9a",           # --color-grey-400
    "text": "#002723",           # --color-primary-900
}

# Deutsche Zahlenschreibweise auf den Achsen (Komma statt Punkt)
KOMMA = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ","))
KOMMA1 = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ","))


def fetch_runs(conn, pipeline: str) -> List[Dict[str, Any]]:
    with conn.cursor(cursor_factory=pgx.RealDictCursor) as cur:
        cur.execute(
            "SELECT run_id, params, metrics FROM eval_runs WHERE pipeline = %s ORDER BY run_id",
            (pipeline,),
        )
        return [dict(r) for r in cur.fetchall()]


def write_table(path_base: str, header: List[str], rows: List[List[Any]]) -> None:
    with open(path_base + ".csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)

    with open(path_base + ".md", "w", encoding="utf-8") as fh:
        fh.write("| " + " | ".join(header) + " |\n")
        fh.write("|" + "|".join(["---"] * len(header)) + "|\n")
        for row in rows:
            fh.write("| " + " | ".join("" if v is None else str(v) for v in row) + " |\n")


def fmt(value: Any, digits: int = 3) -> Any:
    if isinstance(value, (int, float)):
        return round(float(value), digits)
    return value


def report_retrieval(conn, out_dir: str) -> List[Dict[str, Any]]:
    runs = [r for r in fetch_runs(conn, "retrieval_eval") if r["run_id"] in LABELS]
    if not runs:
        print("report: keine Läufe der Messreihe gefunden (run_id m_*)")
        return []

    runs.sort(key=lambda r: list(LABELS).index(r["run_id"]))
    header = ["Variante"] + RETRIEVAL_COLUMNS
    rows = [[LABELS[r["run_id"]]] + [fmt(r["metrics"].get(c)) for c in RETRIEVAL_COLUMNS] for r in runs]
    write_table(os.path.join(out_dir, "tabelle_retrieval"), header, rows)

    labels = [LABELS[r["run_id"]] for r in runs]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = range(len(runs))
    width = 0.38
    ax.bar([i - width / 2 for i in x], [r["metrics"].get("hit@10", 0) for r in runs],
           width, label="hit@10 (exakter Abschnitt)", color=FOM["primaer_dunkel"])
    ax.bar([i + width / 2 for i in x], [r["metrics"].get("case_hit@10", 0) for r in runs],
           width, label="case_hit@10 (richtige Entscheidung)", color=FOM["primaer_hell"])
    for i, r in enumerate(runs):
        for versatz, schluessel in ((-width / 2, "hit@10"), (width / 2, "case_hit@10")):
            wert = r["metrics"].get(schluessel, 0)
            ax.text(i + versatz, wert + 0.012, f"{wert:.2f}".replace(".", ","),
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("Trefferquote")
    ax.set_ylim(0, 0.95)  # Platz für Legende und Beschriftungen
    ax.yaxis.set_major_formatter(KOMMA1)
    ax.set_title("Retrievalqualität über 60 Goldstandard-Fragen (k = 10)")
    ax.legend(loc="upper left", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "abb_retrieval_qualitaet.png"), dpi=200)
    plt.close(fig)
    return runs


def report_answers(conn, out_dir: str) -> None:
    runs = fetch_runs(conn, "answer_eval")
    runs = [r for r in runs if not r["run_id"].endswith("smoke")]
    if not runs:
        print("report: keine Antwortbewertung gefunden")
        return

    header = ["Lauf", "Bedingung", "Korrektheit (0-2)", "Anteil voll korrekt", "Fundierung (0-2)", "Anteil erfunden"]
    rows: List[List[Any]] = []
    for r in runs:
        m = r["metrics"]
        for cond, name in (("rag", "mit Retrieval"), ("norag", "ohne Retrieval")):
            rows.append([
                r["run_id"], name,
                fmt(m.get(f"{cond}_korrektheit_mittel")),
                fmt(m.get(f"{cond}_korrekt_anteil")),
                fmt(m.get(f"{cond}_fundierung_mittel")),
                fmt(m.get(f"{cond}_erfunden_anteil")),
            ])
    write_table(os.path.join(out_dir, "tabelle_antworten"), header, rows)

    latest = runs[-1]["metrics"]
    fig, ax = plt.subplots(figsize=(6, 4))
    conds = ["mit Retrieval", "ohne Retrieval"]
    values = [latest.get("rag_korrektheit_mittel") or 0, latest.get("norag_korrektheit_mittel") or 0]
    ax.bar(conds, values, color=[FOM["primaer"], FOM["grau"]], width=0.5)
    ax.set_ylim(0, 2)
    ax.set_ylabel("mittlere Korrektheit (0 bis 2)")
    ax.yaxis.set_major_formatter(KOMMA1)
    ax.set_title("Antwortqualität mit und ohne Retrieval")
    for i, v in enumerate(values):
        ax.text(i, v + 0.05, f"{v:.2f}".replace(".", ","), ha="center")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "abb_antwortqualitaet.png"), dpi=200)
    plt.close(fig)


def report_bench(bench_dir: str, out_dir: str) -> None:
    paths = sorted(glob.glob(os.path.join(bench_dir, "*.json")))
    if not paths:
        print("report: keine Benchmark-Artefakte gefunden")
        return

    header = ["Lauf", "Modus", "p50 (ms)", "p95 (ms)", "Recall@10"]
    rows: List[List[Any]] = []
    kurven: Dict[str, List[tuple]] = {}
    kurvenfarben = [FOM["primaer_dunkel"], FOM["sekundaer"], FOM["gruen"]]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    for path in paths:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        run_id = data.get("run_id", os.path.basename(path))
        points: List[tuple] = []
        for entry in data.get("results") or []:
            variant = entry.get("variant", "?")
            latency = entry.get("latency_ms") or {}
            recall = entry.get("recall_at_k")
            rows.append([run_id, variant, fmt(latency.get("p50"), 2), fmt(latency.get("p95"), 2), fmt(recall)])
            if entry.get("ef_search") is not None and recall is not None:
                points.append((int(entry["ef_search"]), latency.get("p50"), recall))
        if points:
            points.sort()
            farbe = kurvenfarben[len(kurven) % len(kurvenfarben)]
            kurven[run_id] = points
            ax.plot([p[1] for p in points], [p[2] for p in points], marker="o",
                    color=farbe, label=BENCH_LABELS.get(run_id, run_id))

    # ef-Werte nur an EINER Kurve beschriften, sonst überlagern sich die Labels
    referenz = "tuned_full_restart" if "tuned_full_restart" in kurven else next(iter(kurven), None)
    if referenz:
        for ef, lat, rec in kurven[referenz]:
            ax.annotate(f"ef_search = {ef}", (lat, rec), textcoords="offset points",
                        xytext=(8, -4), fontsize=8)

    write_table(os.path.join(out_dir, "tabelle_bench"), header, rows)
    ax.set_xlabel("Antwortzeit p50 in ms (logarithmisch)")
    ax.set_ylabel("Recall@10 gegen exakte Suche")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 3, 5, 8])
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(0.9, 11)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(KOMMA1)
    ax.set_title("Genauigkeit gegen Geschwindigkeit der Vektorsuche (5,16 Mio. Abschnitte)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right", title="Serverkonfiguration")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "abb_recall_latenz.png"), dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Messergebnisse zu Tabellen und Abbildungen zusammenfassen")
    ap.add_argument("--out-dir", default="artifacts/report")
    ap.add_argument("--bench-dir", default="artifacts/bench")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    conn = pg_connect(PgConfig())
    try:
        report_retrieval(conn, args.out_dir)
        report_answers(conn, args.out_dir)
    finally:
        conn.close()
    report_bench(args.bench_dir, args.out_dir)

    print(f"report fertig. Verzeichnis: {args.out_dir}")
    for name in sorted(os.listdir(args.out_dir)):
        print(f"  {name}")


if __name__ == "__main__":
    main()
