#!/usr/bin/env python3
"""
Stage 2.5: Build a merged knowledge graph from per-issue entity/relation extractions.

Reads kg_raw_entities and kg_raw_relations from harvest.db (populated by
analyzer.py), merges entities by normalized name, aggregates relations,
and produces:
  - Merged SQLite tables (kg_entities, kg_relations) in harvest.db
  - GraphML file for visualization (knowledge_graph.graphml)
  - Summary statistics
"""
import argparse
import json
import os
import re
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = os.path.join(os.path.dirname(__file__), "harvest.db")
OUTPUT_DIR = Path(os.path.dirname(__file__)).parent / "knowledge"


# ── Entity name normalization ────────────────────────────────

def normalize_entity_name(name: str, entity_type: str) -> str:
    """Normalize entity name for merging.

    - modem_model / config_option: preserve case, strip whitespace
    - everything else: lowercase, collapse whitespace, strip punctuation
    """
    name = name.strip()
    if entity_type in ("modem_model", "config_option"):
        return re.sub(r"\s+", " ", name)
    return re.sub(r"\s+", " ", name.lower().strip())


def entity_id(name: str, entity_type: str) -> str:
    """Generate a stable entity ID from normalized name and type."""
    norm = normalize_entity_name(name, entity_type)
    slug = re.sub(r"[^a-zA-Z0-9_]", "_", norm).strip("_")
    slug = re.sub(r"_+", "_", slug)
    return f"{entity_type}:{slug}"


# ── Database ─────────────────────────────────────────────────

def init_merged_tables(conn: sqlite3.Connection) -> None:
    """Create merged KG tables (dropped and rebuilt each run)."""
    conn.execute("DROP TABLE IF EXISTS kg_relations")
    conn.execute("DROP TABLE IF EXISTS kg_entities")

    conn.execute("""
        CREATE TABLE kg_entities (
            id              TEXT    PRIMARY KEY,
            name            TEXT    NOT NULL,
            type            TEXT    NOT NULL,
            description     TEXT,
            issue_refs      TEXT,
            mention_count   INTEGER DEFAULT 1,
            updated_at      TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE kg_relations (
            source_id       TEXT    NOT NULL,
            target_id       TEXT    NOT NULL,
            type            TEXT    NOT NULL,
            description     TEXT,
            weight          REAL    DEFAULT 1.0,
            issue_refs      TEXT,
            mention_count   INTEGER DEFAULT 1,
            updated_at       TEXT,
            PRIMARY KEY (source_id, target_id, type)
        )
    """)
    conn.commit()


def load_raw_entities(conn: sqlite3.Connection) -> list[dict]:
    """Load all raw entity extractions."""
    cols = ["repo", "issue_number", "name", "type", "description"]
    rows = conn.execute(
        "SELECT repo, issue_number, name, type, description FROM kg_raw_entities"
    ).fetchall()
    return [dict(zip(cols, row)) for row in rows]


def load_raw_relations(conn: sqlite3.Connection) -> list[dict]:
    """Load all raw relation extractions."""
    cols = ["repo", "issue_number", "source_name", "target_name",
            "type", "description", "weight"]
    rows = conn.execute(
        "SELECT repo, issue_number, source_name, target_name, type, "
        "description, weight FROM kg_raw_relations"
    ).fetchall()
    return [dict(zip(cols, row)) for row in rows]


# ── Merging ──────────────────────────────────────────────────

def issue_ref(repo: str, issue_number: int) -> str:
    """Create a short issue reference like 'esp-protocols#1003'."""
    slug = repo.split("/")[-1]
    return f"{slug}#{issue_number}"


def merge_descriptions(descriptions: list[str]) -> str:
    """Merge multiple descriptions into one, deduplicating."""
    seen = set()
    unique = []
    for d in descriptions:
        d = d.strip()
        if not d:
            continue
        key = d.lower()
        if key not in seen:
            seen.add(key)
            unique.append(d)
    if not unique:
        return ""
    if len(unique) == 1:
        return unique[0]
    return " | ".join(unique)


def build_merged_graph(raw_entities: list[dict],
                       raw_relations: list[dict]) -> tuple[dict, dict]:
    """Merge raw per-issue entities and relations into a unified graph.

    Returns (merged_entities, merged_relations) dicts.
    """
    # ── Merge entities ────────────────────────────────────────
    # Key: entity_id → {name, type, descriptions[], issue_refs set}
    ent_map: dict[str, dict] = {}
    # Also build a lookup: (raw_name) → entity_id for relation resolution
    name_to_id: dict[str, str] = {}

    for raw in raw_entities:
        eid = entity_id(raw["name"], raw["type"])
        ref = issue_ref(raw["repo"], raw["issue_number"])

        if eid not in ent_map:
            ent_map[eid] = {
                "id": eid,
                "name": normalize_entity_name(raw["name"], raw["type"]),
                "type": raw["type"],
                "descriptions": [],
                "issue_refs": set(),
            }
        ent_map[eid]["descriptions"].append(raw.get("description") or "")
        ent_map[eid]["issue_refs"].add(ref)

        name_to_id[raw["name"]] = eid
        name_to_id[normalize_entity_name(raw["name"], raw["type"])] = eid

    # Finalize entities
    merged_entities = {}
    for eid, data in ent_map.items():
        merged_entities[eid] = {
            "id": eid,
            "name": data["name"],
            "type": data["type"],
            "description": merge_descriptions(data["descriptions"]),
            "issue_refs": sorted(data["issue_refs"]),
            "mention_count": len(data["issue_refs"]),
        }

    # ── Merge relations ───────────────────────────────────────
    # Key: (source_id, target_id, type) → {descriptions[], weights[], issue_refs set}
    rel_map: dict[tuple, dict] = {}

    for raw in raw_relations:
        # Resolve entity names to IDs (try exact match, then normalized)
        src_name = raw["source_name"]
        tgt_name = raw["target_name"]

        src_id = name_to_id.get(src_name)
        tgt_id = name_to_id.get(tgt_name)

        if not src_id:
            # Try normalized lookup across all types
            for etype in ("modem_model", "config_option", "component",
                          "error_pattern", "symptom", "root_cause_cat",
                          "solution_pattern", "idf_version"):
                candidate = entity_id(src_name, etype)
                if candidate in merged_entities:
                    src_id = candidate
                    break
        if not tgt_id:
            for etype in ("modem_model", "config_option", "component",
                          "error_pattern", "symptom", "root_cause_cat",
                          "solution_pattern", "idf_version"):
                candidate = entity_id(tgt_name, etype)
                if candidate in merged_entities:
                    tgt_id = candidate
                    break

        if not src_id or not tgt_id:
            continue

        ref = issue_ref(raw["repo"], raw["issue_number"])
        rtype = raw["type"]
        key = (src_id, tgt_id, rtype)

        if key not in rel_map:
            rel_map[key] = {
                "descriptions": [],
                "weights": [],
                "issue_refs": set(),
            }
        rel_map[key]["descriptions"].append(raw.get("description") or "")
        rel_map[key]["weights"].append(raw.get("weight", 1.0))
        rel_map[key]["issue_refs"].add(ref)

    # Finalize relations
    merged_relations = {}
    for (src_id, tgt_id, rtype), data in rel_map.items():
        avg_weight = sum(data["weights"]) / len(data["weights"])
        merged_relations[(src_id, tgt_id, rtype)] = {
            "source_id": src_id,
            "target_id": tgt_id,
            "type": rtype,
            "description": merge_descriptions(data["descriptions"]),
            "weight": round(avg_weight, 3),
            "issue_refs": sorted(data["issue_refs"]),
            "mention_count": len(data["issue_refs"]),
        }

    return merged_entities, merged_relations


# ── Storage ──────────────────────────────────────────────────

def save_merged_graph(conn: sqlite3.Connection,
                      entities: dict, relations: dict) -> None:
    """Write merged entities and relations to SQLite."""
    now = datetime.now(timezone.utc).isoformat()
    init_merged_tables(conn)

    for eid, ent in entities.items():
        conn.execute("""
            INSERT INTO kg_entities
                (id, name, type, description, issue_refs, mention_count, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            ent["id"], ent["name"], ent["type"], ent["description"],
            json.dumps(ent["issue_refs"]), ent["mention_count"], now,
        ))

    for key, rel in relations.items():
        conn.execute("""
            INSERT INTO kg_relations
                (source_id, target_id, type, description, weight,
                 issue_refs, mention_count, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            rel["source_id"], rel["target_id"], rel["type"],
            rel["description"], rel["weight"],
            json.dumps(rel["issue_refs"]), rel["mention_count"], now,
        ))

    conn.commit()


# ── Export ────────────────────────────────────────────────────

def export_graphml(entities: dict, relations: dict, path: Path) -> None:
    """Export the KG as GraphML for visualization tools (yEd, Gephi, etc.)."""
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphstruct.org/xmlns"',
        '  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
        '  xsi:schemaLocation="http://graphml.graphstruct.org/xmlns '
        'http://graphml.graphstruct.org/xmlns/1.0/graphml.xsd">',
        '  <key id="d0" for="node" attr.name="label" attr.type="string"/>',
        '  <key id="d1" for="node" attr.name="type" attr.type="string"/>',
        '  <key id="d2" for="node" attr.name="description" attr.type="string"/>',
        '  <key id="d3" for="node" attr.name="mention_count" attr.type="int"/>',
        '  <key id="d4" for="edge" attr.name="label" attr.type="string"/>',
        '  <key id="d5" for="edge" attr.name="weight" attr.type="double"/>',
        '  <key id="d6" for="edge" attr.name="mention_count" attr.type="int"/>',
        '  <graph id="G" edgedefault="directed">',
    ]

    def xml_escape(s: str) -> str:
        return (s.replace("&", "&amp;").replace("<", "&lt;")
                 .replace(">", "&gt;").replace('"', "&quot;"))

    for eid, ent in entities.items():
        lines.append(f'    <node id="{xml_escape(eid)}">')
        lines.append(f'      <data key="d0">{xml_escape(ent["name"])}</data>')
        lines.append(f'      <data key="d1">{xml_escape(ent["type"])}</data>')
        lines.append(f'      <data key="d2">{xml_escape(ent["description"])}</data>')
        lines.append(f'      <data key="d3">{ent["mention_count"]}</data>')
        lines.append('    </node>')

    edge_id = 0
    for key, rel in relations.items():
        lines.append(
            f'    <edge id="e{edge_id}" '
            f'source="{xml_escape(rel["source_id"])}" '
            f'target="{xml_escape(rel["target_id"])}">'
        )
        lines.append(f'      <data key="d4">{xml_escape(rel["type"])}</data>')
        lines.append(f'      <data key="d5">{rel["weight"]}</data>')
        lines.append(f'      <data key="d6">{rel["mention_count"]}</data>')
        lines.append('    </edge>')
        edge_id += 1

    lines.append('  </graph>')
    lines.append('</graphml>')

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def export_dot(entities: dict, relations: dict, path: Path) -> None:
    """Export the KG as Graphviz DOT format."""
    type_colors = {
        "modem_model":      "#4FC3F7",
        "component":        "#81C784",
        "error_pattern":    "#E57373",
        "config_option":    "#FFB74D",
        "symptom":          "#F06292",
        "root_cause_cat":   "#BA68C8",
        "solution_pattern": "#AED581",
        "idf_version":      "#90A4AE",
    }
    type_shapes = {
        "modem_model":      "box",
        "component":        "ellipse",
        "error_pattern":    "octagon",
        "config_option":    "parallelogram",
        "symptom":          "diamond",
        "root_cause_cat":   "hexagon",
        "solution_pattern": "house",
        "idf_version":      "tab",
    }

    def dot_escape(s: str) -> str:
        return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

    lines = [
        "digraph KnowledgeGraph {",
        '  rankdir=LR;',
        '  node [style=filled, fontname="Helvetica", fontsize=10];',
        '  edge [fontname="Helvetica", fontsize=8];',
        "",
    ]

    for eid, ent in entities.items():
        color = type_colors.get(ent["type"], "#BDBDBD")
        shape = type_shapes.get(ent["type"], "ellipse")
        label = dot_escape(ent["name"])
        tooltip = dot_escape(f"[{ent['type']}] {ent['description'][:100]}")
        node_id = dot_escape(eid)
        lines.append(
            f'  "{node_id}" [label="{label}", shape={shape}, '
            f'fillcolor="{color}", tooltip="{tooltip}"];'
        )

    lines.append("")
    for key, rel in relations.items():
        src = dot_escape(rel["source_id"])
        tgt = dot_escape(rel["target_id"])
        label = dot_escape(rel["type"])
        penwidth = max(1.0, rel["weight"] * 3)
        lines.append(
            f'  "{src}" -> "{tgt}" [label="{label}", '
            f'penwidth={penwidth:.1f}];'
        )

    lines.append("}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def export_html(entities: dict, relations: dict, path: Path) -> None:
    """Export an interactive HTML visualization (uses vis-network from CDN)."""
    type_colors = {
        "modem_model":      {"background": "#4FC3F7", "border": "#0288D1"},
        "component":        {"background": "#81C784", "border": "#388E3C"},
        "error_pattern":    {"background": "#E57373", "border": "#D32F2F"},
        "config_option":    {"background": "#FFB74D", "border": "#F57C00"},
        "symptom":          {"background": "#F06292", "border": "#C2185B"},
        "root_cause_cat":   {"background": "#BA68C8", "border": "#7B1FA2"},
        "solution_pattern": {"background": "#AED581", "border": "#689F38"},
        "idf_version":      {"background": "#90A4AE", "border": "#546E7A"},
    }
    type_shapes = {
        "modem_model":      "box",
        "component":        "ellipse",
        "error_pattern":    "diamond",
        "config_option":    "box",
        "symptom":          "triangle",
        "root_cause_cat":   "hexagon",
        "solution_pattern": "star",
        "idf_version":      "dot",
    }

    nodes_js = []
    for eid, ent in entities.items():
        colors = type_colors.get(ent["type"], {"background": "#BDBDBD", "border": "#757575"})
        shape = type_shapes.get(ent["type"], "ellipse")
        size = 15 + ent["mention_count"] * 5
        title_html = (
            f"<b>[{ent['type']}] {ent['name']}</b><br>"
            f"{ent['description'][:200]}<br>"
            f"<i>Issues: {', '.join(ent['issue_refs'])}</i>"
        ).replace("'", "\\'").replace("\n", "<br>")
        label = ent["name"].replace("'", "\\'")
        nodes_js.append(
            f"  {{id: '{eid}', label: '{label}', shape: '{shape}', "
            f"size: {size}, "
            f"color: {{background: '{colors['background']}', border: '{colors['border']}'}}, "
            f"title: '{title_html}', "
            f"group: '{ent['type']}'}}"
        )

    edges_js = []
    for i, (key, rel) in enumerate(relations.items()):
        label = rel["type"].replace("'", "\\'")
        width = max(1, rel["weight"] * 3)
        title = (
            f"{rel['type']}<br>"
            f"weight: {rel['weight']}<br>"
            f"Issues: {', '.join(rel['issue_refs'])}"
        ).replace("'", "\\'")
        edges_js.append(
            f"  {{from: '{rel['source_id']}', to: '{rel['target_id']}', "
            f"label: '{label}', width: {width:.1f}, "
            f"arrows: 'to', title: '{title}', "
            f"font: {{size: 8, align: 'middle'}}}}"
        )

    legend_items = "".join(
        f'<span style="display:inline-block;margin:2px 8px;">'
        f'<span style="display:inline-block;width:12px;height:12px;'
        f'background:{c["background"]};border:2px solid {c["border"]};'
        f'border-radius:2px;vertical-align:middle;margin-right:4px;"></span>'
        f'{t}</span>'
        for t, c in type_colors.items()
    )

    html = f"""\
<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>esp-modem Knowledge Graph</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
  body {{ margin: 0; font-family: Helvetica, Arial, sans-serif; background: #1a1a2e; color: #e0e0e0; }}
  #graph {{ width: 100vw; height: calc(100vh - 40px); }}
  #legend {{ height: 40px; display: flex; align-items: center; justify-content: center;
             background: #16213e; font-size: 13px; flex-wrap: wrap; padding: 0 16px; }}
</style>
</head><body>
<div id="legend">{legend_items}</div>
<div id="graph"></div>
<script>
var nodes = new vis.DataSet([
{(",\n".join(nodes_js))}
]);
var edges = new vis.DataSet([
{(",\n".join(edges_js))}
]);
var container = document.getElementById('graph');
var data = {{nodes: nodes, edges: edges}};
var options = {{
  physics: {{
    solver: 'forceAtlas2Based',
    forceAtlas2Based: {{ gravitationalConstant: -60, centralGravity: 0.008, springLength: 140, damping: 0.5 }},
    stabilization: {{ iterations: 200 }}
  }},
  interaction: {{ hover: true, tooltipDelay: 100, navigationButtons: true }},
  edges: {{
    color: {{color: '#556677', highlight: '#FFD54F', hover: '#80CBC4'}},
    smooth: {{type: 'cubicBezier', forceDirection: 'horizontal', roundness: 0.4}}
  }},
  nodes: {{
    font: {{color: '#e0e0e0', size: 11}},
    borderWidth: 2
  }}
}};
new vis.Network(container, data, options);
</script>
</body></html>"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def export_summary_json(entities: dict, relations: dict, path: Path) -> None:
    """Export a compact JSON summary of the KG."""
    data = {
        "entities": list(entities.values()),
        "relations": list(relations.values()),
        "stats": {
            "total_entities": len(entities),
            "total_relations": len(relations),
            "entities_by_type": {},
            "relations_by_type": {},
            "most_connected": [],
        },
    }

    # Entity type breakdown
    by_type: dict[str, int] = defaultdict(int)
    for ent in entities.values():
        by_type[ent["type"]] += 1
    data["stats"]["entities_by_type"] = dict(sorted(by_type.items()))

    # Relation type breakdown
    rel_by_type: dict[str, int] = defaultdict(int)
    for rel in relations.values():
        rel_by_type[rel["type"]] += 1
    data["stats"]["relations_by_type"] = dict(sorted(rel_by_type.items()))

    # Most connected entities (by degree)
    degree: dict[str, int] = defaultdict(int)
    for rel in relations.values():
        degree[rel["source_id"]] += 1
        degree[rel["target_id"]] += 1
    top = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:20]
    data["stats"]["most_connected"] = [
        {"id": eid, "name": entities[eid]["name"],
         "type": entities[eid]["type"], "degree": deg}
        for eid, deg in top if eid in entities
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ── Print stats ──────────────────────────────────────────────

def print_stats(entities: dict, relations: dict) -> None:
    """Print summary statistics to stderr."""
    print(f"\n{'='*60}", file=sys.stderr)
    print("Knowledge Graph Summary", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"  Entities: {len(entities)}", file=sys.stderr)
    print(f"  Relations: {len(relations)}", file=sys.stderr)

    by_type: dict[str, int] = defaultdict(int)
    for ent in entities.values():
        by_type[ent["type"]] += 1
    print("\n  Entity types:", file=sys.stderr)
    for t, c in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"    {t:20s} {c:4d}", file=sys.stderr)

    rel_by_type: dict[str, int] = defaultdict(int)
    for rel in relations.values():
        rel_by_type[rel["type"]] += 1
    print("\n  Relation types:", file=sys.stderr)
    for t, c in sorted(rel_by_type.items(), key=lambda x: -x[1]):
        print(f"    {t:20s} {c:4d}", file=sys.stderr)

    # Most mentioned entities
    top = sorted(entities.values(), key=lambda e: -e["mention_count"])[:10]
    print("\n  Most referenced entities:", file=sys.stderr)
    for ent in top:
        print(f"    [{ent['type']}] {ent['name']} "
              f"(mentioned in {ent['mention_count']} issue(s))", file=sys.stderr)

    # Most connected
    degree: dict[str, int] = defaultdict(int)
    for rel in relations.values():
        degree[rel["source_id"]] += 1
        degree[rel["target_id"]] += 1
    top_conn = sorted(degree.items(), key=lambda x: -x[1])[:10]
    print("\n  Most connected entities (by relation degree):", file=sys.stderr)
    for eid, deg in top_conn:
        if eid in entities:
            ent = entities[eid]
            print(f"    [{ent['type']}] {ent['name']}  degree={deg}",
                  file=sys.stderr)


# ── Main ─────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a merged knowledge graph from per-issue KG extractions")
    parser.add_argument("--db", type=str, default=DB_PATH,
                        help="Path to harvest.db (default: ./harvest.db)")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Output directory for GraphML and JSON files")
    parser.add_argument("--no-graphml", action="store_true",
                        help="Skip GraphML export")
    parser.add_argument("--no-json", action="store_true",
                        help="Skip JSON summary export")
    parser.add_argument("--no-dot", action="store_true",
                        help="Skip DOT (Graphviz) export")
    parser.add_argument("--no-html", action="store_true",
                        help="Skip interactive HTML export")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)

    # Check if raw KG data exists
    try:
        n_raw_ent = conn.execute("SELECT COUNT(*) FROM kg_raw_entities").fetchone()[0]
        n_raw_rel = conn.execute("SELECT COUNT(*) FROM kg_raw_relations").fetchone()[0]
    except sqlite3.OperationalError:
        print("Error: kg_raw_entities/kg_raw_relations tables not found. "
              "Run analyzer.py --extract-kg first.", file=sys.stderr)
        conn.close()
        return 1

    print(f"Raw data: {n_raw_ent} entities, {n_raw_rel} relations",
          file=sys.stderr, flush=True)

    if n_raw_ent == 0:
        print("No raw KG data found. Run analyzer.py --extract-kg first.",
              file=sys.stderr)
        conn.close()
        return 1

    # Load and merge
    raw_entities = load_raw_entities(conn)
    raw_relations = load_raw_relations(conn)

    entities, relations = build_merged_graph(raw_entities, raw_relations)

    # Save to SQLite
    save_merged_graph(conn, entities, relations)
    print(f"Saved merged graph to SQLite: {len(entities)} entities, "
          f"{len(relations)} relations", file=sys.stderr, flush=True)

    # Export files
    out_dir = Path(args.output_dir)

    if not args.no_graphml:
        graphml_path = out_dir / "knowledge_graph.graphml"
        export_graphml(entities, relations, graphml_path)
        print(f"Exported GraphML: {graphml_path}", file=sys.stderr, flush=True)

    if not args.no_dot:
        dot_path = out_dir / "knowledge_graph.dot"
        export_dot(entities, relations, dot_path)
        print(f"Exported DOT: {dot_path}", file=sys.stderr, flush=True)

    if not args.no_html:
        html_path = out_dir / "knowledge_graph.html"
        export_html(entities, relations, html_path)
        print(f"Exported HTML: {html_path}", file=sys.stderr, flush=True)

    if not args.no_json:
        json_path = out_dir / "knowledge_graph.json"
        export_summary_json(entities, relations, json_path)
        print(f"Exported JSON: {json_path}", file=sys.stderr, flush=True)

    # Print stats
    print_stats(entities, relations)

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
