#!/usr/bin/env node
/**
 * Erzeugt aus docs/projektarbeit2_rag_mietrecht.md ein FOM-konformes Word-Dokument.
 *
 *   node scripts/md_to_docx.js [eingabe.md] [ausgabe.docx]
 *
 * Das Markdown bleibt die führende Quelle: nach inhaltlichen Änderungen einfach
 * erneut ausführen. Verzeichnisse und Abbildungsnummern sind Word-Feldfunktionen
 * und werden erst beim Öffnen gefüllt (Strg+A, dann F9).
 */

const fs = require("fs");
const path = require("path");
const {
  AlignmentType, Document, Footer, HeadingLevel, ImageRun, LevelFormat,
  PageBreak, PageNumber, Packer, Paragraph, SectionType, SequentialIdentifier,
  ShadingType, Table, TableCell, TableOfContents, TableRow, TextRun, WidthType,
} = require("docx");

const IN = process.argv[2] || "docs/projektarbeit2_rag_mietrecht.md";
const OUT = process.argv[3] || "docs/Projektarbeit2_RAG_Mietrecht.docx";

const FONT = "Arial";
const SIZE = 22;            // 11 pt in Halbpunkten
const LINE = 360;           // 1,5-zeilig
const TEXT_WIDTH_DXA = 9070; // 16 cm Satzspiegel bei A4 und 2,5 cm Rändern

// ---------------------------------------------------------------- Hilfsmittel

/** Liest Breite und Höhe aus dem IHDR-Chunk einer PNG-Datei. */
function pngSize(file) {
  const buf = fs.readFileSync(file);
  return { width: buf.readUInt32BE(16), height: buf.readUInt32BE(20) };
}

/** Zerlegt Inline-Markup (**fett**, *kursiv*, `code`) in TextRuns. */
function runs(text) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g;
  let last = 0, m;
  const push = (t, opts) => { if (t) out.push(new TextRun({ text: t, font: FONT, size: SIZE, ...opts })); };
  while ((m = re.exec(text)) !== null) {
    push(text.slice(last, m.index), {});
    const tok = m[0];
    if (tok.startsWith("**")) push(tok.slice(2, -2), { bold: true });
    else if (tok.startsWith("`")) push(tok.slice(1, -1), { font: "Consolas", size: SIZE - 2 });
    else push(tok.slice(1, -1), { italics: true });
    last = m.index + tok.length;
  }
  push(text.slice(last), {});
  return out.length ? out : [new TextRun({ text: "", font: FONT, size: SIZE })];
}

const body = (text, extra = {}) => new Paragraph({
  children: runs(text),
  alignment: AlignmentType.JUSTIFIED,
  spacing: { line: LINE, after: 160 },
  ...extra,
});

const plain = (text, opts = {}) => new Paragraph({
  children: [new TextRun({ text, font: FONT, size: SIZE, ...opts })],
  spacing: { line: LINE, after: opts.after ?? 120 },
  alignment: opts.alignment ?? AlignmentType.LEFT,
});

/** Baut eine Tabelle aus Markdown-Zeilen; erste Zeile ist die Kopfzeile. */
function buildTable(lines) {
  const rows = lines
    .map((l) => l.trim())
    .filter((l) => l && !/^\|[\s|:-]+\|$/.test(l))
    .map((l) => l.replace(/^\||\|$/g, "").split("|").map((c) => c.trim()));
  if (!rows.length) return null;

  // Zellenzahl an die Kopfzeile angleichen, damit keine Spalte ohne Breite bleibt
  const cols = rows[0].length;
  for (const row of rows) {
    while (row.length < cols) row.push("");
    row.length = cols;
  }
  const colWidth = Math.floor(TEXT_WIDTH_DXA / cols);
  const widths = Array(cols).fill(colWidth);
  widths[cols - 1] = TEXT_WIDTH_DXA - colWidth * (cols - 1);

  return new Table({
    columnWidths: widths,
    width: { size: TEXT_WIDTH_DXA, type: WidthType.DXA },
    rows: rows.map((cells, r) => new TableRow({
      tableHeader: r === 0,
      children: cells.map((cell, c) => new TableCell({
        width: { size: widths[c], type: WidthType.DXA },
        shading: r === 0 ? { type: ShadingType.CLEAR, fill: "EFEFEF" } : undefined,
        margins: { top: 60, bottom: 60, left: 80, right: 80 },
        children: [new Paragraph({
          children: runs(cell).map((run) => {
            if (r === 0) run.root.forEach?.(() => {});
            return run;
          }),
          spacing: { line: 240, after: 0 },
          alignment: c === 0 ? AlignmentType.LEFT : AlignmentType.RIGHT,
        })],
      })),
    })),
  });
}

/** Beschriftung mit SEQ-Feld, damit Word das Verzeichnis erzeugen kann. */
function caption(label, text) {
  return new Paragraph({
    style: "Caption",
    spacing: { before: 60, after: 240 },
    alignment: AlignmentType.LEFT,
    children: [
      new TextRun({ text: `${label} `, font: FONT, size: SIZE - 2 }),
      new SequentialIdentifier(label),
      new TextRun({ text: `: ${text}`, font: FONT, size: SIZE - 2 }),
    ],
  });
}

// ------------------------------------------------------------------- Parsing

const raw = fs.readFileSync(IN, "utf8");
const lines = raw.split("\n");

const abbrev = [];   // Abkürzungsverzeichnis
const elements = []; // Fließtext
let skipSection = false;
let i = 0;

while (i < lines.length) {
  const line = lines[i];

  if (/^## /.test(line)) {
    const title = line.replace(/^## /, "").trim();
    // Hinweisblock und Abkürzungsverzeichnis wandern in den Vorspann
    skipSection = title === "Hinweise zum Entwurf" || title === "Abkürzungsverzeichnis";
    if (title === "Abkürzungsverzeichnis") {
      let j = i + 1;
      while (j < lines.length && !/^## /.test(lines[j])) {
        const row = lines[j];
        if (/^\|/.test(row) && !/^\|[\s|:-]+\|$/.test(row)) {
          const cells = row.replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
          if (cells[0] && cells[0] !== "Abkürzung") abbrev.push(cells);
        }
        j++;
      }
    }
    if (!skipSection) {
      elements.push(new Paragraph({
        text: title,
        heading: HeadingLevel.HEADING_1,
        pageBreakBefore: true,
        spacing: { after: 240 },
      }));
    }
    i++;
    continue;
  }

  if (skipSection) { i++; continue; }

  if (/^### /.test(line)) {
    elements.push(new Paragraph({
      text: line.replace(/^### /, "").trim(),
      heading: HeadingLevel.HEADING_2,
      spacing: { before: 240, after: 160 },
    }));
    i++;
    continue;
  }

  // Abbildung
  const img = line.match(/^!\[([^\]]*)\]\(([^)]+)\)/);
  if (img) {
    const file = path.resolve(path.dirname(IN), img[2]);
    if (fs.existsSync(file)) {
      const { width, height } = pngSize(file);
      const maxW = 600; // Punkte im Satzspiegel
      const scale = Math.min(1, maxW / width);
      elements.push(new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 200, after: 60 },
        children: [new ImageRun({
          type: "png",
          data: fs.readFileSync(file),
          transformation: { width: Math.round(width * scale), height: Math.round(height * scale) },
        })],
      }));
      // die folgende Zeile ist die kursive Beschriftung
      const next = (lines[i + 1] || "").trim() || (lines[i + 2] || "").trim();
      const cap = next.match(/^\*Abbildung \d+:\s*(.+?)\*$/);
      if (cap) {
        elements.push(caption("Abbildung", cap[1]));
        i += (lines[i + 1] || "").trim() ? 2 : 3;
        continue;
      }
    }
    i++;
    continue;
  }

  // Tabellenbeschriftung
  const tcap = line.match(/^\*\*Tabelle \d+:\s*(.+?)\*\*$/);
  if (tcap) {
    elements.push(caption("Tabelle", tcap[1]));
    i++;
    continue;
  }

  // Tabelle
  if (/^\|/.test(line)) {
    const block = [];
    while (i < lines.length && /^\|/.test(lines[i])) block.push(lines[i++]);
    const table = buildTable(block);
    if (table) {
      elements.push(table);
      elements.push(new Paragraph({ text: "", spacing: { after: 240 } }));
    }
    continue;
  }

  if (/^---+$/.test(line.trim()) || /^# /.test(line)) { i++; continue; }

  // Blockzitat (Forschungsfrage)
  if (/^> /.test(line)) {
    elements.push(new Paragraph({
      children: runs(line.replace(/^> /, "")),
      spacing: { line: LINE, before: 160, after: 200 },
      indent: { left: 480, right: 480 },
      alignment: AlignmentType.JUSTIFIED,
    }));
    i++;
    continue;
  }

  if (line.trim()) elements.push(body(line.trim()));
  i++;
}

// ------------------------------------------------------------------ Vorspann

const titlePage = [
  plain("FOM Hochschule für Oekonomie & Management", { bold: true, alignment: AlignmentType.CENTER, after: 1200 }),
  plain("Retrieval-Augmented Generation auf deutschen Mietrechtsurteilen", {
    bold: true, size: 34, alignment: AlignmentType.CENTER, after: 200,
  }),
  plain("Aufbau und empirische Evaluation eines Retrievalsystems in PostgreSQL/pgvector", {
    size: 26, alignment: AlignmentType.CENTER, after: 1600,
  }),
  plain("Dedner, Nils (727367)", { alignment: AlignmentType.CENTER, after: 120 }),
  plain("31.08.2026", { alignment: AlignmentType.CENTER, after: 1600 }),
  plain("Art der Arbeit: Projektarbeit"),
  plain("Matrikelnummer: 727367"),
  plain("Name des Lehrenden: [BITTE ERGÄNZEN]"),
  plain("Modulname: Big Data Analytics"),
  new Paragraph({ children: [new PageBreak()] }),
];

const disclaimer = [
  new Paragraph({ text: "Erklärung zum Einsatz digitaler Unterstützung", heading: HeadingLevel.HEADING_1, spacing: { after: 240 } }),
  body("Im Rahmen dieser Arbeit wurden digitale Hilfsmittel und KI-basierte Systeme eingesetzt. Dabei wurden die Vorgaben der FOM Hochschule für Oekonomie und Management, die DFG-Leitlinien zur Sicherung guter wissenschaftlicher Praxis sowie die Bestimmungen der DSGVO eingehalten."),
  body("Die eingesetzten Werkzeuge dienten der Unterstützung folgender Arbeitsschritte:"),
  body("• Implementierung: Erstellung und Überarbeitung der Auswertungsskripte mit Claude Code"),
  body("• Datenerhebung: Generierung des Goldstandards aus Urteilstexten mit gpt-4o-mini"),
  body("• Auswertung: Bewertung der erzeugten Antworten durch ein Sprachmodell als Teil des Untersuchungsdesigns (siehe Abschnitt 4.3)"),
  body("• Textarbeit: Unterstützung bei Strukturierung, Formulierung und sprachlicher Prüfung"),
  body("Der Einsatz von Sprachmodellen als Bewertungsinstanz ist Bestandteil der Methodik und wird in Abschnitt 4.3 beschrieben sowie in Kapitel 6 kritisch eingeordnet. Sämtliche inhaltlichen Entscheidungen, die Interpretation der Messergebnisse und die kritische Prüfung wurden von mir vorgenommen. Alle Messwerte stammen aus protokollierten Läufen und sind über die angegebenen Kennungen reproduzierbar."),
  body("Ich versichere, dass die vorliegende Arbeit trotz der digitalen Unterstützung meine eigenständige wissenschaftliche Leistung darstellt."),
  plain("Ort, Datum: Waiblingen, 31.08.2026", { after: 400 }),
  plain("Unterschrift: ________________________"),
  new Paragraph({ children: [new PageBreak()] }),
];

const tocSection = [
  new Paragraph({ text: "Inhaltsverzeichnis", heading: HeadingLevel.HEADING_1, spacing: { after: 240 } }),
  new TableOfContents("Inhalt", { hyperlink: true, headingStyleRange: "1-3" }),
  new Paragraph({ children: [new PageBreak()] }),

  new Paragraph({ text: "Abkürzungsverzeichnis", heading: HeadingLevel.HEADING_1, spacing: { after: 240 } }),
  ...(abbrev.length ? [buildTable([
    "| Abkürzung | Bedeutung |",
    "|---|---|",
    ...abbrev.map((a) => `| ${a[0]} | ${a[1]} |`),
  ])] : []),
  new Paragraph({ children: [new PageBreak()] }),

  new Paragraph({ text: "Abbildungsverzeichnis", heading: HeadingLevel.HEADING_1, spacing: { after: 240 } }),
  new TableOfContents("Abbildungen", { hyperlink: true, captionLabel: "Abbildung" }),
  new Paragraph({ children: [new PageBreak()] }),

  new Paragraph({ text: "Tabellenverzeichnis", heading: HeadingLevel.HEADING_1, spacing: { after: 240 } }),
  new TableOfContents("Tabellen", { hyperlink: true, captionLabel: "Tabelle" }),
];

const declaration = [
  new Paragraph({
    text: "Erklärung über die eigenständige Erstellung der Arbeit",
    heading: HeadingLevel.HEADING_1,
    pageBreakBefore: true,
    spacing: { after: 240 },
  }),
  body("Hiermit versichere ich, dass ich die vorliegende Arbeit selbständig verfasst und keine anderen als die angegebenen Hilfsmittel benutzt habe."),
  plain("", { after: 600 }),
  plain("31.08.2026, Waiblingen"),
  plain("Datum, Ort                                                            Unterschrift [Dedner, Nils]"),
];

// -------------------------------------------------------------- Dokumentbau

const doc = new Document({
  creator: "Nils Dedner",
  title: "Retrieval-Augmented Generation auf deutschen Mietrechtsurteilen",
  styles: {
    default: {
      document: { run: { font: FONT, size: SIZE }, paragraph: { spacing: { line: LINE } } },
    },
    paragraphStyles: [
      { id: "Caption", name: "Caption", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: SIZE - 2, italics: true } },
    ],
  },
  sections: [{
    properties: {
      type: SectionType.CONTINUOUS,
      page: { margin: { top: 1418, right: 1418, bottom: 1418, left: 1418 } },
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: SIZE - 4 })],
        })],
      }),
    },
    children: [...titlePage, ...disclaimer, ...tocSection, ...elements, ...declaration],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  fs.writeFileSync(OUT, buf);
  console.log(`geschrieben: ${OUT}`);
  console.log(`  Abkürzungen: ${abbrev.length}`);
  console.log(`  Textelemente: ${elements.length}`);
  console.log("  Hinweis: In Word einmal Strg+A und F9 drücken, damit Verzeichnisse und");
  console.log("  Abbildungsnummern gefüllt werden.");
});
