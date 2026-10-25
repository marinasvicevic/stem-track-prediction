const docx = require('docx');
const fs = require('fs');

const { Document, Packer, Table, TableRow, TableCell, Paragraph, TextRun, WidthType, AlignmentType, BorderStyle } = docx;

const COL1_W = 1200;  // Model
const COL2_W = 7200;  // Hyperparameter search space
const TABLE_W = COL1_W + COL2_W;

function cell(text, opts = {}) {
    const runs = [];
    // Parse simple italic markers *text*
    const parts = text.split(/(\*[^*]+\*)/);
    for (const p of parts) {
        if (p.startsWith('*') && p.endsWith('*')) {
            runs.push(new TextRun({ text: p.slice(1, -1), italics: true, font: 'Times New Roman', size: 20 }));
        } else {
            runs.push(new TextRun({ text: p, font: 'Times New Roman', size: 20, bold: opts.bold || false }));
        }
    }
    return new TableCell({
        width: { size: opts.width || COL2_W, type: WidthType.DXA },
        verticalAlign: 'center',
        margins: { top: 40, bottom: 40, left: 80, right: 80 },
        borders: {
            top: { style: opts.topBorder ? BorderStyle.SINGLE : BorderStyle.NONE, size: opts.topBorder || 0 },
            bottom: { style: opts.bottomBorder ? BorderStyle.SINGLE : BorderStyle.NONE, size: opts.bottomBorder || 0 },
            left: { style: BorderStyle.NONE },
            right: { style: BorderStyle.NONE },
        },
        shading: opts.shading ? { type: docx.ShadingType.CLEAR, fill: opts.shading } : undefined,
        children: [new Paragraph({
            alignment: opts.align || AlignmentType.LEFT,
            children: runs,
        })],
    });
}

function headerCell(text, width) {
    return new TableCell({
        width: { size: width, type: WidthType.DXA },
        verticalAlign: 'center',
        margins: { top: 60, bottom: 60, left: 80, right: 80 },
        borders: {
            top: { style: BorderStyle.SINGLE, size: 12 },
            bottom: { style: BorderStyle.SINGLE, size: 6 },
            left: { style: BorderStyle.NONE },
            right: { style: BorderStyle.NONE },
        },
        children: [new Paragraph({
            alignment: AlignmentType.LEFT,
            children: [new TextRun({ text, bold: true, font: 'Times New Roman', size: 20 })],
        })],
    });
}

const rows = [
    // Header
    new TableRow({
        children: [
            headerCell('Model', COL1_W),
            headerCell('Hyperparameter search space', COL2_W),
        ],
    }),
    // LR
    new TableRow({
        children: [
            cell('LR', { width: COL1_W, bold: true }),
            cell('*C* ∈ {0.01, 0.1, 1, 10, 100}; penalty ∈ {L1, L2}'),
        ],
    }),
    // KNN
    new TableRow({
        children: [
            cell('KNN', { width: COL1_W, bold: true }),
            cell('*n* ∈ {3, 5, 7, 9, 11, 15}; weights ∈ {uniform, distance}; metric ∈ {Euclidean, Manhattan}'),
        ],
    }),
    // SVM
    new TableRow({
        children: [
            cell('SVM', { width: COL1_W, bold: true }),
            cell('*C* ∈ {0.1, 1, 10}; kernel ∈ {linear, RBF}'),
        ],
    }),
    // RF
    new TableRow({
        children: [
            cell('RF', { width: COL1_W, bold: true }),
            cell('trees ∈ {100, 200}; max depth ∈ {∞, 10, 20}; min samples split ∈ {2, 5}'),
        ],
    }),
    // XGBoost
    new TableRow({
        children: [
            cell('XGBoost', { width: COL1_W, bold: true, bottomBorder: 12 }),
            cell('trees ∈ {100, 200}; max depth ∈ {3, 5}; learning rate ∈ {0.05, 0.1, 0.2}', { bottomBorder: 12 }),
        ],
    }),
];

const table = new Table({
    rows,
    width: { size: TABLE_W, type: WidthType.DXA },
    columnWidths: [COL1_W, COL2_W],
});

// Note paragraph
const note = new Paragraph({
    spacing: { before: 80 },
    children: [
        new TextRun({ text: 'Note. ', italics: true, font: 'Times New Roman', size: 18 }),
        new TextRun({ text: 'All models include feature count ', font: 'Times New Roman', size: 18 }),
        new TextRun({ text: 'k', italics: true, font: 'Times New Roman', size: 18 }),
        new TextRun({ text: ' ∈ {10, 30, 50, 70, 90, all} selected via ANOVA F-test.', font: 'Times New Roman', size: 18 }),
    ],
});

// Caption above table
const caption = new Paragraph({
    spacing: { after: 120 },
    children: [
        new TextRun({ text: 'Table 1. ', bold: true, font: 'Times New Roman', size: 20 }),
        new TextRun({ text: 'Hyperparameter Search Spaces for the Evaluated Machine Learning Models.', font: 'Times New Roman', size: 20 }),
    ],
});

const doc = new Document({
    sections: [{
        properties: {
            page: { size: { width: 12240, height: 15840 } },
        },
        children: [caption, table, note],
    }],
});

Packer.toBuffer(doc).then(buf => {
    fs.writeFileSync('table1_hyperparameters.docx', buf);
    console.log('Done: table1_hyperparameters.docx');
});
