const docx = require('docx');
const fs = require('fs');

const { Document, Packer, Table, TableRow, TableCell, Paragraph, TextRun, WidthType, AlignmentType, BorderStyle } = docx;

// Column widths in DXA (total ~9000 for good fit)
const COL_W = [1100, 1100, 950, 950, 850, 1050, 900, 1000]; // Model, CV F1, Acc, Prec, Recall, F1w, F1m, AUC
const TABLE_W = COL_W.reduce((a, b) => a + b, 0);

function makeCell(text, opts = {}) {
    const runs = [];
    const parts = text.split(/(\*[^*]+\*)/);
    for (const p of parts) {
        if (p.startsWith('*') && p.endsWith('*')) {
            runs.push(new TextRun({ text: p.slice(1, -1), italics: true, font: 'Times New Roman', size: 18 }));
        } else {
            runs.push(new TextRun({
                text: p,
                font: 'Times New Roman',
                size: 18,
                bold: opts.bold || false
            }));
        }
    }

    const borders = {
        top: { style: opts.topBorder ? BorderStyle.SINGLE : BorderStyle.NONE, size: opts.topBorder || 0 },
        bottom: { style: opts.bottomBorder ? BorderStyle.SINGLE : BorderStyle.NONE, size: opts.bottomBorder || 0 },
        left: { style: BorderStyle.NONE },
        right: { style: BorderStyle.NONE },
    };

    return new TableCell({
        width: { size: opts.width || COL_W[1], type: WidthType.DXA },
        verticalAlign: 'center',
        margins: { top: 30, bottom: 30, left: 50, right: 50 },
        borders,
        children: [new Paragraph({
            alignment: opts.align || AlignmentType.CENTER,
            children: runs,
        })],
    });
}

function headerCell(text, width) {
    return new TableCell({
        width: { size: width, type: WidthType.DXA },
        verticalAlign: 'center',
        margins: { top: 40, bottom: 40, left: 50, right: 50 },
        borders: {
            top: { style: BorderStyle.SINGLE, size: 12 },
            bottom: { style: BorderStyle.SINGLE, size: 6 },
            left: { style: BorderStyle.NONE },
            right: { style: BorderStyle.NONE },
        },
        children: [new Paragraph({
            alignment: AlignmentType.CENTER,
            children: [new TextRun({ text, bold: true, font: 'Times New Roman', size: 18 })],
        })],
    });
}

// Data rows: Model, CV F1 ± SD, Accuracy, Precision, Recall, F1(w), F1(m), ROC-AUC
// Bold = best in column (among ML models)
const data = [
    ['SVM',       '0.748 ± 0.014', '**0.787**', '**0.769**', '**0.698**', '**0.786**', '**0.778**', '0.807'],
    ['Random Forest', '0.748 ± 0.019', '0.754', '0.754', '0.605', '0.748', '0.737', '**0.812**'],
    ['XGBoost',   '0.752 ± 0.021', '0.734', '0.691', '0.651', '0.733', '0.724', '0.810'],
    ['Logistic Regression', '**0.754 ± 0.019**', '0.729', '0.674', '0.674', '0.729', '0.722', '0.800'],
    ['KNN',       '0.745 ± 0.018', '0.696', '0.635', '0.628', '0.695', '0.686', '0.766'],
    ['Baseline',  '—',             '0.585', '0.000', '0.000', '0.431', '0.369', '0.500'],
];

const dataRows = data.map((row, idx) => {
    const isLast = idx === data.length - 1;
    return new TableRow({
        children: row.map((cellText, col) => {
            const isBold = cellText.startsWith('**') && cellText.endsWith('**');
            const cleanText = isBold ? cellText.slice(2, -2) : cellText;
            return makeCell(cleanText, {
                width: COL_W[col],
                bold: isBold || col === 0, // model name always bold
                align: col === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
                bottomBorder: isLast ? 12 : 0,
            });
        }),
    });
});

const headerRow = new TableRow({
    children: [
        headerCell('Model', COL_W[0]),
        headerCell('CV F1 (weighted) ± SD', COL_W[1]),
        headerCell('Accuracy', COL_W[2]),
        headerCell('Precision', COL_W[3]),
        headerCell('Recall', COL_W[4]),
        headerCell('F1 (weighted)', COL_W[5]),
        headerCell('F1 (macro)', COL_W[6]),
        headerCell('ROC-AUC', COL_W[7]),
    ],
});

const table = new Table({
    rows: [headerRow, ...dataRows],
    width: { size: TABLE_W, type: WidthType.DXA },
    columnWidths: COL_W,
});

const caption = new Paragraph({
    spacing: { after: 120 },
    children: [
        new TextRun({ text: 'Table 2. ', bold: true, font: 'Times New Roman', size: 20 }),
        new TextRun({ text: 'Comparative Performance of Machine Learning Models on Cross-Validation and the Independent Test Set.', font: 'Times New Roman', size: 20 }),
    ],
});

const note = new Paragraph({
    spacing: { before: 80 },
    children: [
        new TextRun({ text: 'Note. ', italics: true, font: 'Times New Roman', size: 18 }),
        new TextRun({ text: 'Precision and recall refer to the STEM class. CV F1 represents the mean weighted F1 score and standard deviation across five stratified cross-validation folds. All remaining metrics were calculated on the independent test set (', font: 'Times New Roman', size: 18 }),
        new TextRun({ text: 'N', italics: true, font: 'Times New Roman', size: 18 }),
        new TextRun({ text: ' = 207). Bold values indicate the best result among the five ML models in each column.', font: 'Times New Roman', size: 18 }),
    ],
});

const doc = new Document({
    sections: [{
        properties: {
            page: {
                size: { width: 12240, height: 15840 },
                margin: { top: 1440, bottom: 1440, left: 1080, right: 1080 },
            },
        },
        children: [caption, table, note],
    }],
});

Packer.toBuffer(doc).then(buf => {
    fs.writeFileSync('table2_performance.docx', buf);
    console.log('Done: table2_performance.docx');
});
