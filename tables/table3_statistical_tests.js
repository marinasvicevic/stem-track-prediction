const docx = require('docx');
const fs = require('fs');

const { Document, Packer, Table, TableRow, TableCell, Paragraph, TextRun, WidthType, AlignmentType, BorderStyle } = docx;

// Column widths in DXA (total ~9000)
const COL_W = [1700, 1400, 1800, 1200, 1900]; // Comparison, Mean diff, 95% CI, Raw p, Holm p
const TABLE_W = COL_W.reduce((a, b) => a + b, 0);

function makeCell(text, opts = {}) {
    const runs = [];
    // Support italic via *...*
    const parts = text.split(/(\*[^*]+\*)/);
    for (const p of parts) {
        if (p.startsWith('*') && p.endsWith('*')) {
            runs.push(new TextRun({ text: p.slice(1, -1), italics: true, font: 'Times New Roman', size: 18 }));
        } else {
            runs.push(new TextRun({
                text: p,
                font: 'Times New Roman',
                size: 18,
                bold: opts.bold || false,
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

// Data: Model comparison (A−B), Mean F1 difference, Approx. 95% CI, Raw p-value, Holm-adjusted p-value
const data = [
    ['LR − KNN',       '0.0266',  '[−0.0062, 0.0594]',  '0.1072',  '1.000'],
    ['LR − SVM',       '0.0075',  '[−0.0170, 0.0319]',  '0.5346',  '1.000'],
    ['LR − RF',        '−0.0014', '[−0.0318, 0.0289]',  '0.9237',  '1.000'],
    ['LR − XGBoost',   '0.0091',  '[−0.0224, 0.0405]',  '0.5583',  '1.000'],
    ['KNN − SVM',      '−0.0191', '[−0.0481, 0.0099]',  '0.1859',  '1.000'],
    ['KNN − RF',       '−0.0280', '[−0.0628, 0.0068]',  '0.1094',  '1.000'],
    ['KNN − XGBoost',  '−0.0175', '[−0.0536, 0.0185]',  '0.3251',  '1.000'],
    ['SVM − RF',       '−0.0089', '[−0.0420, 0.0242]',  '0.5847',  '1.000'],
    ['SVM − XGBoost',  '0.0016',  '[−0.0333, 0.0365]',  '0.9257',  '1.000'],
    ['RF − XGBoost',   '0.0105',  '[−0.0193, 0.0402]',  '0.4742',  '1.000'],
];

const headerRow = new TableRow({
    children: [
        headerCell('Model comparison (A − B)', COL_W[0]),
        headerCell('Mean F1 difference', COL_W[1]),
        headerCell('Approx. 95% CI', COL_W[2]),
        headerCell('Raw p-value', COL_W[3]),
        headerCell('Holm-adjusted p-value', COL_W[4]),
    ],
});

const dataRows = data.map((row, idx) => {
    const isLast = idx === data.length - 1;
    return new TableRow({
        children: row.map((cellText, col) => {
            return makeCell(cellText, {
                width: COL_W[col],
                bold: col === 0, // model comparison always bold
                align: col === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
                bottomBorder: isLast ? 12 : 0,
            });
        }),
    });
});

const table = new Table({
    rows: [headerRow, ...dataRows],
    width: { size: TABLE_W, type: WidthType.DXA },
    columnWidths: COL_W,
});

const caption = new Paragraph({
    spacing: { after: 120 },
    children: [
        new TextRun({ text: 'Table 3. ', bold: true, font: 'Times New Roman', size: 20 }),
        new TextRun({ text: 'Pairwise Comparisons of ML Models Using the Approximate Corrected Resampled Paired t-Test with Holm Adjustment.', font: 'Times New Roman', size: 20 }),
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
        children: [caption, table],
    }],
});

Packer.toBuffer(doc).then(buf => {
    fs.writeFileSync('table3_statistical_tests.docx', buf);
    console.log('Done: table3_statistical_tests.docx');
});
