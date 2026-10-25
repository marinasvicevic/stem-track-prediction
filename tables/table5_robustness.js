const docx = require('docx');
const fs = require('fs');

const { Document, Packer, Table, TableRow, TableCell, Paragraph, TextRun, WidthType, AlignmentType, BorderStyle } = docx;

// Column widths in DXA
const COL_W = [3600, 1400, 1600, 1400]; // Feature config, Accuracy, F1(w), ROC-AUC
const TABLE_W = COL_W.reduce((a, b) => a + b, 0);

function makeCell(text, opts = {}) {
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
            children: [new TextRun({
                text,
                font: 'Times New Roman',
                size: 18,
                bold: opts.bold || false,
            })],
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

// Data: Feature configuration, Accuracy, F1 (weighted), ROC-AUC
// Bold the best value in each metric column
const data = [
    ['All features',                        '0.754',     '0.748',     '0.812'],
    ['Without physics interestingness',     '**0.758**', '**0.754**', '**0.817**'],
    ['Without enrollment decision factors', '0.710',     '0.706',     '0.756'],
];

const headerRow = new TableRow({
    children: [
        headerCell('Feature configuration', COL_W[0]),
        headerCell('Accuracy', COL_W[1]),
        headerCell('F1 (weighted)', COL_W[2]),
        headerCell('ROC-AUC', COL_W[3]),
    ],
});

const dataRows = data.map((row, idx) => {
    const isLast = idx === data.length - 1;
    return new TableRow({
        children: row.map((cellText, col) => {
            const isBold = cellText.startsWith('**') && cellText.endsWith('**');
            const cleanText = isBold ? cellText.slice(2, -2) : cellText;
            return makeCell(cleanText, {
                width: COL_W[col],
                bold: isBold,
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
        new TextRun({ text: 'Table 5. ', bold: true, font: 'Times New Roman', size: 20 }),
        new TextRun({ text: 'Random Forest Robustness under Feature Ablation.', font: 'Times New Roman', size: 20 }),
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
    fs.writeFileSync('table5_robustness.docx', buf);
    console.log('Done: table5_robustness.docx');
});
