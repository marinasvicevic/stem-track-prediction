const docx = require('docx');
const fs = require('fs');

const { Document, Packer, Table, TableRow, TableCell, Paragraph, TextRun, WidthType, AlignmentType, BorderStyle } = docx;

// Column widths in DXA
const COL_W = [1500, 1000, 1100, 1300, 1300, 1100]; // Model, Gender, Accuracy, Above baseline, F1(w), ROC-AUC
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

// Data: Model, Gender, Accuracy, Above Baseline, F1 (weighted), ROC-AUC
// Bold the best accuracy, F1, and ROC-AUC per model (between genders)
const data = [
    ['LR',      'Male',   '0.750', '0.141', '0.752', '0.773'],
    ['',        'Female', '0.720', '0.049', '0.722', '0.779'],
    ['KNN',     'Male',   '0.656', '0.047', '0.660', '0.708'],
    ['',        'Female', '0.713', '0.042', '0.717', '0.781'],
    ['SVM',     'Male',   '0.766', '0.156', '0.768', '0.779'],
    ['',        'Female', '0.797', '0.126', '0.797', '0.799'],
    ['RF',      'Male',   '0.688', '0.078', '0.691', '0.779'],
    ['',        'Female', '0.783', '0.112', '0.775', '0.793'],
    ['XGBoost', 'Male',   '0.734', '0.125', '0.733', '0.781'],
    ['',        'Female', '0.734', '0.063', '0.729', '0.790'],
];

const headerRow = new TableRow({
    children: [
        headerCell('Model', COL_W[0]),
        headerCell('Gender', COL_W[1]),
        headerCell('Accuracy', COL_W[2]),
        headerCell('Above baseline', COL_W[3]),
        headerCell('F1 (weighted)', COL_W[4]),
        headerCell('ROC-AUC', COL_W[5]),
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
                bold: isBold || col === 0, // model name always bold
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
        new TextRun({ text: 'Table 4. ', bold: true, font: 'Times New Roman', size: 20 }),
        new TextRun({ text: 'Model Performance by Gender on the Independent Test Set.', font: 'Times New Roman', size: 20 }),
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
    fs.writeFileSync('table4_fairness.docx', buf);
    console.log('Done: table4_fairness.docx');
});
