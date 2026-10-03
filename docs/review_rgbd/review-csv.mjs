export function csvCell(value) {
  return `"${String(value ?? '').replaceAll('"', '""')}"`;
}

export function serializeCSV(header, rows) {
  return '\ufeff' + [header, ...rows].map(row => row.map(csvCell).join(',')).join('\r\n') + '\r\n';
}

export function parseCSV(text) {
  const table = [];
  let row = [], cell = '', quoted = false;
  text = text.replace(/^\ufeff/, '');
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (char === '"') {
      if (quoted && text[i + 1] === '"') { cell += '"'; i++; }
      else quoted = !quoted;
    } else if (!quoted && (char === ',' || char === '\n' || char === '\r')) {
      row.push(cell); cell = '';
      if (char !== ',') {
        if (row.some(value => value !== '')) table.push(row);
        row = [];
        if (char === '\r' && text[i + 1] === '\n') i++;
      }
    } else cell += char;
  }
  if (quoted) throw new Error('CSV引号未闭合。');
  row.push(cell);
  if (row.some(value => value !== '')) table.push(row);
  if (!table.length) throw new Error('CSV没有内容。');
  const header = table.shift();
  if (new Set(header).size !== header.length) throw new Error('CSV列名重复。');
  return table.map((values, index) => {
    if (values.length !== header.length) throw new Error(`CSV第${index + 2}条记录列数不符。`);
    return Object.fromEntries(header.map((name, i) => [name, values[i]]));
  });
}
