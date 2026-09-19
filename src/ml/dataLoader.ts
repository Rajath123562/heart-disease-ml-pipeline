/**
 * Data Loader and CSV Parser with Schema Auto-Detection
 * Supports:
 *  1. heart.csv: numeric 14-column format
 *  2. heart_disease_uci.csv: mixed text/boolean format with num (0-4)
 *  3. Custom uploaded CSVs with user-chosen target
 */

import { RawDataRow, DatasetMeta } from '../types';
import { DEFAULT_UCI_HEART_CSV } from '../data/heartData';

export interface ParseResult {
  data: RawDataRow[];
  meta: DatasetMeta;
  warnings: string[];
}

/**
 * Robust CSV string parser handling quotes, spaces, and edge cases
 */
export function parseCSV(csvText: string): { headers: string[]; rows: (string | null)[][] } {
  const cleanText = csvText.replace(/\r\n/g, '\n').replace(/\r/g, '\n').trim();
  if (!cleanText) {
    throw new Error('The CSV file is empty. Please provide a valid CSV file.');
  }

  const lines = cleanText.split('\n');
  if (lines.length < 2) {
    throw new Error('CSV must contain a header row and at least one data row.');
  }

  function parseLine(line: string): (string | null)[] {
    const result: (string | null)[] = [];
    let current = '';
    let inQuotes = false;

    for (let i = 0; i < line.length; i++) {
      const char = line[i];
      if (char === '"') {
        if (inQuotes && line[i + 1] === '"') {
          current += '"';
          i++; // Skip escaped quote
        } else {
          inQuotes = !inQuotes;
        }
      } else if (char === ',' && !inQuotes) {
        const val = current.trim();
        result.push(isMissing(val) ? null : val);
        current = '';
      } else {
        current += char;
      }
    }
    const lastVal = current.trim();
    result.push(isMissing(lastVal) ? null : lastVal);
    return result;
  }

  function isMissing(val: string): boolean {
    const lower = val.toLowerCase();
    return val === '' || lower === '?' || lower === 'na' || lower === 'null' || lower === 'nan' || lower === 'none';
  }

  const headers = parseLine(lines[0]).map((h, i) => (h ? h.trim() : `col_${i}`));
  const rows: (string | null)[][] = [];

  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    const parsed = parseLine(line);
    // Pad or slice to match headers length
    while (parsed.length < headers.length) parsed.push(null);
    rows.push(parsed.slice(0, headers.length));
  }

  return { headers, rows };
}

/**
 * Normalizes categorical text values from heart_disease_uci.csv into standard numerical representations
 */
function normalizeUCIValue(colName: string, rawVal: string | null): number | string | null {
  if (rawVal === null) return null;
  const col = colName.toLowerCase();
  const val = rawVal.toLowerCase().trim();

  // Try parsing float first
  const num = Number(val);
  if (!isNaN(num) && val !== '') {
    return num;
  }

  // Text mappings for heart_disease_uci.csv schema
  if (col === 'sex') {
    if (val.startsWith('m')) return 1;
    if (val.startsWith('f')) return 0;
  }

  if (col === 'fbs') {
    if (val === 'true' || val === 't' || val === '1' || val === 'yes') return 1;
    if (val === 'false' || val === 'f' || val === '0' || val === 'no') return 0;
  }

  if (col === 'exang') {
    if (val === 'true' || val === 't' || val === '1' || val === 'yes') return 1;
    if (val === 'false' || val === 'f' || val === '0' || val === 'no') return 0;
  }

  if (col === 'cp') {
    if (val.includes('typical')) return 0;
    if (val.includes('atypical')) return 1;
    if (val.includes('non-anginal') || val.includes('non anginal')) return 2;
    if (val.includes('asymptomatic')) return 3;
  }

  if (col === 'restecg') {
    if (val.includes('normal')) return 0;
    if (val.includes('st-t') || val.includes('wave') || val.includes('abnormality')) return 1;
    if (val.includes('hypertrophy') || val.includes('lvh')) return 2;
  }

  if (col === 'slope') {
    if (val.includes('up') || val === 'upsloping') return 0;
    if (val.includes('flat')) return 1;
    if (val.includes('down') || val === 'downsloping') return 2;
  }

  if (col === 'thal') {
    if (val.includes('normal')) return 1;
    if (val.includes('fixed')) return 2;
    if (val.includes('revers') || val.includes('defect')) return 3;
  }

  // Boolean general check
  if (val === 'true' || val === 't') return 1;
  if (val === 'false' || val === 'f') return 0;

  return rawVal;
}

/**
 * Auto-detect schema, clean columns, drop id/dataset, binarize target, compute metadata
 */
export function processRawCSV(
  csvText: string,
  datasetName: string = 'Cleveland Heart Disease',
  sourceType: 'uci_authentic' | 'uploaded' | 'synthetic' = 'uci_authentic',
  targetOverride?: string
): ParseResult {
  const { headers, rows } = parseCSV(csvText);
  const warnings: string[] = [];

  // Determine target column
  let detectedTarget = targetOverride;
  if (!detectedTarget) {
    const lowerHeaders = headers.map(h => h.toLowerCase());
    if (lowerHeaders.includes('target')) {
      detectedTarget = headers[lowerHeaders.indexOf('target')];
    } else if (lowerHeaders.includes('num')) {
      detectedTarget = headers[lowerHeaders.indexOf('num')];
    } else if (lowerHeaders.includes('diagnosis')) {
      detectedTarget = headers[lowerHeaders.indexOf('diagnosis')];
    } else if (lowerHeaders.includes('disease')) {
      detectedTarget = headers[lowerHeaders.indexOf('disease')];
    } else {
      // Pick last column
      detectedTarget = headers[headers.length - 1];
      warnings.push(`Target column not explicitly identified as 'target' or 'num'; defaulting to '${detectedTarget}'.`);
    }
  }

  // Identify columns to drop: id, dataset (common in heart_disease_uci.csv)
  const colsToDrop = new Set<string>();
  for (const h of headers) {
    const lh = h.toLowerCase();
    if (lh === 'id' || lh === 'dataset' || lh === 'unnamed: 0' || lh === 'index') {
      colsToDrop.add(h);
    }
  }

  // Map alternative column names (e.g. thalch -> thalach)
  const renameMap: Record<string, string> = {
    thalch: 'thalach',
    pain_type: 'cp',
    chest_pain: 'cp',
    max_heart_rate: 'thalach',
    st_depression: 'oldpeak'
  };

  const finalHeaders = headers
    .filter(h => !colsToDrop.has(h))
    .map(h => renameMap[h.toLowerCase()] || h);

  const cleanRows: RawDataRow[] = [];
  let totalMissing = 0;

  for (const row of rows) {
    const obj: RawDataRow = {};
    for (let j = 0; j < headers.length; j++) {
      const origHeader = headers[j];
      if (colsToDrop.has(origHeader)) continue;
      
      const newHeader = renameMap[origHeader.toLowerCase()] || origHeader;
      let rawVal = row[j];

      // If this is the target column
      if (origHeader === detectedTarget) {
        if (rawVal === null) {
          obj[newHeader] = null;
          totalMissing++;
        } else {
          const numVal = Number(rawVal);
          // If 'num' 0..4, binarize: 0 = no disease, >0 = disease
          if (origHeader.toLowerCase() === 'num') {
            obj[newHeader] = isNaN(numVal) ? (rawVal.toLowerCase() === 'true' ? 1 : 0) : (numVal > 0 ? 1 : 0);
          } else {
            obj[newHeader] = isNaN(numVal) ? (rawVal.toLowerCase().includes('disease') || rawVal.toLowerCase() === 'true' ? 1 : 0) : (numVal > 0 ? 1 : 0);
          }
        }
      } else {
        const normalized = normalizeUCIValue(newHeader, rawVal);
        if (normalized === null) {
          totalMissing++;
        }
        obj[newHeader] = normalized;
      }
    }
    cleanRows.push(obj);
  }

  // Duplicate detection
  let duplicateCount = 0;
  const seenRowStrings = new Set<string>();
  for (const row of cleanRows) {
    const key = JSON.stringify(row);
    if (seenRowStrings.has(key)) {
      duplicateCount++;
    } else {
      seenRowStrings.add(key);
    }
  }

  const finalTarget = renameMap[detectedTarget.toLowerCase()] || detectedTarget;

  const meta: DatasetMeta = {
    name: datasetName,
    source: sourceType,
    rowCount: cleanRows.length,
    columnCount: finalHeaders.length,
    columns: finalHeaders,
    targetColumn: finalTarget,
    hasMissing: totalMissing > 0,
    missingCount: totalMissing,
    duplicateCount,
    isSynthetic: sourceType === 'synthetic'
  };

  return {
    data: cleanRows,
    meta,
    warnings
  };
}

export function loadDefaultDataset(): { data: RawDataRow[]; meta: DatasetMeta } {
  const parsed = processRawCSV(DEFAULT_UCI_HEART_CSV, 'Cleveland Heart Disease (UCI)', 'uci_authentic');
  return {
    data: parsed.data,
    meta: parsed.meta
  };
}

export function parseCsvString(
  csvText: string,
  datasetName: string = 'Uploaded Dataset',
  sourceType: 'uci_authentic' | 'uploaded' | 'synthetic' = 'uploaded',
  targetOverride?: string
): ParseResult {
  return processRawCSV(csvText, datasetName, sourceType, targetOverride);
}

export function getInitialDatasetMeta(): DatasetMeta {
  return loadDefaultDataset().meta;
}
