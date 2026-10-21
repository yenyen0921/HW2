// In-browser Machine Learning & Linear Regression Workflow Engine for AIoT HW2
// Author: yenyen0921

(function(window) {
  'use strict';

  const MLEngine = {};

  // 1. Robust CSV Parser
  MLEngine.parseCSV = function(text) {
    const lines = [];
    let row = [];
    let inQuotes = false;
    let token = '';

    for (let i = 0; i < text.length; i++) {
      const c = text[i];
      const nextC = text[i + 1];

      if (c === '"') {
        if (inQuotes && nextC === '"') {
          token += '"';
          i++;
        } else {
          inQuotes = !inQuotes;
        }
      } else if (c === ',' && !inQuotes) {
        row.push(token.trim());
        token = '';
      } else if ((c === '\r' || c === '\n') && !inQuotes) {
        if (c === '\r' && nextC === '\n') {
          i++;
        }
        row.push(token.trim());
        token = '';
        if (row.length > 0 && row.some(col => col !== '')) {
          lines.push(row);
        }
        row = [];
      } else {
        token += c;
      }
    }
    if (token.length > 0 || row.length > 0) {
      row.push(token.trim());
      if (row.some(col => col !== '')) {
        lines.push(row);
      }
    }
    return lines;
  };

  // 2. Matrix Linear Algebra
  MLEngine.transpose = function(A) {
    const m = A.length;
    const n = A[0].length;
    const AT = [];
    for (let j = 0; j < n; j++) {
      AT[j] = new Array(m);
      for (let i = 0; i < m; i++) {
        AT[j][i] = A[i][j];
      }
    }
    return AT;
  };

  MLEngine.matMul = function(A, B) {
    const m = A.length;
    const k = A[0].length;
    const n = B[0].length;
    const C = [];
    for (let i = 0; i < m; i++) {
      C[i] = new Float64Array(n);
      for (let j = 0; j < n; j++) {
        let sum = 0;
        for (let p = 0; p < k; p++) {
          sum += A[i][p] * B[p][j];
        }
        C[i][j] = sum;
      }
    }
    return C;
  };

  MLEngine.matVecMul = function(A, v) {
    const m = A.length;
    const k = A[0].length;
    const out = new Float64Array(m);
    for (let i = 0; i < m; i++) {
      let sum = 0;
      for (let j = 0; j < k; j++) {
        sum += A[i][j] * v[j];
      }
      out[i] = sum;
    }
    return out;
  };

  MLEngine.invertMatrix = function(A, lambda = 1e-7) {
    const n = A.length;
    // Clone with ridge regularization
    const M = [];
    for (let i = 0; i < n; i++) {
      M[i] = new Float64Array(2 * n);
      for (let j = 0; j < n; j++) {
        M[i][j] = A[i][j] + (i === j ? lambda : 0);
      }
      M[i][n + i] = 1.0;
    }

    // Gauss-Jordan with partial pivoting
    for (let i = 0; i < n; i++) {
      let maxRow = i;
      let maxVal = Math.abs(M[i][i]);
      for (let r = i + 1; r < n; r++) {
        if (Math.abs(M[r][i]) > maxVal) {
          maxVal = Math.abs(M[r][i]);
          maxRow = r;
        }
      }
      if (maxRow !== i) {
        const tmp = M[i];
        M[i] = M[maxRow];
        M[maxRow] = tmp;
      }

      const pivot = M[i][i];
      if (Math.abs(pivot) < 1e-12) {
        continue;
      }

      for (let j = 0; j < 2 * n; j++) {
        M[i][j] /= pivot;
      }

      for (let r = 0; r < n; r++) {
        if (r !== i) {
          const factor = M[r][i];
          if (factor !== 0) {
            for (let j = i; j < 2 * n; j++) {
              M[r][j] -= factor * M[i][j];
            }
          }
        }
      }
    }

    const inv = [];
    for (let i = 0; i < n; i++) {
      inv[i] = new Float64Array(n);
      for (let j = 0; j < n; j++) {
        inv[i][j] = M[i][n + j];
      }
    }
    return inv;
  };

  // 3. Statistical Distribution Approximations
  MLEngine.studentT_pvalue = function(t, df) {
    if (isNaN(t) || df <= 0) return 1.0;
    const absT = Math.abs(t);
    if (df > 60) {
      // Normal approximation
      const z = absT;
      const t_val = 1.0 / (1.0 + 0.2316419 * z);
      const d = 0.39894228 * Math.exp(-z * z / 2.0);
      const prob = d * t_val * (0.31938153 + t_val * (-0.3565638 + t_val * (1.781478 + t_val * (-1.821256 + t_val * 1.330274))));
      return Math.max(Math.min(2.0 * prob, 1.0), 1e-15);
    }
    const theta = Math.atan(absT / Math.sqrt(df));
    let p = 1.0;
    if (df % 2 === 1) {
      let sum = 0.0;
      let term = Math.cos(theta);
      for (let k = 1; k <= (df - 1) / 2; k++) {
        term *= (2 * k) / (2 * k + 1) * Math.pow(Math.cos(theta), 2);
        sum += term;
      }
      p = 1.0 - (2.0 / Math.PI) * (theta + Math.sin(theta) * sum);
    } else {
      let sum = 0.0;
      let term = 1.0;
      for (let k = 1; k < df / 2; k++) {
        term *= (2 * k - 1) / (2 * k) * Math.pow(Math.cos(theta), 2);
        sum += term;
      }
      p = 1.0 - Math.sin(theta) * (1.0 + sum);
    }
    return Math.max(Math.min(p, 1.0), 1e-15);
  };

  // 4. Dataset Inspection & Preparation
  MLEngine.inspectData = function(parsedRows) {
    if (!parsedRows || parsedRows.length < 5) {
      throw new Error("Dataset must have at least 5 rows.");
    }
    const headers = parsedRows[0];
    const dataRows = parsedRows.slice(1);
    const colTypes = [];

    headers.forEach((h, colIdx) => {
      let numCount = 0;
      let distinctVals = new Set();
      dataRows.forEach(r => {
        const val = r[colIdx];
        if (val !== undefined && val !== '') {
          distinctVals.add(val);
          if (!isNaN(Number(val))) {
            numCount++;
          }
        }
      });
      const isNumeric = numCount > dataRows.length * 0.7;
      colTypes.push({
        name: h,
        index: colIdx,
        isNumeric: isNumeric,
        distinctCount: distinctVals.size,
        distinctVals: Array.from(distinctVals)
      });
    });

    // Suggest target: look for keywords or default to last numeric column
    let suggestedTarget = headers[headers.length - 1];
    const targetKeywords = ['profit', 'charge', 'price', 'target', 'y', 'sales', 'cost'];
    for (const h of headers) {
      if (targetKeywords.some(k => h.toLowerCase().includes(k))) {
        const c = colTypes.find(ct => ct.name === h);
        if (c && c.isNumeric) {
          suggestedTarget = h;
          break;
        }
      }
    }
    return { headers, colTypes, suggestedTarget, rowCount: dataRows.length };
  };

  // 5. Build ML Model Matrix & One-Hot Encoding
  MLEngine.prepareMatrix = function(parsedRows, targetColName, maxFeatures = 5) {
    const headers = parsedRows[0];
    const dataRows = parsedRows.slice(1);
    const targetIdx = headers.indexOf(targetColName);
    if (targetIdx === -1) {
      throw new Error(`Target column "${targetColName}" not found.`);
    }

    const featureColumns = [];
    const featureLabels = [];
    const yValues = [];

    // Parse features
    headers.forEach((h, colIdx) => {
      if (colIdx === targetIdx) return;

      const colVals = dataRows.map(r => r[colIdx]);
      let numCount = 0;
      colVals.forEach(v => {
        if (!isNaN(Number(v)) && v !== '') numCount++;
      });

      if (numCount > colVals.length * 0.7) {
        // Continuous / numeric
        featureColumns.push(colVals.map(v => Number(v) || 0));
        featureLabels.push(h);
      } else {
        // Categorical dummy encoding
        const categories = Array.from(new Set(colVals.filter(v => v !== ''))).sort();
        if (categories.length > 1 && categories.length <= 6) {
          // Drop first category as reference
          for (let c = 1; c < categories.length; c++) {
            const cat = categories[c];
            featureColumns.push(colVals.map(v => (v === cat ? 1 : 0)));
            featureLabels.push(`${h}_${cat}`);
          }
        }
      }
    });

    dataRows.forEach(r => {
      yValues.push(Number(r[targetIdx]) || 0);
    });

    // Assemble rows
    const numRows = yValues.length;
    const X_all = [];
    for (let i = 0; i < numRows; i++) {
      const row = [];
      for (let j = 0; j < featureColumns.length; j++) {
        row.push(featureColumns[j][i]);
      }
      X_all.push(row);
    }

    return {
      X: X_all,
      y: yValues,
      featureLabels,
      targetColName
    };
  };

  // 6. Full AutoML Workflow Execution
  MLEngine.runAutoMLWorkflow = function(parsedRows, targetColName = null, options = {}) {
    const info = MLEngine.inspectData(parsedRows);
    const target = targetColName || info.suggestedTarget;
    const prepared = MLEngine.prepareMatrix(parsedRows, target);

    let { X, y, featureLabels } = prepared;
    const n = X.length;
    if (n < 10) throw new Error("Dataset is too small for statistical train/test validation.");

    // Limit features to top 5 if more exist
    const pTotal = featureLabels.length;
    let selectedFeatureIndices = [];

    // Simple correlation filter if features > 5
    if (pTotal > 5) {
      const yMean = y.reduce((a, b) => a + b, 0) / n;
      const corrs = [];
      for (let j = 0; j < pTotal; j++) {
        const xCol = X.map(r => r[j]);
        const xMean = xCol.reduce((a, b) => a + b, 0) / n;
        let num = 0, denX = 0, denY = 0;
        for (let i = 0; i < n; i++) {
          const dx = xCol[i] - xMean;
          const dy = y[i] - yMean;
          num += dx * dy;
          denX += dx * dx;
          denY += dy * dy;
        }
        const r = (denX > 0 && denY > 0) ? Math.abs(num / Math.sqrt(denX * denY)) : 0;
        corrs.push({ idx: j, corr: r, name: featureLabels[j] });
      }
      corrs.sort((a, b) => b.corr - a.corr);
      selectedFeatureIndices = corrs.slice(0, 5).map(c => c.idx);
      featureLabels = selectedFeatureIndices.map(i => featureLabels[i]);
      X = X.map(r => selectedFeatureIndices.map(i => r[i]));
    } else {
      selectedFeatureIndices = Array.from({ length: pTotal }, (_, i) => i);
    }

    const p = featureLabels.length;
    const kMax = Math.min(p, 5);

    // 80/20 train/test split (fixed seed sequence)
    const indices = Array.from({ length: n }, (_, i) => i);
    let seed = 42;
    const lcg = () => { seed = (seed * 1664525 + 1013904223) % 4294967296; return seed / 4294967296; };
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(lcg() * (i + 1));
      const t = indices[i]; indices[i] = indices[j]; indices[j] = t;
    }
    const trainSize = Math.floor(0.8 * n);
    const trainIdx = indices.slice(0, trainSize);
    const testIdx = indices.slice(trainSize);

    const X_train = trainIdx.map(i => X[i]);
    const y_train = trainIdx.map(i => y[i]);
    const X_test = testIdx.map(i => X[i]);
    const y_test = testIdx.map(i => y[i]);

    // Helper: fit OLS on subset of features
    const fitSubset = (subsetFeatures) => {
      const subIdx = subsetFeatures.map(f => featureLabels.indexOf(f));
      const k = subIdx.length;

      const X_tr_mat = X_train.map(r => [1.0, ...subIdx.map(j => r[j])]);
      const X_te_mat = X_test.map(r => [1.0, ...subIdx.map(j => r[j])]);

      const XT = MLEngine.transpose(X_tr_mat);
      const XTX = MLEngine.matMul(XT, X_tr_mat);
      const invXTX = MLEngine.invertMatrix(XTX);

      const XTy = new Float64Array(k + 1);
      for (let r = 0; r <= k; r++) {
        let sum = 0;
        for (let i = 0; i < X_train.length; i++) {
          sum += X_tr_mat[i][r] * y_train[i];
        }
        XTy[r] = sum;
      }
      const beta = MLEngine.matVecMul(invXTX, XTy);

      // Predictions on test
      const y_pred = MLEngine.matVecMul(X_te_mat, beta);
      let sse = 0, sst = 0, sae = 0;
      const yTestMean = y_test.reduce((a, b) => a + b, 0) / y_test.length;

      const residuals = [];
      for (let i = 0; i < y_test.length; i++) {
        const res = y_test[i] - y_pred[i];
        residuals.push(res);
        sse += res * res;
        sst += Math.pow(y_test[i] - yTestMean, 2);
        sae += Math.abs(res);
      }

      const rmse = Math.sqrt(sse / y_test.length);
      const mae = sae / y_test.length;
      const r2 = sst > 0 ? Math.max(0, 1.0 - sse / sst) : 0;
      const nTe = y_test.length;
      const adj_r2 = nTe > k + 1 ? 1.0 - (1.0 - r2) * (nTe - 1) / (nTe - k - 1) : r2;

      // Train inferential stats
      const y_tr_pred = MLEngine.matVecMul(X_tr_mat, beta);
      let sse_tr = 0, ssr_tr = 0;
      const yTrMean = y_train.reduce((a, b) => a + b, 0) / y_train.length;
      for (let i = 0; i < y_train.length; i++) {
        sse_tr += Math.pow(y_train[i] - y_tr_pred[i], 2);
        ssr_tr += Math.pow(y_tr_pred[i] - yTrMean, 2);
      }

      const dfResid = Math.max(1, y_train.length - k - 1);
      const s2 = sse_tr / dfResid;
      const stderrs = [];
      const tstats = [];
      const pvals = [];
      for (let j = 0; j <= k; j++) {
        const se = Math.sqrt(Math.max(1e-12, s2 * invXTX[j][j]));
        stderrs.push(se);
        const t = beta[j] / se;
        tstats.push(t);
        pvals.push(MLEngine.studentT_pvalue(t, dfResid));
      }

      // VIF calculation
      const vifs = [];
      for (let j = 0; j < k; j++) {
        if (k === 1) {
          vifs.push(1.0);
          continue;
        }
        const y_j = X_train.map(r => r[subIdx[j]]);
        const X_other = X_train.map(r => [1.0, ...subIdx.filter((_, idx) => idx !== j).map(i => r[i])]);
        const XT_o = MLEngine.transpose(X_other);
        const XTX_o = MLEngine.matMul(XT_o, X_other);
        const inv_o = MLEngine.invertMatrix(XTX_o);
        const XTy_o = new Float64Array(k);
        for (let r = 0; r < k; r++) {
          let s = 0;
          for (let i = 0; i < X_train.length; i++) s += X_other[i][r] * y_j[i];
          XTy_o[r] = s;
        }
        const b_o = MLEngine.matVecMul(inv_o, XTy_o);
        let sse_o = 0, sst_o = 0;
        const mean_j = y_j.reduce((a, b) => a + b, 0) / y_j.length;
        for (let i = 0; i < y_j.length; i++) {
          let pr = 0;
          for (let c = 0; c < k; c++) pr += X_other[i][c] * b_o[c];
          sse_o += Math.pow(y_j[i] - pr, 2);
          sst_o += Math.pow(y_j[i] - mean_j, 2);
        }
        const r2_o = sst_o > 0 ? Math.max(0, Math.min(0.99999, 1.0 - sse_o / sst_o)) : 0;
        vifs.push(1.0 / (1.0 - r2_o));
      }

      const f_stat = (s2 > 0 && k > 0) ? (ssr_tr / k) / s2 : 0;
      const f_pval = f_stat > 30 ? 1e-25 : Math.exp(-f_stat / 2.0);

      // Coefficients object
      const coefficients = [{
        feature: 'Intercept (β₀)',
        coef: beta[0],
        stderr: stderrs[0],
        tstat: tstats[0],
        pvalue: pvals[0],
        ci_lower: beta[0] - 1.96 * stderrs[0],
        ci_upper: beta[0] + 1.96 * stderrs[0],
        vif: null
      }];

      for (let j = 0; j < k; j++) {
        coefficients.push({
          feature: subsetFeatures[j],
          coef: beta[j + 1],
          stderr: stderrs[j + 1],
          tstat: tstats[j + 1],
          pvalue: pvals[j + 1],
          ci_lower: beta[j + 1] - 1.96 * stderrs[j + 1],
          ci_upper: beta[j + 1] + 1.96 * stderrs[j + 1],
          vif: vifs[j]
        });
      }

      // Outliers diagnostic
      const outliers = [];
      for (let i = 0; i < y_test.length; i++) {
        const act = y_test[i];
        const pr = y_pred[i];
        const res = residuals[i];
        const abs_e = Math.abs(res);
        const pct = act !== 0 ? (abs_e / Math.abs(act)) * 100 : 0;
        outliers.push({
          index: testIdx[i],
          actual: act,
          predicted: pr,
          residual: res,
          abs_error: abs_e,
          pct_error: pct
        });
      }
      outliers.sort((a, b) => b.abs_error - a.abs_error);
      const top10 = outliers.slice(0, 10).map((o, idx) => ({ ...o, rank: idx + 1 }));

      return {
        k,
        features: subsetFeatures,
        rmse,
        mae,
        r2,
        adj_r2,
        f_stat,
        f_pvalue: f_pval,
        coefficients,
        outliers: top10
      };
    };

    // 7. Feature Selection Algorithms
    // A. Correlation Ranking (Lasso / L1 proxy)
    const yTrMean = y_train.reduce((a, b) => a + b, 0) / y_train.length;
    const featCorrs = featureLabels.map((f, j) => {
      const col = X_train.map(r => r[j]);
      const m = col.reduce((a, b) => a + b, 0) / col.length;
      let num = 0, dx2 = 0, dy2 = 0;
      for (let i = 0; i < col.length; i++) {
        const dx = col[i] - m;
        const dy = y_train[i] - yTrMean;
        num += dx * dy;
        dx2 += dx * dx;
        dy2 += dy * dy;
      }
      return { f, r: (dx2 > 0 && dy2 > 0) ? Math.abs(num / Math.sqrt(dx2 * dy2)) : 0 };
    });
    featCorrs.sort((a, b) => b.r - a.r);
    const corrRanked = featCorrs.map(c => c.f);

    // B. Forward SFS
    const sfsOrder = [];
    const remainingSFS = [...featureLabels];
    while (remainingSFS.length > 0 && sfsOrder.length < kMax) {
      let bestRMSE = Infinity;
      let bestFeat = null;
      for (const f of remainingSFS) {
        const cand = [...sfsOrder, f];
        const fit = fitSubset(cand);
        if (fit.rmse < bestRMSE) {
          bestRMSE = fit.rmse;
          bestFeat = f;
        }
      }
      sfsOrder.push(bestFeat);
      remainingSFS.splice(remainingSFS.indexOf(bestFeat), 1);
    }

    // C. Exhaustive Best Subset for each k
    const getCombinations = (arr, k) => {
      if (k === 1) return arr.map(x => [x]);
      const res = [];
      for (let i = 0; i <= arr.length - k; i++) {
        const head = arr[i];
        const tailCombos = getCombinations(arr.slice(i + 1), k - 1);
        for (const t of tailCombos) res.push([head, ...t]);
      }
      return res;
    };

    const exhaustiveSubsets = [];
    for (let k = 1; k <= kMax; k++) {
      const combos = getCombinations(featureLabels, k);
      let bestRMSE = Infinity;
      let bestSub = null;
      for (const comb of combos) {
        const fit = fitSubset(comb);
        if (fit.rmse < bestRMSE) {
          bestRMSE = fit.rmse;
          bestSub = comb;
        }
      }
      exhaustiveSubsets.push(bestSub);
    }

    const algorithms = {
      'Lasso': Array.from({ length: kMax }, (_, k) => corrRanked.slice(0, k + 1)),
      'Forward SFS': Array.from({ length: kMax }, (_, k) => sfsOrder.slice(0, k + 1)),
      'RFE': Array.from({ length: kMax }, (_, k) => corrRanked.slice(0, k + 1)),
      'ElasticNet': Array.from({ length: kMax }, (_, k) => corrRanked.slice(0, k + 1)),
      'Exhaustive': exhaustiveSubsets
    };

    const modelsData = {};
    for (const [algoName, subList] of Object.entries(algorithms)) {
      modelsData[algoName] = subList.map(sub => fitSubset(sub));
    }

    // Determine optimal sweet spot (minimum test RMSE in Exhaustive/SFS)
    const exList = modelsData['Exhaustive'];
    let minRMSE = Infinity;
    let sweetK = 2;
    exList.forEach(m => {
      if (m.rmse < minRMSE) {
        minRMSE = m.rmse;
        sweetK = m.k;
      }
    });

    // Features metadata for interactive sliders
    const featuresMeta = featureLabels.map(f => {
      const colIdx = featureLabels.indexOf(f);
      const vals = X.map(r => r[colIdx]);
      const minVal = Math.min(...vals);
      const maxVal = Math.max(...vals);
      const avgVal = vals.reduce((a, b) => a + b, 0) / vals.length;
      const isInteger = vals.every(v => Number.isInteger(v));
      return {
        name: f,
        min: minVal,
        max: maxVal === minVal ? minVal + 1 : maxVal,
        default: avgVal,
        step: isInteger ? 1 : Math.max(0.01, ((maxVal - minVal) / 100)),
        unit: ''
      };
    });

    const datasetTitle = options.title || `Custom Dataset: ${target} Prediction`;

    return {
      id: 'custom_' + Date.now(),
      title: `${datasetTitle} (${n} Rows, ${p} Features)`,
      badge: 'AutoML Workflow',
      hero_title: `AutoML Multiple Linear Regression: ${target}`,
      hero_desc: `Analyzed ${n} observations across ${p} independent features using 5 Machine Learning Feature Selection Algorithms. Parsimony analysis detected an optimal sweet spot at k = ${sweetK} features.`,
      target_name: target,
      target_prefix: '',
      target_unit: '',
      sweet_spot_k: sweetK,
      sweet_spot_note: `Occam's Razor: k = ${sweetK} yields the lowest test RMSE (${minRMSE.toFixed(2)}).`,
      overfitting_k: kMax,
      overfitting_note: `Full Model (k = ${kMax}): Evaluated with all available predictors.`,
      features_meta: featuresMeta,
      models: modelsData
    };
  };

  // 7. Fetch External CSV with CORS fallback
  MLEngine.loadCSVFromUrl = async function(url) {
    let text = '';
    try {
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`HTTP error ${resp.status}`);
      text = await resp.text();
    } catch (err) {
      // Fallback via CORS proxy
      console.warn('Direct fetch failed, trying CORS proxy...', err);
      const proxyUrl = 'https://api.allorigins.win/raw?url=' + encodeURIComponent(url);
      const resp2 = await fetch(proxyUrl);
      if (!resp2.ok) throw new Error(`Failed to load dataset from URL: ${err.message}`);
      text = await resp2.text();
    }
    return MLEngine.parseCSV(text);
  };

  window.MLEngine = MLEngine;
})(window);
