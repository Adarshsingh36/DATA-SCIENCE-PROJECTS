const pptxgen = require("pptxgenjs");
const { applyTheme } = require("/mnt/skills/public/pptx/scripts/apply_theme.js");
const D = require("./deck_data.json");

const THEME = { name: "Walmart Forecasting", headFontFace: "Georgia", bodyFontFace: "Calibri",
  colors: { dk1: "1B2631", lt1: "FFFFFF", dk2: "12304A", lt2: "F2F5F8", accent1: "1C7293", accent2: "E09F3E",
            accent3: "2E8B57", accent4: "C0392B", accent5: "7F8C8D", accent6: "5B7DB1", hlink: "1C7293", folHlink: "7F8C8D" } };
const HEX = { navy: "12304A", teal: "1C7293", amber: "E09F3E", green: "2E8B57", red: "C0392B", grey: "7F8C8D", light: "F2F5F8", ink: "1B2631" };

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

pres.defineSlideMaster({ title: "DARK", background: { color: HEX.navy },
  objects: [], slideNumber: { x: 9.2, y: 5.25, color: "CADCFC", fontSize: 9 } });
pres.defineSlideMaster({ title: "CONTENT", background: { color: "FFFFFF" },
  objects: [{ rect: { x: 0, y: 0, w: 10, h: 0.95, fill: { color: HEX.navy } } }],
  slideNumber: { x: 9.2, y: 5.25, color: HEX.grey, fontSize: 9 },
  placeholder: { options: { name: "title", type: "title", x: 0.45, y: 0.12, w: 9.1, h: 0.7, fontSize: 24, bold: true, color: "FFFFFF", fontFace: "Georgia", valign: "middle", margin: 0 }, text: "Title" } });

const tx = (s, text, o) => o.placeholder ? s.addText(text, { placeholder: o.placeholder, color: "FFFFFF", bold: true, fontFace: "Georgia", fontSize: 20 }) : s.addText(text, Object.assign({ isTextBox: true, fontFace: "Calibri", color: HEX.ink, fontSize: 14, margin: 0 }, o));
const bullets = (arr, size = 14) => arr.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < arr.length - 1, paraSpaceAfter: 6, fontSize: size } }));
const sec = (t) => pres.addSection({ title: t });
const card = (s, x, y, w, h, big, small, col) => {
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h, fill: { color: HEX.light }, line: { color: "DDE3EA", width: 1 }, rectRadius: 0.08, objectName: "card" });
  tx(s, big, { x: x + 0.15, y: y + 0.1, w: w - 0.3, h: h * 0.5, fontSize: 28, bold: true, color: col, fontFace: "Georgia", valign: "middle" });
  tx(s, small, { x: x + 0.15, y: y + h * 0.55, w: w - 0.3, h: h * 0.4, fontSize: 11, color: HEX.ink, valign: "top" });
};
const L = (m) => ({ naive: "Naive", snaive: "Seasonal naive", lgbm: "LightGBM", lgbm_bottom_up: "LGBM + bottom-up", lgbm_wls_struct: "LGBM + WLS", lgbm_ols: "LGBM + OLS",
  lgbm_mint_shrink: "LGBM + MinT", prophet: "Prophet", tft: "TFT" }[m] || m);
const chartCommon = { catAxisLabelColor: HEX.ink, valAxisLabelColor: HEX.grey, valGridLine: { color: "E3E8ED", size: 0.5 }, catGridLine: { style: "none" },
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", titleFontFace: "+mj-lt", showLegend: false,
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", dataLabelFontSize: 11, catAxisLabelFontSize: 11, valAxisLabelFontSize: 10, showTitle: false, valAxisMinVal: 0 };
function bar(s, labels, vals, o, colors) {
  s.addChart(pres.charts.BAR, [{ name: "WAPE %", labels, values: vals }], Object.assign({ barDir: "col", chartColors: colors, ...chartCommon }, o));
}

// 1 Title
sec("Introduction");
let s = pres.addSlide({ masterName: "DARK", sectionTitle: "Introduction" });
tx(s, "Forecasting weekly demand across Walmart's store-department hierarchy", { x: 0.7, y: 1.2, w: 8.6, h: 1.6, fontSize: 34, bold: true, color: "FFFFFF", fontFace: "Georgia", valign: "top" });
tx(s, "Leakage-safe backtests, reconciliation, prediction intervals and an inventory optimiser", { x: 0.7, y: 3.0, w: 8.6, h: 0.6, fontSize: 18, color: "CADCFC" });
tx(s, "Kaggle Walmart Store Sales Forecasting | 3,331 store-department series | 143 weeks", { x: 0.7, y: 4.6, w: 8.6, h: 0.4, fontSize: 12, color: HEX.amber });
s.addNotes("Open with the one-line promise: a forecasting system you can trust because every number is a chronological, out-of-sample result, including the unflattering ones. About 10 minutes: problem, method, results, inventory, what to take away.");

// 2 Problem and data
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
tx(s, "Planning needs weekly forecasts at every level, with uncertainty", { placeholder: "title" });
card(s, 0.45, 1.25, 2.9, 1.35, "3,331", "store-department weekly series, 45 stores, 81 departments", HEX.teal);
card(s, 3.55, 1.25, 2.9, 1.35, "143 weeks", "2010-02-05 to 2012-10-26; 11.5% of the grid unobserved", HEX.teal);
card(s, 6.65, 1.25, 2.9, 1.35, "1 year", "of prior holiday history in the hardest test window", HEX.amber);
tx(s, bullets(["Target: Weekly_Sales in dollars, 13-week horizon, forecast from the last observed week", "Features: lags 1/2/4/8/12/52, past-only rolling means and volatility, holidays, weather, fuel, markdowns, CPI, unemployment, store type and size",
  "The files hold no state and no item identifiers, so neither is invented: the geographic level is store Type, labelled as a proxy"], 14), { x: 0.45, y: 2.9, w: 9.1, h: 2.2, valign: "top" });
s.addNotes("Explain scale and the two honesty points: no state in the data, so the Group level is store type; no item IDs, so no item-level forecasts. Also mention the holiday window has only one earlier holiday season to learn from.");

// 3 Architecture
sec("Method");
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
tx(s, "One pipeline, five stages, every stage past-only", { placeholder: "title" });
s.addImage({ path: "../docs/architecture.png", x: 0.35, y: 1.15, w: 9.3, h: 1.95, altText: "Architecture diagram of the forecasting pipeline" });
tx(s, bullets(["Direct 1-13 week forecasts from each origin; lag 52 aligned to the target week", "Dollar target: divided by a trailing 13-week level inside the model, multiplied back for output",
  "Original notebook not reused: random split, whole-dataset series statistics as features and target scaling were removed (docs/notebook_review.md)"], 13), { x: 0.45, y: 3.3, w: 9.1, h: 1.8, valign: "top" });
s.addNotes("Walk the diagram left to right. Stress the leakage control: a unit test destroys all sales after the forecast origin and asserts the features do not change.");

// 4 Evaluation design
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
tx(s, "Evaluation: chronological folds, test window touched once", { placeholder: "title" });
const rows = [["", "Training targets", "Validation (13 wk)", "Test (13 wk)"],
  ["holiday fold", "before 2011-07-29", "2011-07-29 to 2011-10-21", "2011-10-28 to 2012-01-20"],
  ["latest fold", "before 2012-05-04", "2012-05-04 to 2012-07-27", "2012-08-03 to 2012-10-26"]];
s.addTable(rows.map((r, i) => r.map((c) => ({ text: c, options: { bold: i === 0, color: i === 0 ? "FFFFFF" : HEX.ink, fill: { color: i === 0 ? HEX.teal : HEX.light }, fontSize: 12, align: "center", valign: "middle" } }))),
  { x: 0.45, y: 1.25, w: 9.1, colW: [1.6, 2.2, 2.65, 2.65], rowH: 0.45, border: { type: "solid", color: "DDE3EA", pt: 1 }, objectName: "fold table" });
tx(s, bullets(["Validation window: early stopping, interval calibration, method selection. Refit on all data before the test window", "Metrics: WAPE (primary), RMSSE, MAE, RMSE, signed bias, interval coverage. R-squared is never computed",
  "Baselines on every level: naive and seasonal naive; Prophet and a Temporal Fusion Transformer on 100 representative series"], 13), { x: 0.45, y: 2.9, w: 9.1, h: 2.2, valign: "top" });
s.addNotes("Two windows on purpose: the latest 13 weeks, and the Thanksgiving to Christmas season where forecasting is hardest.");

// 5 Results latest
sec("Results");
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
tx(s, "Latest window: LightGBM beats every baseline", { placeholder: "title" });
const Ls = D.latest.store_dept;
const order1 = ["naive", "snaive", "lgbm"];
bar(s, order1.map(L), order1.map((m) => Ls[m]), { x: 0.45, y: 1.15, w: 5.2, h: 3.9, valAxisMaxVal: 18 }, [HEX.grey, HEX.amber, HEX.teal]);
tx(s, "Store x Department WAPE %, 13 weeks ahead", { x: 0.45, y: 5.0, w: 5.2, h: 0.3, fontSize: 10, color: HEX.grey });
tx(s, bullets([`LightGBM ${Ls.lgbm}% vs seasonal naive ${Ls.snaive}% and naive ${Ls.naive}%`, `On 100 identical series: LightGBM ${D.rep.latest.lgbm}%, Prophet ${D.rep.latest.prophet}%, TFT ${D.rep.latest.tft}%, seasonal naive ${D.rep.latest.snaive}%`,
  "TFT ran on one CPU core, 8 epochs, no tuning: a baseline, not its best case"], 13), { x: 5.9, y: 1.3, w: 3.7, h: 3.6, valign: "top" });
s.addNotes("Headline result. Be explicit that Prophet and TFT were deliberately small and are scored on the same 100 series as LightGBM, so the comparison is fair but not a verdict on the best achievable TFT.");

// 6 Holiday honesty
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
tx(s, "Holiday window: seasonal naive is hard to beat", { placeholder: "title" });
const Hs = D.holiday.store_dept, Ht = D.holiday.total;
const o2 = ["snaive", "lgbm"];
bar(s, ["Seasonal naive", "LightGBM"], [Hs.snaive, Hs.lgbm], { x: 0.45, y: 1.15, w: 3.0, h: 3.7, valAxisMaxVal: 16, barGapWidthPct: 60 }, [HEX.amber, HEX.teal]);
bar(s, ["Seasonal naive", "LGBM bottom-up"], [Ht.snaive, Ht.lgbm_bottom_up], { x: 3.5, y: 1.15, w: 3.0, h: 3.7, valAxisMaxVal: 16, barGapWidthPct: 60 }, [HEX.amber, HEX.teal]);
tx(s, "Store x Dept WAPE %", { x: 0.45, y: 4.85, w: 3, h: 0.3, fontSize: 10, color: HEX.grey, align: "center" });
tx(s, "Total WAPE %", { x: 3.5, y: 4.85, w: 3, h: 0.3, fontSize: 10, color: HEX.grey, align: "center" });
tx(s, bullets(["Only one earlier holiday season in training, and no lag-52 for it", "So series-specific holiday lifts cannot be learned; last year's same week is a strong rule", `Prophet ${D.rep.holiday.prophet}% and TFT ${D.rep.holiday.tft}% were worse than seasonal naive ${D.rep.holiday.snaive}% on the subset`], 13), { x: 6.7, y: 1.3, w: 2.9, h: 3.6, valign: "top" });
s.addNotes("This is the slide that builds trust. We report that the learned model does not clearly beat seasonal naive in the holiday window, and explain why. With another year of data the picture should change; we have not tested that.");

// 7 Hierarchy
sec("Hierarchy and uncertainty");
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Hierarchy and uncertainty" });
tx(s, "Reconciliation sharpens aggregates; MinT was unstable", { placeholder: "title" });
const lv = ["total", "group", "store", "dept", "store_dept"], lvl = ["Total", "Type group", "Store", "Dept", "Store x Dept"];
s.addChart(pres.charts.BAR, ["snaive", "lgbm", "lgbm_bottom_up"].map((m) => ({ name: L(m), labels: lvl, values: lv.map((k) => D.latest[k][m]) })),
  Object.assign({}, chartCommon, { x: 0.45, y: 1.15, w: 5.8, h: 3.9, barDir: "col", chartColors: [HEX.amber, HEX.grey, HEX.teal], showLegend: true, legendPos: "b", legendFontFace: "+mn-lt", legendFontSize: 11, showValue: false }));
tx(s, bullets([`Bottom-up cuts Dept WAPE from ${D.latest.dept.lgbm}% to ${D.latest.dept.lgbm_bottom_up}% and total from ${D.latest.total.lgbm}% to ${D.latest.total.lgbm_bottom_up}% (latest window)`,
  `MinT won on validation but reached ${D.holiday.store_dept.lgbm_mint_shrink}% store-dept WAPE in the holiday window (covariance from 28 samples, 3,461 nodes)`, "Default switched to bottom-up after seeing this; disclosed in the docs"], 13), { x: 6.4, y: 1.3, w: 3.2, h: 3.7, valign: "top" });
s.addNotes("Reconciliation makes the forecasts add up and helps the aggregates. The covariance-based MinT overfit its tiny validation sample, so the operational default is bottom-up. We disclose that this default was chosen after observing test results.");

// 8 Intervals
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Hierarchy and uncertainty" });
tx(s, "Prediction intervals need calibration", { placeholder: "title" });
const cl = D.latest.cov, ch = D.holiday.cov;
s.addChart(pres.charts.BAR, [{ name: "Latest", labels: ["Raw quantile model", "Conformal (CQR)"], values: [cl["lgbm|raw_quantile"], cl["lgbm|calibrated"]] },
  { name: "Holiday", labels: ["Raw quantile model", "Conformal (CQR)"], values: [ch["lgbm|raw_quantile"], ch["lgbm|calibrated"]] }],
  Object.assign({}, chartCommon, { x: 0.45, y: 1.15, w: 5.4, h: 3.9, barDir: "col", chartColors: [HEX.teal, HEX.amber], showLegend: true, legendPos: "b", legendFontFace: "+mn-lt", valAxisMinVal: 50, valAxisMaxVal: 90 }));
tx(s, "Empirical coverage % of the nominal 80% interval, Store x Dept", { x: 0.45, y: 5.0, w: 5.4, h: 0.3, fontSize: 10, color: HEX.grey });
tx(s, bullets(["Raw q10/q90 models under-cover (about 71-72%)", "Conformalised quantile regression, calibrated on the validation window, lands at about 79%", `Reconciled bottom-up intervals: ${cl["lgbm_bottom_up|calibrated"]}% (latest), ${ch["lgbm_bottom_up|calibrated"]}% (holiday): still short in the holiday regime`], 13), { x: 6.1, y: 1.3, w: 3.5, h: 3.7, valign: "top" });
s.addNotes("Coverage is the honest test of an interval. The raw quantile models are too narrow; conformal calibration fixes that for the base model. Reconciled intervals are wider and in the holiday window still under-cover.");

// 9 Inventory
sec("Decisions");
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Decisions" });
tx(s, "Inventory optimiser: a decision tool driven by assumptions", { placeholder: "title" });
const inv = Object.fromEntries(D.inv.map((r) => [r.policy, r]));
const fmt = (v) => "$" + (v / 1e6).toFixed(2) + "M";
const trows = [["Policy", "Orders", "Modelled total cost", "Fill rate"],
  ["Optimised plan (OR-Tools SCIP)", String(inv.optimised_plan.n_orders), fmt(inv.optimised_plan.total), (inv.optimised_plan.fill_rate * 100).toFixed(1) + "%"],
  ["Weekly base-stock reference", String(inv.base_stock_reference.n_orders), fmt(inv.base_stock_reference.total), (inv.base_stock_reference.fill_rate * 100).toFixed(1) + "%"],
  ["No reordering", "0", fmt(inv.no_reorder.total), (inv.no_reorder.fill_rate * 100).toFixed(1) + "%"]];
s.addTable(trows.map((r, i) => r.map((c, j) => ({ text: c, options: { bold: i === 0, color: i === 0 ? "FFFFFF" : HEX.ink, fill: { color: i === 0 ? HEX.teal : (i === 1 ? "E3F1EA" : HEX.light) }, fontSize: 12, align: j === 0 ? "left" : "center", valign: "middle" } }))),
  { x: 0.45, y: 1.2, w: 9.1, colW: [3.9, 1.3, 2.2, 1.7], rowH: 0.42, border: { type: "solid", color: "DDE3EA", pt: 1 }, objectName: "inventory table" });
tx(s, bullets(["Per-store MILP over 13 weeks: p10/p50/p90 demand scenarios, lead time, fixed ordering cost, holding cost, stockout penalty, storage capacity", "Fewer orders and lower modelled cost, at a slightly lower fill rate",
  "Every cost, lead time, capacity and opening stock is an assumption in config/default.yaml; the data has none. Sales stand in for demand"], 13), { x: 0.45, y: 3.1, w: 9.1, h: 2.0, valign: "top" });
s.addNotes("Be clear this is a demonstration under stated assumptions: the Walmart files contain no unit costs, stock, suppliers or capacities. The saving comes mostly from fewer orders under an assumed fixed ordering cost.");

// 10 Dashboard + takeaways
s = pres.addSlide({ masterName: "DARK", sectionTitle: "Decisions" });
tx(s, "Takeaways", { x: 0.7, y: 0.5, w: 8.6, h: 0.7, fontSize: 30, bold: true, color: "FFFFFF", fontFace: "Georgia" });
tx(s, bullets(["A global LightGBM beats seasonal naive by about 2 points of WAPE in the latest window, and is on par with it in the holiday season",
  "Bottom-up reconciliation gives coherent forecasts and sharper aggregates; covariance-based MinT needs more validation data",
  "Conformal calibration is needed for honest 80% intervals",
  "Dashboard: weekly forecasts, accuracy, hierarchy and inventory tabs (python -m walmartfc.dashboard.app)",
  "Next: add a year of history, a true forward forecast with future calendar inputs, real state mapping and real cost data"], 16).map((b) => { b.options.color = "FFFFFF"; return b; }), { x: 0.7, y: 1.4, w: 8.6, h: 3.6, valign: "top" });
s.addNotes("Close on what is solid and what is not. Invite questions on the holiday window and the inventory assumptions.");

(async () => {
  await pres.writeFile({ fileName: "walmart_forecasting_10min.pptx" });
  await applyTheme("walmart_forecasting_10min.pptx", THEME);
  console.log("ok");
})();
