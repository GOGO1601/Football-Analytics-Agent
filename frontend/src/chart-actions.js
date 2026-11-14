/**
 * chart-actions.js
 * Applies agent-returned ChartAction objects to the existing D3 pages
 * WITHOUT modifying main.js / q2-chart.js / q3-chart.js.
 *
 * Works by setting the pages' existing controls and dispatching 'change':
 *   Q1: #leagueSelect, #metricSelect, team multi-select
 *   Q2: #q2LeagueFilter, x-metric dropdown
 *   Q3: league filter, x-metric dropdown, colour mode
 * Cross-page actions navigate with query params (e.g. q3-disciplinary.html?league=Serie%20A)
 * and are applied on load.
 *
 * Example action: { page: "q1", league: "Premier League", metric: "Points", teams: ["Arsenal","Tottenham"] }
 */
window.ChartActions = { apply(actions) { /* TODO */ } };
