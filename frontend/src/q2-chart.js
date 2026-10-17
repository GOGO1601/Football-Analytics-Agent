/**
 * q2-chart.js — Q2: Offensive Performance vs League Success
 *
 * Features:
 *  - Dark pitch-inspired scatterplot
 *  - X-axis: GoalsFor | AvgShotsOnTarget | AvgShots
 *  - Y-axis: Points
 *  - Colour by league (5-colour map)
 *  - Hover tooltip with football ⚽ icon, team, season, exact values
 *  - D3 brush → filtered team table
 *  - League filter + season range
 *  - Linear regression trend line with R² annotation
 */

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let allTeamSeasons      = [];
let currentXMetric      = "GoalsFor";
let brushSelection      = null;

// ---------------------------------------------------------------------------
// Colour map — one per league
// ---------------------------------------------------------------------------
const LEAGUE_COLOUR_MAP = {
    "Premier League": ThemeHelper.pick("#4da6ff","#2563a8"),
    "La Liga":        ThemeHelper.pick("#ff6b6b","#c0392b"),
    "Bundesliga":     ThemeHelper.pick("#3ef07a","#27805a"),
    "Serie A":        ThemeHelper.pick("#ffcc00","#b87a20"),
    "Ligue 1":        ThemeHelper.pick("#c084fc","#7b52c4"),
};

// ---------------------------------------------------------------------------
// Label maps
// ---------------------------------------------------------------------------
const X_AXIS_LABELS = {
    GoalsFor:         "Goals Scored (season total)",
    AvgShotsOnTarget: "Avg Shots on Target / Game",
    AvgShots:         "Avg Shots / Game",
};

const CHART_TITLE_MAP = {
    GoalsFor:         "Goals Scored vs Points",
    AvgShotsOnTarget: "Avg Shots on Target vs Points",
    AvgShots:         "Avg Shots vs Points",
};

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------
const xAxisSelect    = document.getElementById("q2XAxisSelect");
const leagueFilter   = document.getElementById("q2LeagueFilter");
const seasonStartSel = document.getElementById("q2SeasonStart");
const seasonEndSel   = document.getElementById("q2SeasonEnd");
const resetBtn       = document.getElementById("q2ResetButton");
const chartContainer = document.getElementById("q2Chart");
const loadingMsg     = document.getElementById("q2LoadingMsg");
const brushPanel     = document.getElementById("q2BrushPanel");
const brushTable     = document.getElementById("q2BrushTable");
const clearBrushBtn  = document.getElementById("q2ClearBrush");
const tooltip        = d3.select("#tooltip");

const summaryCount   = document.getElementById("q2SummaryCount");
const summarySeasons = document.getElementById("q2SummarySeasons");
const summaryMissing = document.getElementById("q2SummaryMissing");
const summaryBrush   = document.getElementById("q2SummaryBrush");

// ---------------------------------------------------------------------------
// Season dropdowns
// ---------------------------------------------------------------------------
function populateSeasonDropdowns(teamSeasons) {
    const seasons = [...new Set(teamSeasons.map(d => d.Season))]
        .sort((a, b) => getSeasonSortKey(a) - getSeasonSortKey(b));

    [seasonStartSel, seasonEndSel].forEach(sel => {
        sel.innerHTML = "";
        seasons.forEach(s => {
            const opt = document.createElement("option");
            opt.value = s; opt.textContent = s;
            sel.appendChild(opt);
        });
    });

    seasonStartSel.value = seasons[0];
    seasonEndSel.value   = seasons[seasons.length - 1];
}

// ---------------------------------------------------------------------------
// Data filtering
// ---------------------------------------------------------------------------
function getFilteredData() {
    const xMetric     = xAxisSelect.value;
    const league      = leagueFilter.value;
    const startKey    = getSeasonSortKey(seasonStartSel.value);
    const endKey      = getSeasonSortKey(seasonEndSel.value);

    return allTeamSeasons.filter(d => {
        if (d[xMetric] == null || d.Points == null) return false;
        if (league !== "all" && d.League !== league) return false;
        const sk = getSeasonSortKey(d.Season);
        return sk >= startKey && sk <= endKey;
    });
}

// ---------------------------------------------------------------------------
// Summary cards
// ---------------------------------------------------------------------------
function updateSummaryCards(filtered) {
    const missingCount = allTeamSeasons.filter(d => d[currentXMetric] == null).length;
    summaryCount.textContent   = filtered.length.toLocaleString();
    summarySeasons.textContent = new Set(filtered.map(d => d.Season)).size;
    summaryMissing.textContent = missingCount.toLocaleString();
    summaryBrush.textContent   = brushSelection ? "Active" : "All";
}

// ---------------------------------------------------------------------------
// Linear regression helper
// ---------------------------------------------------------------------------
function linearRegression(data, xKey, yKey) {
    const n  = data.length;
    if (n < 2) return null;
    const mx = d3.mean(data, d => d[xKey]);
    const my = d3.mean(data, d => d[yKey]);
    const ssxx = d3.sum(data, d => (d[xKey] - mx) ** 2);
    const ssxy = d3.sum(data, d => (d[xKey] - mx) * (d[yKey] - my));
    if (ssxx === 0) return null;
    const slope     = ssxy / ssxx;
    const intercept = my - slope * mx;

    // R²
    const ssTot = d3.sum(data, d => (d[yKey] - my) ** 2);
    const ssRes = d3.sum(data, d => (d[yKey] - (slope * d[xKey] + intercept)) ** 2);
    const r2    = ssTot === 0 ? 0 : 1 - ssRes / ssTot;

    return { slope, intercept, r2 };
}

// ---------------------------------------------------------------------------
// Brush table
// ---------------------------------------------------------------------------
function renderBrushTable(selectedPoints) {
    if (!selectedPoints || !selectedPoints.length) {
        brushPanel.style.display = "none";
        return;
    }

    brushPanel.style.display = "block";
    summaryBrush.textContent = `${selectedPoints.length} teams`;

    const xLabel = X_AXIS_LABELS[currentXMetric];
    const sorted = [...selectedPoints].sort((a, b) => b.Points - a.Points);

    brushTable.innerHTML = `
        <table class="brush-data-table">
            <thead>
                <tr>
                    <th>Team</th><th>League</th><th>Season</th>
                    <th>${xLabel}</th><th>Points</th>
                </tr>
            </thead>
            <tbody>
                ${sorted.slice(0, 50).map(d => `
                    <tr>
                        <td><strong>${d.Team}</strong></td>
                        <td><span class="league-badge" style="background:${LEAGUE_COLOUR_MAP[d.League]}22;color:${LEAGUE_COLOUR_MAP[d.League]}">${d.League}</span></td>
                        <td>${d.Season}</td>
                        <td>${d[currentXMetric].toFixed(2)}</td>
                        <td><strong>${d.Points}</strong></td>
                    </tr>`).join("")}
                ${sorted.length > 50 ? `<tr><td colspan="5" class="table-overflow">…and ${sorted.length - 50} more</td></tr>` : ""}
            </tbody>
        </table>`;
}

// ---------------------------------------------------------------------------
// Legend
// ---------------------------------------------------------------------------
function renderLegend() {
    const legendEl = document.getElementById("q2Legend");
    legendEl.innerHTML = "";
    Object.entries(LEAGUE_COLOUR_MAP).forEach(([league, colour]) => {
        const item = document.createElement("div");
        item.className = "legend-item";
        item.style.cursor = "pointer";
        item.innerHTML = `
            <div class="legend-swatch" style="background:${colour};width:10px;height:10px;border-radius:50%"></div>
            <span>${league}</span>`;
        item.addEventListener("click", () => {
            leagueFilter.value = league;
            drawScatterplot();
        });
        legendEl.appendChild(item);
    });
}

// ---------------------------------------------------------------------------
// Main chart draw
// ---------------------------------------------------------------------------
function drawScatterplot() {
    currentXMetric = xAxisSelect.value;
    const filtered = getFilteredData();

    d3.select("#q2Chart").html("");
    brushSelection = null;
    summaryBrush.textContent = "All";
    brushPanel.style.display = "none";

    updateSummaryCards(filtered);
    document.getElementById("q2ChartTitle").textContent = CHART_TITLE_MAP[currentXMetric];

    if (filtered.length === 0) {
        d3.select("#q2Chart").append("div").attr("class", "empty-message")
            .text("No data for the current filters. Try switching the X-axis or widening the season range.");
        return;
    }

    // ---- Dimensions ----
    const margin     = { top: 36, right: 180, bottom: 70, left: 70 };
    const fullWidth  = 1140;
    const fullHeight = 580;
    const W = fullWidth  - margin.left - margin.right;
    const H = fullHeight - margin.top  - margin.bottom;

    const svg = d3.select("#q2Chart")
        .append("svg")
        .attr("viewBox", `0 0 ${fullWidth} ${fullHeight}`)
        .append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

    // Pitch-style background
    svg.append("rect")
        .attr("x", 0).attr("y", 0)
        .attr("width", W).attr("height", H)
        .attr("fill", ThemeHelper.pick("rgba(62,240,122,0.015)","rgba(30,60,100,0.025)"))
        .attr("rx", 8);

    // ---- Scales ----
    const xExtent = d3.extent(filtered, d => d[currentXMetric]);
    const yExtent = d3.extent(filtered, d => d.Points);
    const xPad = (xExtent[1] - xExtent[0]) * 0.04;
    const yPad = (yExtent[1] - yExtent[0]) * 0.04;

    const x = d3.scaleLinear().domain([xExtent[0] - xPad, xExtent[1] + xPad]).range([0, W]);
    const y = d3.scaleLinear().domain([yExtent[0] - yPad, yExtent[1] + yPad]).nice().range([H, 0]);

    // ---- Grid lines ----
    svg.append("g").attr("class", "grid")
        .call(d3.axisLeft(y).tickSize(-W).tickFormat("").ticks(8))
        .call(g => g.select(".domain").remove())
        .call(g => g.selectAll("line")
            .style("stroke",ThemeHelper.pick("rgba(255,255,255,0.055)","rgba(30,60,100,0.10)"))
            .style("stroke-dasharray","4,5"));

    svg.append("g").attr("class", "grid")
        .attr("transform", `translate(0,${H})`)
        .call(d3.axisBottom(x).tickSize(-H).tickFormat("").ticks(8))
        .call(g => g.select(".domain").remove())
        .call(g => g.selectAll("line")
            .style("stroke",ThemeHelper.pick("rgba(255,255,255,0.04)","rgba(30,60,100,0.07)"))
            .style("stroke-dasharray","4,5"));

    // ---- Axes ----
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H})`)
        .call(d3.axisBottom(x).ticks(8))
        .call(g => g.select(".domain").style("stroke",ThemeHelper.pick("rgba(255,255,255,0.1)","rgba(30,60,100,0.15)")))
        .selectAll("text").style("fill",ThemeHelper.pick("rgba(255,255,255,0.35)","#1a2a3a")).style("font-size","10.5px");

    svg.append("g").attr("class", "axis")
        .call(d3.axisLeft(y).ticks(8))
        .call(g => g.select(".domain").style("stroke",ThemeHelper.pick("rgba(255,255,255,0.1)","rgba(30,60,100,0.15)")))
        .selectAll("text").style("fill",ThemeHelper.pick("rgba(255,255,255,0.35)","#1a2a3a")).style("font-size","10.5px");

    // Axis labels
    svg.append("text")
        .attr("x", W / 2).attr("y", H + 56)
        .attr("text-anchor", "middle")
        .style("font-size","12px").style("fill",ThemeHelper.pick("rgba(255,255,255,0.35)","#1a2a3a"))
        .text(X_AXIS_LABELS[currentXMetric]);

    svg.append("text")
        .attr("transform","rotate(-90)")
        .attr("x", -H / 2).attr("y", -52)
        .attr("text-anchor","middle")
        .style("font-size","12px").style("fill",ThemeHelper.pick("rgba(255,255,255,0.35)","#1a2a3a"))
        .text("Points");

    // ---- Regression line ----
    const reg = linearRegression(filtered, currentXMetric, "Points");
    if (reg) {
        const x0 = xExtent[0] - xPad;
        const x1 = xExtent[1] + xPad;
        const y0 = reg.slope * x0 + reg.intercept;
        const y1 = reg.slope * x1 + reg.intercept;

        svg.append("line")
            .attr("class","regression-line")
            .attr("x1", x(x0)).attr("y1", y(y0))
            .attr("x2", x(x1)).attr("y2", y(y1))
            .style("stroke",ThemeHelper.pick("rgba(255,255,255,0.22)","rgba(30,60,100,0.25)"))
            .style("stroke-width","1.5")
            .style("stroke-dasharray","7,5");

        // R² annotation
        svg.append("text")
            .attr("x", W - 6).attr("y", 14)
            .attr("text-anchor","end")
            .style("font-size","10.5px")
            .style("fill",ThemeHelper.pick("rgba(255,255,255,0.35)","#1a2a3a"))
            .text(`R² = ${reg.r2.toFixed(3)}`);
    }

    // ---- Dots ----
    const dotsGroup = svg.append("g").attr("class","dots-group");

    const dots = dotsGroup.selectAll("circle.scatter-dot")
        .data(filtered, d => `${d.League}-${d.Team}-${d.Season}`)
        .enter()
        .append("circle")
        .attr("class","scatter-dot")
        .attr("cx", d => x(d[currentXMetric]))
        .attr("cy", d => y(d.Points))
        .attr("r", 5.5)
        .attr("fill", d => LEAGUE_COLOUR_MAP[d.League] || "#888")
        .attr("fill-opacity", 0.55)
        .attr("stroke", d => LEAGUE_COLOUR_MAP[d.League] || "#888")
        .attr("stroke-width", 0.8)
        .attr("stroke-opacity", 0.8)
        .on("mousemove", function (event, d) {
            d3.select(this).raise()
                .attr("fill-opacity", 1)
                .attr("r", 8)
                .attr("filter", `drop-shadow(0 0 6px ${LEAGUE_COLOUR_MAP[d.League]})`);
            tooltip
                .style("opacity", 1)
                .html(
                    `<strong>⚽ ${d.Team}</strong><br>` +
                    `<span style="color:"+ThemeHelper.pick("rgba(255,255,255,0.4)","#2b3a52")+"">${d.League} · ${d.Season}</span><br>` +
                    `<span style="color:"+ThemeHelper.pick("rgba(255,255,255,0.4)","#2b3a52")+"">${X_AXIS_LABELS[currentXMetric]}:</span> ` +
                    `<strong style="color:var(--accent)">${d[currentXMetric].toFixed(2)}</strong><br>` +
                    `<span style="color:"+ThemeHelper.pick("rgba(255,255,255,0.4)","#2b3a52")+"">Points:</span> <strong>${d.Points}</strong>`
                )
                .style("left", `${event.pageX + 14}px`)
                .style("top",  `${event.pageY - 28}px`);
        })
        .on("mouseleave", function (event, d) {
            d3.select(this)
                .attr("fill-opacity", 0.55)
                .attr("r", 5.5)
                .attr("filter", null);
            tooltip.style("opacity", 0);
        });

    // ---- Brush ----
    const brush = d3.brush()
        .extent([[0, 0], [W, H]])
        .on("start", () => {
            dotsGroup.selectAll(".scatter-dot").attr("fill-opacity", 0.12);
        })
        .on("brush", function (event) {
            if (!event.selection) return;
            const [[x0, y0], [x1, y1]] = event.selection;
            dotsGroup.selectAll(".scatter-dot")
                .attr("fill-opacity", d => {
                    const cx = x(d[currentXMetric]);
                    const cy = y(d.Points);
                    return (cx >= x0 && cx <= x1 && cy >= y0 && cy <= y1) ? 1 : 0.08;
                })
                .attr("r", d => {
                    const cx = x(d[currentXMetric]);
                    const cy = y(d.Points);
                    return (cx >= x0 && cx <= x1 && cy >= y0 && cy <= y1) ? 7 : 4;
                });
        })
        .on("end", function (event) {
            if (!event.selection) {
                dotsGroup.selectAll(".scatter-dot").attr("fill-opacity", 0.55).attr("r", 5.5);
                brushPanel.style.display = "none";
                brushSelection = null;
                summaryBrush.textContent = "All";
                return;
            }
            const [[x0, y0], [x1, y1]] = event.selection;
            brushSelection = { x0, y0, x1, y1 };
            const selected = filtered.filter(d => {
                const cx = x(d[currentXMetric]);
                const cy = y(d.Points);
                return cx >= x0 && cx <= x1 && cy >= y0 && cy <= y1;
            });
            renderBrushTable(selected);
            summaryBrush.textContent = `${selected.length} selected`;
        });

    const brushGroup = svg.append("g").attr("class","brush").call(brush);
    brushGroup.select(".selection")
        .attr("fill",ThemeHelper.pick("rgba(62,240,122,0.08)","rgba(26,111,168,0.10)"))
        .attr("stroke",ThemeHelper.pick("#3ef07a","#1a6fa8"))
        .attr("stroke-width","1");

    // ---- Legend (inside SVG, right side) ----
    const leagues     = Object.keys(LEAGUE_COLOUR_MAP);
    const legendX     = W + 24;
    const legendGroup = svg.append("g").attr("class","q2-legend");

    legendGroup.append("text")
        .attr("x", legendX).attr("y", 12)
        .style("font-size","10px").style("font-weight","700")
        .style("text-transform","uppercase").style("letter-spacing","0.08em")
        .style("fill",ThemeHelper.pick("rgba(255,255,255,0.35)","#1a2a3a"))
        .text("League");

    leagues.forEach((league, i) => {
        const itemY = 28 + i * 24;

        legendGroup.append("circle")
            .attr("cx", legendX + 6).attr("cy", itemY)
            .attr("r", 5.5)
            .attr("fill", LEAGUE_COLOUR_MAP[league])
            .attr("fill-opacity", 0.75);

        legendGroup.append("text")
            .attr("x", legendX + 18).attr("y", itemY + 4)
            .style("font-size","11.5px").style("fill",ThemeHelper.pick("rgba(255,255,255,0.55)","#0f1a2e"))
            .style("cursor","pointer")
            .text(league)
            .on("click", () => { leagueFilter.value = league; drawScatterplot(); });
    });

    legendGroup.append("text")
        .attr("x", legendX).attr("y", 28 + leagues.length * 24 + 14)
        .style("font-size","10px").style("fill",ThemeHelper.pick("rgba(255,255,255,0.22)","#4a5568")).style("cursor","pointer")
        .text("click to filter")
        .on("click", () => { leagueFilter.value = "all"; drawScatterplot(); });

    // Regression label in legend
    if (reg) {
        legendGroup.append("line")
            .attr("x1", legendX).attr("x2", legendX + 22)
            .attr("y1", 28 + leagues.length * 24 + 44)
            .attr("y2", 28 + leagues.length * 24 + 44)
            .style("stroke",ThemeHelper.pick("rgba(255,255,255,0.3)","rgba(30,60,100,0.30)")).style("stroke-width","1.5")
            .style("stroke-dasharray","5,4");
        legendGroup.append("text")
            .attr("x", legendX + 26)
            .attr("y", 28 + leagues.length * 24 + 48)
            .style("font-size","10.5px").style("fill",ThemeHelper.pick("rgba(255,255,255,0.35)","#1a2a3a"))
            .text("Trend line");
    }
}

// ---------------------------------------------------------------------------
// Event listeners
// ---------------------------------------------------------------------------
xAxisSelect.addEventListener("change",    drawScatterplot);
leagueFilter.addEventListener("change",   drawScatterplot);
seasonStartSel.addEventListener("change", drawScatterplot);
seasonEndSel.addEventListener("change",   drawScatterplot);

resetBtn.addEventListener("click", () => {
    xAxisSelect.value    = "GoalsFor";
    leagueFilter.value   = "all";
    const seasons = Array.from(seasonStartSel.options).map(o => o.value);
    seasonStartSel.value = seasons[0];
    seasonEndSel.value   = seasons[seasons.length - 1];
    brushSelection       = null;
    drawScatterplot();
});

clearBrushBtn.addEventListener("click", () => { drawScatterplot(); });

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
(async function init() {
    const updateLoading = (done, total) => {
        const pct = Math.round((done / total) * 100);
        loadingMsg.textContent = `Loading data… ${pct}%`;
    };

    allTeamSeasons = await loadAllLeagues(updateLoading);
    loadingMsg.style.display    = "none";
    chartContainer.style.display = "block";

    populateSeasonDropdowns(allTeamSeasons);
    renderLegend();
    drawScatterplot();

    // Re-render when theme is toggled
    window.__themeRedraw = drawScatterplot;
})();
