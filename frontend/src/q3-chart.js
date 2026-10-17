/**
 * q3-chart.js — Q3: Disciplinary Impact on Match Outcomes
 *
 * Features:
 *  - Bubble chart: X = AvgFouls or AvgYellowCards, Y = Points
 *  - Bubble SIZE = AvgRedCards per game (teams with more red cards = bigger bubbles)
 *  - Bubble COLOUR = League OR sequential scale (yellow/red card intensity)
 *  - Only uses team-seasons where disciplinary data is non-null
 *  - Animated card-flip icons on hover tooltip (🟨 / 🟥)
 *  - League filter dropdown
 *  - Linear regression trend line with R²
 *  - "Danger zone" annotation for high-foul, low-points quadrant
 */

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let allTeamSeasons   = [];
let currentXMetric   = "AvgFouls";
let currentColorMode = "league";

// ---------------------------------------------------------------------------
// League colours
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
const X_LABEL_MAP = {
    AvgFouls:      "Avg Fouls per Game",
    AvgYellowCards:"Avg Yellow Cards per Game",
};

const CHART_TITLE_MAP = {
    AvgFouls:      "Avg Fouls vs Points — Bubble Size = Red Cards",
    AvgYellowCards:"Avg Yellow Cards vs Points — Bubble Size = Red Cards",
};

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------
const xMetricSelect = document.getElementById("q3XMetric");
const leagueFilter  = document.getElementById("q3LeagueFilter");
const colorModeEl   = document.getElementById("q3ColorMode");
const resetBtn      = document.getElementById("q3ResetButton");
const chartDiv      = document.getElementById("q3Chart");
const loadingMsg    = document.getElementById("q3LoadingMsg");
const tooltip       = d3.select("#tooltip");

const summaryCount   = document.getElementById("q3SummaryCount");
const summarySeasons = document.getElementById("q3SummarySeasons");
const summaryFouls   = document.getElementById("q3SummaryFouls");
const summaryYellow  = document.getElementById("q3SummaryYellow");

// ---------------------------------------------------------------------------
// Data filtering — only rows where ALL disciplinary fields are non-null
// ---------------------------------------------------------------------------
function getFilteredData() {
    const xMetric = xMetricSelect.value;
    const league  = leagueFilter.value;

    return allTeamSeasons.filter(d => {
        // Require fouls, yellow cards, red cards to all be present
        if (d.AvgFouls == null || d.AvgYellowCards == null || d.AvgRedCards == null) return false;
        if (d.Points == null) return false;
        if (d[xMetric] == null) return false;
        if (league !== "all" && d.League !== league) return false;
        return true;
    });
}

// ---------------------------------------------------------------------------
// Summary cards
// ---------------------------------------------------------------------------
function updateSummaryCards(filtered) {
    const seasons = new Set(filtered.map(d => d.Season));
    summaryCount.textContent   = filtered.length.toLocaleString();
    summarySeasons.textContent = seasons.size;
    summaryFouls.textContent   = filtered.length
        ? d3.mean(filtered, d => d.AvgFouls).toFixed(1)
        : "—";
    summaryYellow.textContent  = filtered.length
        ? d3.mean(filtered, d => d.AvgYellowCards).toFixed(2)
        : "—";
}

// ---------------------------------------------------------------------------
// Linear regression
// ---------------------------------------------------------------------------
function linearRegression(data, xKey, yKey) {
    const n = data.length;
    if (n < 2) return null;
    const mx = d3.mean(data, d => d[xKey]);
    const my = d3.mean(data, d => d[yKey]);
    const ssxx = d3.sum(data, d => (d[xKey] - mx) ** 2);
    const ssxy = d3.sum(data, d => (d[xKey] - mx) * (d[yKey] - my));
    if (ssxx === 0) return null;
    const slope     = ssxy / ssxx;
    const intercept = my - slope * mx;
    const ssTot = d3.sum(data, d => (d[yKey] - my) ** 2);
    const ssRes = d3.sum(data, d => (d[yKey] - (slope * d[xKey] + intercept)) ** 2);
    const r2    = ssTot === 0 ? 0 : 1 - ssRes / ssTot;
    return { slope, intercept, r2 };
}

// ---------------------------------------------------------------------------
// Bubble colour resolver
// ---------------------------------------------------------------------------
function makeBubbleColorFn(filtered, colorMode) {
    if (colorMode === "league") {
        return d => LEAGUE_COLOUR_MAP[d.League] || "#888";
    }
    if (colorMode === "yellowCards") {
        const ext = d3.extent(filtered, d => d.AvgYellowCards);
        const scale = d3.scaleSequential(d3.interpolateYlOrBr).domain(ext);
        return d => scale(d.AvgYellowCards);
    }
    if (colorMode === "redCards") {
        const ext = d3.extent(filtered, d => d.AvgRedCards);
        const scale = d3.scaleSequential(d3.interpolateOrRd).domain(ext);
        return d => scale(d.AvgRedCards);
    }
    return () => "#888";
}

// ---------------------------------------------------------------------------
// Legend
// ---------------------------------------------------------------------------
function renderLegend(filtered, colorMode) {
    const legendEl = document.getElementById("q3Legend");
    legendEl.innerHTML = "";

    if (colorMode === "league") {
        Object.entries(LEAGUE_COLOUR_MAP).forEach(([league, colour]) => {
            const item = document.createElement("div");
            item.className = "legend-item";
            item.style.cursor = "pointer";
            item.innerHTML = `
                <div class="legend-swatch" style="background:${colour};width:10px;height:10px;border-radius:50%"></div>
                <span>${league}</span>`;
            item.addEventListener("click", () => {
                leagueFilter.value = league;
                drawBubbleChart();
            });
            legendEl.appendChild(item);
        });

        // "All" reset
        const allItem = document.createElement("div");
        allItem.className = "legend-item";
        allItem.style.cursor = "pointer";
        allItem.style.opacity = "0.45";
        allItem.innerHTML = `<span style="font-size:11px">Click league to filter</span>`;
        allItem.addEventListener("click", () => { leagueFilter.value = "all"; drawBubbleChart(); });
        legendEl.appendChild(allItem);
    } else {
        // Gradient swatch for sequential scales
        const label = colorMode === "yellowCards" ? "🟨 Yellow Cards" : "🟥 Red Cards";
        const item = document.createElement("div");
        item.className = "legend-item";
        item.innerHTML = `<span style="font-size:12px;color:"+ThemeHelper.pick("rgba(255,255,255,0.45)","#2b3a52")+"">${label} intensity (low → high)</span>`;
        legendEl.appendChild(item);
    }

    // Bubble size key
    const sizeItem = document.createElement("div");
    sizeItem.className = "legend-item";
    sizeItem.innerHTML = `
        <span class="size-legend-circle" style="width:8px;height:8px;border:1.5px solid "+ThemeHelper.pick("rgba(255,255,255,0.3)","rgba(30,60,100,0.30)")+";border-radius:50%;display:inline-block"></span>
        <span class="size-legend-circle" style="width:14px;height:14px;border:1.5px solid "+ThemeHelper.pick("rgba(255,255,255,0.3)","rgba(30,60,100,0.30)")+";border-radius:50%;display:inline-block"></span>
        <span class="size-legend-circle" style="width:20px;height:20px;border:1.5px solid "+ThemeHelper.pick("rgba(255,255,255,0.3)","rgba(30,60,100,0.30)")+";border-radius:50%;display:inline-block"></span>
        <span style="font-size:11px;color:"+ThemeHelper.pick("rgba(255,255,255,0.45)","#2b3a52")+"">Bubble = Avg Red Cards</span>`;
    legendEl.appendChild(sizeItem);
}

// ---------------------------------------------------------------------------
// Main chart draw
// ---------------------------------------------------------------------------
function drawBubbleChart() {
    currentXMetric   = xMetricSelect.value;
    currentColorMode = colorModeEl.value;
    const filtered   = getFilteredData();

    d3.select("#q3Chart").html("");
    updateSummaryCards(filtered);
    document.getElementById("q3ChartTitle").textContent = CHART_TITLE_MAP[currentXMetric];
    renderLegend(filtered, currentColorMode);

    if (filtered.length === 0) {
        d3.select("#q3Chart").append("div").attr("class","empty-message")
            .text("No disciplinary data available for these filters. Try selecting a different league or switching the X metric.");
        return;
    }

    const getColor = makeBubbleColorFn(filtered, currentColorMode);

    // ---- Dimensions ----
    const margin     = { top: 40, right: 60, bottom: 74, left: 72 };
    const fullWidth  = 1100;
    const fullHeight = 600;
    const W = fullWidth  - margin.left - margin.right;
    const H = fullHeight - margin.top  - margin.bottom;

    const svg = d3.select("#q3Chart")
        .append("svg")
        .attr("viewBox", `0 0 ${fullWidth} ${fullHeight}`)
        .append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

    // Pitch-style background
    svg.append("rect")
        .attr("x",0).attr("y",0)
        .attr("width",W).attr("height",H)
        .attr("fill",ThemeHelper.pick("rgba(62,240,122,0.012)","rgba(30,60,100,0.025)"))
        .attr("rx",8);

    // ---- Scales ----
    const xExt = d3.extent(filtered, d => d[currentXMetric]);
    const yExt = d3.extent(filtered, d => d.Points);
    const rExt = d3.extent(filtered, d => d.AvgRedCards);

    const xPad = (xExt[1] - xExt[0]) * 0.05;
    const yPad = (yExt[1] - yExt[0]) * 0.04;

    const x = d3.scaleLinear().domain([xExt[0] - xPad, xExt[1] + xPad]).range([0, W]);
    const y = d3.scaleLinear().domain([yExt[0] - yPad, yExt[1] + yPad]).nice().range([H, 0]);

    // Bubble radius: sqrt scale so area is proportional to value
    // Min radius 4, max 18; for teams with 0 red cards → min radius
    const rMin = 4;
    const rMax = 18;
    const rScale = rExt[1] > 0
        ? d3.scaleSqrt().domain([0, rExt[1]]).range([rMin, rMax]).clamp(true)
        : () => rMin;

    // ---- Grid lines ----
    svg.append("g").attr("class","grid")
        .call(d3.axisLeft(y).tickSize(-W).tickFormat("").ticks(8))
        .call(g => g.select(".domain").remove())
        .call(g => g.selectAll("line")
            .style("stroke",ThemeHelper.pick("rgba(255,255,255,0.055)","rgba(30,60,100,0.10)"))
            .style("stroke-dasharray","4,5"));

    svg.append("g").attr("class","grid")
        .attr("transform",`translate(0,${H})`)
        .call(d3.axisBottom(x).tickSize(-H).tickFormat("").ticks(8))
        .call(g => g.select(".domain").remove())
        .call(g => g.selectAll("line")
            .style("stroke",ThemeHelper.pick("rgba(255,255,255,0.04)","rgba(30,60,100,0.07)"))
            .style("stroke-dasharray","4,5"));

    // ---- "Danger zone" quadrant — high fouls, low points ----
    // Top-right of X axis, bottom of Y axis
    const dangerXThreshold = xExt[0] + (xExt[1] - xExt[0]) * 0.70;
    const dangerYThreshold = yExt[0] + (yExt[1] - yExt[0]) * 0.35;

    svg.append("rect")
        .attr("x", x(dangerXThreshold))
        .attr("y", y(dangerYThreshold))
        .attr("width", x(xExt[1] + xPad) - x(dangerXThreshold))
        .attr("height", y(yExt[0] - yPad) - y(dangerYThreshold))
        .attr("fill","rgba(255,69,69,0.06)")
        .attr("stroke","rgba(255,69,69,0.18)")
        .attr("stroke-width","1")
        .attr("stroke-dasharray","4,4")
        .attr("rx","4");

    svg.append("text")
        .attr("class","danger-zone-label")
        .attr("x", x(dangerXThreshold) + 8)
        .attr("y", y(yExt[0] - yPad) - 10)
        .style("font-size","9.5px")
        .style("font-family","'Barlow Condensed',sans-serif")
        .style("font-weight","700")
        .style("text-transform","uppercase")
        .style("letter-spacing","0.08em")
        .style("fill","rgba(255,69,69,0.5)")
        .text("⚠ High Fouls · Low Points");

    // ---- Axes ----
    svg.append("g").attr("class","axis").attr("transform",`translate(0,${H})`)
        .call(d3.axisBottom(x).ticks(8))
        .call(g => g.select(".domain").style("stroke",ThemeHelper.pick("rgba(255,255,255,0.1)","rgba(30,60,100,0.15)")))
        .selectAll("text").style("fill",ThemeHelper.pick("rgba(255,255,255,0.35)","#1a2a3a")).style("font-size","10.5px");

    svg.append("g").attr("class","axis")
        .call(d3.axisLeft(y).ticks(8))
        .call(g => g.select(".domain").style("stroke",ThemeHelper.pick("rgba(255,255,255,0.1)","rgba(30,60,100,0.15)")))
        .selectAll("text").style("fill",ThemeHelper.pick("rgba(255,255,255,0.35)","#1a2a3a")).style("font-size","10.5px");

    // Axis labels
    svg.append("text")
        .attr("x", W / 2).attr("y", H + 58)
        .attr("text-anchor","middle")
        .style("font-size","12px").style("fill",ThemeHelper.pick("rgba(255,255,255,0.35)","#1a2a3a"))
        .text(X_LABEL_MAP[currentXMetric]);

    svg.append("text")
        .attr("transform","rotate(-90)")
        .attr("x", -H / 2).attr("y", -54)
        .attr("text-anchor","middle")
        .style("font-size","12px").style("fill",ThemeHelper.pick("rgba(255,255,255,0.35)","#1a2a3a"))
        .text("Points");

    // ---- Regression line ----
    const reg = linearRegression(filtered, currentXMetric, "Points");
    if (reg) {
        const rx0 = xExt[0] - xPad;
        const rx1 = xExt[1] + xPad;
        svg.append("line")
            .attr("x1", x(rx0)).attr("y1", y(reg.slope * rx0 + reg.intercept))
            .attr("x2", x(rx1)).attr("y2", y(reg.slope * rx1 + reg.intercept))
            .style("stroke",ThemeHelper.pick("rgba(255,255,255,0.2)","rgba(30,60,100,0.25)"))
            .style("stroke-width","1.5")
            .style("stroke-dasharray","7,5");

        svg.append("text")
            .attr("x", W - 6).attr("y", 16)
            .attr("text-anchor","end")
            .style("font-size","10.5px")
            .style("fill",ThemeHelper.pick("rgba(255,255,255,0.35)","#1a2a3a"))
            .text(`R² = ${reg.r2.toFixed(3)} · ${reg.slope < 0 ? "↘ neg" : "↗ pos"} correlation`);
    }

    // ---- Bubbles ----
    // Sort by radius desc so smaller bubbles render on top
    const sorted = [...filtered].sort((a, b) => rScale(b.AvgRedCards) - rScale(a.AvgRedCards));

    const bubblesGroup = svg.append("g").attr("class","bubbles-group");

    bubblesGroup.selectAll("circle.discipline-bubble")
        .data(sorted, d => `${d.League}-${d.Team}-${d.Season}`)
        .enter()
        .append("circle")
        .attr("class","discipline-bubble")
        .attr("cx", d => x(d[currentXMetric]))
        .attr("cy", d => y(d.Points))
        .attr("r",  d => rScale(d.AvgRedCards))
        .attr("fill", d => getColor(d))
        .attr("fill-opacity", 0.5)
        .attr("stroke", d => getColor(d))
        .attr("stroke-width", 0.8)
        .attr("stroke-opacity", 0.7)
        .on("mousemove", function (event, d) {
            d3.select(this).raise()
                .attr("fill-opacity", 0.9)
                .attr("stroke-width", 2)
                .attr("filter", `drop-shadow(0 0 8px ${getColor(d)})`);

            // Yellow and red card emoji indicators
            const yellowBars  = Math.round((d.AvgYellowCards || 0) * 4);
            const redBars     = Math.round((d.AvgRedCards   || 0) * 20);
            const yellowEmoji = "🟨".repeat(Math.min(yellowBars, 5));
            const redEmoji    = "🟥".repeat(Math.min(redBars,    3)) || "—";

            tooltip
                .style("opacity", 1)
                .html(
                    `<strong>${d.Team}</strong><br>` +
                    `<span style="color:"+ThemeHelper.pick("rgba(255,255,255,0.45)","#2b3a52")+"">${d.League} · ${d.Season}</span><br>` +
                    `<hr style="border:0;border-top:1px solid "+ThemeHelper.pick("rgba(255,255,255,0.1)","rgba(30,60,100,0.15)")+";margin:6px 0">` +
                    `<span style="color:"+ThemeHelper.pick("rgba(255,255,255,0.45)","#2b3a52")+"">Points:</span> <strong style="color:"+ThemeHelper.pick("#3ef07a","#1a6fa8")+"">${d.Points}</strong><br>` +
                    `<span style="color:"+ThemeHelper.pick("rgba(255,255,255,0.45)","#2b3a52")+"">${X_LABEL_MAP[currentXMetric]}:</span> <strong>${d[currentXMetric].toFixed(2)}</strong><br>` +
                    `<span style="color:"+ThemeHelper.pick("rgba(255,255,255,0.45)","#2b3a52")+"">Yellow / game:</span> ${d.AvgYellowCards != null ? d.AvgYellowCards.toFixed(2) : "—"} ${yellowEmoji}<br>` +
                    `<span style="color:"+ThemeHelper.pick("rgba(255,255,255,0.45)","#2b3a52")+"">Red / game:</span> ${d.AvgRedCards != null ? d.AvgRedCards.toFixed(3) : "—"} ${redEmoji}`
                )
                .style("left", `${event.pageX + 14}px`)
                .style("top",  `${event.pageY - 28}px`);
        })
        .on("mouseleave", function (event, d) {
            d3.select(this)
                .attr("fill-opacity", 0.5)
                .attr("stroke-width", 0.8)
                .attr("filter", null);
            tooltip.style("opacity", 0);
        });

    // ---- Bubble size reference legend (drawn in SVG bottom-right) ----
    // Show 3 reference circles: small, medium, large red card rate
    const refValues = [
        rExt[1] * 0.15,
        rExt[1] * 0.5,
        rExt[1],
    ].filter(v => v > 0);

    if (refValues.length > 0 && rExt[1] > 0) {
        const legX = W - 60;
        const legY = H - 20;

        refValues.forEach((v, i) => {
            const r = rScale(v);
            svg.append("circle")
                .attr("cx", legX).attr("cy", legY - r)
                .attr("r", r)
                .attr("fill","none")
                .attr("stroke",ThemeHelper.pick("rgba(255,255,255,0.18)","rgba(30,60,100,0.25)"))
                .attr("stroke-width","1");
            svg.append("text")
                .attr("x", legX + r + 4).attr("y", legY - r + 4)
                .style("font-size","9px").style("fill",ThemeHelper.pick("rgba(255,255,255,0.35)","#1a2a3a"))
                .text(v.toFixed(2));
        });

        svg.append("text")
            .attr("x", legX).attr("y", legY + 14)
            .attr("text-anchor","middle")
            .style("font-size","9px").style("fill",ThemeHelper.pick("rgba(255,255,255,0.25)","#4a5568"))
            .text("Red cards/game");
    }
}

// ---------------------------------------------------------------------------
// Event listeners
// ---------------------------------------------------------------------------
xMetricSelect.addEventListener("change",  drawBubbleChart);
leagueFilter.addEventListener("change",   drawBubbleChart);
colorModeEl.addEventListener("change",    drawBubbleChart);
resetBtn.addEventListener("click", () => {
    xMetricSelect.value  = "AvgFouls";
    leagueFilter.value   = "all";
    colorModeEl.value    = "league";
    drawBubbleChart();
});

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
(async function init() {
    const updateLoading = (done, total) => {
        const pct = Math.round((done / total) * 100);
        loadingMsg.textContent = `Loading disciplinary data… ${pct}%`;
    };

    allTeamSeasons = await loadAllLeagues(updateLoading);
    loadingMsg.style.display = "none";
    chartDiv.style.display   = "block";

    drawBubbleChart();

    // Re-render when theme is toggled
    window.__themeRedraw = drawBubbleChart;
})();
