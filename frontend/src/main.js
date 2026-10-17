// =============================================================================
// main.js — Q1: Team Performance Over Time
// Dark theme redesign: pitch-inspired grid, football icon tooltip markers,
// football shirt colours, animated line draws, improved empty states.
// =============================================================================

const MAX_TEAMS = 4;

const leagueConfigs = {
    "Premier League": { folder: "premier-league" },
    "La Liga":        { folder: "la-liga" },
    "Bundesliga":     { folder: "bundesliga" },
    "Serie A":        { folder: "serie-a" },
    "Ligue 1":        { folder: "ligue-1" },
};

// Full season list shared across all leagues
const ALL_SEASONS = [
    ["season-9394.csv","1993-94"],["season-9495.csv","1994-95"],
    ["season-9596.csv","1995-96"],["season-9697.csv","1996-97"],
    ["season-9798.csv","1997-98"],["season-9899.csv","1998-99"],
    ["season-9900.csv","1999-00"],["season-0001.csv","2000-01"],
    ["season-0102.csv","2001-02"],["season-0203.csv","2002-03"],
    ["season-0304.csv","2003-04"],["season-0405.csv","2004-05"],
    ["season-0506.csv","2005-06"],["season-0607.csv","2006-07"],
    ["season-0708.csv","2007-08"],["season-0809.csv","2008-09"],
    ["season-0910.csv","2009-10"],["season-1011.csv","2010-11"],
    ["season-1112.csv","2011-12"],["season-1213.csv","2012-13"],
    ["season-1314.csv","2013-14"],["season-1415.csv","2014-15"],
    ["season-1516.csv","2015-16"],["season-1617.csv","2016-17"],
    ["season-1718.csv","2017-18"],["season-1819.csv","2018-19"],
    ["season-1920.csv","2019-20"],["season-2021.csv","2020-21"],
    ["season-2122.csv","2021-22"],["season-2223.csv","2022-23"],
    ["season-2324.csv","2023-24"],["season-2425.csv","2024-25"],
    ["season-2526.csv","2025-26"],
];

// ---------------------------------------------------------------------------
// Colour palette — professional, on light background
// ---------------------------------------------------------------------------
const TEAM_COLOURS = [
    ThemeHelper.pick("#3ef07a","#27805a"),  // green
    ThemeHelper.pick("#4da6ff","#2563a8"),  // blue
    ThemeHelper.pick("#ff6b6b","#c0392b"),  // red
    ThemeHelper.pick("#ffcc00","#b87a20"),  // amber
    ThemeHelper.pick("#c084fc","#7b52c4"),  // violet
    ThemeHelper.pick("#06d6a0","#1a7a6e"),  // teal
    ThemeHelper.pick("#ff9f43","#c0621a"),  // orange
    ThemeHelper.pick("#fd79a8","#b0406a"),  // pink
];

// Single shared scale — domain set explicitly before every draw
let colorScale = d3.scaleOrdinal();

let aggregatedDataByLeague = {};
let currentLeague = "Premier League";

// ---------------------------------------------------------------------------
// DOM references
// ---------------------------------------------------------------------------
const leagueSelect = d3.select("#leagueSelect");
const metricSelect = d3.select("#metricSelect");
const teamSelect   = d3.select("#teamSelect");
const resetButton  = d3.select("#resetButton");
const teamWarning  = d3.select("#teamWarning");
const tooltip      = d3.select("#tooltip");

// ---------------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------------
function loadSeason(file, season, league) {
    return d3.csv(file).then(data => {
        data.forEach(d => {
            d.Season = season;
            d.League = league;
            d.FTHG = +d.FTHG;
            d.FTAG = +d.FTAG;
            const nullable = ["HS","AS","HST","AST","HF","AF","HY","AY","HR","AR"];
            nullable.forEach(f => { d[f] = (d[f] === "" || d[f] == null) ? null : +d[f]; });
        });
        return data;
    }).catch(() => []);
}

function transformToTeamMatchData(matches) {
    const rows = [];
    matches.forEach(d => {
        if (!d.HomeTeam || !d.AwayTeam || !d.FTR) return;
        rows.push({
            League: d.League, Season: d.Season,
            Team: d.HomeTeam, GoalsFor: d.FTHG, GoalsAgainst: d.FTAG,
            Result: d.FTR === "H" ? "Win" : d.FTR === "D" ? "Draw" : "Loss",
            Points: d.FTR === "H" ? 3 : d.FTR === "D" ? 1 : 0,
            Shots: d.HS, ShotsOnTarget: d.HST, Fouls: d.HF,
            YellowCards: d.HY, RedCards: d.HR,
        });
        rows.push({
            League: d.League, Season: d.Season,
            Team: d.AwayTeam, GoalsFor: d.FTAG, GoalsAgainst: d.FTHG,
            Result: d.FTR === "A" ? "Win" : d.FTR === "D" ? "Draw" : "Loss",
            Points: d.FTR === "A" ? 3 : d.FTR === "D" ? 1 : 0,
            Shots: d.AS, ShotsOnTarget: d.AST, Fouls: d.AF,
            YellowCards: d.AY, RedCards: d.AR,
        });
    });
    return rows;
}

function aggregateTeamSeason(teamMatches) {
    const grouped = d3.group(teamMatches, d => d.League, d => d.Team, d => d.Season);
    const aggregated = [];
    grouped.forEach((teamsMap, league) => {
        teamsMap.forEach((seasonMap, team) => {
            seasonMap.forEach((matches, season) => {
                const wins   = matches.filter(d => d.Result === "Win").length;
                const draws  = matches.filter(d => d.Result === "Draw").length;
                const losses = matches.filter(d => d.Result === "Loss").length;
                const goalsFor     = d3.sum(matches, d => d.GoalsFor);
                const goalsAgainst = d3.sum(matches, d => d.GoalsAgainst);
                const nullMean = (arr, acc) => {
                    const v = arr.filter(m => acc(m) != null);
                    return v.length ? d3.mean(v, acc) : null;
                };
                aggregated.push({
                    League: league, Team: team, Season: season,
                    Played: matches.length, Wins: wins, Draws: draws, Losses: losses,
                    Points: wins * 3 + draws,
                    GoalsFor: goalsFor, GoalsAgainst: goalsAgainst,
                    GoalDiff: goalsFor - goalsAgainst,
                    AvgShots:         nullMean(matches, m => m.Shots),
                    AvgShotsOnTarget: nullMean(matches, m => m.ShotsOnTarget),
                    AvgFouls:         nullMean(matches, m => m.Fouls),
                    AvgYellowCards:   nullMean(matches, m => m.YellowCards),
                    AvgRedCards:      nullMean(matches, m => m.RedCards),
                });
            });
        });
    });
    return aggregated;
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------
function getSeasonSortValue(s) { return +s.split("-")[0]; }
function cssSafeClass(t) { return t.replace(/[^a-zA-Z0-9_-]/g, "_"); }

// ---------------------------------------------------------------------------
// UI population
// ---------------------------------------------------------------------------
function populateLeagueDropdown() {
    leagueSelect.selectAll("option")
        .data(Object.keys(leagueConfigs))
        .enter().append("option")
        .attr("value", d => d).text(d => d);
    leagueSelect.property("value", currentLeague);
}

function populateTeamMultiSelect(leagueName) {
    const data  = aggregatedDataByLeague[leagueName] || [];
    const teams = [...new Set(data.map(d => d.Team))].sort();

    teamSelect.html("");
    teamSelect.selectAll("option")
        .data(teams).enter().append("option")
        .attr("value", d => d).text(d => d);

    const defaults = {
        "Premier League": ["Manchester United", "Arsenal", "Liverpool"],
        "La Liga":        ["Barcelona", "Real Madrid", "Atletico Madrid"],
        "Bundesliga":     ["Bayern Munich", "Borussia Dortmund", "Werder Bremen"],
        "Serie A":        ["Juventus", "AC Milan", "Inter"],
        "Ligue 1":        ["Paris SG", "Lyon", "Marseille"],
    };

    const preferred = (defaults[leagueName] || []).filter(t => teams.includes(t));
    const toSelect  = preferred.length ? preferred : teams.slice(0, 3);
    teamSelect.selectAll("option").property("selected", d => toSelect.includes(d));
    teamWarning.text("");
}

function getSelectedTeams() {
    return Array.from(teamSelect.node().selectedOptions).map(o => o.value);
}

function enforceTeamSelectionLimit() {
    const selected = getSelectedTeams();
    if (selected.length <= MAX_TEAMS) { teamWarning.text(""); return selected; }
    const allowed = selected.slice(0, MAX_TEAMS);
    teamSelect.selectAll("option").property("selected", function () {
        return allowed.includes(this.value);
    });
    teamWarning.text(`Max ${MAX_TEAMS} teams. Extra selections removed.`);
    return allowed;
}

function updateSummaryCards(leagueName, metricName, selectedTeams, data) {
    d3.select("#summaryLeague").text(leagueName.split(" ")[0]); // short form
    d3.select("#summaryMetric").text(metricName === "GoalDiff" ? "GD" : metricName);
    d3.select("#summaryTeams").text(selectedTeams.length);
    d3.select("#summarySeasons").text(new Set(data.map(d => d.Season)).size);
}

// ---------------------------------------------------------------------------
// Legend
// ---------------------------------------------------------------------------
function renderLegend(selectedTeams) {
    const legend = d3.select("#legend").html("");
    const items = legend.selectAll(".legend-item")
        .data(selectedTeams).enter()
        .append("div").attr("class", "legend-item");

    // Football shirt SVG icon in the legend colour
    items.append("div")
        .attr("class", "legend-swatch")
        .style("background-color", d => colorScale(d));

    items.append("span").text(d => d);
}

// ---------------------------------------------------------------------------
// Metric metadata
// ---------------------------------------------------------------------------
const METRIC_LABELS = {
    Points:   "Points",
    GoalsFor: "Goals Scored",
    GoalDiff: "Goal Difference",
    Wins:     "Wins",
};

const CHART_TITLES = {
    Points:   "Points Across Seasons",
    GoalsFor: "Goals Scored Across Seasons",
    GoalDiff: "Goal Difference Across Seasons",
    Wins:     "Wins Across Seasons",
};

// ---------------------------------------------------------------------------
// Football icon for data points — SVG path of a football
// We use a small ⚽ unicode emoji approach via foreignObject for simplicity,
// or fall back to styled circle with a "stitching" cross pattern.
// ---------------------------------------------------------------------------
function appendFootballMarker(svg, cx, cy, r, color, data, metric) {
    // Outer circle
    const g = svg.append("g")
        .attr("transform", `translate(${cx},${cy})`)
        .style("cursor", "pointer");

    g.append("circle")
        .attr("r", r)
        .attr("fill", color)
        .attr("fill-opacity", 0.9)
        .attr("stroke", "white")
        .attr("stroke-width", 1);

    // Pentagon star pattern (simplified football stitch lines)
    g.append("line")
        .attr("x1", -r * 0.5).attr("y1", 0)
        .attr("x2", r * 0.5).attr("y2", 0)
        .attr("stroke", "rgba(0,0,0,0.35)").attr("stroke-width", 0.8);
    g.append("line")
        .attr("x1", 0).attr("y1", -r * 0.5)
        .attr("x2", 0).attr("y2", r * 0.5)
        .attr("stroke", "rgba(0,0,0,0.35)").attr("stroke-width", 0.8);

    return g;
}

// ---------------------------------------------------------------------------
// Main chart — redesigned with dark pitch aesthetic
// ---------------------------------------------------------------------------
function drawMultiLineChart(data, selectedTeams, metric) {
    d3.select("#chart").html("");

    // Set colour scale domain before rendering anything
    colorScale
        .domain(selectedTeams)
        .range(TEAM_COLOURS.slice(0, selectedTeams.length));

    if (!selectedTeams.length) {
        d3.select("#chart").append("div").attr("class", "empty-message")
            .text("Select at least one team to begin.");
        d3.select("#legend").html("");
        return;
    }

    const filtered = data.filter(d => selectedTeams.includes(d.Team));
    if (!filtered.length) {
        d3.select("#chart").append("div").attr("class", "empty-message")
            .text("No data found for the selected teams.");
        d3.select("#legend").html("");
        return;
    }

    d3.select("#chartTitle").text(CHART_TITLES[metric]);

    // Dimensions
    const margin     = { top: 36, right: 100, bottom: 74, left: 64 };
    const fullWidth  = 1160;
    const fullHeight = 520;
    const W = fullWidth  - margin.left - margin.right;
    const H = fullHeight - margin.top  - margin.bottom;

    const svg = d3.select("#chart")
        .append("svg")
        .attr("viewBox", `0 0 ${fullWidth} ${fullHeight}`)
        .append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

    // Pitch-style background: very dark green tint
    svg.append("rect")
        .attr("x", 0).attr("y", 0)
        .attr("width", W).attr("height", H)
        .attr("fill", ThemeHelper.pick("rgba(62,240,122,0.018)","rgba(30,60,100,0.03)"))
        .attr("rx", 8);

    // Seasons sorted chronologically
    const seasons = [...new Set(filtered.map(d => d.Season))]
        .sort((a, b) => getSeasonSortValue(a) - getSeasonSortValue(b));

    const x = d3.scalePoint().domain(seasons).range([0, W]).padding(0.05);

    const yExtent = d3.extent(filtered, d => d[metric]);
    const yFloor  = metric === "GoalDiff" ? yExtent[0] : Math.min(0, yExtent[0]);
    const y = d3.scaleLinear()
        .domain([yFloor, yExtent[1]]).nice()
        .range([H, 0]);

    // ---- Pitch horizontal stripes (alternating faint bands) ----
    const yTicks = y.ticks(7);
    yTicks.forEach((tick, i) => {
        if (i % 2 === 0) {
            svg.append("rect")
                .attr("x", 0)
                .attr("y", y(yTicks[i + 1] || yExtent[1]))
                .attr("width", W)
                .attr("height", y(tick) - y(yTicks[i + 1] || yExtent[1]))
                .attr("fill", ThemeHelper.pick("rgba(255,255,255,0.015)","rgba(30,60,100,0.06)"));
        }
    });

    // ---- Grid lines (horizontal only, very subtle) ----
    svg.append("g").attr("class", "grid-lines")
        .call(d3.axisLeft(y).tickSize(-W).tickFormat("").ticks(7))
        .call(g => g.select(".domain").remove())
        .call(g => g.selectAll("line")
            .style("stroke", ThemeHelper.pick("rgba(255,255,255,0.06)","rgba(30,60,100,0.10)"))
            .style("stroke-dasharray", "4,5"));

    // ---- Zero line for GoalDiff ----
    if (metric === "GoalDiff") {
        svg.append("line")
            .attr("x1", 0).attr("x2", W)
            .attr("y1", y(0)).attr("y2", y(0))
            .style("stroke", "rgba(255,204,0,0.4)")
            .style("stroke-width", 1.5)
            .style("stroke-dasharray", "6,5");
        svg.append("text")
            .attr("x", W + 6).attr("y", y(0) + 4)
            .style("font-size", "10px")
            .style("fill", "rgba(255,204,0,0.5)")
            .text("0");
    }

    // ---- Axes ----
    const tickStep = Math.max(1, Math.floor(seasons.length / 14));
    svg.append("g")
        .attr("class", "axis")
        .attr("transform", `translate(0,${H})`)
        .call(d3.axisBottom(x)
            .tickValues(seasons.filter((_, i) => i % tickStep === 0)))
        .call(g => g.select(".domain").style("stroke",ThemeHelper.pick("rgba(255,255,255,0.1)","rgba(30,60,100,0.15)")))
        .selectAll("text")
        .attr("transform", "rotate(-42)")
        .style("text-anchor", "end")
        .style("font-size", "10.5px")
        .style("fill", ThemeHelper.pick("rgba(255,255,255,0.35)","#1a2a3a"));

    svg.append("g")
        .attr("class", "axis")
        .call(d3.axisLeft(y).ticks(7))
        .call(g => g.select(".domain").style("stroke",ThemeHelper.pick("rgba(255,255,255,0.1)","rgba(30,60,100,0.15)")))
        .selectAll("text")
        .style("font-size", "10.5px")
        .style("fill", ThemeHelper.pick("rgba(255,255,255,0.35)","#1a2a3a"));

    // Axis labels
    svg.append("text")
        .attr("x", W / 2).attr("y", H + 62)
        .attr("text-anchor", "middle")
        .style("font-size", "11px").style("fill", ThemeHelper.pick("rgba(255,255,255,0.35)","#1a2a3a"))
        .text("Season");

    svg.append("text")
        .attr("transform", "rotate(-90)")
        .attr("x", -H / 2).attr("y", -50)
        .attr("text-anchor", "middle")
        .style("font-size", "11px").style("fill", ThemeHelper.pick("rgba(255,255,255,0.35)","#1a2a3a"))
        .text(METRIC_LABELS[metric]);

    // ---- Build series ----
    const grouped = d3.group(filtered, d => d.Team);
    const teamSeries = Array.from(grouped, ([team, values]) => ({
        team,
        values: values.sort((a, b) => getSeasonSortValue(a.Season) - getSeasonSortValue(b.Season))
    }));

    const lineGen = d3.line()
        .defined(d => d[metric] != null)
        .x(d => x(d.Season))
        .y(d => y(d[metric]))
        .curve(d3.curveMonotoneX);

    const linesGroup = svg.append("g").attr("class", "lines-group");

    // ---- Area fills (glow under lines) ----
    const areaGen = d3.area()
        .defined(d => d[metric] != null)
        .x(d => x(d.Season))
        .y0(y(Math.max(yFloor, 0)))
        .y1(d => y(d[metric]))
        .curve(d3.curveMonotoneX);

    linesGroup.selectAll(".team-area")
        .data(teamSeries).enter()
        .append("path")
        .attr("class", d => `team-area area-${cssSafeClass(d.team)}`)
        .attr("fill", d => colorScale(d.team))
        .attr("fill-opacity", 0.07)
        .attr("d", d => areaGen(d.values));

    // ---- Lines ----
    linesGroup.selectAll(".team-line")
        .data(teamSeries).enter()
        .append("path")
        .attr("class", "team-line")
        .attr("fill", "none")
        .attr("stroke", d => colorScale(d.team))
        .attr("stroke-width", 2.5)
        .attr("stroke-linejoin", "round")
        .attr("stroke-linecap", "round")
        .attr("opacity", 0.9)
        .attr("filter", d => `drop-shadow(0 0 4px ${colorScale(d.team)}66)`)
        .attr("d", d => lineGen(d.values))
        .on("mouseenter", function (event, d) {
            linesGroup.selectAll(".team-line").attr("opacity", 0.1).attr("stroke-width", 2);
            linesGroup.selectAll(".team-area").attr("fill-opacity", 0.01);
            d3.select(this).attr("opacity", 1).attr("stroke-width", 3.5);
            svg.selectAll(`.area-${cssSafeClass(d.team)}`).attr("fill-opacity", 0.14);
            svg.selectAll(`.point-${cssSafeClass(d.team)}`).raise().attr("opacity", 1);
        })
        .on("mouseleave", function () {
            linesGroup.selectAll(".team-line").attr("opacity", 0.9).attr("stroke-width", 2.5);
            linesGroup.selectAll(".team-area").attr("fill-opacity", 0.07);
            svg.selectAll(".data-point").attr("opacity", 1);
        });

    // ---- Data point circles (football-style) ----
    teamSeries.forEach(series => {
        svg.selectAll(`.point-${cssSafeClass(series.team)}`)
            .data(series.values.filter(d => d[metric] != null))
            .enter()
            .append("circle")
            .attr("class", `data-point point-${cssSafeClass(series.team)}`)
            .attr("cx", d => x(d.Season))
            .attr("cy", d => y(d[metric]))
            .attr("r", 5)
            .attr("fill", colorScale(series.team))
            .attr("stroke", ThemeHelper.pick("#0b0f19","#f0f4f8"))
            .attr("stroke-width", 1.5)
            .style("cursor", "pointer")
            .on("mousemove", function (event, d) {
                d3.select(this).attr("r", 7)
                    .attr("filter", `drop-shadow(0 0 6px ${colorScale(series.team)})`);
                tooltip
                    .style("opacity", 1)
                    .html(
                        `<strong>${d.Team}</strong><br>` +
                        `<span style="color:"+ThemeHelper.pick("rgba(255,255,255,0.45)","#2b3a52")+"">${d.League} · ${d.Season}</span><br>` +
                        `<span style="color:"+ThemeHelper.pick("rgba(255,255,255,0.45)","#2b3a52")+"">${METRIC_LABELS[metric]}:</span> ` +
                        `<strong style="color:var(--accent)">${d[metric]}</strong>`
                    )
                    .style("left", `${event.pageX + 14}px`)
                    .style("top",  `${event.pageY - 30}px`);
            })
            .on("mouseleave", function () {
                d3.select(this).attr("r", 5).attr("filter", null);
                tooltip.style("opacity", 0);
            });
    });

    // ---- End-of-line team labels ----
    teamSeries.forEach(series => {
        const last = series.values.filter(d => d[metric] != null).at(-1);
        if (!last) return;
        svg.append("text")
            .attr("x", x(last.Season) + 8)
            .attr("y", y(last[metric]) + 4)
            .style("font-size", "10.5px")
            .style("font-weight", "700")
            .style("fill", colorScale(series.team))
            .style("pointer-events", "none")
            .text(series.team.split(" ").at(-1));
    });

    renderLegend(selectedTeams);
}

// ---------------------------------------------------------------------------
// Dashboard orchestration
// ---------------------------------------------------------------------------
function renderDashboard() {
    const leagueName    = leagueSelect.property("value");
    const metric        = metricSelect.property("value");
    const selectedTeams = enforceTeamSelectionLimit();
    const leagueData    = aggregatedDataByLeague[leagueName] || [];

    updateSummaryCards(leagueName, metric, selectedTeams, leagueData);
    drawMultiLineChart(leagueData, selectedTeams, metric);
}

function resetControls() {
    currentLeague = "Premier League";
    leagueSelect.property("value", currentLeague);
    metricSelect.property("value", "Points");
    populateTeamMultiSelect(currentLeague);
    renderDashboard();
}

// ---------------------------------------------------------------------------
// App init
// ---------------------------------------------------------------------------
async function loadLeagueData(leagueName, config) {
    const promises = ALL_SEASONS.map(([filename, season]) =>
        loadSeason(`../datasets/${config.folder}/${filename}`, season, leagueName)
    );
    const allSeasons  = await Promise.all(promises);
    const matchData   = allSeasons.flat();
    const teamMatches = transformToTeamMatchData(matchData);
    return aggregateTeamSeason(teamMatches);
}

async function initializeApp() {
    populateLeagueDropdown();

    const leagueNames = Object.keys(leagueConfigs);
    for (let i = 0; i < leagueNames.length; i++) {
        const name = leagueNames[i];
        d3.select("#chart").html(
            `<div class="loading-message">Loading ${name} (${i + 1}/${leagueNames.length})</div>`
        );
        try {
            aggregatedDataByLeague[name] = await loadLeagueData(name, leagueConfigs[name]);
        } catch (err) {
            console.warn(`Failed to load ${name}:`, err);
            aggregatedDataByLeague[name] = [];
        }
    }

    populateTeamMultiSelect(currentLeague);
    renderDashboard();

    leagueSelect.on("change", function () {
        currentLeague = this.value;
        populateTeamMultiSelect(currentLeague);
        renderDashboard();
    });
    metricSelect.on("change", renderDashboard);
    teamSelect.on("change", renderDashboard);
    resetButton.on("click", resetControls);
}

initializeApp();

// Re-render chart when theme is toggled
window.__themeRedraw = renderDashboard;
