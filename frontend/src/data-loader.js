/**
 * data-loader.js
 * Shared data pipeline used by Q2 and Q3 pages.
 * Exposes: loadAllLeagues(onProgress) → Promise<Object[]>
 */

const LEAGUE_CONFIGS = {
    "Premier League": { folder: "premier-league" },
    "La Liga":        { folder: "la-liga"        },
    "Bundesliga":     { folder: "bundesliga"     },
    "Serie A":        { folder: "serie-a"        },
    "Ligue 1":        { folder: "ligue-1"        },
};

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
// Step 1 — fetch + parse one CSV
// ---------------------------------------------------------------------------
function loadSeasonCSV(path, seasonLabel, leagueName) {
    return d3.csv(path).then(rows => {
        rows.forEach(d => {
            d.Season = seasonLabel;
            d.League = leagueName;
            d.FTHG = +d.FTHG;
            d.FTAG = +d.FTAG;
            const nullableFields = ["HS","AS","HST","AST","HF","AF","HY","AY","HR","AR"];
            nullableFields.forEach(field => {
                d[field] = (d[field] === "" || d[field] == null) ? null : +d[field];
            });
        });
        return rows;
    }).catch(() => []);
}

// ---------------------------------------------------------------------------
// Step 2 — explode match rows into per-team rows
// ---------------------------------------------------------------------------
function expandMatchesToTeamRows(matches) {
    const teamRows = [];
    matches.forEach(d => {
        if (!d.HomeTeam || !d.AwayTeam || !d.FTR) return;
        teamRows.push({
            League: d.League, Season: d.Season, Team: d.HomeTeam,
            GoalsFor: d.FTHG, GoalsAgainst: d.FTAG,
            Result: d.FTR === "H" ? "Win" : d.FTR === "D" ? "Draw" : "Loss",
            Points: d.FTR === "H" ? 3 : d.FTR === "D" ? 1 : 0,
            Shots: d.HS, ShotsOnTarget: d.HST, Fouls: d.HF,
            YellowCards: d.HY, RedCards: d.HR,
        });
        teamRows.push({
            League: d.League, Season: d.Season, Team: d.AwayTeam,
            GoalsFor: d.FTAG, GoalsAgainst: d.FTHG,
            Result: d.FTR === "A" ? "Win" : d.FTR === "D" ? "Draw" : "Loss",
            Points: d.FTR === "A" ? 3 : d.FTR === "D" ? 1 : 0,
            Shots: d.AS, ShotsOnTarget: d.AST, Fouls: d.AF,
            YellowCards: d.AY, RedCards: d.AR,
        });
    });
    return teamRows;
}

// ---------------------------------------------------------------------------
// Step 3 — aggregate per-team rows into team-season summary
// ---------------------------------------------------------------------------
function aggregateToTeamSeasons(teamRows) {
    const grouped = d3.group(teamRows, d => d.League, d => d.Team, d => d.Season);
    const result  = [];

    grouped.forEach((teamsMap, league) => {
        teamsMap.forEach((seasonsMap, team) => {
            seasonsMap.forEach((matches, season) => {
                const wins   = matches.filter(m => m.Result === "Win").length;
                const draws  = matches.filter(m => m.Result === "Draw").length;
                const losses = matches.filter(m => m.Result === "Loss").length;
                const goalsFor     = d3.sum(matches, m => m.GoalsFor);
                const goalsAgainst = d3.sum(matches, m => m.GoalsAgainst);

                // Only average over non-null values; return null if none exist
                const nullMean = (arr, accessor) => {
                    const valid = arr.filter(m => accessor(m) != null);
                    return valid.length ? d3.mean(valid, accessor) : null;
                };

                result.push({
                    League: league, Season: season, Team: team,
                    Played: matches.length,
                    Wins: wins, Draws: draws, Losses: losses,
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

    return result;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
async function loadAllLeagues(onProgress) {
    const tasks = [];
    for (const [leagueName, cfg] of Object.entries(LEAGUE_CONFIGS)) {
        for (const [filename, seasonLabel] of ALL_SEASONS) {
            tasks.push({ leagueName, cfg, filename, seasonLabel });
        }
    }

    const total = tasks.length;
    let done = 0;
    const allMatchRows = [];

    const BATCH_SIZE = 15;
    for (let i = 0; i < tasks.length; i += BATCH_SIZE) {
        const batch = tasks.slice(i, i + BATCH_SIZE);
        const batchResults = await Promise.all(
            batch.map(({ leagueName, cfg, filename, seasonLabel }) =>
                loadSeasonCSV(`../datasets/${cfg.folder}/${filename}`, seasonLabel, leagueName)
            )
        );
        batchResults.forEach(rows => allMatchRows.push(...rows));
        done += batch.length;
        if (onProgress) onProgress(done, total);
    }

    const teamRows    = expandMatchesToTeamRows(allMatchRows);
    const teamSeasons = aggregateToTeamSeasons(teamRows);
    return teamSeasons;
}

// Utility: sort key for season strings like "1993-94"
function getSeasonSortKey(seasonStr) {
    return parseInt(seasonStr.split("-")[0], 10);
}
