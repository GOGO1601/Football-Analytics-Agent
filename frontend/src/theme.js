/* theme.js — shared dark/light mode toggle, persists across pages */
(function () {
    const STORAGE_KEY = "fb-dashboard-theme";

    function getTheme() {
        return localStorage.getItem(STORAGE_KEY) || "light";
    }

    function applyTheme(theme) {
        document.documentElement.setAttribute("data-theme", theme);
        // Update all toggle buttons on the page
        document.querySelectorAll(".theme-toggle").forEach(btn => {
            const icon = btn.querySelector(".toggle-icon");
            const label = btn.querySelector(".toggle-label");
            if (theme === "dark") {
                if (icon)  icon.textContent  = "☀️";
                if (label) label.textContent = "Light Mode";
            } else {
                if (icon)  icon.textContent  = "🌙";
                if (label) label.textContent = "Dark Mode";
            }
        });
    }

    function toggleTheme() {
        const next = getTheme() === "dark" ? "light" : "dark";
        localStorage.setItem(STORAGE_KEY, next);
        applyTheme(next);
        // Re-render charts if a global redraw function is registered
        if (typeof window.__themeRedraw === "function") window.__themeRedraw();
    }

    // Apply on load immediately (before paint)
    applyTheme(getTheme());

    // Wire up all .theme-toggle buttons once DOM is ready
    document.addEventListener("DOMContentLoaded", function () {
        applyTheme(getTheme()); // re-apply so labels are set
        document.querySelectorAll(".theme-toggle").forEach(btn => {
            btn.addEventListener("click", toggleTheme);
        });
    });

    // Expose helpers globally for JS chart files
    window.ThemeHelper = {
        isDark:  () => getTheme() === "dark",
        getTheme,
        // Return the right value for dark vs light
        pick: (darkVal, lightVal) => getTheme() === "dark" ? darkVal : lightVal,
    };
})();
