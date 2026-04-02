function toggleSidebar() {
    const sidebar = document.getElementById("sidebar");
    sidebar.classList.toggle("collapsed");
    const collapsed = sidebar.classList.contains("collapsed");
    localStorage.setItem("sidebarCollapsed", collapsed);
}

function showRiskLoader() {
    const overlay = document.getElementById("riskLoadingOverlay");
    if (overlay) {
        overlay.classList.add("active");
        overlay.setAttribute("aria-hidden", "false");
    }
}

function hideRiskLoader() {
    const overlay = document.getElementById("riskLoadingOverlay");
    if (overlay) {
        overlay.classList.remove("active");
        overlay.setAttribute("aria-hidden", "true");
    }
}

document.addEventListener("DOMContentLoaded", () => {
    const collapsed = localStorage.getItem("sidebarCollapsed") === "true";
    if (collapsed) {
        document.getElementById("sidebar").classList.add("collapsed");
    }

    const riskForm = document.querySelector("form.risk-filter-panel");
    const dropdown = document.getElementById("gn_dropdown");

    if (riskForm) {
        riskForm.addEventListener("submit", (e) => {
            if (!dropdown || !dropdown.value) {
                e.preventDefault();
                alert("Please select an area and crime type first.");
                return;
            }
            showRiskLoader();
        });
    }

    window.addEventListener("pageshow", () => {
        hideRiskLoader();
    });
});