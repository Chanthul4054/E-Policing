function toggleSidebar() {
    const sidebar = document.getElementById("sidebar");
    sidebar.classList.toggle("collapsed");
    const collapsed = sidebar.classList.contains("collapsed");
    localStorage.setItem("sidebarCollapsed", collapsed);

}

// Restore state
document.addEventListener("DOMContentLoaded", () => {
    const collapsed = localStorage.getItem("sidebarCollapsed") === "true";
    if (collapsed) {
        document.getElementById("sidebar").classList.add("collapsed");
    }
});