let totalChart = null;
let marginalChart = null;

function getParamsFromForm() {
    const form = document.getElementById("controls");
    const params = new URLSearchParams(new FormData(form));
    return params.toString();
}

async function loadCharts() {
    const btn = document.querySelector('button[onclick="loadCharts()"]');
    if (btn) {
        var originalText = btn.innerHTML;
        btn.innerHTML = '<span class="btn-icon">⌛</span> Updating...';
        btn.disabled = true;
    }

    try {
        const qs = getParamsFromForm();
        const res = await fetch("/api/diminishing?" + qs);
        const data = await res.json();

        const totalCtx = document.getElementById("totalChart");
        const marginalCtx = document.getElementById("marginalChart");

        if (totalChart) totalChart.destroy();
        if (marginalChart) marginalChart.destroy();

        const commonOptions = {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 1000,
                easing: 'easeOutQuart'
            },
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: {
                    display: true,
                    labels: { color: '#94a3b8' }
                },
                tooltip: {
                    backgroundColor: 'rgba(30, 41, 59, 0.9)',
                    titleColor: '#f8fafc',
                    bodyColor: '#cbd5e1',
                    borderColor: '#334155',
                    borderWidth: 1,
                    padding: 10,
                    displayColors: true,
                }
            },
            scales: {
                x: {
                    grid: { color: '#334155' },
                    ticks: { color: '#94a3b8' }
                },
                y: {
                    grid: { color: '#334155' },
                    ticks: { color: '#94a3b8' }
                }
            }
        };

        totalChart = new Chart(totalCtx, {
            type: "line",
            data: {
                labels: data.totals,
                datasets: [{
                    label: "Total Benefit",
                    data: data.total_benefits,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 6
                }]
            },
            options: commonOptions
        });

        marginalChart = new Chart(marginalCtx, {
            type: "line",
            data: {
                labels: data.totals,
                datasets: [{
                    label: "Marginal Benefit",
                    data: data.marginal_benefits,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 6
                }]
            },
            options: commonOptions
        });

    } catch (e) {
        console.error("Error loading charts:", e);
    } finally {
        if (btn) {
            btn.innerHTML = originalText;
            btn.disabled = false;
        }
    }
}




function toggleSidebar() {
    const sidebar = document.getElementById("sidebar");
    sidebar.classList.toggle("collapsed");
    const collapsed = sidebar.classList.contains("collapsed");
    localStorage.setItem("sidebarCollapsed", collapsed);

    // Trigger chart resize after transition
    setTimeout(() => {
        if (totalChart) totalChart.resize();
        if (marginalChart) marginalChart.resize();
    }, 450);
}

// Restore state
document.addEventListener("DOMContentLoaded", () => {
    const collapsed = localStorage.getItem("sidebarCollapsed") === "true";
    if (collapsed) {
        document.getElementById("sidebar").classList.add("collapsed");
    }
});

//Load charts on page load
loadCharts();