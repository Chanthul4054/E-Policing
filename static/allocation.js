let totalChart = null;
let marginalChart = null;

function getParamsFromForm() {
    const form = document.getElementById("controls");
    const params = new URLSearchParams(new FormData(form));
    return params.toString();
}

async function loadCharts() {
    const qs = getParamsFromForm();
    const res = await fetch("/api/diminishing?" + qs);
    const data = await res.json();

    const totalCtx = document.getElementById("totalChart");
    const marginalCtx = document.getElementById("marginalChart");

    if (totalChart) totalChart.destroy();
    if (marginalChart) marginalChart.destroy();

    totalChart = new Chart(totalCtx, {
        type: "line",
        data: {
            labels: data.totals,
            datasets: [{
                label: "Total Benefit",
                data: data.total_benefits,
                tension: 0.25
            }]
        },
        options: {
            responsive: true,
            animation: { duration: 900 },
            plugins: { legend: { display: true } }
        }
    });

    marginalChart = new Chart(marginalCtx, {
        type: "line",
        data: {
            labels: data.totals,
            datasets: [{
                label: "Marginal Benefit",
                data: data.marginal_benefits,
                tension: 0.25
            }]
        },
        options: {
            responsive: true,
            animation: { duration: 900 },
            plugins: { legend: { display: true } }
        }
    });
}

//Load charts on page load
loadCharts();