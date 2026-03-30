//Yearly trend chart 
var trendChart = new Chart(document.getElementById("trendChart"), {
    type: "line",
    data: {
        labels: ["2020","2021","2022","2023","2024","2025"],
        datasets: [{
            label: "Crime Frequency",
            data: [0,0,0,0,0,0],
            tension: 0.4,
            fill: true,
            borderColor: "#3b82f6",
            backgroundColor: "rgba(59,130,246,0.15)"
        }]
    },
    options: {
        plugins: { legend: { display: false } },
        scales: {
            x: {
                title: {
                    display: true,
                    text: 'Year',
                    color: '#94a3b8',
                    font: { size: 12 }
                },
                ticks: { color: '#94a3b8' },
                grid: { color: 'rgba(255,255,255,0.05)' }
            },
            y: {
                beginAtZero: true,
                title: {
                    display: true,
                    text: 'Number of Incidents',
                    color: '#94a3b8',
                    font: { size: 12 }
                },
                ticks: { color: '#94a3b8' },
                grid: { color: 'rgba(255,255,255,0.05)' }
            }
        }
    }
});

function updateChart(crimeType) {
    fetch('/pattern/get-trend?crime_type=' + encodeURIComponent(crimeType))
    .then(function(r) { return r.json(); })
    .then(function(d) {
        trendChart.data.labels            = d.labels;
        trendChart.data.datasets[0].data  = d.data;
        trendChart.data.datasets[0].label = crimeType + " frequency";
        trendChart.update();
    });
}

//Pattern strength helpers
function patternColor(strength) {
    if (strength === "high")   return "#ef4444";
    if (strength === "medium") return "#f97316";
    if (strength === "low")    return "#eab308";
    return "#64748b";
}

function patternLabel(strength) {
    if (strength === "high")   return "Strong Pattern";
    if (strength === "medium") return "Moderate Pattern";
    if (strength === "low")    return "Weak Pattern";
    return "No Pattern";
}

// ── Heatmap ──────────────────────────────────────────────────────
var TIMES    = ["morning", "afternoon", "night"];
var DAY_CATS = ["holiday", "non holiday"];

function parsePatternText(text) {
    if (!text) return null;
    text = text.toLowerCase().replace("time late night", "time night");

    var time = null;
    for (var i = 0; i < TIMES.length; i++) {
        if (text.indexOf("time " + TIMES[i]) !== -1) { time = TIMES[i]; break; }
    }

    var day = null;
    if      (text.indexOf("day non holiday") !== -1) day = "non holiday";
    else if (text.indexOf("day holiday")     !== -1) day = "holiday";
    else if (text.indexOf("day weekday")     !== -1) day = "weekday";
    else if (text.indexOf("day weekend")     !== -1) day = "weekend";

    if (!time || !day) return null;
    return { time: time, day: day };
}

function buildHeatmapMatrix(patterns) {
    var matrix = {};
    TIMES.forEach(function(t) {
        matrix[t] = {};
        DAY_CATS.forEach(function(d) { matrix[t][d] = 0; });
    });
    patterns.forEach(function(p) {
        var parsed = parsePatternText(p.pattern_text);
        if (!parsed) return;
        matrix[parsed.time][parsed.day] += p.confidence;
    });
    return matrix;
}

function renderHeatmap(gnName, patterns) {
    document.getElementById('heatmapGN').textContent     = gnName;
    document.getElementById('heatmapHint').style.display = 'none';

    var matrix = buildHeatmapMatrix(patterns);
    var maxVal = 0;
    TIMES.forEach(function(t) {
        DAY_CATS.forEach(function(d) { if (matrix[t][d] > maxVal) maxVal = matrix[t][d]; });
    });

    var grid    = document.getElementById('heatmapGrid');
    var xlabels = document.getElementById('heatmapXLabels');
    grid.innerHTML = '';
    grid.style.gridTemplateColumns = 'repeat(' + DAY_CATS.length + ', 1fr)';

    TIMES.forEach(function(t) {
        DAY_CATS.forEach(function(d) {
            var val       = matrix[t][d];
            var intensity = maxVal > 0 ? val / maxVal : 0;
            var cell      = document.createElement('div');
            cell.className        = 'heatmap-cell';
            cell.style.background = getCellColor(intensity);
            cell.title            = val > 0
                ? (t + ' / ' + d + '\nConf. weight: ' + val.toFixed(2))
                : (t + ' / ' + d + '\nNo patterns');
            grid.appendChild(cell);
        });
    });

    xlabels.innerHTML = '';
    xlabels.style.gridTemplateColumns = 'repeat(' + DAY_CATS.length + ', 1fr)'; // ← add this
    DAY_CATS.forEach(function(d) {
        var span = document.createElement('div');
        span.textContent = d.charAt(0).toUpperCase() + d.slice(1);
        xlabels.appendChild(span);
    });
}

function getCellColor(intensity) {
    if (intensity === 0)  return "#1e293b";
    if (intensity < 0.25) return "#1e3a5f";
    if (intensity < 0.5)  return "#1d4ed8";
    if (intensity < 0.75) return "#f97316";
    return "#ef4444";
}

function clearHeatmap() {
    document.getElementById('heatmapGN').textContent     = 'Select a GN division on the map';
    document.getElementById('heatmapHint').style.display = 'block';
    document.getElementById('heatmapGrid').innerHTML     = '';
    document.getElementById('heatmapXLabels').innerHTML  = '';
}

// ── Map setup ────────────────────────────────────────────────────
var map = L.map('map', { minZoom: 11, maxZoom: 17, zoomControl: true })
           .setView([7.2906, 80.6337], 11);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap'
}).addTo(map);

var kandyLayer = null;
var riskLayer  = null;
var allGeoData = null;
var riskData   = [];

var legend = L.control({ position: "bottomright" });
legend.onAdd = function() {
    var div = L.DomUtil.create("div", "map-legend");
    div.innerHTML =
        "<b>Pattern Strength</b><br>" +
        "<span style='color:#ef4444;font-size:16px;'>⬤</span> Strong &nbsp;(Lift≥1.5, Conf≥55%)<br>" +
        "<span style='color:#f97316;font-size:16px;'>⬤</span> Moderate<br>" +
        "<span style='color:#eab308;font-size:16px;'>⬤</span> Weak<br>" +
        "<span style='color:#64748b;font-size:16px;'>⬤</span> No pattern matched";
    return div;
};

fetch('/pattern/map-data')
.then(function(res) {
    if (!res.ok) throw new Error("GeoJSON fetch failed: " + res.status);
    return res.json();
})
.then(function(geoData) {
    allGeoData = geoData;

    kandyLayer = L.geoJSON(geoData, {
        filter: function(feature) {
            var props = feature.properties;
            if (!props) return false;
            var name = (props.ADM2_EN    || "").toLowerCase();
            var code = (props.ADM2_PCODE || "").toLowerCase();
            return name.includes("kandy") || code.includes("lk23");
        },
        style: function() {
            return { color: "#2563eb", weight: 1.5, fillColor: "#1d4ed8", fillOpacity: 0.1 };
        },
        onEachFeature: function(feature, layer) {
            var gnName = feature.properties.ADM4_EN || feature.properties.GN_NAME || "Unknown";
            layer.bindTooltip(gnName, { sticky: true, className: 'gn-tooltip' });
        }
    }).addTo(map);

    if (kandyLayer.getLayers().length > 0) {
        var bounds = kandyLayer.getBounds();
        map.fitBounds(bounds, { padding: [10, 10] });
        map.setMaxBounds(bounds.pad(0.05));
        map.setMinZoom(map.getBoundsZoom(bounds));
        legend.addTo(map);
    }
})
.catch(function(err) {
    console.error("Map load error:", err);
    document.getElementById('map').innerHTML =
        '<p style="color:#ef4444;padding:20px;">Could not load map. Check console.</p>';
});

// ── Apply Analytics button ────────────────────────────────────────
document.getElementById('applyBtn').addEventListener('click', function() {

    var crimeType    = document.getElementById('crimeTypeBadge').textContent.trim().toLowerCase();
    var locationType = document.getElementById('locationTypeSelect').value;
    var timeFilter   = document.getElementById('timeSelect').value;

    document.getElementById('loadingMsg').style.display = 'block';
    updateChart(crimeType);
    clearHeatmap();

    fetch('/pattern/get-risk-data' +
          '?location_type=' + encodeURIComponent(locationType) +
          '&time_filter='   + encodeURIComponent(timeFilter))
    .then(function(r) { return r.json(); })
    .then(function(data) {
        document.getElementById('loadingMsg').style.display = 'none';

        if (data.status !== 'success') { console.error("API error:", data); return; }

        riskData = data.predictions;

        if (riskLayer) { map.removeLayer(riskLayer); riskLayer = null; }
        if (!allGeoData) return;

        var riskLookup = {};
        riskData.forEach(function(p) { riskLookup[p.gn_pcode] = p; });

        riskLayer = L.geoJSON(allGeoData, {

            filter: function(feature) {
                var props  = feature.properties;
                if (!props) return false;
                var name   = (props.ADM2_EN    || "").toLowerCase();
                var code   = (props.ADM2_PCODE || "").toLowerCase();
                var gnCode = props.ADM4_PCODE  || props.GN_PCODE || "";
                return (name.includes("kandy") || code.includes("lk23")) && !!riskLookup[gnCode];
            },

            style: function(feature) {
                var gnCode   = feature.properties.ADM4_PCODE || feature.properties.GN_PCODE || "";
                var pred     = riskLookup[gnCode];
                var strength = pred ? pred.pattern_strength : "none";
                return { color: "#ffffff", weight: 1.5, fillColor: patternColor(strength), fillOpacity: 0.65 };
            },

            onEachFeature: function(feature, layer) {
                var props       = feature.properties;
                var gnCode      = props.ADM4_PCODE  || props.GN_PCODE || "";
                var pred        = riskLookup[gnCode];
                if (!pred) return;

                var gnName      = pred.gn_division      || gnCode;
                var score       = (pred.risk_score * 100).toFixed(1);
                var patterns    = pred.top_patterns     || [];
                var allPatterns = pred.all_patterns     || patterns;
                var strength    = pred.pattern_strength || "none";

                var popupHtml =
                    '<div class="risk-popup">' +
                    '<div class="popup-header">' +
                        '<span class="popup-gn">' + gnName + '</span>' +
                        '<span class="popup-score" style="background:' + patternColor(strength) + '">' +
                            patternLabel(strength) + '</span>' +
                    '</div>' +
                    '<div class="popup-risk-row">Risk Score: <b>' + score + '%</b></div>';

                if (patterns.length === 0) {
                    popupHtml += '<p style="color:#94a3b8;font-size:12px;margin-top:8px;">No patterns matched for selected filters.</p>';
                } else {
                    popupHtml += '<table class="popup-table"><thead><tr><th>#</th><th>When</th><th>Location</th><th>How Often</th><th>Risk</th></tr></thead><tbody>';
                    patterns.forEach(function(p, i) {
                        var text = (p.pattern_text || '').toLowerCase();
                        // When it happens
                        var parts = [];
                        if      (text.includes("non holiday")) parts.push("Non-holiday");
                        else if (text.includes("holiday"))     parts.push("Holiday");
                        if      (text.includes("weekday"))     parts.push("Weekday");
                        else if (text.includes("weekend"))     parts.push("Weekend");
                        if      (text.includes("late night") || text.includes("time night")) parts.push("Night");
                        else if (text.includes("morning"))    parts.push("Morning");
                        else if (text.includes("afternoon"))  parts.push("Afternoon");

                        if      (text.includes("weather rainy"))   parts.push("Rainy");
                        else if (text.includes("weather cloudy"))  parts.push("Cloudy");
                        else if (text.includes("weather sunny"))   parts.push("Sunny");
                        else if (text.includes("weather windy"))   parts.push("Windy");
                        var when = parts.join(', ') || '-';
                        // How often
                        var freq = p.support >= 0.3  ? "Very often"
                        : p.support >= 0.15 ? "Often"
                        : p.support >= 0.07 ? "Sometimes"
                        : "Rarely";
                        
                        // Risk level
                        var risk = (p.lift > 2.0 && p.confidence >= 0.7) ? "🔴 High"
                        : (p.lift >= 1.5 || p.confidence >= 0.5) ? "🟡 Moderate"
                        : "🟢 Low";
                        
                        popupHtml += '<tr>' +
                        '<td>' + (i+1) + '</td>' +
                        '<td>' + when + '</td>' +
                        '<td>' + (p.location_type||'-') + '</td>' +
                        '<td>' + freq + '</td>' +
                        '<td>' + risk + '</td>' +
                        '</tr>';
                    });
                    popupHtml += '</tbody></table>';
                }
                popupHtml += '</div>';

                layer.bindPopup(popupHtml, { maxWidth: 400, className: 'risk-popup-wrap' });

                layer.on('mouseover', function(e) {
                    layer.setStyle({ fillOpacity: 0.88, weight: 2.5 });
                    layer.openPopup(e.latlng);
                    updateSummaryTable(gnName, patterns, strength);
                    renderHeatmap(gnName, allPatterns);
                });

                layer.on('mouseout', function() {
                    riskLayer.resetStyle(layer);
                });

                layer.on('click', function() {
                    updateSummaryTable(gnName, patterns, strength);
                    renderHeatmap(gnName, allPatterns);
                });
            }

        }).addTo(map);
    })
    .catch(function(err) {
        document.getElementById('loadingMsg').style.display = 'none';
        console.error("Risk data fetch error:", err);
    });
});

// ── Pattern summary table ─────────────────────────────────────────
function updateSummaryTable(gnName, patterns, strength) {
    document.getElementById('selectedGN').textContent = gnName;

    var badge = document.getElementById('strengthBadge');
    if (badge) {
        badge.textContent      = patternLabel(strength || "none");
        badge.style.background = patternColor(strength || "none");
    }

    var tbody = document.getElementById('patternTableBody');

    if (!patterns || patterns.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" style="color:#64748b;text-align:center;padding:15px;">No patterns found for this division with selected filters.</td></tr>';
        return;
    }

    function freqLabel(support) {
        if (support >= 0.3)  return { text: "Very often", cls: "freq-high" };
        if (support >= 0.15) return { text: "Often",      cls: "freq-high" };
        if (support >= 0.07) return { text: "Sometimes",  cls: "freq-med"  };
        return                      { text: "Rarely",     cls: "freq-low"  };
    }

    function riskLabel(lift, confidence) {
        if (lift > 2.0 && confidence >= 0.7) return { text: "🔴 High",     cls: "lift-high"   };
        if (lift >= 1.5 || confidence >= 0.5) return { text: "🟡 Moderate", cls: "lift-medium" };
        return                                       { text: "🟢 Low",      cls: "lift-low"    };
    }

    function parseWhen(patternText) {
        if (!patternText) return "—";
        var text  = patternText.toLowerCase().replace(/_/g, ' ');
        var parts = [];

        if      (text.includes("non holiday")) parts.push("Non-holiday");
        else if (text.includes("holiday"))     parts.push("Holiday");
        if      (text.includes("weekday"))     parts.push("Weekday");
        else if (text.includes("weekend"))     parts.push("Weekend");

        if      (text.includes("late night") || text.includes("time night")) parts.push("Night");
        else if (text.includes("morning"))    parts.push("Morning");
        else if (text.includes("afternoon"))  parts.push("Afternoon");

        if      (text.includes("weather rainy"))   parts.push("Rainy");
        else if (text.includes("weather cloudy"))  parts.push("Cloudy");
        else if (text.includes("weather sunny"))   parts.push("Sunny");
        else if (text.includes("weather windy"))   parts.push("Windy");

        return parts.map(function(p) {
            return '<span class="time-badge">' + p + '</span>';
        }).join(' ');
    }

    function buildDescription(p) {
        var text       = (p.pattern_text || '').toLowerCase().replace(/_/g, ' ');
        var loc        = (p.location_type || 'unknown location').toLowerCase();
        var crime      = document.getElementById('crimeTypeBadge').textContent.trim();
        var crimeLabel = crime.charAt(0).toUpperCase() + crime.slice(1).toLowerCase();

        var timeDesc = text.includes("late night") || text.includes("time night") ? "at night"
                     : text.includes("morning")   ? "in the morning"
                     : text.includes("afternoon") ? "in the afternoon" : "";

        var dayDesc  = text.includes("non holiday")  ? "on non-holiday days"
                     : text.includes("holiday")      ? "on public holidays"
                     : text.includes("weekend")      ? "on weekends"
                     : text.includes("weekday")      ? "on weekdays" : "";
        
        var weatherDesc = text.includes("weather rainy")  ? "during rainy weather"
                    : text.includes("weather cloudy") ? "during cloudy weather"
                    : text.includes("weather sunny")  ? "during sunny weather"
                    : text.includes("weather windy")  ? "during windy weather" : "";

        return crimeLabel + " is reported in <b>" + loc + "</b> areas " + timeDesc + " " + dayDesc + 
           (weatherDesc ? " " + weatherDesc : "") + ".";
    }

    var html = '';
    patterns.forEach(function(p, i) {
        var freq  = freqLabel(p.support);
        var risk  = riskLabel(p.lift, p.confidence);
        var when  = parseWhen(p.pattern_text);
        var desc  = buildDescription(p);
        var loc   = p.location_type || '—';

        html += '<tr>' +
            '<td style="font-weight:500;color:#94a3b8;">' + (i + 1) + '</td>' +
            '<td>' + when + '</td>' +
            '<td style="font-size:13px;">' + loc + '</td>' +
            '<td><span class="' + freq.cls + '" style="display:inline-block;padding:2px 8px;border-radius:6px;font-size:12px;font-weight:500;">' + freq.text + '</span></td>' +
            '<td style="font-size:12px;color:#94a3b8;line-height:1.5;max-width:260px;">' + desc + '</td>' +
            '<td class="' + risk.cls + '">' + risk.text + '</td>' +
            '</tr>';
    });

    tbody.innerHTML = html;
}