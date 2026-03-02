//--------------Variable initialization-------------

// Initialize the map globally, for the instance to be accessible
const map = L.map('map').setView([7.2906, 80.6337], 13);
//Keep track of the layer 
let currentGeoLayer = null 
//For the chart
let riskChart = null;
 // Global variable for the second chart
let distributionChart = null;
// Toggle state
let showAllResults = false; 
// Store data globally to reuse on toggle
let latestPredictions = []; 


//Base map layer

L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
    subdomains: 'abcd',
    maxZoom: 20
}).addTo(map);

//Loading the Kandy Map

async function loadKandyMap(crimeType='burglary') {
    try {
        //Fetch both the GeoJSON and the Mapping filter simultaneously
        const [geoResponse, mappingResponse, predictResponse] = await Promise.all([
            fetch('/static/gn_div_info/Kandy_geo.json'),
            fetch('/static/gn_div_info/gn_name_mapping.json'),
            fetch(`/hotspot/predict?type=${crimeType}`)// Calling your Flask route
        ]);

        if (!geoResponse.ok || !mappingResponse.ok) {
            throw new Error("One or more data files not found on server");
        }

        const kandyGeoJSON = await geoResponse.json();
        const mappingData = await mappingResponse.json();
        const predictData = await predictResponse.json(); // The results from generate_risk_scores

        if (predictData.predictions) {

            updateRiskChart(predictData.predictions);
            // Update the distribution summary chart
            updateDistributionChart(predictData.predictions);

        }

        //Convert the mapping values into an Array 
        const allowedCodes = Object.values(mappingData);

        //A quick-lookup object for risk scores
        const riskLookup = {};
        predictData.predictions.forEach(item => {
        riskLookup[item.gn_name] = item.risk_score;
        });        


        // Filter the GeoJSON features
        const filteredFeatures = kandyGeoJSON.features.filter(feature => {
            const featureCode = feature.properties.ADM4_PCODE;
            return allowedCodes.includes(featureCode);
        });

        //Create a new GeoJSON object with only the 72 divisions
        const filteredGeoJSON = {
            type: "FeatureCollection",
        
            features: filteredFeatures.map(feature => {
                const featureCode = feature.properties.ADM4_PCODE;
        
                //Pull the real risk score using the ADM4_PCODE
                feature.properties.risk = riskLookup[featureCode] || 0;
                
                return feature;
            })   
        };     


        //Pass the filtered data to the renderer
        if (filteredFeatures.length === 0) {
            console.warn("No matching GN divisions found. Check if property names match.");
            return;
        }        
        renderMapLayer(filteredGeoJSON);

    } catch (err) {
        console.error("Error loading/filtering Kandy GeoJSON:", err);
    }
}



// -----------------Rendering logic--------------------------
function renderMapLayer(geoData) {
    //Remove the old layer if it exists
    if (currentGeoLayer) {
        map.removeLayer(currentGeoLayer);
    }

    //Turn the JSON coordinates into interactive SVG shapes
    const geoLayer = L.geoJson(geoData, {
        //Determine how each ploygon looks
        style: function(feature) {
            const riskScore = feature.properties.risk || 0;
            return {
                fillColor: getColor(riskScore), // Default green== low risk
                weight: 1.5,
                opacity: 1,
                color: '#4a5568',
                fillOpacity: 0.4
            };
        },
        //Attch interactively to every single GN division
        onEachFeature: function(feature, layer) {

            // Match the property name to the GeoJSON
            const gnName = feature.properties.ADM4_EN || 'Unknown';

            //Bind a popup that appears when clicking a division
            layer.bindPopup(`
                <div style="font-family: Arial;">
                    <b>GN Division:</b> ${gnName}<br>
                    <hr>
                    <b>Predicted Risk:</b> <span style="color:${getColor(feature.properties.risk)}">
                        ${(feature.properties.risk * 100).toFixed(1)}%
                    </span>
                </div>
            `);
            
            //Highlight the boundary when the mouse hovers over it
            layer.on('mouseover', function() {
                this.setStyle({ 
                    fillOpacity: 0.7, 
                    weight: 3,
                    color: '#1b345d' // Changes border
                });
            });

            // Reset style when the mouse leaves
            layer.on('mouseout', function() {
                this.setStyle({ 
                    fillOpacity: 0.4, 
                    weight: 1.5,
                    color: '#4a5568' 
                });
            });
        }
    }).addTo(map);

    //Store this layer so we can clear it next time
    currentGeoLayer = geoLayer;

    if (geoLayer) {
        map.fitBounds(geoLayer.getBounds(),{
        padding: [20, 20], 
        maxZoom: 14
    
    });

    }
    //Recalculate the square dimensions
    setTimeout(() => {
        map.invalidateSize();
    }, 100);
    
}

//Colour helper function
function getColor(risk) {
    return risk > 0.75 ? '#ef4444' : // Red (Critical)
           risk > 0.50 ? '#f97316' : // Orange (High)
           risk > 0.25 ? '#f1c40f' : // Yellow (Medium)
                         '#10b981';  // Green (Low)
}

//Trigger everything on the page load
document.addEventListener("DOMContentLoaded", () => {
    loadKandyMap('burglary');

    // Listener for dropdown changes
    const selector = document.getElementById('crimeTypeSelect');
    if (selector) {
        selector.addEventListener('change', (e) => {
            const selectedType = e.target.value;
            console.log("Switching to:", selectedType);
            loadKandyMap(selectedType); // Re-fetch and re-render
        });
    }


    //Populates other parts of the dashboard
    if (typeof displayDummyData === 'function') {
        displayDummyData();
    }
});


//Map Legend
function addLegend() {
    const legend = L.control({ position: 'topright' });

    legend.onAdd = function (map) {
        const div = L.DomUtil.create('div', 'info legend');
        const grades = [0, 0.25, 0.50, 0.75]; 
        const labels = ['Low Risk', 'Medium Risk', 'High Risk', 'Critical'];

        div.innerHTML = '<h4>Risk Intensity</h4>';

        for (let i = 0; i < grades.length; i++) {
            div.innerHTML +=
                '<i style="background:' + getColor(grades[i] + 0.01) + '"></i> ' +
                labels[i] + '<br>';
        }

        return div;
    };

    legend.addTo(map);
}

// Update the DOMContentLoaded to trigger the legend
document.addEventListener("DOMContentLoaded", () => {
    loadKandyMap('burglary');
    addLegend(); 

    const selector = document.getElementById('crimeTypeSelect');
    if (selector) {
        selector.addEventListener('change', (e) => {
            loadKandyMap(e.target.value);
        });
    }

    const toggleBtn = document.getElementById('toggleChartBtn');
    if (toggleBtn) {
        toggleBtn.addEventListener('click', () => {
            showAllResults = !showAllResults; // Flip the state
            updateRiskChart(latestPredictions); // Re-draw with the same data
        });
    }


});


//--------------------The Top 10 division chart---------------
function updateRiskChart(predictions) {
    latestPredictions = predictions; // Save for toggling
    const canvas = document.getElementById('priorityCharts');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    // Destroy old instance
    if (riskChart) {
        riskChart.destroy();
    }
    // Determine slice based on toggle state
    const limit = showAllResults ? 10 : 5;
    //Sort data
    const sortedData = [...predictions]
        .sort((a, b) => b.risk_score - a.risk_score)
        .slice(0, limit); // Top 10 GN Divisions

    const labels = sortedData.map(item => item.gn_name);
    const scores = sortedData.map(item => item.risk_score * 100);

    // Update the Title and Button Text
    document.getElementById('chartTitle').innerText = `Top ${limit} Priority Zones`;
    document.getElementById('toggleChartBtn').innerText = showAllResults ? "View Less" : "View More";
    //Create the Chart
    riskChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Risk Level (%)',
                data: scores,
                // Synchronize colors with the map
                backgroundColor: sortedData.map(item => getColor(item.risk_score)),
                borderRadius: 5,
                borderWidth: 0,
                barThickness: showAllResults ? 15 : 25 
            }]
        },
        options: {
            indexAxis: 'y', // Makes the bar chart horizontal
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }, // Hide legend for cleaner look
                tooltip: {
                    callbacks: {
                        label: (context) => `Risk: ${context.raw.toFixed(1)}%`
                    }
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    max: 100,
                    grid: { display: false },
                    ticks: { color: '#ffffff', font: { size: 10 } }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#ffffff', font: { size: 11 } }
                }
            }
        }
    });
}

//--------------------The risk distribution chart -------------------------------
function updateDistributionChart(predictions) {
    const canvas = document.getElementById('totalChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    if (distributionChart) {
        distributionChart.destroy();
    }

    //Initialize the counts object 
    const counts = { 
        critical: 0, 
        high: 0, 
        medium: 0, 
        low: 0 
    };

    //Aggregate the data
    predictions.forEach(item => {
        const score = item.risk_score;
        if (score > 0.75) counts.critical++;
        else if (score > 0.50) counts.high++;
        else if (score > 0.25) counts.medium++;
        else counts.low++;
    });

    const totalGNs = predictions.length;

    //Update the HTML cards 
    if (document.getElementById('count-critical')) {
        document.getElementById('count-critical').innerText = counts.critical.toString().padStart(2, '0');
        document.getElementById('count-high').innerText = counts.high.toString().padStart(2, '0');
        document.getElementById('count-medium').innerText = counts.medium.toString().padStart(2, '0');
        document.getElementById('count-low').innerText = counts.low.toString().padStart(2, '0');
    }

    //Create the Chart
    distributionChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Critical', 'High', 'Medium', 'Low'],
            datasets: [{
                data: [counts.critical, counts.high, counts.medium, counts.low],
                backgroundColor: ['#ef4444', '#f97316', '#f1c40f', '#6bfdcc'],
                borderWidth: 0,
                cutout: '70%'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            }
        },
        plugins: [{
            id: 'centerText',
            beforeDraw: (chart) => {
                const { width, height, ctx } = chart;
                ctx.restore();
                
                // Draw Total Number
                ctx.font = "bold 2rem sans-serif";
                ctx.textBaseline = "middle";
                ctx.fillStyle = "white";
                const text = totalGNs.toString();
                const textX = Math.round((width - ctx.measureText(text).width) / 2);
                ctx.fillText(text, textX, height / 2 - 10);

                // Draw "TOTAL GNs" label
                ctx.font = "12px sans-serif";
                ctx.fillStyle = "#94a3b8";
                const subText = "TOTAL GNs";
                const subX = Math.round((width - ctx.measureText(subText).width) / 2);
                ctx.fillText(subText, subX, height / 2 + 20);
                
                ctx.save();
            }
        }]
    });
}