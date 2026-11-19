
const startDate = document.getElementById("startDate");
const endDate = document.getElementById("endDate");

const predictButton = document.getElementById("predictButton");
const message = document.getElementById("message");

const resultBody = document.getElementById("resultBody");
const summaryGrid = document.getElementById("summaryGrid");

let pollutionChart = null;
let weatherChart = null;


// Үзүүлэлтийн нэр болон нэгж
const indicators = [
    {
        key: "PM10",
        label: "PM10",
        unit: "µg/m³"
    },
    {
        key: "PM2.5",
        label: "PM2.5",
        unit: "µg/m³"
    },
    {
        key: "SO2",
        label: "SO2",
        unit: "µg/m³"
    },
    {
        key: "temperature",
        label: "Температур",
        unit: "°C"
    },
    {
        key: "max_wind_speed",
        label: "Салхины хурд",
        unit: ""
    },
    {
        key: "precipitation",
        label: "Хур тунадас",
        unit: ""
    }
];


// Утгыг харуулах
function formatValue(value) {

    if (value === null || value === undefined) {
        return "—";
    }

    return Number(value).toFixed(2);
}


// Эх үүсвэрийн CSS class
function getSourceClass(source) {

    if (source === "actual") {
        return "actual";
    }

    if (source === "forecast") {
        return "forecast";
    }

    return "missing";
}


// Нэгтгэсэн үзүүлэлтүүдийг харуулах
function displaySummary(summary) {

    const summaryItems = [
        {
            key: "PM10",
            label: "Дундаж PM10",
            unit: "µg/m³"
        },
        {
            key: "PM2.5",
            label: "Дундаж PM2.5",
            unit: "µg/m³"
        },
        {
            key: "SO2",
            label: "Дундаж SO2",
            unit: "µg/m³"
        },
        {
            key: "temperature",
            label: "Дундаж температур",
            unit: "°C"
        },
        {
            key: "max_wind_speed",
            label: "Дундаж салхины хурд",
            unit: ""
        },
        {
            key: "precipitation",
            label: "Нийт хур тунадас",
            unit: ""
        }
    ];

    summaryGrid.innerHTML = "";

    summaryItems.forEach(item => {

        const value = summary[item.key];

        const card = document.createElement("div");
        card.className = "summary-item";

        card.innerHTML = `
            <h3>${item.label}</h3>
            <p>${formatValue(value)}</p>
            <span>${item.unit}</span>
        `;

        summaryGrid.appendChild(card);

    });

}


// Сарын хүснэгт харуулах
function displayTable(rows) {

    resultBody.innerHTML = "";

    rows.forEach(row => {

        const tr = document.createElement("tr");

        let html = `<td><strong>${row.month}</strong></td>`;

        indicators.forEach(item => {

            const data = row[item.key];

            html += `
                <td class="${getSourceClass(data.source)}">
                    ${formatValue(data.value)}
                </td>
            `;

        });

        tr.innerHTML = html;

        resultBody.appendChild(tr);

    });

}


// Графикийн эх үүсвэрийг ялгах
function getChartValues(rows, key, source) {

    return rows.map(row => {

        const item = row[key];

        if (item.source === source) {
            return item.value;
        }

        return null;

    });

}


// График байгуулах
function displayCharts(rows) {

    const labels = rows.map(row => row.month);

    if (pollutionChart) {
        pollutionChart.destroy();
    }

    if (weatherChart) {
        weatherChart.destroy();
    }


    // Агаарын бохирдлын график
    const pollutionContext = document
        .getElementById("pollutionChart")
        .getContext("2d");

    pollutionChart = new Chart(pollutionContext, {

        type: "line",

        data: {
            labels: labels,

            datasets: [

                {
                    label: "PM10 - Бодит",
                    data: getChartValues(rows, "PM10", "actual"),
                    borderColor: "#2563eb",
                    tension: 0.3,
                    spanGaps: false
                },

                {
                    label: "PM10 - Прогноз",
                    data: getChartValues(rows, "PM10", "forecast"),
                    borderColor: "#2563eb",
                    borderDash: [6, 5],
                    tension: 0.3,
                    spanGaps: false
                },

                {
                    label: "PM2.5 - Бодит",
                    data: getChartValues(rows, "PM2.5", "actual"),
                    borderColor: "#16a34a",
                    tension: 0.3
                },

                {
                    label: "PM2.5 - Прогноз",
                    data: getChartValues(rows, "PM2.5", "forecast"),
                    borderColor: "#16a34a",
                    borderDash: [6, 5],
                    tension: 0.3
                },

                {
                    label: "SO2 - Бодит",
                    data: getChartValues(rows, "SO2", "actual"),
                    borderColor: "#ea580c",
                    tension: 0.3
                },

                {
                    label: "SO2 - Прогноз",
                    data: getChartValues(rows, "SO2", "forecast"),
                    borderColor: "#ea580c",
                    borderDash: [6, 5],
                    tension: 0.3
                }

            ]
        },

        options: {
            responsive: true,
            maintainAspectRatio: false,

            interaction: {
                intersect: false,
                mode: "index"
            },

            plugins: {
                legend: {
                    position: "bottom"
                }
            },

            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: "µg/m³"
                    }
                }
            }
        }

    });


    // Цаг уурын график
    const weatherContext = document
        .getElementById("weatherChart")
        .getContext("2d");

    weatherChart = new Chart(weatherContext, {

        type: "line",

        data: {
            labels: labels,

            datasets: [

                {
                    label: "Температур",
                    data: rows.map(row => row.temperature.value),
                    borderColor: "#dc2626",
                    tension: 0.3,
                    yAxisID: "y"
                },

                {
                    label: "Салхины хурд",
                    data: rows.map(row => row.max_wind_speed.value),
                    borderColor: "#2563eb",
                    tension: 0.3,
                    yAxisID: "y1"
                },

                {
                    label: "Хур тунадас",
                    data: rows.map(row => row.precipitation.value),
                    borderColor: "#16a34a",
                    tension: 0.3,
                    yAxisID: "y1"
                }

            ]
        },

        options: {
            responsive: true,
            maintainAspectRatio: false,

            interaction: {
                intersect: false,
                mode: "index"
            },

            plugins: {
                legend: {
                    position: "bottom"
                }
            },

            scales: {

                y: {
                    type: "linear",
                    position: "left",
                    title: {
                        display: true,
                        text: "Температур (°C)"
                    }
                },

                y1: {
                    type: "linear",
                    position: "right",
                    grid: {
                        drawOnChartArea: false
                    },
                    title: {
                        display: true,
                        text: "Салхи / Хур тунадас"
                    }
                }

            }
        }

    });

}


// Тооцоолох товч
predictButton.addEventListener("click", async () => {

    const start = startDate.value;
    const end = endDate.value;

    message.textContent = "";

    if (!start || !end) {
        message.textContent =
            "Эхлэх болон дуусах огноог сонгоно уу.";
        return;
    }

    if (start > end) {
        message.textContent =
            "Эхлэх огноо дуусах огнооноос хойш байж болохгүй.";
        return;
    }

    predictButton.disabled = true;
    predictButton.textContent = "Тооцоолж байна...";

    try {

        const response = await fetch("/predict", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                start_date: start,
                end_date: end
            })

        });

        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(
                data.message || "Тооцоолол амжилтгүй боллоо."
            );
        }

        displaySummary(data.summary);

        displayTable(data.monthly_results);

        displayCharts(data.monthly_results);

        document.getElementById("summarySection").hidden = false;
        document.getElementById("tableSection").hidden = false;
        document.getElementById("chartSection").hidden = false;

        message.textContent =
            "Амжилттай тооцооллоо!";

        message.className = "success-message";

    } catch (error) {

        message.textContent = error.message;
        message.className = "error-message";

    } finally {

        predictButton.disabled = false;
        predictButton.textContent = "Тооцоолох";

    }

});
