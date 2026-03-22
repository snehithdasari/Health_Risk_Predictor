// Initialize charts on Results page if data exists
document.addEventListener('DOMContentLoaded', () => {
    if (typeof resultData !== 'undefined') {
        initCharts();
    }
});

let charts = {};

function initCharts() {
    // Chart Defaults
    Chart.defaults.font.family = "'Inter', sans-serif";
    Chart.defaults.color = '#64748b'; // slate-500
    
    const commonOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { display: false },
            tooltip: {
                backgroundColor: '#1e293b',
                padding: 12,
                titleFont: { size: 14, weight: 'bold' },
                bodyFont: { size: 14 },
                callbacks: {
                    label: (context) => ` Risk: ${context.raw}%`
                }
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                max: 100,
                grid: {
                    color: '#e2e8f0',
                    drawBorder: false,
                    borderDash: [5, 5]
                },
                ticks: {
                    callback: (value) => value + '%'
                },
                title: {
                    display: true,
                    text: 'Predicted Risk Probability (%)',
                    font: { weight: 'bold' }
                }
            },
            x: {
                grid: { display: false }
            }
        }
    };

    // Helper to extract data
    const getPlotData = (diseaseKey) => {
        const d = resultData[diseaseKey] || {};
        return {
            labels: ['Gradient Boosting', 'Random Forest', 'SVM'],
            data: [d['Gradient Boosting'] || 0, d['Random Forest'] || 0, d['SVM'] || 0]
        };
    };

    // 1. Diabetes Chart (Purple)
    const ctxDiab = document.getElementById('chart-diabetes');
    if(ctxDiab) {
        const dataDiab = getPlotData('diabetes');
        charts.diabetes = new Chart(ctxDiab, {
            type: 'bar',
            data: {
                labels: dataDiab.labels,
                datasets: [{
                    data: dataDiab.data,
                    backgroundColor: '#a855f7', // purple-500
                    borderRadius: 6,
                    barPercentage: 0.5
                }]
            },
            options: commonOptions
        });
    }

    // 2. Hypertension Chart (Orange)
    const ctxHyper = document.getElementById('chart-hypertension');
    if(ctxHyper) {
        const dataHyper = getPlotData('hypertension');
        charts.hypertension = new Chart(ctxHyper, {
            type: 'bar',
            data: {
                labels: dataHyper.labels,
                datasets: [{
                    data: dataHyper.data,
                    backgroundColor: '#f97316', // orange-500
                    borderRadius: 6,
                    barPercentage: 0.5
                }]
            },
            options: commonOptions
        });
    }

    // 3. Heart Disease Chart (Blue)
    const ctxHeart = document.getElementById('chart-heart');
    if(ctxHeart) {
        const dataHeart = getPlotData('heart_disease');
        charts.heart = new Chart(ctxHeart, {
            type: 'bar',
            data: {
                labels: dataHeart.labels,
                datasets: [{
                    data: dataHeart.data,
                    backgroundColor: '#3b82f6', // blue-500
                    borderRadius: 6,
                    barPercentage: 0.5
                }]
            },
            options: commonOptions
        });
    }
}

// Tab Switching Logic
function switchTab(tabName) {
    // 1. Update Buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('bg-white', 'shadow', 'font-bold', 'text-slate-900');
        btn.classList.add('font-semibold', 'text-slate-500', 'hover:text-slate-700');
        btn.setAttribute('aria-selected', 'false');
    });
    
    const activeBtn = document.getElementById(`tab-${tabName}`);
    activeBtn.classList.add('bg-white', 'shadow', 'font-bold', 'text-slate-900');
    activeBtn.classList.remove('font-semibold', 'text-slate-500', 'hover:text-slate-700');
    activeBtn.setAttribute('aria-selected', 'true');

    // 2. Update Panels
    document.querySelectorAll('.tab-panel').forEach(panel => {
        panel.classList.add('hidden');
        panel.classList.remove('block');
    });
    
    document.getElementById(`panel-${tabName}`).classList.remove('hidden');
    document.getElementById(`panel-${tabName}`).classList.add('block');
}
