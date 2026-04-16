const UPDATE_HZ = 100;
const UPDATE_INTERVAL_MS = 1000 / UPDATE_HZ;

let updateTimer = null;

function startUpdateLoop() {
  if (updateTimer) return;
  updateTimer = setInterval(fetchDataAndUpdateCharts, UPDATE_INTERVAL_MS);
}
startUpdateLoop();

const chartMap = {
  yaw: { label: "Yaw" },
  pitch: { label: "Pitch" },
  target_yaw: { label: "Target Yaw" },
  target_pitch: { label: "Target Pitch" },
  gimbal_yaw: { label: "Gimbal Yaw" },
  gimbal_pitch: { label: "Gimbal Pitch" },
  control_v_yaw: { label: "Control V Yaw" },
  control_v_pitch: { label: "Control V Pitch" },
  control_a_yaw: { label: "Control A Yaw" },
  control_a_pitch: { label: "Control A Pitch" },
  fly_time: { label: "Fly Time" },
  target_v_yaw: { label: "Target V Yaw" },
};

const mainCtx = document.getElementById("mainChart").getContext("2d");
let individualCharts = {};
let individualRanges = {};

const commonChartOptions = {
  animation: false,
  responsive: false,
  interaction: {
    mode: "nearest",
    axis: "x",
    intersect: false,
  },
  elements: {
    line: {
      tension: 0,
    },
    point: {
      radius: 0,
      hoverRadius: 0,
    },
  },
  plugins: {
    tooltip: {
      enabled: true,
      mode: "nearest",
      intersect: false,
      backgroundColor: "rgba(0, 188, 212, 0.85)",
      padding: 8,
      cornerRadius: 6,
      titleFont: {
        size: 12,
        family: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
      },
      bodyFont: {
        size: 16,
        family: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
        weight: "bold",
      },
      callbacks: {
        title: () => "",
        label: (context) => `值: ${context.parsed.y.toFixed(3)}`,
      },
    },
    legend: {
      labels: {
        font: {
          size: 14,
          family: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
        },
        color: "#00bcd4",
      },
    },
  },
  scales: {
    x: {
      display: false,   
    },
    y: {
      title: { display: true, text: "Value" },
      ticks: {
        color: "#00bcd4",
        font: {
          size: 14,
          family: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
        },
      },
      grid: { color: "#2f3241" },
    },
  },
};

const mainChart = new Chart(mainCtx, {
  type: "line",
  data: { labels: [], datasets: [] },
  options: commonChartOptions,
});

function updateMainRange() {
  const maxPts = parseInt(document.getElementById("mainMaxPts").value) || 100;
  mainChart._maxPoints = maxPts;
}

function updateCharts() {
  const showMulti = document.getElementById("multiLineChart").checked;
  const selected = Array.from(
    document.querySelectorAll(
      '.chart-select-controls input[type="checkbox"]:checked'
    )
  )
    .map((cb) => cb.dataset.key)
    .filter(Boolean);

  const container = document.getElementById("individualCharts");
  container.innerHTML = "";
  individualCharts = {};
  individualRanges = {};

  if (showMulti) {
    mainChart.data.datasets = selected.map((key) => ({
      label: chartMap[key]?.label || key,
      data: [],
      fill: false,
    }));
  } else {
    mainChart.data.datasets = [];
  }
  mainChart.update();

  selected.forEach((key) => {
    const box = document.createElement("div");
    box.className = "chart-box";
    const title = document.createElement("h4");
    title.textContent = chartMap[key]?.label || key;
    box.appendChild(title);

    const rdiv = document.createElement("div");
    rdiv.className = "range-controls";
    rdiv.innerHTML = `
      <label><input type="checkbox" class="childEnable" /> 固定范围</label>
      min:<input type="number" class="childMin" value="0" step="0.1" />
      max:<input type="number" class="childMax" value="1" step="0.1" />
      <button type="button" class="applyRange">应用</button>
    `;
    box.appendChild(rdiv);

    const canvas = document.createElement("canvas");
    canvas.width = 440;
    canvas.height = 280;
    box.appendChild(canvas);
    container.appendChild(box);

    const ctx = canvas.getContext("2d");
    const chart = new Chart(ctx, {
      type: "line",
      data: {
        labels: [],
        datasets: [
          {
            label: chartMap[key]?.label || key,
            data: [],
            fill: false,
          },
        ],
      },
      options: commonChartOptions,
    });
    individualCharts[key] = chart;

    const applyBtn = rdiv.querySelector(".applyRange");
    applyBtn.addEventListener("click", () => {
      const enabled = rdiv.querySelector(".childEnable").checked;
      const minVal = parseFloat(rdiv.querySelector(".childMin").value);
      const maxVal = parseFloat(rdiv.querySelector(".childMax").value);
      chart.options.scales.y.min = enabled ? minVal : undefined;
      chart.options.scales.y.max = enabled ? maxVal : undefined;
      chart.update("none");
    });
  });
}

async function fetchDataAndUpdateCharts() {
  try {
    const res = await fetch("/data");
    const json = await res.json();
    const time = json.time;
    if (!time) return;

    const maxPts = mainChart._maxPoints || 100;
    const start = time.length > maxPts ? time.length - maxPts : 0;
    const slicedTime = time.slice(start);

    if (document.getElementById("multiLineChart").checked) {
      mainChart.data.labels = slicedTime;
      const keys = Object.keys(individualCharts);
      mainChart.data.datasets.forEach((ds, i) => {
        const key = keys[i];
        ds.data = json[key]?.slice(start) || [];
      });

      const allValues = mainChart.data.datasets.flatMap(ds => ds.data);
      if (allValues.length > 0) {
        const minVal = Math.min(...allValues);
        const maxVal = Math.max(...allValues);
        const padding = (maxVal - minVal) * 0.1 || 1; // 至少留1的余量
        mainChart.options.scales.y.min = minVal - padding;
        mainChart.options.scales.y.max = maxVal + padding;
      }
      mainChart.update();
    }

    Object.entries(individualCharts).forEach(([key, ch]) => {
      ch.data.labels = slicedTime;
      ch.data.datasets[0].data = json[key]?.slice(start) || [];

      // 每个子图也加 margin，避免线条贴边
      const arr = ch.data.datasets[0].data;
      if (arr.length > 0) {
        const minVal = Math.min(...arr);
        const maxVal = Math.max(...arr);
        const padding = (maxVal - minVal) * 0.1 || 1;
        ch.options.scales.y.min = minVal - padding;
        ch.options.scales.y.max = maxVal + padding;
      }
      ch.update();
    });
  } catch (e) {
    console.error("fetch error:", e);
  }
}
