function showTab(tab) {
  document.getElementById("video-tab").style.display = tab === "video" ? "flex" : "none";
}

document.addEventListener("DOMContentLoaded", () => {
  updateCharts();
  updateMainRange();
  setInterval(() => {
    fetchDataAndUpdateCharts();
    fetchAndDisplayJsonWithTree("json-log", "/log");
  }, 200);
});
function toggleFullscreen() {
  const container = document.querySelector('.video-container');

  if (!document.fullscreenElement) {
    container.requestFullscreen().then(() => {
      document.body.classList.add('fullscreen-mode');
    }).catch((err) => {
      alert(`无法进入全屏模式: ${err.message}`);
    });
  } else {
    document.exitFullscreen().then(() => {
      document.body.classList.remove('fullscreen-mode');
    });
  }
}

document.addEventListener('fullscreenchange', () => {
  if (!document.fullscreenElement) {
    document.body.classList.remove('fullscreen-mode');
  }
});
