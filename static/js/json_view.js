function jsonToHtml(data, container) {
  const title = container.querySelector("h3");
  container.innerHTML = "";
  if (title) container.appendChild(title);
  const contentDiv = document.createElement("div");
  container.appendChild(contentDiv);

  function buildTree(d, parent) {
    if (typeof d !== "object" || d === null) {
      parent.textContent = String(d);
      return;
    }
    const ul = document.createElement("ul");
    ul.className = "json-tree";
    const entries = Array.isArray(d) ? d.map((v, i) => [i, v]) : Object.entries(d);
    entries.forEach(([k, v]) => {
      const li = document.createElement("li");
      if (typeof v === "object" && v !== null) {
        const details = document.createElement("details");
        details.open = true;
        const summary = document.createElement("summary");
        summary.textContent = k;
        details.appendChild(summary);
        const childUl = document.createElement("ul");
        childUl.className = "json-tree";
        buildTree(v, childUl);
        details.appendChild(childUl);
        li.appendChild(details);
      } else {
        li.textContent = `${k}: ${v}`;
      }
      ul.appendChild(li);
    });
    parent.appendChild(ul);
  }
  buildTree(data, contentDiv);
}

let lastSerialData = null, lastTargetData = null;

async function fetchAndDisplayJsonWithTree(id, url) {
  const parent = document.getElementById(id + "-container");
  const cont = document.getElementById(id);
  try {
    parent.classList.add("json-updating");
    const res = await fetch(url);
    if (!res.ok) throw new Error(res.statusText);
    const data = await res.json();

    const prev = id === "json-serial" ? lastSerialData : lastTargetData;
    if (JSON.stringify(data) !== JSON.stringify(prev)) {
      jsonToHtml(data, cont);
      if (id === "json-serial") lastSerialData = data;
      else lastTargetData = data;
    }
  } catch (e) {
    console.warn(`请求失败(${url}): ${e.message}`);
  } finally {
    parent.classList.remove("json-updating");
  }
}
