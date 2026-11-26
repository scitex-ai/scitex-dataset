// SciTeX Dataset app: copy buttons and the fetch-progress poller.
// The pages work without this script (forms post and redirect); with it,
// "Record and fetch" runs as a background job with a live progress bar.

const csrf = () => document.querySelector('meta[name="csrf-token"]')?.content || "";

function wireCopyButtons() {
  for (const button of document.querySelectorAll("[data-copy]")) {
    button.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(button.dataset.copy);
        const label = button.textContent;
        button.textContent = "Copied";
        setTimeout(() => (button.textContent = label), 1200);
      } catch (error) {
        button.textContent = "Copy failed";
      }
    });
  }
}

function showProgress(box, job) {
  box.hidden = false;
  const fill = box.querySelector("[data-dataset-progress-fill]");
  const text = box.querySelector("[data-dataset-progress-text]");
  const percent = job.percent ?? 0;
  fill.style.width = `${Math.max(2, Math.min(100, percent))}%`;
  text.textContent = `${job.state}: ${job.message || ""}${job.percent != null ? ` (${percent}%)` : ""}`;
  box.dataset.state = job.state;
}

const API = "/api/v1/dataset";

async function api(method, path, body) {
  const headers = { "X-CSRFToken": csrf(), Accept: "application/json" };
  const init = { method, credentials: "same-origin", headers };
  if (body) {
    headers["Content-Type"] = "application/json";
    headers["Idempotency-Key"] = crypto.randomUUID();
    init.body = JSON.stringify(body);
  }
  const response = await fetch(`${API}${path}`, init);
  const data = response.status === 204 ? {} : await response.json();
  return { ok: response.ok, status: response.status, data };
}

async function poll(jobId, project, box) {
  for (;;) {
    const { ok, data } = await api("GET", `/jobs/${encodeURIComponent(jobId)}?project=${encodeURIComponent(project)}`);
    if (!ok) {
      showProgress(box, { state: "failed", message: data.detail || "job unavailable", percent: 0 });
      return;
    }
    showProgress(box, data.data);
    if (!["queued", "running"].includes(data.data.state)) {
      if (data.data.state === "done") setTimeout(() => window.location.reload(), 800);
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 700));
  }
}

// "Record and fetch" through the REST API (the same operation programs and
// agents call); the form post remains the no-JavaScript path.
function wireFetch() {
  const form = document.querySelector("[data-dataset-use-form]");
  const button = form?.querySelector("[data-dataset-fetch]");
  if (!form || !button) return;
  const box = form.querySelector("[data-dataset-progress]");
  const { datasetId, project } = form.dataset;
  button.addEventListener("click", async (event) => {
    event.preventDefault();
    button.disabled = true;
    showProgress(box, { state: "queued", message: "asking the host for storage", percent: 0 });
    const localDir = form.querySelector("[name=local_dir]").value;
    const { ok, data } = await api("POST", `/project-datasets/${encodeURIComponent(datasetId)}/fetch`, {
      project,
      local_dir: localDir,
    });
    if (!ok) {
      showProgress(box, { state: "refused", message: data.detail || data.title, percent: 0 });
      button.disabled = false;
      return;
    }
    await poll(data.job.id, project, box);
    button.disabled = false;
  });
}

wireCopyButtons();
wireFetch();
