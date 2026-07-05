"use strict";

const STORAGE_KEY = "attributionGalleryState";
const MAX_COLUMNS = 5;
const MIN_CARD_WIDTH = 300;
const EXPLANATIONS_ROOT = "./explanations";
const PYTHON_TOKEN_PATTERN =
  /(\"\"\"[\s\S]*?\"\"\")|('''[\s\S]*?''')|("(?:\\.|[^"\\])*")|('(?:\\.|[^'\\])*')|(#.*$)|\b(False|None|True|and|as|assert|async|await|break|class|continue|def|del|elif|else|except|finally|for|from|global|if|import|in|is|lambda|nonlocal|not|or|pass|raise|return|try|while|with|yield)\b|(\b\d+(?:\.\d+)?\b)/gm;

const elements = {
  page: document.querySelector(".page"),
  taskSelect: document.getElementById("task-select"),
  modelSelect: document.getElementById("model-select"),
  typeSelect: document.getElementById("type-select"),
  scopeSelect: document.getElementById("scope-select"),
  sampleSelect: document.getElementById("sample-select"),
  metricSelect: document.getElementById("metric-select"),
  metricSortToggle: document.getElementById("metric-sort-toggle"),
  metricSortIcon: document.querySelector(".metric-sort-icon"),
  methodsSummary: document.querySelector(".methods-dropdown summary"),
  methodsList: document.getElementById("methods-list"),
  methodsSelectAll: document.getElementById("methods-select-all"),
  methodsClear: document.getElementById("methods-clear"),
  cards: document.getElementById("cards"),
  status: document.getElementById("status"),
};

const state = {
  manifest: null,
  entries: [],
  models: [],
  metricsMeta: {},
  metricsSummary: [],
  task: null,
  model: null,
  type: null,
  scope: null,
  sample: null,
  methods: [],
  availableMethods: [],
  metric: null,
  sortDirection: null,
};

let eventsBound = false;
let cardsObserver = null;
let currentColumns = 1;
const iframeObservers = new WeakMap();

document.addEventListener("DOMContentLoaded", init);

async function init() {
  setStatus("Loading manifest...");
  const manifest = await loadManifest();
  const normalized = normalizeManifest(manifest);
  if (!normalized) {
    setStatus(
      "manifest.json is missing or invalid. Run scripts/build_manifest.py.",
      "error"
    );
    setControlsDisabled(true);
    return;
  }

  state.manifest = normalized.manifest;
  state.entries = normalized.entries;
  state.models = normalized.models;
  state.metricsMeta = normalized.metricsMeta || {};
  state.metricsSummary = normalized.metricsSummary || [];

  renderInterpretoVersion(normalized.manifest);

  if (!state.entries.length || !state.models.length) {
    setStatus(
      "manifest.json is empty. Add explanation files and rebuild the manifest.",
      "error"
    );
    setControlsDisabled(true);
    return;
  }

  const hydrated = hydrateState();
  if (!hydrated) {
    setStatus(
      "No explanations found for the selected model. Check the manifest content.",
      "error"
    );
    setControlsDisabled(true);
    return;
  }

  bindEvents();
  observeCardsResize();
  updateControls();
  renderMethodsList();
  renderCards();
  syncState();
  setStatus("");
}

async function loadManifest() {
  try {
    const response = await fetch("manifest.json", { cache: "no-store" });
    if (!response.ok) {
      return null;
    }
    return await response.json();
  } catch (error) {
    return null;
  }
}

function renderInterpretoVersion(manifest) {
  const badge = document.getElementById("interpreto-version");
  if (!badge) {
    return;
  }
  const version =
    manifest && typeof manifest.interpreto_version === "string"
      ? manifest.interpreto_version
      : null;
  if (!version) {
    badge.textContent = "";
    badge.hidden = true;
    return;
  }
  badge.textContent = `interpreto v${version}`;
  badge.hidden = false;
}

function normalizeManifest(manifest) {
  if (!manifest || typeof manifest !== "object") {
    return null;
  }

  if (Array.isArray(manifest.explanations)) {
    const models = normalizeModels(manifest.models);
    const taskLookup = new Map(models.map((model) => [model.id, model.task]));
    const entries = manifest.explanations
      .filter((entry) => entry && entry.model && entry.type && entry.scope)
      .map((entry) => ({
        model: entry.model,
        task:
          entry.task || taskLookup.get(entry.model) || inferModelTask(entry.model),
        type: entry.type,
        scope: entry.scope,
        sample: entry.sample || null,
        methods: Array.isArray(entry.methods) ? entry.methods.slice() : [],
        metricScores: normalizeMetricScores(entry.metric_scores),
      }))
      .filter((entry) => entry.methods.length);

    return {
      manifest,
      models: models.length ? models : deriveModelsFromEntries(entries),
      entries: sortEntries(entries),
      metricsMeta: normalizeMetricsMeta(manifest.metrics_meta),
      metricsSummary: normalizeMetricsSummary(manifest.metrics_summary),
    };
  }

  if (manifest.models && typeof manifest.models === "object") {
    return normalizeLegacyManifest(manifest);
  }

  return null;
}

function normalizeMetricScores(rawScores) {
  if (!rawScores || typeof rawScores !== "object") {
    return {};
  }
  const out = {};
  Object.entries(rawScores).forEach(([method, values]) => {
    if (!values || typeof values !== "object") {
      return;
    }
    const clean = {};
    Object.entries(values).forEach(([metric, value]) => {
      if (typeof value === "number" && Number.isFinite(value)) {
        clean[metric] = value;
      }
    });
    if (Object.keys(clean).length) {
      out[method] = clean;
    }
  });
  return out;
}

function normalizeMetricsMeta(rawMeta) {
  if (!rawMeta || typeof rawMeta !== "object") {
    return {};
  }
  const out = {};
  Object.entries(rawMeta).forEach(([name, meta]) => {
    if (!meta || typeof meta !== "object") {
      return;
    }
    const label = typeof meta.label === "string" && meta.label ? meta.label : name;
    const direction =
      meta.direction === "higher_better" || meta.direction === "lower_better"
        ? meta.direction
        : "higher_better";
    const appliesTo =
      typeof meta.applies_to === "string" ? meta.applies_to : null;
    out[name] = { label, direction, appliesTo };
  });
  return out;
}

function normalizeMetricsSummary(rawSummary) {
  if (!Array.isArray(rawSummary)) {
    return [];
  }
  return rawSummary.filter(
    (name) => typeof name === "string" && name.length > 0
  );
}

function normalizeModels(modelsData) {
  if (!modelsData || typeof modelsData !== "object") {
    return [];
  }

  return Object.entries(modelsData)
    .map(([id, meta]) => {
      const inferred = parseModelId(id);
      return {
        id,
        task:
          meta && typeof meta.task === "string" ? meta.task : inferred.task,
        dataset:
          meta && typeof meta.dataset === "string" ? meta.dataset : inferred.dataset,
      };
    })
    .sort((a, b) => a.id.localeCompare(b.id));
}

function normalizeLegacyManifest(manifest) {
  const entries = [];
  const models = [];

  Object.entries(manifest.models || {}).forEach(([modelId, modelData]) => {
    const inferred = parseModelId(modelId);
    models.push({
      id: modelId,
      task: inferred.task,
      dataset: inferred.dataset,
    });

    const samples = modelData && modelData.samples ? modelData.samples : {};
    Object.entries(samples).forEach(([sampleId, sampleData]) => {
      const methods =
        sampleData && Array.isArray(sampleData.methods)
          ? sampleData.methods.slice()
          : [];
      if (!methods.length) {
        return;
      }
      entries.push({
        model: modelId,
        task: inferred.task,
        type: "attribution",
        scope: "all-classes",
        sample: sampleId,
        methods,
        metricScores: {},
      });
    });
  });

  return {
    manifest,
    models: models.sort((a, b) => a.id.localeCompare(b.id)),
    entries: sortEntries(entries),
    metricsMeta: {},
    metricsSummary: [],
  };
}

function deriveModelsFromEntries(entries) {
  const byId = new Map();
  entries.forEach((entry) => {
    if (!entry.model) {
      return;
    }
    if (!byId.has(entry.model)) {
      const inferred = parseModelId(entry.model);
      byId.set(entry.model, {
        id: entry.model,
        task: entry.task || inferred.task,
        dataset: inferred.dataset,
      });
    }
  });
  return Array.from(byId.values()).sort((a, b) => a.id.localeCompare(b.id));
}

function sortEntries(entries) {
  return entries.slice().sort((a, b) => {
    const keyA = [
      a.model,
      a.type,
      a.scope,
      a.sample || "",
    ];
    const keyB = [
      b.model,
      b.type,
      b.scope,
      b.sample || "",
    ];
    return keyA.join("::").localeCompare(keyB.join("::"));
  });
}

function parseModelId(modelId) {
  const parts = String(modelId).split(":");
  if (parts[0] === "clf") {
    return {
      task: "classification",
      dataset: parts.length > 1 ? parts[1] : null,
    };
  }
  if (parts[0] === "gen") {
    return {
      task: "generation",
      dataset: null,
    };
  }
  return {
    task: "classification",
    dataset: null,
  };
}

function inferModelTask(modelId) {
  return parseModelId(modelId).task;
}

function hydrateState() {
  const tasks = listTasks();
  if (!tasks.length) {
    return false;
  }

  const urlState = getUrlState();
  const storedState = loadStoredState();

  state.task = chooseFromList([urlState.task, storedState.task], tasks, tasks[0]);

  const models = listModels(state.task);
  if (!models.length) {
    return false;
  }
  state.model = chooseFromList(
    [urlState.model, storedState.model],
    models,
    models[0]
  );

  const types = listTypes(state.model);
  if (!types.length) {
    return false;
  }
  state.type = chooseFromList(
    [urlState.type, storedState.type],
    types,
    types[0]
  );

  const scopes = listScopes(state.model, state.type);
  if (!scopes.length) {
    return false;
  }
  state.scope = chooseFromList(
    [urlState.scope, storedState.scope],
    scopes,
    scopes[0]
  );

  const samples = listSamples(state.model, state.type, state.scope);
  state.sample = chooseFromList(
    [urlState.sample, storedState.sample],
    samples,
    samples[0] || null
  );

  state.availableMethods = listMethodsForSelection();
  state.methods = chooseMethods(
    urlState.methods,
    storedState.methods,
    state.availableMethods
  );

  if (state.methods.length === 0) {
    state.methods = state.availableMethods.slice();
  }

  const availableMetrics = listMetricsForCurrentEntry(getCurrentEntry());
  state.metric = chooseFromList(
    [urlState.metric, storedState.metric],
    availableMetrics,
    null
  );
  state.sortDirection = chooseSortDirection(
    [urlState.sortDirection, storedState.sortDirection],
    state.metric
  );

  applyMethodSort();

  return true;
}

function applyMethodSort() {
  const entry = getCurrentEntry();
  const rawMethods = entry ? entry.methods.slice() : [];
  state.availableMethods = sortMethodsByMetric(
    rawMethods,
    entry,
    state.metric,
    state.sortDirection
  );
  const selected = new Set(state.methods);
  state.methods = state.availableMethods.filter((method) => selected.has(method));
}

function bindEvents() {
  if (eventsBound) {
    return;
  }
  eventsBound = true;

  elements.taskSelect.addEventListener("change", () => {
    state.task = elements.taskSelect.value;
    refreshAfterSelection();
  });

  elements.modelSelect.addEventListener("change", () => {
    state.model = elements.modelSelect.value;
    refreshAfterSelection();
  });

  elements.typeSelect.addEventListener("change", () => {
    state.type = elements.typeSelect.value;
    refreshAfterSelection();
  });

  elements.scopeSelect.addEventListener("change", () => {
    state.scope = elements.scopeSelect.value;
    refreshAfterSelection();
  });

  elements.sampleSelect.addEventListener("change", () => {
    state.sample = elements.sampleSelect.value;
    refreshAfterSelection();
  });

  elements.metricSelect.addEventListener("change", () => {
    const value = elements.metricSelect.value || null;
    state.metric = value;
    state.sortDirection = chooseSortDirection([null], value);
    refreshMethodSort();
  });

  elements.metricSortToggle.addEventListener("click", () => {
    if (!state.metric) {
      return;
    }
    state.sortDirection = state.sortDirection === "asc" ? "desc" : "asc";
    refreshMethodSort();
  });

  elements.methodsSelectAll.addEventListener("click", () => {
    state.methods = state.availableMethods.slice();
    renderMethodsList();
    renderCards();
    syncState();
  });

  elements.methodsClear.addEventListener("click", () => {
    state.methods = [];
    renderMethodsList();
    renderCards();
    syncState();
  });

  window.addEventListener("resize", () => {
    updateCardsLayout();
  });
}

function refreshAfterSelection() {
  const tasks = listTasks();
  state.task = chooseFromList([state.task], tasks, tasks[0]);

  const models = listModels(state.task);
  state.model = chooseFromList([state.model], models, models[0]);

  const types = listTypes(state.model);
  state.type = chooseFromList([state.type], types, types[0]);

  const scopes = listScopes(state.model, state.type);
  state.scope = chooseFromList([state.scope], scopes, scopes[0]);

  const samples = listSamples(state.model, state.type, state.scope);
  state.sample = chooseFromList([state.sample], samples, samples[0] || null);

  updateControls();

  state.availableMethods = listMethodsForSelection();
  state.methods = chooseMethods([], state.methods, state.availableMethods);

  if (state.methods.length === 0) {
    state.methods = state.availableMethods.slice();
  }

  const availableMetrics = listMetricsForCurrentEntry(getCurrentEntry());
  state.metric = chooseFromList([state.metric], availableMetrics, null);
  state.sortDirection = chooseSortDirection(
    [state.sortDirection],
    state.metric
  );

  applyMethodSort();

  renderMethodsList();
  renderCards();
  syncState();
}

function updateControls() {
  populateSelect(
    elements.taskSelect,
    listTasks(),
    state.task,
    "No tasks",
    formatTaskLabel
  );
  populateSelect(
    elements.modelSelect,
    listModels(state.task),
    state.model,
    "No models",
    formatModelLabel
  );
  populateSelect(
    elements.typeSelect,
    listTypes(state.model),
    state.type,
    "No explanations",
    formatTitleLabel
  );
  populateSelect(
    elements.scopeSelect,
    listScopes(state.model, state.type),
    state.scope,
    "No scopes",
    formatTitleLabel
  );

  const samples = listSamples(state.model, state.type, state.scope);
  populateSelect(
    elements.sampleSelect,
    samples,
    state.sample,
    samples.length ? "Select sample" : "Not applicable",
    formatSampleLabel
  );

  updateMetricControl();
}

function updateMetricControl() {
  const control = document.getElementById("metric-control");
  const hasAnyMetric =
    Array.isArray(state.metricsSummary) && state.metricsSummary.length > 0;
  if (control) {
    control.hidden = !hasAnyMetric;
  }
  if (!hasAnyMetric) {
    elements.metricSortToggle.hidden = true;
    return;
  }

  const availableMetrics = listMetricsForCurrentEntry(getCurrentEntry());
  populateSelect(
    elements.metricSelect,
    availableMetrics,
    state.metric || "",
    availableMetrics.length ? "Alphabetical" : "No metrics",
    formatMetricLabel,
    { includeEmpty: availableMetrics.length ? "Alphabetical" : null }
  );

  const hasMetric = Boolean(state.metric && availableMetrics.includes(state.metric));
  elements.metricSortToggle.hidden = !hasMetric;
  if (hasMetric) {
    const direction = state.sortDirection === "asc" ? "asc" : "desc";
    elements.metricSortIcon.textContent = direction === "asc" ? "↑" : "↓";
    const meta = state.metricsMeta[state.metric];
    const label = meta && meta.label ? meta.label : state.metric;
    elements.metricSortToggle.setAttribute(
      "aria-label",
      `Sort ${label} ${direction === "asc" ? "ascending" : "descending"}`
    );
    elements.metricSortToggle.setAttribute(
      "title",
      `Sort ${label} ${direction === "asc" ? "ascending" : "descending"}`
    );
  }
}

function formatMetricLabel(metric) {
  const meta = state.metricsMeta[metric];
  return meta && meta.label ? meta.label : formatTitleLabel(metric);
}

function getMethodScore(method, entry, metric) {
  if (!entry || !entry.metricScores || !metric) {
    return null;
  }
  const scores = entry.metricScores[method];
  if (!scores || typeof scores[metric] !== "number") {
    return null;
  }
  return scores[metric];
}

function formatMetricValue(value) {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "—";
  }
  if (value === 0) {
    return "0";
  }
  return value.toPrecision(3);
}

function renderMethodsList() {
  elements.methodsList.innerHTML = "";

  if (!state.availableMethods.length) {
    const empty = document.createElement("div");
    empty.className = "methods-empty";
    empty.textContent = "No methods available for this selection.";
    elements.methodsList.appendChild(empty);
    updateMethodsSummary();
    return;
  }

  const entry = getCurrentEntry();

  state.availableMethods.forEach((method) => {
    const item = document.createElement("label");
    item.className = "method-item";
    item.dataset.method = method;

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = method;
    checkbox.checked = state.methods.includes(method);
    checkbox.addEventListener("change", collectSelectedMethods);

    const text = document.createElement("span");
    text.className = "method-item-label";
    text.textContent = prettifyMethodName(method);

    item.appendChild(checkbox);
    item.appendChild(text);

    if (state.metric) {
      const scoreText = formatMetricValue(getMethodScore(method, entry, state.metric));
      const score = document.createElement("span");
      score.className = "method-metric";
      score.textContent = scoreText;
      item.appendChild(score);
    }

    elements.methodsList.appendChild(item);
  });

  updateMethodsSummary();
}

function collectSelectedMethods() {
  const selected = [];
  elements.methodsList
    .querySelectorAll("input[type='checkbox']")
    .forEach((input) => {
      if (input.checked) {
        selected.push(input.value);
      }
    });

  state.methods = selected;
  updateMethodsSummary();
  renderCards();
  syncState();
}

function updateMethodsSummary() {
  if (!elements.methodsSummary) {
    return;
  }
  const total = state.availableMethods.length;
  const selected = state.methods.length;
  if (!total) {
    elements.methodsSummary.textContent = "Select methods";
    return;
  }
  if (!selected) {
    elements.methodsSummary.textContent = "Select methods";
    return;
  }
  elements.methodsSummary.textContent = `Methods (${selected}/${total})`;
}

function renderCards() {
  elements.cards.innerHTML = "";

  if (!state.availableMethods.length) {
    setStatus("No methods available for this selection.", "error");
    updateCardsLayout();
    return;
  }

  if (!state.methods.length) {
    setStatus("Select at least one method to view comparisons.");
    updateCardsLayout();
    return;
  }

  setStatus("");
  const fragment = document.createDocumentFragment();

  state.methods.forEach((method, index) => {
    const card = buildCard(method, index);
    fragment.appendChild(card);
  });

  elements.cards.appendChild(fragment);
  updateCardsLayout();
}

function updateCardsLayout() {
  const desiredColumns = Math.min(
    Math.max(state.methods.length || 1, 1),
    MAX_COLUMNS
  );
  const containerWidth = elements.cards.clientWidth || 0;
  const maxByWidth = containerWidth
    ? Math.max(1, Math.floor(containerWidth / MIN_CARD_WIDTH))
    : desiredColumns;
  const columns = Math.min(desiredColumns, maxByWidth);
  currentColumns = columns;
  elements.cards.style.setProperty("--cards-columns", String(columns));
  updateIframeHeights();
}

function updateIframeHeights() {
  elements.cards.querySelectorAll("iframe").forEach((iframe) => {
    resizeIframeToContent(iframe, false);
  });
  updateRowMinHeights();
}

function updateRowMinHeights() {
  const cards = Array.from(elements.cards.querySelectorAll(".card"));
  if (!cards.length || currentColumns <= 0) {
    cards.forEach((card) => {
      card.style.removeProperty("--row-min-height");
    });
    return;
  }

  const rowMaxHeights = [];
  cards.forEach((card, index) => {
    const iframe = card.querySelector("iframe");
    const measuredHeight = getIframeMeasuredHeight(iframe);
    const rowIndex = Math.floor(index / currentColumns);
    rowMaxHeights[rowIndex] = Math.max(
      rowMaxHeights[rowIndex] || 0,
      measuredHeight
    );
  });

  cards.forEach((card, index) => {
    const rowIndex = Math.floor(index / currentColumns);
    const rowHeight = rowMaxHeights[rowIndex] || 0;
    if (rowHeight > 0) {
      card.style.setProperty("--row-min-height", `${rowHeight}px`);
    } else {
      card.style.removeProperty("--row-min-height");
    }
  });
}

function getIframeMeasuredHeight(iframe) {
  if (!iframe) {
    return 0;
  }
  const dataHeight = Number.parseFloat(iframe.dataset.contentHeight || "");
  if (Number.isFinite(dataHeight) && dataHeight > 0) {
    return dataHeight;
  }
  const inlineHeight = Number.parseFloat(iframe.style.height || "");
  if (Number.isFinite(inlineHeight) && inlineHeight > 0) {
    return inlineHeight;
  }
  const rect = iframe.getBoundingClientRect();
  return rect.height || 0;
}

function observeCardsResize() {
  if (cardsObserver || !window.ResizeObserver) {
    return;
  }
  cardsObserver = new ResizeObserver(() => {
    updateCardsLayout();
  });
  cardsObserver.observe(elements.cards);
}

function buildCard(method, index) {
  const card = document.createElement("article");
  card.className = "card";
  card.style.setProperty("--delay", `${index * 60}ms`);

  const header = document.createElement("div");
  header.className = "card-header";

  const headerText = document.createElement("div");
  headerText.className = "card-header-text";
  const title = document.createElement("div");
  title.className = "card-title";
  title.textContent = prettifyMethodName(method);

  headerText.appendChild(title);

  if (state.metric) {
    const entry = getCurrentEntry();
    const scoreText = formatMetricValue(getMethodScore(method, entry, state.metric));
    const meta = state.metricsMeta[state.metric];
    const label = meta && meta.label ? meta.label : state.metric;
    const badge = document.createElement("span");
    badge.className = "metric-badge";
    badge.textContent = `${label}: ${scoreText}`;
    headerText.appendChild(badge);
  }

  const headerActions = document.createElement("div");
  headerActions.className = "card-actions";

  const codeButton = document.createElement("button");
  codeButton.type = "button";
  codeButton.className = "button button--ghost card-action";
  codeButton.textContent = "View code";
  codeButton.setAttribute("aria-pressed", "false");

  header.appendChild(headerText);
  header.appendChild(headerActions);
  headerActions.appendChild(codeButton);

  const frameWrap = document.createElement("div");
  frameWrap.className = "frame-wrap";

  const iframe = document.createElement("iframe");
  iframe.loading = "lazy";
  iframe.title = `${method} visualization`;
  iframe.setAttribute("scrolling", "no");
  attachIframeAutoResize(iframe);

  const fallback = document.createElement("div");
  fallback.className = "frame-fallback";
  fallback.innerHTML =
    "<strong>Unable to load this visualization.</strong>" +
    "<p>Check that the HTML file exists in the explanations folder.</p>";

  const codePanel = document.createElement("div");
  codePanel.className = "code-panel";
  codePanel.setAttribute("aria-hidden", "true");
  codePanel.setAttribute("role", "region");
  codePanel.setAttribute(
    "aria-label",
    `${prettifyMethodName(method)} python snippet`
  );

  const codePanelId = `code-panel-${index}`;
  codePanel.id = codePanelId;
  codeButton.setAttribute("aria-controls", codePanelId);
  codeButton.setAttribute("aria-expanded", "false");

  const codeHeader = document.createElement("div");
  codeHeader.className = "code-panel-header";

  const codeHeading = document.createElement("div");
  codeHeading.className = "code-panel-heading";

  const codeTitle = document.createElement("div");
  codeTitle.className = "code-panel-title";
  codeTitle.textContent = "Python snippet";

  const codeMeta = document.createElement("div");
  codeMeta.className = "code-panel-meta";
  codeMeta.textContent = methodToCodeFilename(method);

  codeHeading.appendChild(codeTitle);
  codeHeading.appendChild(codeMeta);

  const codeActions = document.createElement("div");
  codeActions.className = "code-panel-actions";

  const copyButton = document.createElement("button");
  copyButton.type = "button";
  copyButton.className = "button button--ghost code-copy";
  copyButton.textContent = "Copy";

  codeActions.appendChild(copyButton);

  codeHeader.appendChild(codeHeading);
  codeHeader.appendChild(codeActions);

  const codeStatus = document.createElement("div");
  codeStatus.className = "code-panel-status";
  codeStatus.setAttribute("role", "status");
  codeStatus.setAttribute("aria-live", "polite");

  const codePre = document.createElement("pre");
  codePre.className = "code-block";
  codePre.style.display = "none";

  const codeElement = document.createElement("code");
  codeElement.className = "language-python";
  codePre.appendChild(codeElement);

  codePanel.appendChild(codeHeader);
  codePanel.appendChild(codeStatus);
  codePanel.appendChild(codePre);

  frameWrap.appendChild(iframe);
  frameWrap.appendChild(fallback);
  frameWrap.appendChild(codePanel);

  card.appendChild(header);
  card.appendChild(frameWrap);

  const url = buildExplanationUrl(method);
  checkFileExists(url).then((exists) => {
    if (exists) {
      iframe.src = url;
    } else {
      card.classList.add("has-error");
    }
  });

  iframe.addEventListener("error", () => {
    card.classList.add("has-error");
  });

  const codeUrl = buildCodeUrl(method);
  let codeLoaded = false;
  let codeLoading = false;
  let codeText = "";
  let copyResetTimer = null;

  const setCodeStatus = (message, isError = false) => {
    codeStatus.textContent = message;
    codeStatus.classList.toggle("is-error", Boolean(isError));
    if (message) {
      codePre.style.display = "none";
      return;
    }
    if (codeLoaded) {
      codePre.style.display = "block";
    }
  };

  const ensureCodeLoaded = async () => {
    if (codeLoaded || codeLoading) {
      return;
    }
    if (!codeUrl) {
      setCodeStatus("Python snippet unavailable for this method.", true);
      return;
    }
    codeLoading = true;
    setCodeStatus("Loading snippet...");
    try {
      const response = await fetch(codeUrl, { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`Failed to load snippet: ${response.status}`);
      }
      const text = await response.text();
      codeText = text;
      codeElement.innerHTML = highlightPython(text);
      codeLoaded = true;
      setCodeStatus("");
    } catch (error) {
      setCodeStatus("Python snippet unavailable for this method.", true);
    } finally {
      codeLoading = false;
    }
  };

  const setCopyButtonState = (label, state) => {
    if (copyResetTimer) {
      window.clearTimeout(copyResetTimer);
      copyResetTimer = null;
    }
    copyButton.textContent = label;
    copyButton.classList.toggle("is-success", state === "success");
    copyButton.classList.toggle("is-error", state === "error");
    if (state) {
      copyResetTimer = window.setTimeout(() => {
        copyButton.textContent = "Copy";
        copyButton.classList.remove("is-success", "is-error");
      }, 1600);
    }
  };

  const copyWithFallback = async (text) => {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    }
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "true");
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.appendChild(textarea);
    textarea.select();
    let copied = false;
    try {
      copied = document.execCommand("copy");
    } catch (error) {
      copied = false;
    }
    document.body.removeChild(textarea);
    return copied;
  };

  copyButton.addEventListener("click", async () => {
    if (!codeLoaded) {
      await ensureCodeLoaded();
    }
    if (!codeLoaded || !codeText) {
      setCopyButtonState("Unavailable", "error");
      return;
    }
    try {
      const copied = await copyWithFallback(codeText);
      if (copied) {
        setCopyButtonState("Copied", "success");
      } else {
        setCopyButtonState("Copy failed", "error");
      }
    } catch (error) {
      setCopyButtonState("Copy failed", "error");
    }
  });

  codeButton.addEventListener("click", () => {
    const isActive = card.classList.toggle("show-code");
    codeButton.textContent = isActive ? "Hide code" : "View code";
    codeButton.setAttribute("aria-pressed", String(isActive));
    codeButton.setAttribute("aria-expanded", String(isActive));
    codePanel.setAttribute("aria-hidden", String(!isActive));
    if (isActive) {
      ensureCodeLoaded();
    }
  });

  return card;
}

function attachIframeAutoResize(iframe) {
  iframe.addEventListener("load", () => {
    resizeIframeToContent(iframe);
    suppressIframeScrollbars(iframe);
    const doc = iframe.contentDocument;
    if (!doc || !window.ResizeObserver) {
      return;
    }
    const existingObserver = iframeObservers.get(iframe);
    if (existingObserver) {
      existingObserver.disconnect();
    }
    const observer = new ResizeObserver(() => {
      resizeIframeToContent(iframe);
    });
    if (doc.documentElement) {
      observer.observe(doc.documentElement);
    }
    if (doc.body) {
      observer.observe(doc.body);
    }
    iframeObservers.set(iframe, observer);
  });
}

function suppressIframeScrollbars(iframe) {
  try {
    const doc = iframe.contentDocument;
    if (!doc) {
      return;
    }
    if (doc.documentElement) {
      doc.documentElement.style.overflow = "hidden";
    }
    if (doc.body) {
      doc.body.style.overflow = "hidden";
    }
  } catch (error) {
    // Ignore cross-origin access issues.
  }
}

function resizeIframeToContent(iframe, shouldUpdateRows = true) {
  let contentHeight = 0;
  try {
    const doc = iframe.contentDocument;
    if (!doc) {
      return 0;
    }
    const body = doc.body;
    const html = doc.documentElement;
    contentHeight = Math.max(
      body ? body.scrollHeight : 0,
      body ? body.offsetHeight : 0,
      html ? html.clientHeight : 0,
      html ? html.scrollHeight : 0,
      html ? html.offsetHeight : 0
    );
  } catch (error) {
    // Ignore cross-origin access issues.
  }

  if (contentHeight) {
    const nextHeightValue = `${contentHeight}px`;
    if (iframe.style.height !== nextHeightValue) {
      iframe.style.height = nextHeightValue;
    }
    iframe.dataset.contentHeight = String(contentHeight);
  }
  if (shouldUpdateRows) {
    updateRowMinHeights();
  }
  return contentHeight;
}

function methodToCodeFilename(method) {
  const value = String(method || "");
  if (!value) {
    return "";
  }
  const lower = value.toLowerCase();
  if (lower.endsWith(".py")) {
    return value;
  }
  if (lower.endsWith(".html")) {
    return `${value.slice(0, -5)}.py`;
  }
  return `${value}.py`;
}

function buildExplanationUrl(method) {
  const entry = getCurrentEntry();
  if (!entry) {
    return "";
  }
  const parts = [
    EXPLANATIONS_ROOT,
    entry.model,
    entry.type,
    entry.scope,
  ];
  if (entry.sample) {
    parts.push(entry.sample);
  }
  parts.push(method);
  return parts.join("/");
}

function buildCodeUrl(method) {
  const codeFile = methodToCodeFilename(method);
  if (!codeFile) {
    return "";
  }
  return buildExplanationUrl(codeFile);
}

async function checkFileExists(url) {
  try {
    const head = await fetch(url, { method: "HEAD", cache: "no-store" });
    if (head.ok) {
      return true;
    }
    if (head.status !== 405 && head.status !== 403) {
      return false;
    }
  } catch (error) {
    // Fall through to GET.
  }

  try {
    const response = await fetch(url, { method: "GET", cache: "no-store" });
    return response.ok;
  } catch (error) {
    return false;
  }
}

function prettifyMethodName(fileName) {
  return prettifyLabel(fileName);
}

function prettifyLabel(value) {
  const withoutExt = String(value || "").replace(/\.html$/i, "");
  const pretty = withoutExt.replace(/[_-]+/g, " ").replace(/\s+/g, " ").trim();
  return pretty || String(value || "");
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function highlightPython(code) {
  const escaped = escapeHtml(code);
  return escaped.replace(
    PYTHON_TOKEN_PATTERN,
    (
      match,
      tripleDouble,
      tripleSingle,
      doubleString,
      singleString,
      comment,
      keyword,
      number
    ) => {
      if (tripleDouble || tripleSingle || doubleString || singleString) {
        return `<span class="code-string">${match}</span>`;
      }
      if (comment) {
        return `<span class="code-comment">${match}</span>`;
      }
      if (keyword) {
        return `<span class="code-keyword">${match}</span>`;
      }
      if (number) {
        return `<span class="code-number">${match}</span>`;
      }
      return match;
    }
  );
}

function formatTitleLabel(value) {
  const pretty = prettifyLabel(value);
  return pretty.replace(/\b\w/g, (char) => char.toUpperCase());
}

function formatTaskLabel(task) {
  if (task === "classification") {
    return "Classification";
  }
  if (task === "generation") {
    return "Generation";
  }
  return formatTitleLabel(task);
}

function formatModelLabel(modelId) {
  return modelId;
}

function formatSampleLabel(sampleId) {
  return sampleId;
}

function listTasks() {
  const order = ["classification", "generation", "unknown"];
  const unique = new Set();
  state.models.forEach((model) => {
    unique.add(model.task || "classification");
  });
  return Array.from(unique).sort((a, b) => {
    const indexA = order.indexOf(a);
    const indexB = order.indexOf(b);
    const rankA = indexA === -1 ? 99 : indexA;
    const rankB = indexB === -1 ? 99 : indexB;
    if (rankA !== rankB) {
      return rankA - rankB;
    }
    return a.localeCompare(b);
  });
}

function listModels(task) {
  return state.models
    .filter((model) => !task || model.task === task)
    .map((model) => model.id)
    .sort();
}

function listTypes(model) {
  return uniqueValues(
    filterEntries({ model }),
    (entry) => entry.type
  );
}

function listScopes(model, type) {
  return uniqueValues(
    filterEntries({ model, type }),
    (entry) => entry.scope
  );
}

function listSamples(model, type, scope) {
  const entries = filterEntries({ model, type, scope });
  return uniqueValues(entries, (entry) => entry.sample).filter(isPresent);
}

function listMethodsForSelection() {
  const entry = getCurrentEntry();
  return entry ? entry.methods.slice() : [];
}

function listMetricsForCurrentEntry(entry) {
  if (!entry || !entry.metricScores) {
    return [];
  }
  const scoreMap = entry.metricScores;
  const scoredMethods = Object.keys(scoreMap);
  if (!scoredMethods.length) {
    return [];
  }
  const perMethodMetrics = scoredMethods.map(
    (method) => new Set(Object.keys(scoreMap[method] || {}))
  );
  const intersection = perMethodMetrics.reduce((acc, set) => {
    if (!acc) {
      return set;
    }
    const next = new Set();
    acc.forEach((name) => {
      if (set.has(name)) {
        next.add(name);
      }
    });
    return next;
  }, null);
  if (!intersection || !intersection.size) {
    return [];
  }
  // Preserve metricsSummary ordering when possible, otherwise sort.
  const ordered = [];
  state.metricsSummary.forEach((name) => {
    if (intersection.has(name)) {
      ordered.push(name);
    }
  });
  Array.from(intersection)
    .filter((name) => !ordered.includes(name))
    .sort()
    .forEach((name) => ordered.push(name));
  return ordered;
}

function chooseSortDirection(candidates, metric) {
  for (const candidate of candidates) {
    if (candidate === "asc" || candidate === "desc") {
      return candidate;
    }
  }
  if (!metric) {
    return null;
  }
  const meta = state.metricsMeta[metric];
  if (meta && meta.direction === "lower_better") {
    return "asc";
  }
  return "desc";
}

function sortMethodsByMetric(methods, entry, metric, direction) {
  if (!metric || !entry || !entry.metricScores) {
    return methods.slice().sort();
  }
  const scoreOf = (method) => {
    const scores = entry.metricScores[method];
    if (!scores || typeof scores[metric] !== "number") {
      return null;
    }
    return scores[metric];
  };
  const sign = direction === "asc" ? 1 : -1;
  return methods.slice().sort((a, b) => {
    const sa = scoreOf(a);
    const sb = scoreOf(b);
    // Unscored methods sink to the bottom regardless of direction.
    if (sa === null && sb === null) {
      return a.localeCompare(b);
    }
    if (sa === null) {
      return 1;
    }
    if (sb === null) {
      return -1;
    }
    if (sa === sb) {
      return a.localeCompare(b);
    }
    return sign * (sa - sb) < 0 ? -1 : 1;
  });
}

function refreshMethodSort() {
  const entry = getCurrentEntry();
  const rawMethods = entry ? entry.methods.slice() : [];
  state.availableMethods = sortMethodsByMetric(
    rawMethods,
    entry,
    state.metric,
    state.sortDirection
  );
  // Preserve the user's method selection but reorder to match availability.
  const selected = new Set(state.methods);
  state.methods = state.availableMethods.filter((method) => selected.has(method));
  updateMetricControl();
  renderMethodsList();
  renderCards();
  syncState();
}

function getCurrentEntry() {
  const entries = filterEntries({
    model: state.model,
    type: state.type,
    scope: state.scope,
    sample: state.sample === undefined ? null : state.sample,
  });
  return entries[0] || null;
}

function filterEntries(criteria) {
  return state.entries.filter((entry) => {
    if (criteria.task && entry.task !== criteria.task) {
      return false;
    }
    if (criteria.model && entry.model !== criteria.model) {
      return false;
    }
    if (criteria.type && entry.type !== criteria.type) {
      return false;
    }
    if (criteria.scope && entry.scope !== criteria.scope) {
      return false;
    }
    if (Object.prototype.hasOwnProperty.call(criteria, "sample")) {
      if (criteria.sample === null) {
        if (entry.sample !== null) {
          return false;
        }
      } else if (entry.sample !== criteria.sample) {
        return false;
      }
    }
    return true;
  });
}

function uniqueValues(entries, accessor) {
  const values = new Set();
  entries.forEach((entry) => {
    const value = accessor(entry);
    if (value !== null && value !== undefined) {
      values.add(value);
    }
  });
  return Array.from(values).sort();
}

function isPresent(value) {
  return value !== null && value !== undefined;
}

function populateSelect(select, options, selectedValue, emptyLabel, formatLabel, extras) {
  select.innerHTML = "";
  const includeEmpty = extras && typeof extras.includeEmpty === "string" ? extras.includeEmpty : null;

  if (!options.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = emptyLabel || "Not available";
    option.selected = true;
    option.disabled = true;
    select.appendChild(option);
    select.disabled = true;
    return;
  }

  if (includeEmpty !== null) {
    const emptyOption = document.createElement("option");
    emptyOption.value = "";
    emptyOption.textContent = includeEmpty;
    emptyOption.selected = !selectedValue;
    select.appendChild(emptyOption);
  }

  options.forEach((option) => {
    const item = document.createElement("option");
    item.value = option;
    item.textContent = formatLabel ? formatLabel(option) : option;
    item.selected = option === selectedValue;
    select.appendChild(item);
  });
  select.disabled = options.length === 0;
}

function chooseFromList(candidates, options, fallback) {
  for (const candidate of candidates) {
    if (candidate !== null && candidate !== undefined && options.includes(candidate)) {
      return candidate;
    }
  }
  return fallback;
}

function chooseMethods(urlMethods, storedMethods, availableMethods) {
  const candidates =
    Array.isArray(urlMethods) && urlMethods.length
      ? urlMethods
      : Array.isArray(storedMethods)
      ? storedMethods
      : [];

  return candidates.filter((method) => availableMethods.includes(method));
}

function getUrlState() {
  const params = new URLSearchParams(window.location.search);
  const methodsParam = params.get("methods") || "";
  return {
    task: params.get("task"),
    model: params.get("model"),
    type: params.get("type"),
    scope: params.get("scope"),
    sample: params.get("sample"),
    methods: methodsParam
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean),
    metric: params.get("metric"),
    sortDirection: params.get("sortDir"),
  };
}

function syncState() {
  updateUrl();
  saveStoredState();
}

function updateUrl() {
  const params = new URLSearchParams();
  if (state.task) {
    params.set("task", state.task);
  }
  if (state.model) {
    params.set("model", state.model);
  }
  if (state.type) {
    params.set("type", state.type);
  }
  if (state.scope) {
    params.set("scope", state.scope);
  }
  if (state.sample) {
    params.set("sample", state.sample);
  }
  if (state.methods.length) {
    params.set("methods", state.methods.join(","));
  }
  if (state.metric) {
    params.set("metric", state.metric);
  }
  if (state.sortDirection) {
    params.set("sortDir", state.sortDirection);
  }
  const query = params.toString();
  const newUrl = query
    ? `${window.location.pathname}?${query}`
    : window.location.pathname;
  window.history.replaceState({}, "", newUrl);
}

function loadStoredState() {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return {};
    }
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") {
      return {};
    }
    return parsed;
  } catch (error) {
    return {};
  }
}

function saveStoredState() {
  try {
    const payload = {
      task: state.task,
      model: state.model,
      type: state.type,
      scope: state.scope,
      sample: state.sample,
      methods: state.methods,
      metric: state.metric,
      sortDirection: state.sortDirection,
    };
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
  } catch (error) {
    // Storage may be unavailable; ignore.
  }
}

function setStatus(message, variant) {
  if (!message) {
    elements.status.textContent = "";
    elements.status.classList.remove("is-visible", "is-error");
    return;
  }
  elements.status.textContent = message;
  elements.status.classList.add("is-visible");
  elements.status.classList.toggle("is-error", variant === "error");
}

function setControlsDisabled(disabled) {
  elements.taskSelect.disabled = disabled;
  elements.modelSelect.disabled = disabled;
  elements.typeSelect.disabled = disabled;
  elements.scopeSelect.disabled = disabled;
  elements.sampleSelect.disabled = disabled;
  elements.metricSelect.disabled = disabled;
  elements.metricSortToggle.disabled = disabled;
  elements.methodsSelectAll.disabled = disabled;
  elements.methodsClear.disabled = disabled;
}
