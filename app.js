"use strict";

const STORAGE_KEY = "attributionGalleryState";
const MAX_COLUMNS = 5;
const MIN_CARD_WIDTH = 300;

const elements = {
  page: document.querySelector(".page"),
  modelSelect: document.getElementById("model-select"),
  sampleSelect: document.getElementById("sample-select"),
  methodsList: document.getElementById("methods-list"),
  methodsSelectAll: document.getElementById("methods-select-all"),
  methodsClear: document.getElementById("methods-clear"),
  cards: document.getElementById("cards"),
  status: document.getElementById("status"),
};

const state = {
  manifest: null,
  model: null,
  sample: null,
  methods: [],
  availableMethods: [],
};

let eventsBound = false;
let cardsObserver = null;
let singleRowMinHeight = 0;
const iframeObservers = new WeakMap();

document.addEventListener("DOMContentLoaded", init);

async function init() {
  setStatus("Loading manifest...");
  const manifest = await loadManifest();
  if (!manifest || !manifest.models) {
    setStatus(
      "manifest.json is missing or invalid. Run tools/build_manifest.py.",
      "error"
    );
    setControlsDisabled(true);
    return;
  }

  if (Object.keys(manifest.models).length === 0) {
    setStatus(
      "manifest.json is empty. Add attribution files and rebuild the manifest.",
      "error"
    );
    setControlsDisabled(true);
    return;
  }

  state.manifest = manifest;
  const hydrated = hydrateState();
  if (!hydrated) {
    setStatus(
      "No samples found for the selected model. Check the manifest content.",
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

function hydrateState() {
  const models = listModels();
  if (!models.length) {
    return false;
  }

  const urlState = getUrlState();
  const storedState = loadStoredState();

  state.model = chooseFromList(
    [urlState.model, storedState.model],
    models,
    models[0]
  );

  const samples = listSamples(state.model);
  if (!samples.length) {
    return false;
  }

  state.sample = chooseFromList(
    [urlState.sample, storedState.sample],
    samples,
    samples[0]
  );

  state.availableMethods = listMethods(state.model, state.sample);
  state.methods = chooseMethods(
    urlState.methods,
    storedState.methods,
    state.availableMethods
  );

  if (state.methods.length === 0) {
    state.methods = state.availableMethods.slice();
  }

  return true;
}

function bindEvents() {
  if (eventsBound) {
    return;
  }
  eventsBound = true;

  elements.modelSelect.addEventListener("change", () => {
    state.model = elements.modelSelect.value;
    state.sample = null;
    refreshAfterSelection();
  });

  elements.sampleSelect.addEventListener("change", () => {
    state.sample = elements.sampleSelect.value;
    refreshAfterSelection();
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
  const models = listModels();
  if (!models.includes(state.model)) {
    state.model = models[0];
  }

  const samples = listSamples(state.model);
  state.sample = chooseFromList([state.sample], samples, samples[0]);

  updateControls();
  state.availableMethods = listMethods(state.model, state.sample);
  state.methods = chooseMethods([], state.methods, state.availableMethods);

  if (state.methods.length === 0) {
    state.methods = state.availableMethods.slice();
  }

  renderMethodsList();
  renderCards();
  syncState();
}

function updateControls() {
  populateSelect(elements.modelSelect, listModels(), state.model);
  populateSelect(elements.sampleSelect, listSamples(state.model), state.sample);
}

function renderMethodsList() {
  elements.methodsList.innerHTML = "";

  if (!state.availableMethods.length) {
    return;
  }

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
    text.textContent = prettifyMethodName(method);

    item.appendChild(checkbox);
    item.appendChild(text);
    elements.methodsList.appendChild(item);
  });
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
  renderCards();
  syncState();
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
  elements.cards.style.setProperty("--cards-columns", String(columns));
  updateSingleRowMinHeight(columns);
}

function updateSingleRowMinHeight(columns) {
  if (!state.methods.length || columns <= 0) {
    setSingleRowMinHeight(0);
    return;
  }

  const rows = Math.ceil(state.methods.length / columns);
  if (rows !== 1) {
    setSingleRowMinHeight(0);
    return;
  }

  const frameGap = getFrameGapValue();
  const cardsTop = elements.cards.getBoundingClientRect().top;
  const headerHeight = getCardHeaderHeight();
  const chromeHeight = getCardChromeHeight();
  const availableCardHeight = window.innerHeight - frameGap - cardsTop;
  const nextMinHeight = Math.max(
    0,
    Math.floor(availableCardHeight - headerHeight - chromeHeight)
  );
  setSingleRowMinHeight(nextMinHeight);
}

function setSingleRowMinHeight(value) {
  const nextValue = Number.isFinite(value) ? Math.max(0, value) : 0;
  if (nextValue === singleRowMinHeight) {
    return;
  }
  singleRowMinHeight = nextValue;
  elements.cards.style.setProperty("--single-row-min-height", `${nextValue}px`);
  updateIframeHeights();
}

function updateIframeHeights() {
  elements.cards.querySelectorAll("iframe").forEach((iframe) => {
    resizeIframeToContent(iframe);
  });
}

function getFrameGapValue() {
  if (!elements.page) {
    return 0;
  }
  const raw = getComputedStyle(elements.page).paddingBottom;
  const parsed = Number.parseFloat(raw);
  return Number.isFinite(parsed) ? parsed : 0;
}

function getCardHeaderHeight() {
  const header = elements.cards.querySelector(".card-header");
  return header ? header.getBoundingClientRect().height : 0;
}

function getCardChromeHeight() {
  const card = elements.cards.querySelector(".card");
  if (!card) {
    return 0;
  }
  const cardStyles = getComputedStyle(card);
  const cardBorderTop = Number.parseFloat(cardStyles.borderTopWidth) || 0;
  const cardBorderBottom = Number.parseFloat(cardStyles.borderBottomWidth) || 0;
  const frameWrap = card.querySelector(".frame-wrap");
  if (!frameWrap) {
    return cardBorderTop + cardBorderBottom;
  }
  const frameStyles = getComputedStyle(frameWrap);
  const frameBorderTop = Number.parseFloat(frameStyles.borderTopWidth) || 0;
  const frameBorderBottom = Number.parseFloat(frameStyles.borderBottomWidth) || 0;
  return cardBorderTop + cardBorderBottom + frameBorderTop + frameBorderBottom;
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
  const title = document.createElement("div");
  title.className = "card-title";
  title.textContent = prettifyMethodName(method);

  headerText.appendChild(title);

  const link = document.createElement("a");
  link.className = "card-link";
  link.textContent = "Open in new tab";
  link.href = buildAttributionUrl(method);
  link.target = "_blank";
  link.rel = "noopener";

  header.appendChild(headerText);
  header.appendChild(link);

  const frameWrap = document.createElement("div");
  frameWrap.className = "frame-wrap";

  const iframe = document.createElement("iframe");
  iframe.loading = "lazy";
  iframe.title = `${method} visualization`;
  attachIframeAutoResize(iframe);

  const fallback = document.createElement("div");
  fallback.className = "frame-fallback";
  fallback.innerHTML =
    "<strong>Unable to load this visualization.</strong>" +
    "<p>Check that the HTML file exists in the attributions folder.</p>";

  const fallbackLink = document.createElement("a");
  fallbackLink.className = "card-link";
  fallbackLink.textContent = "Open in new tab";
  fallbackLink.href = buildAttributionUrl(method);
  fallbackLink.target = "_blank";
  fallbackLink.rel = "noopener";

  fallback.appendChild(fallbackLink);

  frameWrap.appendChild(iframe);
  frameWrap.appendChild(fallback);

  card.appendChild(header);
  card.appendChild(frameWrap);

  const url = buildAttributionUrl(method);
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

  return card;
}

function attachIframeAutoResize(iframe) {
  iframe.addEventListener("load", () => {
    resizeIframeToContent(iframe);
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

function resizeIframeToContent(iframe) {
  const minHeight = singleRowMinHeight || 0;
  let contentHeight = 0;
  try {
    const doc = iframe.contentDocument;
    if (!doc) {
      if (minHeight) {
        const minHeightValue = `${minHeight}px`;
        if (iframe.style.height !== minHeightValue) {
          iframe.style.height = minHeightValue;
        }
      }
      return;
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

  const nextHeight = Math.max(contentHeight, minHeight);
  if (nextHeight) {
    const nextHeightValue = `${nextHeight}px`;
    if (iframe.style.height !== nextHeightValue) {
      iframe.style.height = nextHeightValue;
    }
  }
}

function buildAttributionUrl(method) {
  return `./assets/${state.model}/${state.sample}/${method}`;
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
  const withoutExt = fileName.replace(/\.html$/i, "");
  const pretty = withoutExt.replace(/[_-]+/g, " ").replace(/\s+/g, " ").trim();
  return pretty || fileName;
}

function listModels() {
  return Object.keys(state.manifest.models || {}).sort();
}

function listSamples(model) {
  const modelData = state.manifest.models[model];
  if (!modelData || !modelData.samples) {
    return [];
  }
  return Object.keys(modelData.samples).sort();
}

function listMethods(model, sample) {
  const modelData = state.manifest.models[model];
  if (!modelData || !modelData.samples) {
    return [];
  }
  const sampleData = modelData.samples[sample];
  if (!sampleData || !Array.isArray(sampleData.methods)) {
    return [];
  }
  return sampleData.methods.slice();
}

function populateSelect(select, options, selectedValue) {
  select.innerHTML = "";
  options.forEach((option) => {
    const item = document.createElement("option");
    item.value = option;
    item.textContent = option;
    item.selected = option === selectedValue;
    select.appendChild(item);
  });
  select.disabled = options.length === 0;
}

function chooseFromList(candidates, options, fallback) {
  for (const candidate of candidates) {
    if (candidate && options.includes(candidate)) {
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
    model: params.get("model"),
    sample: params.get("sample"),
    methods: methodsParam
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean),
  };
}

function syncState() {
  updateUrl();
  saveStoredState();
}

function updateUrl() {
  const params = new URLSearchParams();
  if (state.model) {
    params.set("model", state.model);
  }
  if (state.sample) {
    params.set("sample", state.sample);
  }
  if (state.methods.length) {
    params.set("methods", state.methods.join(","));
  }
  const query = params.toString();
  const newUrl = query ? `${window.location.pathname}?${query}` : window.location.pathname;
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
      model: state.model,
      sample: state.sample,
      methods: state.methods,
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
  elements.modelSelect.disabled = disabled;
  elements.sampleSelect.disabled = disabled;
  elements.methodsSelectAll.disabled = disabled;
  elements.methodsClear.disabled = disabled;
}
