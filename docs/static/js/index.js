(function () {
  "use strict";

  function renderMath() {
    if (typeof window.renderMathInElement !== "function") {
      return;
    }
    window.renderMathInElement(document.body, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "\\(", right: "\\)", display: false },
      ],
      throwOnError: false,
    });
  }

  async function copyText(text) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch (error) {
      const area = document.createElement("textarea");
      area.value = text;
      area.setAttribute("readonly", "");
      area.style.position = "fixed";
      area.style.opacity = "0";
      document.body.appendChild(area);
      area.select();
      const ok = document.execCommand("copy");
      area.remove();
      return ok;
    }
  }

  function flash(button, label) {
    const original = button.dataset.label || button.innerHTML;
    button.dataset.label = original;
    button.textContent = label;
    clearTimeout(Number(button.dataset.timer));
    button.dataset.timer = String(
      setTimeout(() => {
        button.innerHTML = original;
      }, 1600),
    );
  }

  function initCopyButtons() {
    for (const button of document.querySelectorAll("[data-copy]")) {
      button.addEventListener("click", async () => {
        const source = document.getElementById(button.dataset.copy);
        if (source) {
          flash(button, (await copyText(source.textContent)) ? "Copied" : "Press Ctrl+C");
        }
      });
    }
  }

  // Figure state lives in the URL hash as #<figure-id>?<query>, so any view can be linked.
  function parseHash() {
    const raw = decodeURIComponent(location.hash.slice(1));
    const cut = raw.indexOf("?");
    return cut < 0 ? { id: raw, query: "" } : { id: raw.slice(0, cut), query: raw.slice(cut + 1) };
  }

  function hashFor(id, query) {
    return `#${id}${query ? `?${query}` : ""}`;
  }

  function initFigures() {
    const F = window.TLSFigures;
    const P = window.TLSPlot;
    if (!F || !P) {
      return;
    }
    for (const ramp of document.querySelectorAll("[data-ramp]")) {
      ramp.style.background = P.gradientCss();
    }
    const target = parseHash();
    const figures = {};
    let hashTimer = 0;
    for (const [id, init] of [
      ["real-tokens", F.initRealTokens],
      ["real-panels", F.initRealPanels],
      ["mano", F.initMano],
    ]) {
      const root = document.getElementById(id);
      if (!root) {
        continue;
      }
      try {
        figures[id] = init(root, {
          initial: target.id === id ? target.query : "",
          onChange(query) {
            clearTimeout(hashTimer);
            hashTimer = setTimeout(() => history.replaceState(null, "", hashFor(id, query)), 300);
          },
        });
      } catch (error) {
        console.error(`Figure ${id} failed`, error);
      }
    }

    for (const button of document.querySelectorAll("[data-action]")) {
      const root = button.closest("figure[id]");
      const figure = root && figures[root.id];
      if (!figure) {
        button.hidden = true;
        continue;
      }
      button.addEventListener("click", async () => {
        if (button.dataset.action === "link") {
          const hash = hashFor(root.id, figure.query());
          history.replaceState(null, "", hash);
          flash(button, (await copyText(`${location.origin}${location.pathname}${hash}`)) ? "Link copied" : "Copy the address bar");
        } else if (button.dataset.action === "svg") {
          P.downloadSvg(figure.svgs(), `${root.id}.svg`, figure.title());
        }
      });
    }

    if (target.query && figures[target.id]) {
      requestAnimationFrame(() => document.getElementById(target.id).scrollIntoView({ block: "start" }));
    }
  }

  function start() {
    renderMath();
    initCopyButtons();
    initFigures();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
