// MathJax configuration for MkDocs Material + pymdownx.arithmatex (generic mode).
//
// The document$.subscribe(...) hook re-typesets math after Material's
// `navigation.instant` page swaps. Without it, equations only render on the
// first full page load and appear as raw \(...\) after client-side navigation.
//
// Loaded BEFORE the engine (see extra_javascript order in mkdocs.yml) so that
// window.MathJax is defined when tex-mml-chtml.js initialises.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
