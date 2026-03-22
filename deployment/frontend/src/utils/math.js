/**
 * Math rendering utility using KaTeX.
 */

import katex from 'katex';

/**
 * Render math in a DOM element.
 * @param {HTMLElement} element - Element to render math in
 */
export function renderMath(element) {
  if (!element) return;

  // Find and render inline math \( ... \) and $ ... $
  const inlineRegex = /\\\((.+?)\\\)|\$([^$]+?)\$/g;

  // Find and render display math \[ ... \] and $$ ... $$
  const displayRegex = /\\\[(.+?)\\\]|\$\$(.+?)\$\$/gs;

  const processNode = (node) => {
    if (node.nodeType === Node.TEXT_NODE) {
      const text = node.textContent;

      // Check for display math first
      if (displayRegex.test(text)) {
        displayRegex.lastIndex = 0;
        const parts = [];
        let lastIndex = 0;
        let match;

        while ((match = displayRegex.exec(text)) !== null) {
          if (match.index > lastIndex) {
            parts.push({ type: 'text', value: text.slice(lastIndex, match.index) });
          }
          parts.push({ type: 'display', value: match[1] || match[2] });
          lastIndex = match.index + match[0].length;
        }

        if (lastIndex < text.length) {
          parts.push({ type: 'text', value: text.slice(lastIndex) });
        }

        if (parts.length > 1 || parts[0]?.type === 'display') {
          const fragment = document.createDocumentFragment();
          parts.forEach(part => {
            if (part.type === 'display') {
              const span = document.createElement('span');
              span.className = 'katex-display';
              try {
                katex.render(part.value, span, { displayMode: true, throwOnError: false });
              } catch (e) {
                span.textContent = part.value;
              }
              fragment.appendChild(span);
            } else {
              fragment.appendChild(document.createTextNode(part.value));
            }
          });
          node.parentNode?.replaceChild(fragment, node);
          return;
        }
      }

      // Check for inline math
      if (inlineRegex.test(text)) {
        inlineRegex.lastIndex = 0;
        const parts = [];
        let lastIndex = 0;
        let match;

        while ((match = inlineRegex.exec(text)) !== null) {
          if (match.index > lastIndex) {
            parts.push({ type: 'text', value: text.slice(lastIndex, match.index) });
          }
          parts.push({ type: 'inline', value: match[1] || match[2] });
          lastIndex = match.index + match[0].length;
        }

        if (lastIndex < text.length) {
          parts.push({ type: 'text', value: text.slice(lastIndex) });
        }

        if (parts.length > 1 || parts[0]?.type === 'inline') {
          const fragment = document.createDocumentFragment();
          parts.forEach(part => {
            if (part.type === 'inline') {
              const span = document.createElement('span');
              span.className = 'katex-inline';
              try {
                katex.render(part.value, span, { displayMode: false, throwOnError: false });
              } catch (e) {
                span.textContent = part.value;
              }
              fragment.appendChild(span);
            } else {
              fragment.appendChild(document.createTextNode(part.value));
            }
          });
          node.parentNode?.replaceChild(fragment, node);
        }
      }
    } else if (node.nodeType === Node.ELEMENT_NODE && node.tagName !== 'SCRIPT') {
      // Process child nodes (create array first to avoid live collection issues)
      Array.from(node.childNodes).forEach(processNode);
    }
  };

  processNode(element);
}

/**
 * Format question text with image placeholders.
 * @param {string} text - Question text
 * @returns {string} Formatted HTML
 */
export function formatQuestionText(text) {
  return text
    .replace(/\n/g, '<br>')
    .replace(
      /\[Image:\s*(.*?)\]/gi,
      '<div class="image-note"><i class="fas fa-image"></i> <em>$1</em></div>'
    );
}
