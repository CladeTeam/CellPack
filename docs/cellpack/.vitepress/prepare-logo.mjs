import { DOMParser, XMLSerializer } from '@xmldom/xmldom';
import { readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';

const numeric = new Set([
  'x', 'y', 'x1', 'x2', 'y1', 'y2', 'cx', 'cy', 'r', 'rx', 'ry', 'width', 'height',
  'opacity', 'fill-opacity', 'stroke-opacity', 'stroke-width', 'stroke-dasharray',
  'stroke-dashoffset', 'offset', 'viewBox', 'points', 'values', 'keyTimes', 'begin',
  'dur', 'stdDeviation', 'fx', 'fy', 'fr',
]);
const transforms = /^(?:(?:translate|scale|rotate|matrix|skewX|skewY)\([^)]*\)\s*)+$/;
const number = /[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?/g;
const identity = /^(?:translate\(0(?:[ ,]+0)?\)|rotate\(0(?:[ ,]+[-+.\d]+[ ,]+[-+.\d]+)?\)|scale\(1(?:[ ,]+1)?\)|matrix\(1[ ,]+0[ ,]+0[ ,]+1[ ,]+0[ ,]+0\))$/;
const animationTags = new Set(['animate', 'animateTransform', 'animateMotion', 'set']);

function compact(value) {
  const candidate = value.replace(number, (token) => {
    if (/[eE]/.test(token)) return token;
    let result = token.includes('.') ? token.replace(/0+$/, '').replace(/\.$/, '') : token;
    result = result.replace(/^(-?)0\./, '$1.');
    return result === '-0' || result === '' || result === '-' || result === '+' ? '0' : result;
  });
  // SVG permits signs and decimal points to delimit adjacent path numbers.
  // Keep the original attribute if shortening would merge those tokens.
  const before = value.match(number) || [];
  const after = candidate.match(number) || [];
  return before.length === after.length && before.every((token, index) => Number(token) === Number(after[index]))
    ? candidate
    : value;
}

function prepare(source, animated) {
  const doc = new DOMParser().parseFromString(source, 'image/svg+xml');
  const root = doc.documentElement;
  const elements = [root, ...Array.from(root.getElementsByTagName('*'))];
  const textSensitive = elements.some(node => ['text', 'tspan', 'textPath', 'foreignObject'].includes(node.localName) || node.hasAttribute('xml:space'));
  const structureSensitive = textSensitive || elements.some(node => ['style', 'script'].includes(node.localName) || node.hasAttribute('style')) || /<\?xml-stylesheet\b/.test(source);
  let duration = 0;
  function visit(node) {
    for (const child of Array.from(node.childNodes || [])) visit(child);
    if (node.nodeType !== 1) return;
    if (animationTags.has(node.localName)) {
      const seconds = (value) => Number.parseFloat(value || '0') / (value?.endsWith('ms') ? 1000 : 1);
      duration = Math.max(duration, seconds(node.getAttribute('begin')) + seconds(node.getAttribute('dur')));
    }
    for (const attr of Array.from(node.attributes)) {
      const value = attr.value;
      if ((numeric.has(attr.name) && /^[-+.\dEe\s,;%ms]+$/.test(value)) || attr.name === 'd' || ((attr.name === 'transform' || attr.name === 'gradientTransform' || attr.name === 'patternTransform') && transforms.test(value))) {
        node.setAttribute(attr.name, compact(value));
      }
    }
    // Only unwrap groups with no effective attributes and no timeline targeting
    // the group itself. Descendant graphics retain their order and inheritance.
    if (!structureSensitive && node.localName === 'g' && (node.attributes.length === 0 || (node.attributes.length === 1 && node.hasAttribute('transform') && identity.test(node.getAttribute('transform')))) && !Array.from(node.childNodes).some(child => animationTags.has(child.localName))) {
      while (node.firstChild) node.parentNode.insertBefore(node.firstChild, node);
      node.parentNode.removeChild(node);
    }
  }
  visit(root);
  if (animated) {
    if (!Number.isFinite(duration) || duration <= 0) throw new Error('Logo animation must have a finite duration');
    root.setAttribute('data-duration', String(duration));
    root.setAttribute('width', '100%');
    root.setAttribute('height', '100%');
  }
  const serialized = new XMLSerializer().serializeToString(doc);
  return textSensitive ? serialized : serialized.replace(/>\s+</g, '><');
}

export function prepareLogo(root) {
  for (const [input, output, animated] of [
    ['cellpack-animation-first-frame.svg', 'cellpack-logo-poster.svg', false],
    ['cellpack-animation.svg', 'cellpack-logo-animation.svg', true],
  ]) {
    const source = readFileSync(path.join(root, '../../assets', input), 'utf8');
    writeFileSync(path.join(root, 'public', output), prepare(source, animated));
  }
}
