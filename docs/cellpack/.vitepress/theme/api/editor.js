import { EditorState } from '@codemirror/state';
import { EditorView, keymap, lineNumbers, drawSelection } from '@codemirror/view';
import { defaultKeymap } from '@codemirror/commands';
import { rust } from '@codemirror/lang-rust';
import { syntaxHighlighting, defaultHighlightStyle, bracketMatching } from '@codemirror/language';

export function createViewer(parent, code) {
  return new EditorView({ parent, state: EditorState.create({ doc: code, extensions: [
    rust(), lineNumbers(), drawSelection(), bracketMatching(), syntaxHighlighting(defaultHighlightStyle),
    EditorState.readOnly.of(true), EditorView.editable.of(false), EditorView.lineWrapping,
    EditorView.contentAttributes.of({ 'aria-label': 'Read-only Rust example', tabindex: '0' }),
    keymap.of(defaultKeymap),
    EditorView.theme({ '&': { fontSize: '13px' }, '.cm-content': { fontFamily: 'var(--vp-font-family-mono)', padding: '18px 0' },
      '.cm-scroller': { maxHeight: '480px', overflow: 'auto' }, '.cm-gutters': { background: 'transparent', color: 'var(--vp-c-text-3)', border: 'none', paddingRight: '8px' },
      '&.cm-focused': { outline: '2px solid var(--vp-c-brand-1)', outlineOffset: '-2px' },
      '&.cm-focused .cm-selectionBackground, .cm-selectionBackground': { background: 'var(--cp-accent-soft)' },
      '.cm-tooltip': { background: 'var(--vp-c-bg-elv)', color: 'var(--vp-c-text-1)', border: '1px solid var(--vp-c-divider)' },
    }),
  ] }) });
}
