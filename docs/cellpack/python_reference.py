"""Render Python declarations without importing the extension or optional packages."""
from __future__ import annotations

import ast
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'src/pycellpack'
# Public entry points are grouped by task, including classes re-exported by PyO3.
GROUPS = {
    'storage': {
        'cellpack/_store.py': 'Store read write compress',
        'cellpack/_file.py': 'File Writer ReadOptions ReadSummary ReadSelection',
        'cellpack/_types.py': 'CSR',
        '_cellpack.pyi': 'GeneMap ReadWorkspace',
        'io/_cpk.py': 'open_cpk read_cpk iter_cpk write_cpk cpk_info CpkInfo',
    },
    'dataloader': {'dataloader/_dataset.py': 'ScDataset', 'dataloader/_accesses.py': 'Accesses', 'dataloader/_loader.py': 'ScDataLoader'},
    'loader': {'loader/_source.py': 'DataSource', 'loader/_schema.py': 'OutputSchema allowed', 'loader/_config.py': 'LoaderConfig IoBackend', 'loader/_loader.py': 'Loader load'},
    'access-functions': {'loader/access_fn/_access_fn.py': 'AccessFn', '_cellpack.pyi': 'Accesses', 'loader/access_fn/_numba.py': 'numba_bindings'},
    'rings': {'_cellpack.pyi': 'RingLayout Ring Batch LeaseObserver Consumer Run'},
    'planning': {'loader/opt/_config.py': 'PlanOptions CostModel InputCost RowCost OutputCost ProcessCost', 'loader/opt/_plan.py': 'Plan PageTrace', 'loader/opt/_stats.py': 'PlanSummary OnlineStats CacheSummary Prices'},
    'process-functions': {'loader/process_fn/_process_fn.py': 'ProcessFn', 'loader/process_fn/_output.py': 'OutputLayout', 'loader/process_fn/_build.py': 'compile_library find_compiler Compiler ProcessBuildError CompilerNotFoundError ProcessLibraryError', 'loader/process_fn/presets.py': 'scgpt geneformer uce state_se'},
    'distributed': {'loader/_distributed.py': 'DistributedLoader DistributedRun', 'loader/_shared.py': 'SharedRing'},
    'statistics': {'loader/_stats.py': 'RunStats IoStats OutputStats MemoryEstimate ProfileReport SiteReport', 'loader/_config.py': 'profile_sites profile_groups', 'loader/costs/_resolve.py': 'resolve keys compose', 'loader/costs/_calibrate.py': 'calibrate', 'loader/costs/_collect.py': 'record_run', 'loader/costs/_store.py': 'CostStore'},
    'anndata': {'io/_anndata.py': 'read_h5 write_h5ad write_h5cp h5ad_to_cpk open_matrix', 'anndata/_core.py': 'AnnData'},
    'codecs': {'cellpack/scans/_types.py': 'Meta EncodeOptions Thresholds EncodeTables DecodeTables IdSpace Fill RecordLayout LayoutChoice ValueChoice PagedRecord Parts Dict ModeDict Ranks Values', 'cellpack/scans/_codec.py': 'ChunkSizes chunk_sizes encode_chunk encode_record decode_chunk decode_dense decode_csr decode_record decode_record_dense decode_parts pack_chunk packed_len record_nnz selection_nnz dense_bytes record_bound encode_workspace_words decode_workspace_bound', 'cellpack/scans/_prepared.py': 'PreparedEncoder Decoder', 'cellpack/scans/_mapping.py': 'compose_mapping cast_values map_parts map_parts_at map_dense', 'cellpack/_types.py': 'SparseRow', 'cellpack/scans/__init__.py': 'kernel_set supported_kernel_sets'},
}
# Native declarations carry types; their ownership semantics are authored here.
NOTES = {
    'Batch.region': 'Return the raw uint8 matrix for the named output, shape (rows, row_bytes). The view retains the slot lease.',
    'Batch.arrays': 'Return a dictionary of all named NumPy output views. Each view retains its slot lease.',
    'Batch.release': 'Release this Batch reference. Exported arrays remain valid and keep the slot leased until dropped.',
    'Batch.observer': 'Return a LeaseObserver for checking whether exported views still hold this slot.',
    'Run.join': 'Drain output before calling. Wait for native completion and return RunStats; validation, I/O and worker errors propagate.',
    'Run.snapshot': 'Return current RunStats without waiting for completion. Requires an open run.',
    'Run.close': 'Cancel unfinished execution, release the iteration-held Batch and wait for cleanup. Exported NumPy arrays remain valid.',
    'Run.wait': 'Acquire the specified batch for this rank, or return None past the end. Release the result explicitly or drop it.',
    'Run.wait_finished': 'Poll for completion, checking Python signals. timeout_ms must be between 0 and 1000 inclusive.',
    'Run.cancel': 'Request cancellation; close or leave the context to wait for cleanup.',
    'Run.consumer': 'Open a Consumer for another rank of this run. Iteration releases that consumer\'s previous Batch.',
    'Run.finished': 'True after success, failure or cancellation. Use succeeded to distinguish success.',
    'Run.succeeded': 'Whether native execution completed successfully; queryable after close.',
    'Ring.abort': 'Permanently abort this ring. For stopping one run, use Run.cancel or Run.close.',
    'Consumer.wait': 'Acquire a batch belonging to this rank; None past the sequence end. The caller must release acquired batches.',
    'Consumer.close': 'Release consumer state and record whether this rank consumed its complete sequence.',
    'Accesses.array': 'Snapshot a one-dimensional uint32 buffer as a native sequence, preserving order and repeats. Loader entry points separately normalize ordinary integer arrays.',
    'Accesses.replicate': 'Expand every position cell into schemas identical cell IDs, ordered position first, schema second.',
    'Accesses.offset': 'Expand every position into one cell per schema by adding its corresponding offset.',
    'Accesses.fill': 'Materialize [start, start + count) as global cell IDs. With count=None, use len(out) when out is supplied, otherwise the sequence remainder.',
    'Accesses.slice': 'Return a lazy interval of the sequence without materializing function-generated cells.',
    'Accesses.function': 'Bind a global-cell C callback address/context for length accesses. owner retains code and configuration storage.',
    'Accesses.corpus_function': 'Bind a corpus callback using per-dataset row counts to translate dataset/local-row pairs to global cell IDs.',
    'Writer.finish': 'Finalize the stream and publish an atomic destination. Returns None; reopen the path as Store to read it.',
    'Writer.abort': 'Close unfinished output and discard an atomic staging file. Returns None.',
    'Store.read_dense': 'Decode this view into a NumPy array. out must match the view shape/dtype and is returned when supplied.',
    'Store.read_csr': 'Decode this view into a NumPy-backed CSR. out accepts CSR or (indptr, indices, data) with sufficient capacity.',
    'ScDataLoader.out_ids': 'Read-only standard IDs in model output order.',
    'ScDataset.store': 'The immutable Store view supplied for this dataset.',
    'ScDataset.ids': 'Read-only uint32 standard IDs, one per visible column; uint32.max marks a dropped dimension.',
    'ScDataset.shape': 'Visible (rows, columns) of the dataset view.',
    'ScDataset.dtype': 'Stored numeric dtype. Loader output conversion is configured separately.',
}


def signature(node, name):
    args = ast.unparse(node.args)
    args = re.sub(r'^(?:self|cls)(?:, )?', '', args)
    result = '' if node.returns is None else ' -> ' + ast.unparse(node.returns)
    return f'{name}({args}){result}'


def prose(node):
    value = ast.get_docstring(node) or ''
    return re.sub(r':(?:class|func|meth|mod|attr):`([^`]+)`', r'`\1`', value)


def render(node):
    lines = [f'### {node.name}', '']
    if isinstance(node, ast.FunctionDef):
        return '\n'.join(lines + ['```python', signature(node, node.name), '```', '', prose(node), ''])
    init = next((m for m in node.body if isinstance(m, ast.FunctionDef) and m.name == '__init__'), None)
    # Constructors accepting private handles are intentionally not advertised.
    if init is not None and not any(a.arg in {'handle', 'capsule', 'shared'} for a in init.args.args):
        lines += ['```python', signature(init, node.name), '```', '']
    if prose(node):
        lines += [prose(node), '']
    fields = [ast.unparse(m) for m in node.body if isinstance(m, (ast.AnnAssign, ast.Assign)) and not ast.unparse(m).startswith('_')]
    if fields:
        lines += ['#### Fields and defaults', '', '```python', *fields, '```', '']
    seen = set()
    for m in node.body:
        if not isinstance(m, ast.FunctionDef) or m.name.startswith('_') or m.name in seen:
            continue
        seen.add(m.name)
        if any(isinstance(d, ast.Name) and d.id == 'overload' for d in m.decorator_list):
            continue
        prop = any(isinstance(d, ast.Name) and d.id == 'property' for d in m.decorator_list)
        label = f'{node.name}.{m.name}'
        lines += [f'#### {label}', '']
        if prop:
            annotation = '' if m.returns is None else ': ' + ast.unparse(m.returns)
            lines += [f'`{m.name}{annotation}` — property.', '']
        else:
            lines += ['```python', signature(m, label), '```', '']
        text = NOTES.get(label) or prose(m)
        if text:
            lines += [text, '']
    return '\n'.join(lines)


def prepare_reference(destination):
    for slug, files in GROUPS.items():
        content = []
        for file, names in files.items():
            tree = ast.parse((SOURCE / file).read_text())
            declarations = {n.name: n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
            for name in names.split():
                if name not in declarations:
                    raise ValueError(f'Missing public declaration: {file}:{name}')
                content.append(render(declarations[name]))
        (destination / f'{slug}-members.md').write_text('\n'.join(content))
