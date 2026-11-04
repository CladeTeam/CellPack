"""Authored explanations supplement declarations extracted from rustdoc."""

EXAMPLES = {
    "write": {
        "title": "Write and reopen a matrix",
        "description": "Write a small matrix and inspect the decoded dense rows and CSR structure. The file becomes readable after finish succeeds.",
        "expected": "shape: 2 × 3\ndense: [0.0, 2.0, 0.0, 1.0, 0.0, 3.0]\nCSR offsets: [0, 1, 3]\nCSR columns: [1, 0, 2]",
        "symbols": ["store::CpkWriter", "store::CpkFile", "store::CpkOptions"],
    },
    "selection": {
        "title": "Gather rows and map columns",
        "description": "Rows [3, 0, 3] preserve order and repetition. The source → output map keeps source column 2 first and column 0 second.",
        "expected": "rows: [3, 0, 3]\noutput: [0.0, 7.0, 0.0, 0.0, 0.0, 7.0]\nCSR offsets: [0, 1, 1, 2]\nactual nnz: 2",
        "symbols": ["store::ReadSelection", "store::ReadWorkspace", "store::RowSelection", "store::ReadSpec", "store::ReadOptions"],
    },
    "loading": {
        "title": "Deliver ordered batches",
        "description": "Two output schemas share each requested position. Missing dimensions use the schema default; present sparse zeros stay zero.",
        "expected": "positions 0..2: panel [0.0, 8.0, NaN, 1.0, 2.0, NaN]\npositions 2..4: panel [6.0, 0.0, NaN, 0.0, 8.0, NaN]\npositions 4..5: panel [0.0, 0.0, NaN]\ndelivered: 5 positions, 10 accesses",
        "symbols": ["loader::Loader", "loader::LoaderConfig", "loader::Run", "loader::OutputSchema", "loader::Replicate", "loader::BatchLease", "loader::Ring", "loader::DataSourceBuilder"],
    },
}

ITEMS = {
    "store::CpkWriter": {
        "when": "Use for incremental writes. Declare the input layout and dtype once, append row batches, then consume the writer with finish.",
        "methods": {
            "create": {
                "parameters": {
                    "path": "Destination file. An existing file is truncated immediately.",
                    "n_cols": "Number of source columns, in elements; every appended row must have this width.",
                    "dtype": "Stored numeric type. Its byte width must match the input values.",
                    "source": "Layout of incoming rows (Dense or Csr); this does not force the encoded record layout.",
                    "options": "Encoding choices, target rows per chunk, and page size in bytes.",
                },
                "returns": "A writer owning staging buffers. The destination is not a valid stream until finish succeeds.",
            },
            "append": {"parameters": {"rows": "Borrowed row-major dense or canonical CSR input. Rows are appended in order; the writer does not retain the borrow after this call."}},
            "finish": {"returns": "Writes the remaining rows, directory, and superblock. Consumes the writer. Success does not imply fsync or atomic publication."},
        },
    },
    "store::CpkFile": {
        "when": "Use for file inspection and prepared reads. Prefer prepare_read or prepare_selection when you need typed output or reusable buffers.",
        "methods": {
            "prepare_read": {"parameters": {"rows": "Half-open file-row interval in u64 row IDs, such as 0..2."}, "returns": "Prepared metadata and output sizing. Record I/O happens when the selection is read."},
            "prepare_selection": {"parameters": {"rows": "File rows. A List preserves order and repeated IDs.", "gene_map": "Optional source-column → output-column map; one entry per source column. Use UNMAPPED to drop a column.", "columns": "Output width in elements, including zero-filled missing output positions.", "options": "Read policy and merge settings. These describe application reads, not physical device traffic."}},
            "read_dense": {"returns": "Stored-dtype bytes (`Vec<u8>`), not a `Vec<f32>`. For typed values use a prepared selection."},
        },
    },
    "store::ReadSelection": {
        "when": "Prepare once, allocate from spec(), then read into your own buffers. Separate concurrent reads need separate mutable workspaces and outputs.",
        "methods": {
            "read_dense_into": {"parameters": {"out": "Exactly spec().dense_len() typed elements; the entire output is filled.", "ws": "Exclusive reusable scratch for this call.", "workers": "Read/decode worker count; at least one worker is used."}, "returns": "Fills the caller's output. On failure discard the output because it may be partially written."},
            "read_csr_into": {"parameters": {"indptr": "Exactly spec().rows + 1 output row offsets. The integer type must represent every required offset.", "indices": "At least spec().nnz output column IDs; the type must represent the largest output column.", "values": "At least spec().nnz typed values. Only the returned actual-nnz prefix is meaningful.", "ws": "Exclusive reusable scratch for this call.", "workers": "Read/decode worker count."}, "returns": "Actual stored-entry count after mapping. Only this prefix of indices and values is meaningful; spec().nnz is an allocation capacity."},
        },
    },
    "store::ReadSpec": {"when": "Use rows × columns to size dense elements and nnz to reserve CSR capacity. Byte sizes describe the stored dtype, which may differ from a requested typed output."},
    "loader::Loader": {"when": "Use when repeated requests should be delivered in order by concurrent workers. Build a source, define schemas, then prepare a run with accesses and a ring.", "methods": {
        "new": {"parameters": {"source": "Shared immutable registered matrices and their metadata.", "schemas": "One or more output declarations with distinct schema IDs, in delivery order.", "config": "Worker count, I/O backend, slot size, and execution window."}},
        "prepare": {"parameters": {"accesses": "Deterministic global cell IDs. Length must be divisible by the schema count. The run borrows this sequence.", "ring": "Shared output storage with a layout matching this loader and no outstanding earlier leases.", "planning": "No cache plan, an exact compatible offline plan, or online compiler options."}, "returns": "A prepared run borrowing this loader and sequence. Use for_each_batch to overlap production and consumption."},
        "plan_options": {"parameters": {"cache_bytes": "Frame capacity in bytes. This is not a bound on total process memory."}},
    }},
    "loader::Run": {
        "when": "Start with for_each_batch for local consumption. With run + wait, production and consumption must overlap; a full ring blocks the producer until leases are released.",
        "methods": {
            "for_each_batch": {"returns": "Final run statistics after scoped production and ordered consumption. Batch bytes are borrowed only for the callback duration."},
            "run": {"returns": "Executes production on the calling thread. Consumers must execute concurrently when the sequence can fill the ring."},
            "wait": {"parameters": {"rank": "Owner rank; batch b belongs to b % world.", "batch": "Global batch index within this run, not an input cell ID."}, "returns": "A lease after the batch is published, or None beyond the end. Retaining a lease prevents its slot from being overwritten."},
        },
    },
    "loader::LoaderConfig": {"when": "For small portable examples use Pread, a few workers, and tiny slots. Defaults target a different workload and should not be read as tuning advice."},
    "loader::OutputSchema": {"when": "List standard-column IDs in output order. A missing dimension uses the default; a present dimension with an omitted sparse value is zero."},
    "loader::BatchLease": {"when": "Read only rows() × bytes-per-row from region(schema). Keep the lease until every reader, including an asynchronous device copy, has finished."},
    "store::CpkOptions": {"when": "Choose page size in bytes and a target number of rows per chunk. The new constructor uses 4096-byte pages and a 4096-row target. Source layout and encoded layout are separate choices."},
    "store::RowSelection": {"when": "Range uses a half-open interval of file-row IDs. List gathers exactly the requested order and permits repeated rows; it does not sort or deduplicate."},
    "store::ReadWorkspace": {"when": "Keep one workspace per concurrent read and reuse it for later reads. clear releases retained scratch; it does not own or invalidate caller output arrays."},
    "loader::DataSourceBuilder": {"when": "Register immutable files in a known order and map each source column into one shared standard vocabulary. All registered streams must use the builder's page size.", "methods": {
        "new": {"parameters": {"vocab": "Number of standard-column IDs; positive and below u32::MAX.", "page_size": "Power-of-two bytes per page, between 512 and 65536. Must match every stream."}},
    }},
    "loader::Ring": {"when": "Allocate from loader.ring_layout so schema extents and alignment agree. All readers must release their leases before a slot can be reused. An aborted ring needs replacement."},
    "loader::Planning": {"when": "Choose None to read privately, Fixed for an offline plan bound to the exact sequence, or Online for finite-lookahead compilation during the run."},
    "loader::Accesses": {"when": "Each index identifies one schema delivery. With S schemas, position k / S uses schema k % S. Implementations must return the same values for repeated and out-of-order range calls."},
    "opt::PlanOptions": {"when": "Use Loader::plan_options as a starting point. gap and horizon count slots; chunk counts accesses. Cache bytes reserve frames and do not include the entire loader's memory."},
}
