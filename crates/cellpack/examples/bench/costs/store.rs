//! pycellpack's cost lookup for the benchmark: the records under
//! `$CELLPACK_HOME` (default `~/.cellpack`) keyed exactly as
//! `pycellpack.loader.costs` keys them, the same defaults where none matches, and
//! saved models (`CostModel.save`). Keys and messages follow
//! `src/pycellpack/loader/costs/` (`_keys.py`, `_code.py`, `_resolve.py`); the
//! defaults are the crate's ([`cellpack::opt::defaults`]), which pycellpack
//! reads too. The crate itself has no such lookup.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

use cellpack::opt::defaults::{self as d, Tier};
use cellpack::opt::{CostModel, InputCost, OutputCost, ProcessCost, RowCost};
use serde_json::{Value, json};

/// A schema as the lookup keys it.
pub struct Schema {
    pub name: String,
    /// numpy's name of the output dtype (`float32`, ...).
    pub dtype: &'static str,
    pub cell_bytes: usize,
    pub scratch_bytes: usize,
    pub process: Option<Process>,
}

/// A process function: its entry addresses (0 for a callback its row format
/// does not use), declared identity, whether it takes partial rows, and its
/// row format.
pub struct Process {
    pub csr: usize,
    pub dense: usize,
    pub identity: u64,
    pub partial_rows: bool,
    /// `auto`, `csr` or `dense` (`pycellpack.loader.ProcessFn.row_format`).
    pub row_format: &'static str,
}

/// How the data source reads its files.
pub struct Source<'a> {
    pub paths: &'a [PathBuf],
    pub backend: &'static str,
    pub direct_io: bool,
    pub page_size: usize,
}

/// A cost model and where each part came from.
pub struct Resolved {
    pub model: CostModel,
    pub provenance: Vec<(String, String)>,
}

const FORMAT: u64 = 1;

// Defaults as record values (`_resolve.py` reads the same table).

fn default_storage(tier: &str) -> Value {
    let s = d::storage(Tier::named(tier));
    json!({"iops": s.iops, "bandwidth": s.bandwidth, "request": s.request, "serial": s.serial})
}

fn default_host() -> Value {
    let h = d::HOST;
    let b = d::builtin();
    json!({
        "access": h.access,
        "compile": h.compile,
        "memory_bandwidth": h.memory_bandwidth,
        "input": {"decode": h.input.decode, "decode_element": h.input.decode_element, "cold": h.input.cold},
        "rows": {"gather": h.rows.gather, "admit": h.rows.admit},
        "convert": h.convert,
        "builtin": {"*": {"map_element": b.map_element, "fill_element": b.fill_element}},
    })
}

fn default_process() -> Value {
    let p = d::process();
    json!({"process": p.process, "map_element": p.map_element, "fill_element": p.fill_element, "resolve": p.resolve, "resolve_element": p.resolve_element, "curve": []})
}

// Keys: `_keys.py` and `_code.py`.

/// `json.dumps(value, sort_keys=True, separators=(",", ":"))`: serde_json's
/// maps are sorted; non-ASCII characters are escaped as Python does.
fn python_json(value: &Value) -> String {
    let mut out = String::new();
    for c in serde_json::to_string(value).unwrap().chars() {
        if c.is_ascii() {
            out.push(c);
        } else {
            let mut units = [0u16; 2];
            for u in c.encode_utf16(&mut units) {
                out.push_str(&format!("\\u{u:04x}"));
            }
        }
    }
    out
}

fn digest(identity: &Value) -> String {
    hex(&blake2b(python_json(identity).as_bytes(), 10))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn cpu_model() -> String {
    if let Ok(text) = std::fs::read_to_string("/proc/cpuinfo") {
        for line in text.lines() {
            if let Some((key, value)) = line.split_once(':')
                && matches!(key.trim(), "model name" | "Hardware" | "Processor")
            {
                return value.trim().to_string();
            }
        }
    }
    std::env::consts::ARCH.to_string()
}

fn host_identity() -> Value {
    let version = env!("CARGO_PKG_VERSION").split('.').take(2).collect::<Vec<_>>().join(".");
    json!({"cpu": cpu_model(), "machine": std::env::consts::ARCH, "kernel_set": cellpack::scans::kernel_set(), "version": version, "row_policy": 1})
}

fn mount(device: u64) -> (String, String) {
    let (major, minor) = (libc::major(device), libc::minor(device));
    let want = format!("{major}:{minor}");
    if let Ok(text) = std::fs::read_to_string("/proc/self/mountinfo") {
        for line in text.lines() {
            let (left, right) = line.split_once(" - ").unwrap_or((line, ""));
            let fields: Vec<&str> = left.split_whitespace().collect();
            if fields.len() >= 3 && fields[2] == want {
                let rest: Vec<&str> = right.split_whitespace().collect();
                return (rest.first().unwrap_or(&"").to_string(), rest.get(1).unwrap_or(&"").to_string());
            }
        }
    }
    (String::new(), String::new())
}

fn block(major: u32, minor: u32, depth: u32) -> (String, Option<bool>) {
    let Ok(real) = std::fs::canonicalize(format!("/sys/dev/block/{major}:{minor}")) else {
        return (String::new(), None);
    };
    let mut queue = real.join("queue");
    if !queue.exists() && real.parent().is_some_and(|p| p.join("queue").exists()) {
        queue = real.parent().unwrap().join("queue");
    }
    let slaves = real.join("slaves");
    if depth < 3 && slaves.is_dir() {
        let mut entries: Vec<PathBuf> = std::fs::read_dir(&slaves).map(|d| d.filter_map(|e| e.ok().map(|e| e.path())).collect()).unwrap_or_default();
        entries.sort();
        if let Some(slave) = entries.first()
            && let Ok(dev) = std::fs::read_to_string(slave.join("dev"))
            && let Some((a, b)) = dev.trim().split_once(':')
            && let (Ok(a), Ok(b)) = (a.parse(), b.parse())
        {
            return block(a, b, depth + 1);
        }
    }
    let rotational = std::fs::read_to_string(queue.join("rotational")).ok().map(|s| s.trim() == "1");
    let name = if real.join("queue").exists() { real.file_name() } else { real.parent().and_then(|p| p.file_name()) };
    (name.map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(), rotational)
}

const NETWORK_FS: [&str; 17] = [
    "nfs", "nfs4", "cifs", "smb3", "smbfs", "lustre", "gpfs", "beegfs", "cephfs", "ceph", "glusterfs", "9p", "afs", "orangefs", "pvfs2", "juicefs", "s3fs",
];

fn storage_tier(fs: &str, device: Option<u64>) -> String {
    if matches!(fs, "tmpfs" | "ramfs" | "devtmpfs") {
        return "memory".into();
    }
    if NETWORK_FS.contains(&fs) || fs.starts_with("fuse") {
        return "network".into();
    }
    let Some(device) = device else { return "unknown".into() };
    let (name, rotational) = block(libc::major(device), libc::minor(device), 0);
    if name.starts_with("nvme") {
        "nvme".into()
    } else {
        match rotational {
            Some(true) => "hdd".into(),
            Some(false) => "ssd".into(),
            None => "unknown".into(),
        }
    }
}

fn storage_identity(source: &Source<'_>, host: &str) -> (Value, String) {
    let dominant = source.paths.iter().filter_map(|p| std::fs::metadata(p).ok().map(|m| (m.len(), p))).max_by_key(|(n, _)| *n);
    let (mut fs, mut src, mut device) = (String::new(), String::new(), None);
    if let Some((_, path)) = dominant {
        let dev = std::fs::metadata(path).unwrap().dev();
        (fs, src) = mount(dev);
        device = Some(dev);
    }
    let tier = storage_tier(&fs, device);
    let identity = json!({"host": host, "fs": fs, "source": src, "backend": source.backend, "direct_io": source.direct_io, "page_size": source.page_size});
    (identity, tier)
}

/// This process's mappings: sorted `[lo, hi)` spans.
struct Mappings(Vec<(u64, u64)>);

impl Mappings {
    fn read() -> Self {
        let mut spans: Vec<(u64, u64)> = std::fs::read_to_string("/proc/self/maps")
            .unwrap_or_default()
            .lines()
            .filter_map(|l| {
                let (lo, hi) = l.split_whitespace().next()?.split_once('-')?;
                Some((u64::from_str_radix(lo, 16).ok()?, u64::from_str_radix(hi, 16).ok()?))
            })
            .collect();
        spans.sort();
        Self(spans)
    }
    fn end(&self, address: u64) -> Option<u64> {
        let i = self.0.partition_point(|&(lo, _)| lo <= address);
        (i > 0 && address < self.0[i - 1].1).then(|| self.0[i - 1].1)
    }
}

fn read_memory(address: u64, length: u64) -> Vec<u8> {
    let mut out = vec![0; length as usize];
    let Ok(mut f) = File::open("/proc/self/mem") else { return Vec::new() };
    if f.seek(SeekFrom::Start(address)).is_err() {
        return Vec::new();
    }
    let mut n = 0;
    while n < out.len() {
        match f.read(&mut out[n..]) {
            Ok(0) | Err(_) => break,
            Ok(k) => n += k,
        }
    }
    out.truncate(n);
    out
}

fn jump_target(address: u64, code: &[u8]) -> Option<u64> {
    let at = if code.starts_with(&[0xf3, 0x0f, 0x1e, 0xfa]) { 4 } else { 0 };
    let rel = |from: usize, n: usize| -> Option<i64> {
        let b = code.get(from..from + n)?;
        Some(if n == 1 { b[0] as i8 as i64 } else { i32::from_le_bytes(b.try_into().ok()?) as i64 })
    };
    match code.get(at) {
        Some(0xe9) => Some((address as i64 + at as i64 + 5 + rel(at + 1, 4)?) as u64),
        Some(0xeb) => Some((address as i64 + at as i64 + 2 + rel(at + 1, 1)?) as u64),
        Some(0xff) if code.get(at + 1) == Some(&0x25) => {
            let slot = (address as i64 + at as i64 + 6 + rel(at + 2, 4)?) as u64;
            let target = read_memory(slot, 8);
            (target.len() == 8).then(|| u64::from_le_bytes(target.try_into().unwrap()))
        }
        _ => None,
    }
}

fn body_length(code: &[u8], machine: &str) -> usize {
    match machine {
        "x86_64" | "amd64" | "i386" | "i686" => {
            for (i, &b) in code.iter().enumerate() {
                if b == 0xc3 {
                    return i + 1;
                }
                if b == 0xc2 && i + 3 <= code.len() {
                    return i + 3;
                }
            }
            code.len()
        }
        "aarch64" | "arm64" => (0..code.len().saturating_sub(3))
            .step_by(4)
            .find(|&i| u32::from_le_bytes(code[i..i + 4].try_into().unwrap()) & 0xFFFF_FC1F == 0xD65F_0000)
            .map_or(code.len(), |i| i + 4),
        _ => code.len().min(256),
    }
}

/// `pycellpack.loader.costs.signature`: the code from a function's entry to its
/// first return (64 to 4096 bytes, after at most two leading jumps), with
/// eight-byte values pointing into this process's mappings zeroed.
fn signature(address: usize, maps: &Mappings) -> String {
    let machine = std::env::consts::ARCH;
    let mut address = address as u64;
    let mut code = Vec::new();
    for _ in 0..3 {
        let Some(end) = maps.end(address) else {
            code.clear();
            break;
        };
        code = read_memory(address, 4096.min(end - address));
        if !matches!(machine, "x86_64" | "amd64") {
            break;
        }
        match jump_target(address, &code) {
            Some(t) if t != address => address = t,
            _ => break,
        }
    }
    let n = body_length(&code, machine).max(64.min(code.len()));
    code.truncate(n);
    let mut i = 0;
    while i + 8 <= code.len() {
        let v = u64::from_le_bytes(code[i..i + 8].try_into().unwrap());
        if maps.end(v).is_some() {
            code[i..i + 8].fill(0);
            i += 8;
        } else {
            i += 1;
        }
    }
    let mut bytes = machine.as_bytes().to_vec();
    bytes.push(0);
    bytes.extend_from_slice(&code);
    hex(&blake2b(&bytes, 16))
}

/// The signature of the function at `address` (as `pycellpack.loader.costs.signature`).
pub fn function_signature(address: usize) -> String {
    signature(address, &Mappings::read())
}

fn output_identity(schema: &Schema, host: &str, maps: &Mappings) -> Option<Value> {
    let p = schema.process.as_ref()?;
    Some(json!({
        "host": host,
        "code": [signature(p.csr, maps), signature(p.dense, maps)],
        "identity": p.identity,
        "partial_rows": p.partial_rows,
        "row_format": p.row_format,
        "dtype": schema.dtype,
        "cell_bytes": schema.cell_bytes,
        "scratch_bytes": schema.scratch_bytes,
    }))
}

/// The record keys: host, storage (with its tier) and each process output.
pub struct Keys {
    pub host: (String, Value),
    pub storage: (String, Value),
    pub tier: String,
    pub outputs: Vec<Option<(String, Value)>>,
}

pub fn keys(source: &Source<'_>, schemas: &[Schema]) -> Keys {
    let host_id = host_identity();
    let host = digest(&host_id);
    let (storage_id, tier) = storage_identity(source, &host);
    let maps = Mappings::read();
    let outputs = schemas.iter().map(|s| output_identity(s, &host, &maps).map(|id| (digest(&id), id))).collect();
    Keys { host: (host, host_id), storage: (digest(&storage_id), storage_id), tier, outputs }
}

// Records: `_store.py`.

/// `$CELLPACK_HOME`, else `~/.cellpack`.
pub fn home() -> PathBuf {
    std::env::var_os("CELLPACK_HOME").filter(|v| !v.is_empty()).map(PathBuf::from).unwrap_or_else(|| {
        Path::new(&std::env::var_os("HOME").unwrap_or_default()).join(".cellpack")
    })
}

fn record(kind: &str, key: &str) -> Option<Value> {
    let path = home().join("costs").join(format!("v{FORMAT}")).join(kind).join(format!("{key}.json"));
    let text = std::fs::read_to_string(&path).ok()?;
    match serde_json::from_str::<Value>(&text) {
        Ok(r) if r["format"] == FORMAT && r["kind"] == kind && r["key"] == key => Some(r["values"].clone()),
        Ok(_) => {
            eprintln!("warning: ignoring cost record {} of another format", path.display());
            None
        }
        Err(e) => {
            eprintln!("warning: ignoring unreadable cost record {}: {e}", path.display());
            None
        }
    }
}

// Models.

fn f(v: &Value, key: &str, default: &Value) -> f64 {
    v.get(key).and_then(Value::as_f64).or_else(|| default.get(key).and_then(Value::as_f64)).unwrap_or(0.0)
}

fn output_cost(v: &Value) -> OutputCost {
    let d = default_process();
    let curve = v
        .get("curve")
        .and_then(Value::as_array)
        .map(|bands| {
            bands
                .iter()
                .map(|b| match b {
                    Value::Array(x) => ProcessCost { max_elements: x[0].as_u64().unwrap() as u32, process: x[1].as_f64().unwrap(), resolve: x.get(2).and_then(Value::as_f64).unwrap_or(0.0) },
                    _ => ProcessCost { max_elements: b["max_elements"].as_u64().unwrap() as u32, process: b["process"].as_f64().unwrap(), resolve: b.get("resolve").and_then(Value::as_f64).unwrap_or(0.0) },
                })
                .collect()
        })
        .unwrap_or_default();
    OutputCost {
        process: f(v, "process", &d),
        map_element: f(v, "map_element", &d),
        fill_element: f(v, "fill_element", &d),
        resolve: f(v, "resolve", &d),
        resolve_element: f(v, "resolve_element", &d),
        curve,
    }
}

fn convert(v: &Value) -> [f64; 6] {
    let mut out = [0.0; 6];
    if let Some(list) = v.as_array() {
        for (o, x) in out.iter_mut().zip(list) {
            *o = x.as_f64().unwrap_or(0.0);
        }
    }
    out
}

/// A cost model from host values, storage values and per-output costs.
fn compose(host: &Value, storage: &Value, outputs: Vec<OutputCost>, cores: f64, provenance: Vec<(String, String)>) -> Resolved {
    let dh = default_host();
    let ds = default_storage("unknown");
    let (input, di) = (host.get("input").unwrap_or(&dh["input"]), &dh["input"]);
    let (rows, dr) = (host.get("rows").unwrap_or(&dh["rows"]), &dh["rows"]);
    Resolved {
        model: CostModel {
            cores,
            iops: f(storage, "iops", &ds),
            bandwidth: f(storage, "bandwidth", &ds),
            memory_bandwidth: f(host, "memory_bandwidth", &dh),
            serial: f(storage, "serial", &ds),
            bucket: d::BUCKET,
            access: f(host, "access", &dh),
            request: f(storage, "request", &ds),
            compile: f(host, "compile", &dh),
            input: InputCost { decode: f(input, "decode", di), decode_element: f(input, "decode_element", di), cold: f(input, "cold", di) },
            rows: RowCost { gather: f(rows, "gather", dr), admit: f(rows, "admit", dr) },
            outputs,
            convert: convert(host.get("convert").unwrap_or(&dh["convert"])),
        },
        provenance,
    }
}

fn builtin(host: &Value, dtype: &str) -> OutputCost {
    let table = host.get("builtin").cloned().unwrap_or(Value::Null);
    let entry = table.get(dtype).or_else(|| table.get("*")).cloned().unwrap_or_else(|| default_host()["builtin"]["*"].clone());
    output_cost(&entry)
}

/// A model saved by `CostModel.save`, its outputs taken in order as the
/// schemas'; `cores` replaces the saved count.
pub fn load(path: &Path, schemas: &[Schema], cores: f64) -> Result<Resolved, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let v: Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    if v.get("format").and_then(Value::as_u64).unwrap_or(FORMAT) != FORMAT {
        return Err(format!("{}: unsupported cost model format", path.display()));
    }
    let outputs: Vec<OutputCost> = v["outputs"].as_array().map(|o| o.iter().map(output_cost).collect()).unwrap_or_default();
    if outputs.len() != schemas.len() {
        return Err(format!("{}: {} outputs for {} schemas", path.display(), outputs.len(), schemas.len()));
    }
    let storage = json!({"iops": v["iops"], "bandwidth": v["bandwidth"], "request": v["request"], "serial": v["serial"]});
    let mut r = compose(&v, &storage, outputs, cores, vec![("model".into(), format!("file:{}", path.display()))]);
    if let Some(b) = v.get("bucket").and_then(Value::as_u64) {
        r.model.bucket = b;
    }
    Ok(r)
}

/// The defaults alone (no records).
pub fn defaults(source: &Source<'_>, schemas: &[Schema], cores: f64) -> Resolved {
    let tier = keys(source, &[]).tier;
    let host = default_host();
    let outputs = schemas.iter().map(|s| if s.process.is_some() { output_cost(&default_process()) } else { builtin(&host, s.dtype) }).collect();
    compose(&host, &default_storage(&tier), outputs, cores, vec![("model".into(), format!("default:{tier}"))])
}

/// `pycellpack.loader.costs.resolve`: `$CELLPACK_COST_MODEL` replaces everything;
/// otherwise each part from its record, else defaults with a warning.
pub fn resolve(source: &Source<'_>, schemas: &[Schema], cores: f64) -> Resolved {
    if let Some(path) = std::env::var_os("CELLPACK_COST_MODEL").filter(|v| !v.is_empty()) {
        match load(Path::new(&path), schemas, cores) {
            Ok(r) => return r,
            Err(e) => eprintln!("warning: ignoring CELLPACK_COST_MODEL ({e})"),
        }
    }
    let k = keys(source, schemas);
    let mut provenance = Vec::new();
    let (host, host_from) = match record("host", &k.host.0) {
        Some(v) => (v, format!("store:host/{}", k.host.0)),
        None => {
            eprintln!("warning: no measured host costs for this CPU and build; using defaults (run pycellpack.loader.costs.calibrate to measure them)");
            (default_host(), "default".to_string())
        }
    };
    provenance.push(("host".into(), host_from.clone()));
    let storage = match record("storage", &k.storage.0) {
        Some(v) => {
            provenance.push(("storage".into(), format!("store:storage/{}", k.storage.0)));
            v
        }
        None => {
            let id = &k.storage.1;
            let fs = id["fs"].as_str().filter(|s| !s.is_empty()).unwrap_or("this");
            let src = id["source"].as_str().filter(|s| !s.is_empty()).unwrap_or("unknown source");
            eprintln!("warning: no measured device costs for {fs} storage ({src}); using {} defaults", k.tier);
            provenance.push(("storage".into(), format!("default:{}", k.tier)));
            default_storage(&k.tier)
        }
    };
    let mut outputs = Vec::new();
    for (schema, entry) in schemas.iter().zip(&k.outputs) {
        let (cost, from) = match entry {
            None => (builtin(&host, schema.dtype), host_from.clone()),
            Some((key, _)) => match record("output", key) {
                Some(v) => (output_cost(&v), format!("store:output/{key}")),
                None => {
                    eprintln!("warning: no measured costs for the process function of output '{}'; using defaults", schema.name);
                    (output_cost(&default_process()), "default".to_string())
                }
            },
        };
        provenance.push((format!("output:{}", schema.name), from));
        outputs.push(cost);
    }
    compose(&host, &storage, outputs, cores, provenance)
}

/// The keys and their identities as JSON, for comparison with
/// `pycellpack.loader.costs.keys`.
pub fn describe(k: &Keys) -> Value {
    let mut outputs = BTreeMap::new();
    for (i, o) in k.outputs.iter().enumerate() {
        outputs.insert(i.to_string(), o.as_ref().map_or(Value::Null, |(key, id)| json!([key, id])));
    }
    json!({"host": [k.host.0, k.host.1], "storage": [k.storage.0, k.storage.1], "tier": k.tier, "outputs": outputs})
}

// BLAKE2b (RFC 7693), unkeyed, for keys matching Python's hashlib.

const IV: [u64; 8] = [
    0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
    0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
];
const SIGMA: [[usize; 16]; 12] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
];

fn compress(h: &mut [u64; 8], block: &[u8; 128], t: u128, last: bool) {
    let m: [u64; 16] = std::array::from_fn(|i| u64::from_le_bytes(block[i * 8..i * 8 + 8].try_into().unwrap()));
    let mut v = [0u64; 16];
    v[..8].copy_from_slice(h);
    v[8..].copy_from_slice(&IV);
    v[12] ^= t as u64;
    v[13] ^= (t >> 64) as u64;
    if last {
        v[14] = !v[14];
    }
    let g = |v: &mut [u64; 16], a: usize, b: usize, c: usize, d: usize, x: u64, y: u64| {
        v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
        v[d] = (v[d] ^ v[a]).rotate_right(32);
        v[c] = v[c].wrapping_add(v[d]);
        v[b] = (v[b] ^ v[c]).rotate_right(24);
        v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
        v[d] = (v[d] ^ v[a]).rotate_right(16);
        v[c] = v[c].wrapping_add(v[d]);
        v[b] = (v[b] ^ v[c]).rotate_right(63);
    };
    for s in &SIGMA {
        g(&mut v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
        g(&mut v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
        g(&mut v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
        g(&mut v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
        g(&mut v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
        g(&mut v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
        g(&mut v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
        g(&mut v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
    }
    for i in 0..8 {
        h[i] ^= v[i] ^ v[i + 8];
    }
}

fn blake2b(data: &[u8], size: usize) -> Vec<u8> {
    let mut h = IV;
    h[0] ^= 0x0101_0000 ^ size as u64;
    let mut t: u128 = 0;
    let mut chunks = data.chunks(128).peekable();
    if data.is_empty() {
        compress(&mut h, &[0; 128], 0, true);
    }
    while let Some(chunk) = chunks.next() {
        let mut block = [0u8; 128];
        block[..chunk.len()].copy_from_slice(chunk);
        t += chunk.len() as u128;
        compress(&mut h, &block, t, chunks.peek().is_none());
    }
    h.iter().flat_map(|w| w.to_le_bytes()).take(size).collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn blake2b_matches_known_digests() {
        // hashlib.blake2b(b"abc").hexdigest() and blake2b(b"", digest_size=10).
        assert_eq!(super::hex(&super::blake2b(b"abc", 64)), "ba80a53f981c4d0d6a2797b69f12f6e94c212f14685ac4b74b12bb6fdbffa2d17d87c5392aab792dc252d5de4533cc9518d38aa8dbf1925ab92386edd4009923");
    }
}
