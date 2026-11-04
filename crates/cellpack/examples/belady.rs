//! Exact page-level locality experiment, run on a worker:
//! `belady /path/to/scratch [pages=4096] [repetitions=16]`.
//! Every trace has the same page histogram; P_r=(0^r ... (U-1)^r)^(R/r).
use cellpack::loader::*;
use cellpack::opt::*;
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "Usage: belady <scratch_directory> [pages=4096] [repetitions=16]\n\
        Compare page-only plans with the independent serial Belady optimum.\n\
        repetitions must be a power of two. Output is CSV; the temporary .cpk is removed.";
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{usage}");
        return;
    }
    if args.is_empty() {
        eprintln!("{usage}");
        std::process::exit(2);
    }
    let root = PathBuf::from(&args[0]);
    std::fs::create_dir_all(&root).unwrap();
    let get = |key: &str, default: usize| {
        args.iter()
            .find_map(|a| a.strip_prefix(key))
            .map_or(default, |s| s.parse().unwrap())
    };
    let pages = get("pages=", 4096);
    let repetitions = get("repetitions=", 16);
    assert!(pages > 0 && pages < u32::MAX as usize && repetitions.is_power_of_two());
    // One distinct-valued 1000-float row per page: each record is stored
    // dense and raw (about 4 KB), so no two records share a page.
    let path = root.join(format!("serial-pages-{}.cpk", std::process::id()));
    let cols = 1000usize;
    let data: Vec<u8> = (0..pages * cols).flat_map(|i| (i as f32 + 0.5).to_le_bytes()).collect();
    let encode = cellpack::scans::EncodeOptions {
        renumber: false,
        thresholds: cellpack::scans::Thresholds { indices: 0.6, values: 0.6 },
        max_lanes: 32,
        lane_min: 32,
        column_modes: None,
        layout: cellpack::scans::LayoutChoice::Dense,
        values: cellpack::scans::ValueChoice::Raw,
    };
    let mut w = cellpack::store::CpkWriter::create(&path, cols as u32, Dtype::F32, cellpack::format::SourceLayout::Dense, cellpack::store::CpkOptions::new(encode)).unwrap();
    w.append(cellpack::scans::DenseRef { n_genes: cols as u32, value_width: 4, data: &data }).unwrap();
    w.finish().unwrap();
    let mut builder = DataSourceBuilder::new(cols, 4096).unwrap();
    builder.add_cpk(&path, &(0..cols as u32).collect::<Vec<_>>()).unwrap();
    let source = std::sync::Arc::new(builder.build().unwrap());
    let schema = OutputSchema::dense(1, (0..cols as u32).collect::<Vec<_>>(), Dtype::F32).unwrap();
    // One position per slot: the reuse gap counts accesses.
    let loader = Loader::new(source, vec![schema], LoaderConfig { slot_cells: 1, ..Default::default() }).unwrap();
    let mut capacities = vec![0, 1, 4, 16, 64, 256, 1024, pages];
    capacities.retain(|&c| c <= pages);
    capacities.sort_unstable();
    capacities.dedup();
    println!("repeat,pages,repetitions,uses,capacity,distance,opt_hits,plan_hits,hit_gap_pp");
    for power in 0..=repetitions.trailing_zeros() {
        let repeat = 1usize << power;
        let trace: Vec<u32> = (0..repetitions / repeat)
            .flat_map(|_| (0..pages as u32).flat_map(|p| std::iter::repeat_n(p, repeat)))
            .collect();
        let oracle = PageTrace::from_pages(&trace, pages).unwrap();
        for &capacity in &capacities {
            let opt = oracle.optimum(capacity);
            for distance in [1, 32, 4096] {
                let mut options = PlanOptions::new(capacity * 4096, distance);
                options.objective = PlanObjective::Misses;
                options.threads = 4;
                let plan = loader.compile(&trace, &options).unwrap();
                let replay = loader.replay(&plan, &trace).unwrap();
                assert_eq!(&replay, plan.summary());
                assert!(replay.page_hits <= opt.hits);
                println!(
                    "{repeat},{pages},{repetitions},{},{capacity},{distance},{},{},{:.6}",
                    trace.len(),
                    opt.hits,
                    replay.page_hits,
                    100.0 * (opt.hits - replay.page_hits) as f64 / trace.len() as f64
                );
            }
        }
    }
    std::fs::remove_file(path).unwrap();
}
