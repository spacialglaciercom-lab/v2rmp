use crossterm::event::{self, Event, KeyCode, KeyModifiers};
use std::time::Duration;

use crate::app::{App, InputField, Status, View};
use crate::core::clean::CleanOptions;

pub fn poll_event(timeout: Duration) -> anyhow::Result<Option<Event>> {
    if event::poll(timeout)? {
        Ok(Some(event::read()?))
    } else {
        Ok(None)
    }
}

pub async fn handle_event(app: &mut App, ev: Event) -> anyhow::Result<()> {
    if let Event::Key(key) = ev {
        if app.input_mode.active {
            handle_input_mode(app, key.code, key.modifiers);
            return Ok(());
        }
        handle_normal_mode(app, key.code, key.modifiers).await;
    }
    Ok(())
}

fn handle_input_mode(app: &mut App, code: KeyCode, _mods: KeyModifiers) {
    match code {
        KeyCode::Enter => app.confirm_input(),
        KeyCode::Esc => app.cancel_input(),
        KeyCode::Backspace => {
            app.input_mode.buffer.pop();
        }
        KeyCode::Char(c) => {
            app.input_mode.buffer.push(c);
        }
        _ => {}
    }
}

async fn handle_normal_mode(app: &mut App, code: KeyCode, mods: KeyModifiers) {
    // Handle file browser separately
    if app.current_view == View::FileBrowser {
        handle_file_browser_keys(app, code);
        return;
    }

    match code {
        KeyCode::Char('q') => {
            app.running = false;
        }
        KeyCode::Esc => {
            app.current_view = View::Home;
            app.workflow_selection = 0;
        }
        KeyCode::Char('h') | KeyCode::F(1) => {
            if app.current_view == View::Help {
                app.current_view = View::Home;
            } else {
                app.current_view = View::Help;
            }
        }
        KeyCode::Up => app.navigate_up(),
        KeyCode::Down => app.navigate_down(),
        _ => handle_view_keys(app, code, mods).await,
    }
}

async fn handle_view_keys(app: &mut App, code: KeyCode, mods: KeyModifiers) {
    match app.current_view {
        View::Home => handle_home_keys(app, code, mods),
        View::Extract => handle_extract_keys(app, code, mods).await,
        View::Compile => handle_compile_keys(app, code, mods),
        View::Optimize => handle_optimize_keys(app, code, mods).await,
        View::Vrp => handle_vrp_keys(app, code, mods).await,
        View::BrowseMaps => handle_browse_maps_keys(app, code),
        View::BrowseRoutes => handle_browse_routes_keys(app, code),
        View::FileBrowser => {} // Handled separately in handle_normal_mode
        View::Help => {}
        View::Clean => handle_clean_keys(app, code, mods),
    }
}

fn handle_home_keys(app: &mut App, code: KeyCode, _mods: KeyModifiers) {
    if code == KeyCode::Enter {
        match app.workflow_selection {
            0 => {
                app.current_view = View::Extract;
                app.log(crate::app::LogLevel::Info, "Switched to Extract Data view");
            }
            1 => {
                app.current_view = View::Clean;
                app.log(crate::app::LogLevel::Info, "Switched to Clean GeoJSON view");
            }
            2 => {
                app.current_view = View::Compile;
                app.log(crate::app::LogLevel::Info, "Switched to Compile Map view");
            }
            3 => {
                app.current_view = View::Optimize;
                app.log(
                    crate::app::LogLevel::Info,
                    "Switched to Optimize Route view",
                );
            }
            4 => {
                app.current_view = View::Vrp;
                app.log(crate::app::LogLevel::Info, "Switched to VRP Solver view");
            }
            5 => {
                app.current_view = View::BrowseMaps;
                app.browse_selection = 0;
                app.log(crate::app::LogLevel::Info, "Switched to Cached Maps view");
            }
            6 => {
                app.current_view = View::BrowseRoutes;
                app.browse_selection = 0;
                app.log(crate::app::LogLevel::Info, "Switched to Saved Routes view");
            }
            _ => {}
        }
    }
}

async fn handle_extract_keys(app: &mut App, code: KeyCode, _mods: KeyModifiers) {
    match code {
        KeyCode::Tab | KeyCode::Char('s') => {
            app.data_source = match app.data_source {
                crate::app::DataSource::Osm => crate::app::DataSource::Overture,
                crate::app::DataSource::Overture => crate::app::DataSource::Osm,
            };
            app.log(
                crate::app::LogLevel::Info,
                format!("Data source: {}", app.data_source),
            );
        }
        KeyCode::Char('b') | KeyCode::Char('B') => {
            app.start_input(InputField::BoundingBox);
        }
        KeyCode::Enter => {
            if let Some(ref bbox) = app.bounding_box {
                let bbox_vals = bbox.clone();
                let source_name = format!("{}", app.data_source);
                app.extract_status = crate::app::Status::Running {
                    progress: 0,
                    message: "Starting extraction...".to_string(),
                };
                app.log(
                    crate::app::LogLevel::Info,
                    format!("Starting extraction from {}", source_name),
                );

                // Build extract request
                use crate::core::extract::{BBoxRequest, ExtractRequest, ExtractSource, RoadClass};

                let source = match app.data_source {
                    crate::app::DataSource::Osm => ExtractSource::Osm,
                    crate::app::DataSource::Overture => ExtractSource::Overture,
                };

                let output_path = format!(
                    "extract_{}.geojson",
                    chrono::Local::now().format("%Y%m%d_%H%M%S")
                );
                let req = ExtractRequest {
                    source,
                    bbox: BBoxRequest {
                        min_lon: bbox_vals.min_lon,
                        min_lat: bbox_vals.min_lat,
                        max_lon: bbox_vals.max_lon,
                        max_lat: bbox_vals.max_lat,
                    },
                    road_classes: RoadClass::all_vehicle(),
                    output_path: output_path.clone(),
                    pbf_path: None,
                };

                // Execute extraction
                match crate::core::extract::run_extract(&req).await {
                    Ok(result) => {
                        app.extract_status = crate::app::Status::Done(format!(
                            "Extracted {} nodes, {} edges, {:.2} km → {}",
                            result.nodes, result.edges, result.total_km, output_path
                        ));
                        app.log(
                            crate::app::LogLevel::Success,
                            format!(
                                "Extraction complete: {} nodes, {} edges, {:.2} km",
                                result.nodes, result.edges, result.total_km
                            ),
                        );
                        app.log(
                            crate::app::LogLevel::Info,
                            format!("Output saved to: {}", output_path),
                        );
                    }
                    Err(e) => {
                        app.extract_status = crate::app::Status::Error(e.to_string());
                        app.log(
                            crate::app::LogLevel::Error,
                            format!("Extraction failed: {}", e),
                        );
                    }
                }
            } else {
                app.log(
                    crate::app::LogLevel::Warn,
                    "Set a bounding box first (press 'b')",
                );
            }
        }
        _ => {}
    }
}

fn handle_compile_keys(app: &mut App, code: KeyCode, _mods: KeyModifiers) {
    match code {
        KeyCode::Char('i') | KeyCode::Char('I') => {
            app.start_file_browser(InputField::InputFile);
        }
        KeyCode::Char('o') | KeyCode::Char('O') => {
            app.start_file_browser(InputField::OutputFile);
        }
        KeyCode::Enter => {
            if let Some(input_path) = app.input_file.clone() {
                app.compile_status = crate::app::Status::Running {
                    progress: 0,
                    message: "Compiling map...".to_string(),
                };
                app.log(crate::app::LogLevel::Info, "Starting map compilation");

                // Build compile request
                use crate::core::compile::{run_compile, CompileRequest};

                let output_path = app.output_file.clone().unwrap_or_else(|| {
                    input_path
                        .replace(".geojson", ".rmp")
                        .replace(".json", ".rmp")
                });

                let req = CompileRequest {
                    input_geojson: input_path.clone(),
                    output_rmp: output_path.clone(),
                    compress: false,
                    road_classes: vec![],
                    clean_options: None,
                    prune_disconnected: false,
                };

                // Execute compilation
                match run_compile(&req) {
                    Ok(result) => {
                        app.compile_status = crate::app::Status::Done(format!(
                            "Compiled {} nodes, {} edges in {}ms → {}",
                            result.node_count, result.edge_count, result.elapsed_ms, output_path
                        ));
                        app.log(
                            crate::app::LogLevel::Success,
                            format!(
                                "Compilation complete: {} nodes, {} edges",
                                result.node_count, result.edge_count
                            ),
                        );
                        app.log(
                            crate::app::LogLevel::Info,
                            format!(
                                "Input: {} bytes → Output: {} bytes ({:.1}% compression)",
                                result.input_size_bytes,
                                result.output_size_bytes,
                                (1.0 - result.output_size_bytes as f64
                                    / result.input_size_bytes as f64)
                                    * 100.0
                            ),
                        );
                        app.log(
                            crate::app::LogLevel::Info,
                            format!("Output saved to: {}", output_path),
                        );
                    }
                    Err(e) => {
                        app.compile_status = crate::app::Status::Error(e.to_string());
                        app.log(
                            crate::app::LogLevel::Error,
                            format!("Compilation failed: {}", e),
                        );
                    }
                }
            } else {
                app.log(
                    crate::app::LogLevel::Warn,
                    "Set an input file first (press 'i')",
                );
            }
        }
        _ => {}
    }
}

async fn handle_optimize_keys(app: &mut App, code: KeyCode, _mods: KeyModifiers) {
    match code {
        KeyCode::Char('c') | KeyCode::Char('C') => {
            app.start_file_browser(InputField::CacheFile);
        }
        KeyCode::Char('r') => {
            app.start_file_browser(InputField::RouteFile);
        }
        KeyCode::Char('l') | KeyCode::Char('L') => {
            app.start_input(InputField::LeftTurnPenalty);
        }
        KeyCode::Char('R') => {
            app.start_input(InputField::RightTurnPenalty);
        }
        KeyCode::Char('u') | KeyCode::Char('U') => {
            app.start_input(InputField::UTurnPenalty);
        }
        KeyCode::Char('d') | KeyCode::Char('D') => {
            app.start_input(InputField::DepotCoordinates);
        }
        KeyCode::Char('v') | KeyCode::Char('V') => {
            app.start_input(InputField::NumVehicles);
        }
        KeyCode::Char('s') | KeyCode::Char('S') => {
            app.start_input(InputField::SolverId);
        }
        KeyCode::Enter => {
            if let Some(cache_path) = app.cache_file.clone() {
                let penalties = app.turn_penalties;
                let depot = app.depot_coords;
                let num_vehicles = app.num_vehicles;
                let solver_id = app.solver_id.clone();

                app.optimize_status = crate::app::Status::Running {
                    progress: 0,
                    message: "Optimizing route...".to_string(),
                };
                app.log(
                    crate::app::LogLevel::Info,
                    format!("Starting optimization (mode: VRP, solver: {})", solver_id),
                );

                // Build optimize request
                use crate::core::optimize::{
                    run_optimize, OnewayMode, OptimizeRequest, SolverMode,
                };

                let route_path = app.route_file.clone().or_else(|| {
                    Some(format!(
                        "route_{}.gpx",
                        chrono::Local::now().format("%Y%m%d_%H%M%S")
                    ))
                });

                let req = OptimizeRequest {
                    cache_file: cache_path,
                    route_file: route_path.clone(),
                    turn_penalties: penalties,
                    depot,
                    oneway_mode: OnewayMode::Respect,
                    mode: SolverMode::Vrp,
                    num_vehicles,
                    solver_id,
                };

                // Execute optimization
                match run_optimize(&req).await {
                    Ok(result) => {
                        app.optimize_status = crate::app::Status::Done(format!(
                            "Route: {:.2} km, {} vehicles, {} stops",
                            result.total_distance_km, result.num_routes, result.total_segments
                        ));
                        app.log(
                            crate::app::LogLevel::Success,
                            format!(
                                "Optimization complete: {:.2} km total distance across {} routes",
                                result.total_distance_km, result.num_routes
                            ),
                        );
                        app.log(
                            crate::app::LogLevel::Info,
                            format!(
                                "Turns: {} left, {} right, {} u-turn, {} straight",
                                result.turns.left,
                                result.turns.right,
                                result.turns.u_turn,
                                result.turns.straight
                            ),
                        );
                        if let Some(ref path) = route_path {
                            app.log(
                                crate::app::LogLevel::Info,
                                format!("Route saved to: {}", path),
                            );
                        }
                    }
                    Err(e) => {
                        app.optimize_status = crate::app::Status::Error(e.to_string());
                        app.log(
                            crate::app::LogLevel::Error,
                            format!("Optimization failed: {}", e),
                        );
                    }
                }
            } else {
                app.log(
                    crate::app::LogLevel::Warn,
                    "Set a cache file first (press 'c')",
                );
            }
        }
        _ => {}
    }
}

async fn handle_vrp_keys(app: &mut App, code: KeyCode, _mods: KeyModifiers) {
    match code {
        KeyCode::Char('i') | KeyCode::Char('I') => {
            app.start_file_browser(InputField::VrpInputFile);
        }
        KeyCode::Char('w') | KeyCode::Char('W') => {
            app.start_file_browser(InputField::VrpWaypointsFile);
        }
        KeyCode::Char('o') | KeyCode::Char('O') => {
            app.start_input(InputField::VrpOutputDir);
        }
        KeyCode::Char('v') | KeyCode::Char('V') => {
            app.start_input(InputField::NumVehicles);
        }
        KeyCode::Char('a') | KeyCode::Char('A') => {
            app.start_input(InputField::VrpAlgorithm);
        }
        KeyCode::Char('c') | KeyCode::Char('C') => {
            app.start_input(InputField::VrpCapacity);
        }
        KeyCode::Char('d') | KeyCode::Char('D') => {
            app.start_input(InputField::VrpDepot);
        }
        KeyCode::Char('x') | KeyCode::Char('X') => {
            app.vrp_depots.clear();
            app.log(crate::app::LogLevel::Info, "VRP depots cleared");
        }
        KeyCode::Enter => {
            if let Some(_input_path) = app.vrp_input_file.clone() {
                if app.vrp_depots.is_empty() {
                    app.log(
                        crate::app::LogLevel::Warn,
                        "Add at least one depot (press 'd')",
                    );
                    return;
                }
                if app.vrp_waypoints_file.is_none() {
                    app.log(
                        crate::app::LogLevel::Warn,
                        "Set a waypoints file (press 'w')",
                    );
                    return;
                }

                let algo = app.vrp_algo.clone();
                let vehicles = app.vrp_vehicles;
                let capacity = app.vrp_capacity.unwrap_or(100.0);
                let depots: Vec<String> = app.vrp_depots.clone();
                let waypoints_path = app.vrp_waypoints_file.clone().unwrap();
                let output_dir = app.vrp_output_dir.clone();

                app.vrp_status = crate::app::Status::Running {
                    progress: 0,
                    message: "Solving VRP...".to_string(),
                };
                app.log(
                    crate::app::LogLevel::Info,
                    format!("Starting VRP solve with algorithm: {}", algo),
                );

                // Parse depots
                let parsed_depots: Result<Vec<(f64, f64)>, _> = depots
                    .iter()
                    .map(|s| {
                        let parts: Vec<f64> = s
                            .split(',')
                            .filter_map(|v| v.parse::<f64>().ok())
                            .collect();
                        if parts.len() == 2 {
                            Ok((parts[0], parts[1]))
                        } else {
                            Err(anyhow::anyhow!("Invalid depot: {}", s))
                        }
                    })
                    .collect();

                match parsed_depots {
                    Ok(parsed) => {
                        // Build stops from waypoints
                        let mut stops = Vec::new();
                        stops.push(crate::core::vrp::types::VRPSolverStop {
                            lat: parsed[0].0,
                            lon: parsed[0].1,
                            label: "Depot".into(),
                            demand: Some(0.0),
                            arrival_time: None,
                        });

                        match std::fs::read_to_string(&waypoints_path) {
                            Ok(wp_data) => {
                                match serde_json::from_str::<Vec<[f64; 2]>>(&wp_data) {
                                    Ok(points) => {
                                        for (i, p) in points.into_iter().enumerate() {
                                            stops.push(crate::core::vrp::types::VRPSolverStop {
                                                lat: p[0],
                                                lon: p[1],
                                                label: format!("WP {}", i),
                                                demand: Some(1.0),
                                                arrival_time: None,
                                            });
                                        }
                                    }
                                    Err(e) => {
                                        app.vrp_status =
                                            crate::app::Status::Error(format!(
                                                "Invalid waypoints JSON: {}",
                                                e
                                            ));
                                        app.log(
                                            crate::app::LogLevel::Error,
                                            format!("VRP failed: invalid waypoints: {}", e),
                                        );
                                        return;
                                    }
                                }
                            }
                            Err(e) => {
                                app.vrp_status = crate::app::Status::Error(format!(
                                    "Failed to read waypoints: {}",
                                    e
                                ));
                                app.log(
                                    crate::app::LogLevel::Error,
                                    format!("VRP failed: cannot read waypoints: {}", e),
                                );
                                return;
                            }
                        }

                        let solver_id = match algo.as_str() {
                            "greedy" => "default",
                            "savings" => "clarke_wright",
                            "local_search" => "two_opt",
                            "simulated_annealing" => "or_opt",
                            other => other,
                        };

                        let matrix = crate::core::vrp::utils::build_haversine_matrix(&stops, 40.0);

                        let vrp_input = crate::core::vrp::types::VRPSolverInput {
                            locations: stops,
                            num_vehicles: vehicles,
                            vehicle_capacity: capacity,
                            objective: crate::core::vrp::types::VrpObjective::MinDistance,
                            matrix: Some(matrix),
                            service_time_secs: Some(30.0),
                            use_time_windows: false,
                            window_open: None,
                            window_close: None,
                        };

                        match crate::core::vrp::registry::solve_with(solver_id, &vrp_input).await {
                            Ok(output) => {
                                if let Some(routes) = output.routes {
                                    let mut log_routes = Vec::new();
                                    // Create output directory
                                    if let Err(e) = std::fs::create_dir_all(&output_dir) {
                                        app.vrp_status = crate::app::Status::Error(format!(
                                            "Failed to create output dir: {}",
                                            e
                                        ));
                                        app.log(
                                            crate::app::LogLevel::Error,
                                            format!("VRP error: {}", e),
                                        );
                                        return;
                                    }

                                    for (i, route) in routes.iter().enumerate() {
                                        let path = format!(
                                            "{}/vehicle_{}.gpx",
                                            output_dir,
                                            i + 1
                                        );
                                        match crate::core::optimize::write_gpx_multi(
                                            &path,
                                            std::slice::from_ref(route),
                                        ) {
                                            Ok(_) => log_routes.push(path),
                                            Err(e) => {
                                                app.log(
                                                    crate::app::LogLevel::Error,
                                                    format!(
                                                        "Failed to write route {}: {}",
                                                        i + 1,
                                                        e
                                                    ),
                                                );
                                            }
                                        }
                                    }

                                    app.vrp_status = crate::app::Status::Done(format!(
                                        "VRP solved: {} routes, {} total distance",
                                        log_routes.len(),
                                        output.total_distance_km
                                    ));
                                    app.log(
                                        crate::app::LogLevel::Success,
                                        format!(
                                            "VRP complete: {} routes written to {}",
                                            log_routes.len(),
                                            output_dir
                                        ),
                                    );
                                } else {
                                    app.vrp_status =
                                        crate::app::Status::Done("VRP solved (no routes)".into());
                                    app.log(
                                        crate::app::LogLevel::Success,
                                        "VRP complete: no routes produced",
                                    );
                                }
                            }
                            Err(e) => {
                                app.vrp_status = crate::app::Status::Error(e.clone());
                                app.log(
                                    crate::app::LogLevel::Error,
                                    format!("VRP failed: {}", e),
                                );
                            }
                        }
                    }
                    Err(e) => {
                        app.vrp_status =
                            crate::app::Status::Error(format!("Invalid depot: {}", e));
                        app.log(
                            crate::app::LogLevel::Error,
                            format!("VRP failed: {}", e),
                        );
                    }
                }
            } else {
                app.log(
                    crate::app::LogLevel::Warn,
                    "Set an input .rmp file first (press 'i')",
                );
            }
        }
        _ => {}
    }
}

fn handle_browse_maps_keys(app: &mut App, code: KeyCode) {
    match code {
        KeyCode::Enter if !app.cached_maps.is_empty() => {
            let selected = app.cached_maps[app.browse_selection].clone();
            app.cache_file = Some(selected.clone());
            app.current_view = View::Optimize;
            app.log(
                crate::app::LogLevel::Info,
                format!("Selected map for optimization: {}", selected),
            );
        }
        KeyCode::Char('d') | KeyCode::Char('D') if !app.cached_maps.is_empty() => {
            let name = app.cached_maps.remove(app.browse_selection);
            app.log(crate::app::LogLevel::Warn, format!("Deleted: {}", name));
            if app.browse_selection >= app.cached_maps.len() && !app.cached_maps.is_empty() {
                app.browse_selection = app.cached_maps.len() - 1;
            }
        }
        KeyCode::Char('r') | KeyCode::Char('R') => {
            app.log(crate::app::LogLevel::Info, "Refreshing cached maps...");
        }
        _ => {}
    }
}

fn handle_browse_routes_keys(app: &mut App, code: KeyCode) {
    match code {
        KeyCode::Enter if !app.saved_routes.is_empty() => {
            let selected = app.saved_routes[app.browse_selection].clone();
            app.log(
                crate::app::LogLevel::Info,
                format!("Viewing route: {}", selected),
            );
        }
        KeyCode::Char('d') | KeyCode::Char('D') if !app.saved_routes.is_empty() => {
            let name = app.saved_routes.remove(app.browse_selection);
            app.log(crate::app::LogLevel::Warn, format!("Deleted: {}", name));
            if app.browse_selection >= app.saved_routes.len() && !app.saved_routes.is_empty() {
                app.browse_selection = app.saved_routes.len() - 1;
            }
        }
        KeyCode::Char('r') | KeyCode::Char('R') => {
            app.log(crate::app::LogLevel::Info, "Refreshing saved routes...");
        }
        _ => {}
    }
}

fn handle_file_browser_keys(app: &mut App, code: KeyCode) {
    match code {
        KeyCode::Up => {
            if let Some(browser) = app.file_browser.as_mut() {
                browser.navigate_up();
            }
        }
        KeyCode::Down => {
            if let Some(browser) = app.file_browser.as_mut() {
                browser.navigate_down();
            }
        }
        KeyCode::Enter => {
            let selected_path = if let Some(browser) = app.file_browser.as_mut() {
                browser.select_entry()
            } else {
                None
            };

            if let Some(path) = selected_path {
                // File selected, close browser and populate field
                app.close_file_browser(Some(path));
            }
        }
        KeyCode::Backspace => {
            if let Some(browser) = app.file_browser.as_mut() {
                browser.go_parent();
            }
        }
        KeyCode::Char('h') | KeyCode::Char('H') => {
            let show_hidden = if let Some(browser) = app.file_browser.as_mut() {
                browser.show_hidden = !browser.show_hidden;
                let _ = browser.refresh_entries();
                browser.show_hidden
            } else {
                return;
            };

            app.log(
                crate::app::LogLevel::Info,
                format!(
                    "Hidden files: {}",
                    if show_hidden { "shown" } else { "hidden" }
                ),
            );
        }
        KeyCode::Esc => {
            // Cancel file browser
            app.close_file_browser(None);
            app.log(crate::app::LogLevel::Info, "File browser cancelled");
        }
        _ => {}
    }
}

fn handle_clean_keys(app: &mut App, code: KeyCode, _mods: KeyModifiers) {
    match code {
        KeyCode::Char('i') | KeyCode::Char('I') => {
            app.start_file_browser(InputField::CleanInputFile);
        }
        KeyCode::Char('o') | KeyCode::Char('O') => {
            app.start_file_browser(InputField::CleanOutputFile);
        }
        KeyCode::Char('r') | KeyCode::Char('R') => {
            app.clean_options = CleanOptions::default();
            app.log(
                crate::app::LogLevel::Info,
                "Clean options reset to defaults",
            );
        }
        KeyCode::Char(' ') => {
            toggle_clean_option(app);
        }
        KeyCode::Enter => {
            run_clean(app);
        }
        KeyCode::Up if app.clean_selection > 0 => {
            app.clean_selection -= 1;
        }
        KeyCode::Down if app.clean_selection < 14 => {
            app.clean_selection += 1;
        }
        _ => {}
    }
}

fn toggle_clean_option(app: &mut App) {
    match app.clean_selection {
        0 => app.clean_options.make_valid = !app.clean_options.make_valid,
        1 => app.clean_options.drop_invalid = !app.clean_options.drop_invalid,
        2 => app.clean_options.remove_selfloops = !app.clean_options.remove_selfloops,
        3 => app.clean_options.dedupe_edges = !app.clean_options.dedupe_edges,
        4 => app.clean_options.remove_isolates = !app.clean_options.remove_isolates,
        5 => app.clean_options.merge_node_positions = !app.clean_options.merge_node_positions,
        6 => app.clean_options.include_polygons = !app.clean_options.include_polygons,
        7 => app.clean_options.include_points = !app.clean_options.include_points,
        8 => app.clean_options.merge_parallel_edges = !app.clean_options.merge_parallel_edges,
        9 => {
            app.clean_options.merge_parallel_edge_properties =
                !app.clean_options.merge_parallel_edge_properties
        }
        10 => {
            app.clean_options.min_length_m += 0.1;
        }
        11 => {
            app.clean_options.node_snap_m += 0.5;
        }
        12 => {
            app.clean_options.max_components = app.clean_options.max_components.saturating_add(1);
        }
        13 => {
            app.clean_options.simplify_tolerance_m += 0.5;
        }
        14 => {
            app.clean_options.node_precision_decimals = app
                .clean_options
                .node_precision_decimals
                .saturating_add(1)
                .min(12);
        }
        _ => {}
    }
}

fn run_clean(app: &mut App) {
    use crate::core::clean::clean_geojson;
    use std::io::Read;

    if let Some(input_path) = app.clean_input_file.clone() {
        app.clean_status = Status::Running {
            progress: 0,
            message: "Cleaning GeoJSON...".to_string(),
        };
        app.log(crate::app::LogLevel::Info, "Starting GeoJSON cleaning");

        let output_path = app.clean_output_file.clone().unwrap_or_else(|| {
            input_path
                .replace(".geojson", ".cleaned.geojson")
                .replace(".json", ".cleaned.json")
        });

        // Read input file
        let mut input_data = Vec::new();
        match std::fs::File::open(input_path) {
            Ok(mut file) => {
                if let Err(e) = file.read_to_end(&mut input_data) {
                    app.clean_status = Status::Error(format!("Failed to read input: {}", e));
                    app.log(crate::app::LogLevel::Error, format!("Read error: {}", e));
                    return;
                }
            }
            Err(e) => {
                app.clean_status = Status::Error(format!("Failed to open input: {}", e));
                app.log(crate::app::LogLevel::Error, format!("Open error: {}", e));
                return;
            }
        };

        // Parse GeoJSON
        let geojson: geojson::FeatureCollection = match serde_json::from_slice(&input_data) {
            Ok(fc) => fc,
            Err(e) => {
                app.clean_status = Status::Error(format!("Failed to parse GeoJSON: {}", e));
                app.log(crate::app::LogLevel::Error, format!("Parse error: {}", e));
                return;
            }
        };

        // Run cleaning
        match clean_geojson(&geojson, &app.clean_options) {
            Ok((cleaned, stats, warnings)) => {
                // Write output
                let output_json = serde_json::to_string_pretty(&cleaned).unwrap_or_default();
                match std::fs::write(&output_path, &output_json) {
                    Ok(_) => {
                        let summary = stats.summary();
                        for warning in &warnings {
                            app.log(crate::app::LogLevel::Warn, warning.clone());
                        }
                        app.clean_status = Status::Done(summary.clone());
                        app.log(
                            crate::app::LogLevel::Success,
                            format!(
                                "Cleaning complete: {} → {} features",
                                stats.input_features, stats.output_features
                            ),
                        );
                        app.log(
                            crate::app::LogLevel::Info,
                            format!("Output saved to: {}", output_path),
                        );
                    }
                    Err(e) => {
                        app.clean_status = Status::Error(format!("Failed to write output: {}", e));
                        app.log(crate::app::LogLevel::Error, format!("Write error: {}", e));
                    }
                }
            }
            Err(e) => {
                app.clean_status = Status::Error(e.to_string());
                app.log(
                    crate::app::LogLevel::Error,
                    format!("Cleaning failed: {}", e),
                );
            }
        }
    } else {
        app.log(
            crate::app::LogLevel::Warn,
            "Set an input file first (press 'i')",
        );
    }
}
