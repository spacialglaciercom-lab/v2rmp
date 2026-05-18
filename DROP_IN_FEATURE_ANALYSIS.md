# Drop-In Feature Feasibility Analysis

## Executive Summary

**YES, it is feasible to add a drop-in feature for OSM and GeoJSON files.** The application already has the core infrastructure to support this functionality. This document outlines the technical analysis and proposed implementation approaches.

---

## Current State Analysis

### Existing File Input Mechanisms

The application currently uses **text-based input prompts** for file paths:

1. **Extract View** (`src/ui/extract.rs`):
   - User presses 'b' to enter bounding box coordinates
   - Extracts data from OSM/Overture APIs
   - Outputs to GeoJSON format

2. **Compile View** (`src/ui/compile.rs`):
   - User presses 'i' to input GeoJSON file path
   - User presses 'o' to set output .rmp file path (optional)
   - Compiles GeoJSON → binary .rmp format

3. **Optimize View** (`src/ui/optimize.rs`):
   - User presses 'c' to input cache file (.rmp)
   - User presses 'r' to input route file

### Supported File Formats

**Currently Supported:**
- **GeoJSON** (`.geojson`, `.json`) - Input for compilation
- **RMP Binary** (`.rmp`) - Compiled map format for optimization
- **JSON** - Route output format

**Dependencies Available:**
```toml
geojson = "0.24"        # GeoJSON parsing
serde_json = "1"        # JSON handling
```

**OSM Support:** The codebase mentions OSM as a data source but currently only extracts from OSM API via bounding box, not from local OSM files (`.osm`, `.pbf`).

### Technical Constraints

**TUI Environment Limitations:**
1. **No Native File Picker**: Terminal UIs cannot display graphical file dialogs
2. **No Drag-and-Drop**: Terminal applications don't support drag-and-drop operations
3. **Text-Based Input Only**: All interactions must be keyboard-driven

**Current Input Flow:**
```
User → Press Key → Input Prompt → Type Path → Enter → Process File
```

---

## Feasibility Assessment

### ✅ What IS Feasible

1. **File Browser Widget** - Navigate filesystem within TUI
2. **Recent Files List** - Quick access to previously used files
3. **File Watcher** - Monitor a designated "drop" directory
4. **Clipboard Integration** - Paste file paths from system clipboard
5. **Autocomplete** - Tab-completion for file paths
6. **Batch Processing** - Process multiple files from a directory

### ❌ What is NOT Feasible (in pure TUI)

1. **True Drag-and-Drop** - Requires GUI environment
2. **Visual File Preview** - Limited by terminal capabilities
3. **Mouse-Based File Selection** - Terminal mouse support is limited

### 🔄 Hybrid Approaches

1. **Watch Directory + TUI** - External process monitors folder, TUI displays available files
2. **CLI + TUI Mode** - Accept file paths as CLI arguments, then enter TUI
3. **Web Interface Bridge** - Companion web UI for file selection, TUI for processing

---

## Proposed Implementation Approaches

### Approach 1: Interactive File Browser (Recommended)

**Description:** Add a built-in file browser widget to navigate and select files within the TUI.

**Implementation:**
```rust
// New view: FileBrowser
pub enum View {
    Home,
    Extract,
    Compile,
    Optimize,
    BrowseMaps,
    BrowseRoutes,
    FileBrowser,  // NEW
    Help,
}

// File browser state
pub struct FileBrowser {
    current_path: PathBuf,
    entries: Vec<DirEntry>,
    selection: usize,
    filter: FileFilter,  // .geojson, .osm, .pbf, etc.
    target_field: InputField,  // Which field to populate
}
```

**User Flow:**
1. User presses 'i' in Compile view
2. Instead of text input, opens file browser view
3. Navigate with arrow keys, Enter to select directory/file
4. Selected file path populates the input field
5. Return to previous view

**Pros:**
- Native TUI experience
- No external dependencies
- Works in any terminal
- Intuitive navigation

**Cons:**
- Requires new UI component development
- More complex than text input
- May be slower for power users who know exact paths

**Effort:** Medium (2-3 days)

---

### Approach 2: Watch Directory Pattern

**Description:** Monitor a designated "drop" directory for new files, automatically detect and offer to process them.

**Implementation:**
```rust
// Add to App state
pub struct App {
    // ... existing fields
    watch_dir: PathBuf,  // e.g., ~/rmpca/drop/
    detected_files: Vec<DetectedFile>,
}

pub struct DetectedFile {
    path: PathBuf,
    file_type: FileType,  // GeoJSON, OSM, PBF
    detected_at: Instant,
}

// Background thread or periodic check
fn check_watch_directory(watch_dir: &Path) -> Vec<DetectedFile> {
    // Scan directory for supported file types
    // Return new files since last check
}
```

**User Flow:**
1. User copies/moves file to `~/rmpca/drop/` directory
2. TUI detects new file and shows notification
3. User presses key to view detected files
4. Select file to process

**Pros:**
- Feels like "drop-in" behavior
- Works with any file manager
- Can process files from external sources easily
- Non-intrusive

**Cons:**
- Requires file system monitoring
- User must know about the watch directory
- Extra step (copy to directory)

**Effort:** Low-Medium (1-2 days)

---

### Approach 3: CLI Arguments + TUI Mode

**Description:** Accept file paths as command-line arguments, then enter TUI for processing.

**Implementation:**
```rust
// Update main.rs
use clap::Parser;

#[derive(Parser)]
struct Args {
    /// GeoJSON file to compile
    #[arg(short, long)]
    input: Option<PathBuf>,
    
    /// Output .rmp file
    #[arg(short, long)]
    output: Option<PathBuf>,
    
    /// Start in specific view
    #[arg(short, long)]
    view: Option<String>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    
    // ... terminal setup
    
    let mut app = App::new();
    
    // Pre-populate from CLI args
    if let Some(input) = args.input {
        app.input_file = Some(input.to_string_lossy().to_string());
        app.current_view = View::Compile;
    }
    
    // ... main loop
}
```

**User Flow:**
```bash
# Drag file onto terminal or use shell completion
rmpca --input /path/to/map.geojson

# Or with output specified
rmpca -i map.geojson -o map.rmp --view compile
```

**Pros:**
- Leverages existing shell features (drag-drop to terminal, tab completion)
- Minimal code changes
- Power user friendly
- Works with scripts and automation

**Cons:**
- Not truly "in-app" drop-in
- Requires restarting app for new files
- Less discoverable for casual users

**Effort:** Low (few hours)

---

### Approach 4: Recent Files & Favorites

**Description:** Track recently used files and allow quick selection from a list.

**Implementation:**
```rust
// Add to App state
pub struct App {
    // ... existing fields
    recent_files: Vec<RecentFile>,
    favorites: Vec<PathBuf>,
}

pub struct RecentFile {
    path: PathBuf,
    file_type: FileType,
    last_used: SystemTime,
}

// New view for browsing recent files
pub enum View {
    // ... existing views
    RecentFiles,
}
```

**User Flow:**
1. User presses 'r' in Compile view
2. Shows list of recently used GeoJSON files
3. Navigate and select with arrow keys + Enter
4. File path populates input field

**Pros:**
- Quick access to frequently used files
- Complements other approaches
- Low friction for repeat workflows

**Cons:**
- Doesn't help with first-time file selection
- Requires persistent storage (config file)

**Effort:** Low (1 day)

---

### Approach 5: Enhanced Path Input with Autocomplete

**Description:** Improve the existing text input with shell-like tab completion and path suggestions.

**Implementation:**
```rust
pub struct InputMode {
    pub active: bool,
    pub field: InputField,
    pub buffer: String,
    pub suggestions: Vec<String>,  // NEW
    pub suggestion_index: usize,   // NEW
}

// On Tab key press
fn handle_tab_completion(app: &mut App) {
    let partial_path = &app.input_mode.buffer;
    app.input_mode.suggestions = get_path_completions(partial_path);
    // Cycle through suggestions or auto-complete if only one match
}
```

**User Flow:**
1. User starts typing path: `/home/user/ma`
2. Press Tab → completes to `/home/user/maps/`
3. Continue typing: `city`
4. Press Tab → shows suggestions: `city.geojson`, `city_roads.json`
5. Arrow keys to select, Enter to confirm

**Pros:**
- Familiar to terminal users
- Minimal UI changes
- Fast for users who know approximate paths

**Cons:**
- Still requires typing
- Not as visual as file browser
- May be confusing for non-technical users

**Effort:** Medium (2 days)

---

## OSM File Format Support

### Current Gap

The application currently only supports:
- **GeoJSON** input for compilation
- **OSM/Overture API** extraction (not local files)

### Adding OSM File Support

To support local OSM files (`.osm` XML or `.pbf` binary), you would need:

**Option A: Convert OSM → GeoJSON (Recommended)**
```rust
// Use existing tools
// 1. osmium: osmium export -f geojson input.osm.pbf -o output.geojson
// 2. ogr2ogr: ogr2ogr -f GeoJSON output.geojson input.osm.pbf

// Or add dependency
[dependencies]
osmpbf = "0.2"  // For .pbf files
quick-xml = "0.31"  // For .osm XML files
```

**Option B: Direct OSM Parsing**
```rust
// Add new compilation path
pub enum InputFormat {
    GeoJson,
    OsmXml,
    OsmPbf,
}

pub fn compile_from_osm(osm_path: &Path) -> Result<RmpData> {
    // Parse OSM file
    // Extract road network (ways with highway tag)
    // Convert to internal graph format
    // Write .rmp binary
}
```

**Recommendation:** Start with Option A (convert to GeoJSON first) as it reuses existing compilation logic. Add native OSM parsing later if needed for performance.

---

## Recommended Implementation Plan

### Phase 1: Quick Wins (Week 1)
1. **CLI Arguments** (Approach 3) - Immediate value, minimal effort
2. **Recent Files** (Approach 4) - Improves workflow for repeat users
3. **Better Error Messages** - Show supported formats, suggest corrections

### Phase 2: Enhanced Input (Week 2)
4. **Path Autocomplete** (Approach 5) - Improves text input experience
5. **File Type Detection** - Auto-detect format from extension/content
6. **Validation** - Check file exists and is readable before processing

### Phase 3: Advanced Features (Week 3-4)
7. **File Browser Widget** (Approach 1) - Full in-app file selection
8. **Watch Directory** (Approach 2) - Optional "drop zone" feature
9. **OSM Format Support** - Add .osm/.pbf parsing

### Phase 4: Polish (Week 5)
10. **Batch Processing** - Process multiple files at once
11. **File Metadata Display** - Show file size, modification date, preview
12. **Keyboard Shortcuts** - Quick access to file operations

---

## Technical Considerations

### Dependencies to Add

```toml
[dependencies]
# For file browser
dirs = "5.0"  # Cross-platform directory paths

# For path autocomplete
shellexpand = "3.1"  # Expand ~ and environment variables

# For OSM support (optional)
osmpbf = "0.2"  # Parse .pbf files
quick-xml = "0.31"  # Parse .osm XML files

# For file watching (optional)
notify = "6.1"  # File system notifications
```

### Cross-Platform Compatibility

All proposed approaches work on:
- ✅ Linux
- ✅ macOS  
- ✅ Windows (with minor path handling adjustments)

### Performance Considerations

- **File Browser**: Lazy-load large directories (>1000 files)
- **Watch Directory**: Use debouncing to avoid excessive checks
- **Autocomplete**: Cache directory listings for faster suggestions

---

## Security Considerations

1. **Path Traversal**: Validate file paths to prevent directory traversal attacks
2. **File Size Limits**: Check file size before loading to prevent memory exhaustion
3. **File Type Validation**: Verify file format matches extension (magic bytes)
4. **Permissions**: Handle permission errors gracefully

```rust
fn validate_input_file(path: &Path) -> Result<()> {
    // Check file exists
    if !path.exists() {
        return Err(anyhow!("File does not exist"));
    }
    
    // Check file size (e.g., max 1GB)
    let metadata = std::fs::metadata(path)?;
    if metadata.len() > 1_000_000_000 {
        return Err(anyhow!("File too large (max 1GB)"));
    }
    
    // Check file extension
    let ext = path.extension()
        .and_then(|e| e.to_str())
        .ok_or_else(|| anyhow!("Invalid file extension"))?;
    
    if !matches!(ext, "geojson" | "json" | "osm" | "pbf") {
        return Err(anyhow!("Unsupported file format: {}", ext));
    }
    
    Ok(())
}
```

---

## User Experience Mockups

### File Browser View
```
┌─ File Browser ─────────────────────────────────────────────────┐
│ Path: /home/user/maps/                                         │
│                                                                 │
│ [📁] ..                                                         │
│ [📁] cities/                                                    │
│ [📄] downtown.geojson                    2.3 MB   2026-05-01   │
│ [📄] suburbs.geojson                     5.1 MB   2026-04-28   │
│ [📄] highways.json                       1.8 MB   2026-04-25   │
│ [📁] regions/                                                   │
│                                                                 │
│ Filter: [*.geojson *.json]                                     │
│                                                                 │
│ [↑↓] Navigate  [Enter] Select  [Backspace] Parent  [Esc] Cancel│
└─────────────────────────────────────────────────────────────────┘
```

### Recent Files View
```
┌─ Recent Files ─────────────────────────────────────────────────┐
│                                                                 │
│ ▸ downtown.geojson                       Used 2 hours ago      │
│   suburbs.geojson                        Used yesterday        │
│   highways.json                          Used 3 days ago       │
│   city_center.geojson                    Used last week        │
│                                                                 │
│ [↑↓] Navigate  [Enter] Select  [d] Remove  [Esc] Cancel        │
└─────────────────────────────────────────────────────────────────┘
```

### Enhanced Input with Autocomplete
```
┌─ Input GeoJSON file path ──────────────────────────────────────┐
│                                                                 │
│ /home/user/maps/city█                                          │
│                                                                 │
│ Suggestions:                                                    │
│   city.geojson                                                  │
│   city_roads.json                                               │
│   city_center.geojson                                           │
│                                                                 │
│ [Tab] Complete  [↑↓] Select  [Enter] Confirm  [Esc] Cancel     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Conclusion

**Drop-in file support is absolutely feasible** for this TUI application. The recommended approach is a **phased implementation**:

1. **Start with CLI arguments** (quick win, immediate value)
2. **Add recent files list** (improves repeat workflows)  
3. **Build file browser widget** (best in-app experience)
4. **Optionally add watch directory** (for true "drop-in" feel)

The existing codebase is well-structured to support these additions with minimal refactoring. The main work involves:
- Creating new UI components (file browser, recent files list)
- Enhancing input handling (autocomplete, validation)
- Adding file system utilities (directory scanning, path completion)

**Estimated Total Effort:** 3-4 weeks for full implementation of all phases.

**Minimum Viable Feature:** CLI arguments + recent files = 2-3 days of work.
