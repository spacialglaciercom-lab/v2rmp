import re

with open("src/app.rs", "r") as f:
    content = f.read()

helper_method = """
    fn handle_file_selection(&mut self, target_field: InputField, path_str: String) {
        match target_field {
            InputField::InputFile => {
                self.input_file = Some(path_str.clone());
                if self.output_file.is_none() {
                    let out = path_str
                        .replace(".geojson", ".rmp")
                        .replace(".json", ".rmp");
                    self.output_file = Some(out);
                }
                self.log(
                    LogLevel::Success,
                    format!("Input file selected: {}", path_str),
                );
                self.current_view = View::Compile;
            }
            InputField::OutputFile => {
                self.output_file = Some(path_str.clone());
                self.log(
                    LogLevel::Success,
                    format!("Output file selected: {}", path_str),
                );
                self.current_view = View::Compile;
            }
            InputField::CacheFile => {
                self.cache_file = Some(path_str.clone());
                self.log(
                    LogLevel::Success,
                    format!("Cache file selected: {}", path_str),
                );
                self.current_view = View::Optimize;
            }
            InputField::RouteFile => {
                self.route_file = Some(path_str.clone());
                self.log(
                    LogLevel::Success,
                    format!("Route file selected: {}", path_str),
                );
                self.current_view = View::Optimize;
            }
            InputField::CleanInputFile => {
                self.clean_input_file = Some(path_str.clone());
                if self.clean_output_file.is_none() {
                    let out = path_str
                        .replace(".geojson", ".cleaned.geojson")
                        .replace(".json", ".cleaned.json");
                    self.clean_output_file = Some(out);
                }
                self.log(
                    LogLevel::Success,
                    format!("Clean input file selected: {}", path_str),
                );
                self.current_view = View::Clean;
            }
            InputField::CleanOutputFile => {
                self.clean_output_file = Some(path_str.clone());
                self.log(
                    LogLevel::Success,
                    format!("Clean output file selected: {}", path_str),
                );
                self.current_view = View::Clean;
            }
            InputField::VrpWaypointsFile => {
                self.vrp_waypoints_file = Some(path_str.clone());
                self.log(
                    LogLevel::Success,
                    format!("VRP waypoints file selected: {}", path_str),
                );
                self.current_view = View::Vrp;
            }
            _ => {
                self.current_view = View::Home;
            }
        }
    }
"""

close_file_browser_pattern = r"""(    pub fn close_file_browser\(&mut self, selected_path: Option<PathBuf>\) \{
        let previous_view = self
            \.file_browser
            \.as_ref\(\)
            \.map\(\|b\| b\.previous_view\.clone\(\)\)
            \.unwrap_or\(View::Home\);

        if let Some\(path\) = selected_path \{
            let path_str = path\.to_string_lossy\(\)\.to_string\(\);

            if let Some\(browser\) = &self\.file_browser \{
                match browser\.target_field \{
(?:                    .*?
)+                \}
            \}
        \} else \{
            // User cancelled — restore the view they came from
            self\.current_view = previous_view;
        \}

        self\.file_browser = None;
    \})"""

new_close_file_browser = """    pub fn close_file_browser(&mut self, selected_path: Option<PathBuf>) {
        let previous_view = self
            .file_browser
            .as_ref()
            .map(|b| b.previous_view.clone())
            .unwrap_or(View::Home);

        if let Some(path) = selected_path {
            let path_str = path.to_string_lossy().to_string();

            if let Some(browser) = &self.file_browser {
                let target_field = browser.target_field.clone();
                self.handle_file_selection(target_field, path_str);
            }
        } else {
            // User cancelled — restore the view they came from
            self.current_view = previous_view;
        }

        self.file_browser = None;
    }"""

match = re.search(close_file_browser_pattern, content)
if match:
    new_content = content[:match.end()] + "\n" + helper_method + content[match.end():]
    new_content = new_content.replace(match.group(1), new_close_file_browser)
    with open("src/app.rs", "w") as f:
        f.write(new_content)
    print("Updated app.rs")
else:
    print("Could not find close_file_browser pattern")
