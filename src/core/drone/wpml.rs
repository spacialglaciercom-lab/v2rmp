use crate::core::drone::mod_types::{DroneVrpInstance, DroneVrpResult};
use anyhow::Result;

pub fn generate_wpml(instance: &DroneVrpInstance, result: &DroneVrpResult) -> Result<String> {
    let mut xml = String::new();
    xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    xml.push_str("<kml xmlns=\"http://www.opengis.net/kml/2.2\" xmlns:wpml=\"http://www.dji.com/wpmz/1.0.3\">\n");
    xml.push_str("  <Document>\n");
    xml.push_str("    <wpml:createTime>1680000000000</wpml:createTime>\n");
    xml.push_str("    <wpml:updateTime>1680000000000</wpml:updateTime>\n");
    xml.push_str("    <Folder>\n");
    xml.push_str("      <wpml:templateId>0</wpml:templateId>\n");
    xml.push_str("      <wpml:executeHeightMode>relativeToGround</wpml:executeHeightMode>\n");
    xml.push_str("      <wpml:waylineId>0</wpml:waylineId>\n");

    for (route_idx, route) in result.routes.iter().enumerate() {
        for (stop_idx, &cust_idx) in route.iter().enumerate() {
            let lat = instance.customers[cust_idx][0];
            let lon = instance.customers[cust_idx][1];

            xml.push_str("      <Placemark>\n");
            xml.push_str(&format!(
                "        <name>Route {} Stop {}</name>\n",
                route_idx + 1,
                stop_idx + 1
            ));
            xml.push_str("        <Point>\n");
            xml.push_str(&format!(
                "          <coordinates>{},{},50</coordinates>\n",
                lon, lat
            ));
            xml.push_str("        </Point>\n");
            xml.push_str("        <wpml:index>");
            xml.push_str(&(route_idx * 100 + stop_idx).to_string());
            xml.push_str("</wpml:index>\n");
            xml.push_str("        <wpml:actionGroup>\n");
            xml.push_str("          <wpml:actionGroupId>0</wpml:actionGroupId>\n");
            xml.push_str("          <wpml:actionGroupTrigger>\n");
            xml.push_str("            <wpml:actionGroupTriggerType>reachPoint</wpml:actionGroupTriggerType>\n");
            xml.push_str("          </wpml:actionGroupTrigger>\n");
            xml.push_str("        </wpml:actionGroup>\n");
            xml.push_str("      </Placemark>\n");
        }
    }

    xml.push_str("    </Folder>\n");
    xml.push_str("  </Document>\n");
    xml.push_str("</kml>");

    Ok(xml)
}
