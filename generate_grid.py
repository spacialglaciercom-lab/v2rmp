import json

features = []
grid_size = 10
spacing = 0.01 # Approx 1km

# Generate Horizontal Streets
for y in range(grid_size):
    oneway = "yes" if y % 2 == 0 else "no"
    for x in range(grid_size - 1):
        features.append({
            "type": "Feature",
            "properties": {
                "name": f"Street H{y}",
                "oneway": oneway,
                "highway": "tertiary" if y % 3 == 0 else "residential",
                "maxspeed": 50 if y % 3 == 0 else 30
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [-74.0 + x * spacing, 40.7 + y * spacing],
                    [-74.0 + (x + 1) * spacing, 40.7 + y * spacing]
                ]
            }
        })

# Generate Vertical Streets
for x in range(grid_size):
    oneway = "yes" if x % 2 == 0 else "no"
    for y in range(grid_size - 1):
        features.append({
            "type": "Feature",
            "properties": {
                "name": f"Avenue V{x}",
                "oneway": oneway,
                "highway": "secondary" if x % 4 == 0 else "residential",
                "maxspeed": 60 if x % 4 == 0 else 30
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [-74.0 + x * spacing, 40.7 + y * spacing],
                    [-74.0 + x * spacing, 40.7 + (y + 1) * spacing]
                ]
            }
        })

geojson = {
    "type": "FeatureCollection",
    "features": features
}

print(json.dumps(geojson, indent=2))
