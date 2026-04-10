import json
import folium

def main():
    file_path = 'A_Nashville_to_Paducah.route.json'
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return

    sections = data.get('Sections', [])
    route_coords = []
    
    # Initialize Map (centered near the start)
    m = folium.Map(location=[36.1468, -86.7755], zoom_start=10)
    
    # Add the full route as a background reference
    for section in sections:
        route_coords.append((section['CoordinatesInitial']['Latitude'], section['CoordinatesInitial']['Longitude']))
    if sections:
        last = sections[-1]['CoordinatesFinal']
        route_coords.append((last['Latitude'], last['Longitude']))
    
    folium.PolyLine(route_coords, color="gray", weight=1, opacity=0.3, tooltip="Full Route").add_to(m)

    # Feature Group for Steep Segments
    steep_group = folium.FeatureGroup(name="Steep Gradients (>50%)").add_to(m)

    count = 0
    for section in sections:
        elev_init = section['ElevationInitialFt']
        elev_final = section['ElevationFinalFt']
        length = section['LengthFt']
        
        gradient = ((elev_final - elev_init) / length * 100) if length > 0 else 0
            
        if abs(gradient) > 50:
            count += 1
            start_pt = [section['CoordinatesInitial']['Latitude'], section['CoordinatesInitial']['Longitude']]
            end_pt = [section['CoordinatesFinal']['Latitude'], section['CoordinatesFinal']['Longitude']]
            
            color = 'red' if gradient > 0 else 'purple'
            
            # Create a shared tooltip for the segment and its points
            info_html = f"""
            <div style='font-family: Arial; width: 180px;'>
                <b style='color:{color};'>Segment #{section['PositionInRoute']}</b><br>
                <b>Gradient:</b> {gradient:.2f}%<br>
                <b>Start Elev:</b> {elev_init:.1f} ft<br>
                <b>End Elev:</b> {elev_final:.1f} ft<br>
                <b>Length:</b> {length:.2f} ft<br>
                <b>Limit:</b> {section['SpeedLimitMph']} mph<br>
                <b>Maneuver:</b> {section['ExitInstruction']}
            </div>
            """
            
            # 1. Draw the segment line connecting start and end
            folium.PolyLine(
                [start_pt, end_pt],
                color=color,
                weight=6,
                opacity=1,
                tooltip=folium.Tooltip(info_html)
            ).add_to(steep_group)

            # 2. Add Start Point Marker
            folium.CircleMarker(
                location=start_pt,
                radius=4,
                color='black',
                fill=True,
                fill_color='white',
                fill_opacity=1,
                tooltip=f"<b>START</b> of Segment #{section['PositionInRoute']}<br>{info_html}"
            ).add_to(steep_group)

            # 3. Add End Point Marker
            folium.CircleMarker(
                location=end_pt,
                radius=4,
                color='black',
                fill=True,
                fill_color=color,
                fill_opacity=1,
                tooltip=f"<b>END</b> of Segment #{section['PositionInRoute']}<br>{info_html}"
            ).add_to(steep_group)

    folium.LayerControl().add_to(m)
    
    output_name = 'steep_segments_pairs.html'
    m.save(output_name)
    print(f"Success! Plotted {count} steep segments with start/end point pairs.")
    print(f"File saved as: {output_name}")

if __name__ == "__main__":
    main()