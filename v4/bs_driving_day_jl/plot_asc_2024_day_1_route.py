import json
import folium
import matplotlib.pyplot as plt
import numpy as np

def main():
    # 1. Load the JSON data
    file_path = 'A_Nashville_to_Paducah.route.json'
    
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"Error: Could not find {file_path}. Ensure it is in the same directory as this script.")
        return

    sections = data.get('Sections', [])
    if not sections:
        print("Error: No sections found in the route data.")
        return

    # Lists to hold our extracted and calculated data
    coordinates = []
    elevations = []
    distances_miles = []
    gradients = []
    lengths_ft = []
    indices = []

    cumulative_distance_ft = 0.0

    # 2. Extract coordinates, elevations, and calculate gradients
    for section in sections:
        start_coord = (section['CoordinatesInitial']['Latitude'], section['CoordinatesInitial']['Longitude'])
        coordinates.append(start_coord)

        elev_initial = section['ElevationInitialFt']
        elev_final = section['ElevationFinalFt']
        length_ft = section['LengthFt']

        elevations.append(elev_initial)
        distances_miles.append(cumulative_distance_ft / 5280.0)  # Convert feet to miles

        # Calculate gradient as a percentage
        if length_ft > 0:
            gradient = ((elev_final - elev_initial) / length_ft) * 100
        else:
            gradient = 0.0

        if np.abs(gradient) > 50:
            print("\nGradient above 50! Printing section...")
            print(section)

        gradients.append(gradient)

        cumulative_distance_ft += length_ft

        # JL ADDED
        lengths_ft.append(length_ft)
        indices.append(section['PositionInRoute'])

    # Add the final endpoint data
    last_section = sections[-1]
    final_coord = (last_section['CoordinatesFinal']['Latitude'], last_section['CoordinatesFinal']['Longitude'])
    coordinates.append(final_coord)
    elevations.append(last_section['ElevationFinalFt'])
    distances_miles.append(cumulative_distance_ft / 5280.0)

    # 3. Create the Folium Map
    # Center map at the midpoint of the route
    mid_index = len(coordinates) // 2
    map_center = coordinates[mid_index]
    route_map = folium.Map(location=map_center, zoom_start=9)

    # Draw the route polyline
    folium.PolyLine(
        locations=coordinates,
        color='blue',
        weight=5,
        opacity=0.8,
        tooltip='Nashville to Paducah Route'
    ).add_to(route_map)

    # Add markers for the Start and End points
    folium.Marker(
        coordinates[0], 
        popup='Start: Nashville', 
        icon=folium.Icon(color='green', icon='play')
    ).add_to(route_map)
    
    folium.Marker(
        coordinates[-1], 
        popup='End: Paducah', 
        icon=folium.Icon(color='red', icon='stop')
    ).add_to(route_map)

    # Save the interactive map to an HTML file
    map_filename = 'nashville_to_paducah_map.html'
    route_map.save(map_filename)
    print(f"Folium map successfully saved to {map_filename}")

    # 4. Plot Elevation and Gradient using Matplotlib
    plt.figure(figsize=(12, 8))

    # Elevation Profile Plot
    plt.subplot(2, 1, 1)
    plt.plot(distances_miles, elevations, color='dodgerblue', linewidth=2)
    plt.title('Elevation Profile: Nashville to Paducah', fontsize=14, fontweight='bold')
    plt.ylabel('Elevation (ft)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    # Fill the area under the elevation curve for a nicer visual
    plt.fill_between(distances_miles, elevations, min(elevations) - 20, color='dodgerblue', alpha=0.2)

    # Gradient Profile Plot
    plt.subplot(2, 1, 2)
    # Use a step plot for gradients since each gradient represents a constant rate over a specific section
    plt.step(distances_miles[:-1], gradients, where='post', color='crimson', linewidth=1.5)
    plt.title('Route Gradient Profile', fontsize=14, fontweight='bold')
    plt.xlabel('Distance (miles)', fontsize=12)
    plt.ylabel('Gradient (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(0, color='black', linewidth=1)  # Add a baseline at 0% gradient for reference

    plt.tight_layout()
    
    # Save the Matplotlib figures
    plot_filename = 'elevation_and_gradient.png'
    plt.savefig(plot_filename, dpi=300)
    print(f"Elevation and gradient plots successfully saved to {plot_filename}")

    # Display the plots
    plt.show()

    # 4. JL ADDED plots
    # plt.plot(indices, lengths_ft, alpha=0.7)
    # plt.xlabel("Index")
    # plt.ylabel("Route Length [ft]")

    # plt.twinx()
    # plt.plot(indices, np.diff(elevations), color='red', alpha=0.7)
    # plt.xlabel("Index")
    # plt.ylabel("Delta Elevation [ft]")

    # plt.show()

if __name__ == "__main__":
    main()