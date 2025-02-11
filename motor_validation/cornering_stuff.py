from haversine import haversine, Unit
import numpy as np
import pickle
from pathlib import Path

# save in this current directory
race_directory = Path.cwd()

def calculate_meter_distance(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    # Base coordinate
    coord_base = (lat1, lon1)
    # Coordinate for latitude difference (keep longitude the same)
    coord_lat = (lat2, lon1)
    # Coordinate for longitude difference (keep latitude the same)
    coord_long = (lat1, lon2)

    # Calculate y distance (latitude difference) using haversine function
    y_distance = haversine(coord_base, coord_lat, unit=Unit.METERS)
    # Calculate x distance (longitude difference) using haversine function
    x_distance = haversine(coord_base, coord_long, unit=Unit.METERS)

    if lat2 < lat1:
        y_distance = -y_distance
    if lon2 < lon1:
        x_distance = -x_distance

    return x_distance, y_distance


# uses circumcircle formula
def radius_of_curvature(x1, y1, x2, y2, x3, y3):
    numerator = np.sqrt(
        ((x3 - x2) ** 2 + (y3 - y2) ** 2) *
        ((x1 - x3) ** 2 + (y1 - y3) ** 2) *
        ((x2 - x1) ** 2 + (y2 - y1) ** 2)
    )

    denominator = 2 * abs(
        ((x2 - x1) * (y1 - y3) - (x1 - x3) * (y2 - y1))
    )

    return numerator / denominator


def write_slip_angles(min_degrees, max_degrees, num_elements):
    # coefficients for pacekja's majick formula
    # https://www.edy.es/dev/docs/pacejka-94-parameters-explained-a-comprehensive-guide/
    B = .5  # Stiffness (Example value for dry tarmac)
    C = 2.2 # Shape (Example value for dry tarmac)
    D = 2.75  # Peak (Example value for dry tarmac)
    E = 1.0  # Curvature (Example value for dry tarmac)

    # HARD CODED MASS OF DAYBREAK - 350 KG
    Fz = 350 * 9.81  # Normal load in Newtons

    slip_angles = np.linspace(min_degrees, max_degrees, num_elements)
    tire_forces = Fz * D * np.sin(C * np.arctan(B * slip_angles - E * (B * slip_angles - np.arctan(B * slip_angles))))

    with open(race_directory / "slip_angle_lookup.pkl", 'wb') as outfile:
        pickle.dump((slip_angles, tire_forces), outfile)


def read_slip_angle_lookup():
    # Deserialize the data points from the file
    with open(race_directory / "slip_angle_lookup.pkl", 'rb') as f:
        slip_angles, tire_forces = pickle.load(f)

    return slip_angles, tire_forces


def get_slip_angle_for_tire_force(desired_tire_force):
    # Read the lookup table data points
    slip_angles, tire_forces = read_slip_angle_lookup()

    # Use the numpy interpolation function to find slip angle for the given tire force
    # interpolation estimates unknown slip angle from a tire force that lies between known tire forces (from the lookup table)
    estimated_slip_angle = np.interp(desired_tire_force, tire_forces, slip_angles)

    return estimated_slip_angle


def calculate_radii(waypoints):
    # pop off last coordinate if first and last coordinate are the same
    repeated_last_coordinate = False
    if np.array_equal(waypoints[0], waypoints[len(waypoints) - 1]):
        waypoints = waypoints[:-1]
        repeated_last_coordinate = True

    cornering_radii = np.empty(len(waypoints))
    for i in range(len(waypoints)):
        # if the next point or previous point is out of bounds, wrap the index around the array
        i2 = (i - 1) % len(waypoints)
        i3 = (i + 1) % len(waypoints)
        current_point = waypoints[i]
        previous_point = waypoints[i2]
        next_point = waypoints[i3]

        x1 = 0
        y1 = 0
        x2, y2 = calculate_meter_distance(current_point, previous_point)
        x3, y3 = calculate_meter_distance(current_point, next_point)
        cornering_radii[i] = radius_of_curvature(x1, y1, x2, y2, x3, y3)

    # If the last coordinate was removed, duplicate the first radius value to the end of the array
    if repeated_last_coordinate:
        cornering_radii = np.append(cornering_radii, cornering_radii[0])

    # ensure that super large radii are bounded by a large number, like 1000
    cornering_radii = np.where(np.isnan(cornering_radii), 10000, cornering_radii)
    cornering_radii = np.where(cornering_radii > 10000, 10000, cornering_radii)

    # uncomment the line below to create the map.html file
    plot_coordinates(waypoints[:1000], cornering_radii[:1000])
    return cornering_radii


# This function is a debugging tool that generates a map.html file in the root directory of simulation
# The file will plot each waypoint of the race, hovering over a waypoint will display the cornering radius calculated at that point
# There is a measure tool in the top left corner which can be used to measure distances on the map
# be sure to import folium to use this plotting function

import folium
from folium.plugins import MeasureControl

def plot_coordinates(coords, data):
    # Calculate the center of your map
    center_lat = sum([coord[0] for coord in coords]) / len(coords)
    center_lon = sum([coord[1] for coord in coords]) / len(coords)

    # Create the map
    my_map = folium.Map(location=[center_lat, center_lon], zoom_start=6)

    # Add a measurement tool to the map for users to measure distance
    my_map.add_child(MeasureControl())

    # Add points with tooltips
    for coord, datum in zip(coords, data):
        folium.Marker(
            [coord[0], coord[1]],
            tooltip=folium.Tooltip(datum)
        ).add_to(my_map)

    # Save the map to an HTML file
    my_map.save("map.html")


if __name__ == "__main__":
    write_slip_angles(0, 80, 100000)