import numpy as np
import matplotlib.pyplot as plt

def plot_forces(
    drag_forces,
    g_forces,
    acceleration_force,
    net_force,
    cornering_force,
    required_speed_kmh,
    gradients
):
    """
    Plots the calculated forces (drag, gravity, road friction, acceleration) on one axis,
    with speed and gradient on their own separate y-axes.

    :param np.ndarray ticks: Time steps or indices.
    :param np.ndarray drag_forces: Drag forces acting on the vehicle.
    :param np.ndarray g_forces: Gravity forces due to gradients.
    :param np.ndarray road_friction_array: Road friction forces.
    :param np.ndarray acceleration_force: Forces due to acceleration.
    :param np.ndarray net_force: Total net force acting on the vehicle.
    :param np.ndarray required_speed_kmh: Vehicle speed in km/h.
    :param np.ndarray gradients: Road gradient values.
    """

    # --- PLOTTING ---
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ticks = np.arange(len(drag_forces))
    # Plot forces on the primary y-axis
    ax1.plot(ticks, drag_forces, linestyle='-', label="Drag Force", color='blue')
    ax1.plot(ticks, g_forces, linestyle='-', label="Gravity Force", color='red')
    ax1.plot(ticks, acceleration_force, linestyle='-', label="Acceleration Force", color='orange')
    ax1.plot(ticks, net_force, linestyle='-', label="Net Force", color='black', linewidth=2)
    ax1.plot(ticks, cornering_force, linestyle='-', label="Cornering Force", color='purple', linewidth=2)

    ax1.set_xlabel("Tick")
    ax1.set_ylabel("Force (N)")
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True)

    # --- SECONDARY Y-AXIS for Speed ---
    ax2 = ax1.twinx()
    ax2.plot(ticks, required_speed_kmh, linestyle='-', label="Vehicle Speed (km/h)", color='green', alpha=0.7)
    ax2.set_ylabel("Speed (km/h)", color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    # --- THIRD Y-AXIS for Gradient ---
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 60))  # Offset third y-axis to the right
    ax3.plot(ticks, gradients, linestyle='-', label="Gradient", color='brown', alpha=0.7)
    ax3.set_ylabel("Gradient", color='brown')
    ax3.tick_params(axis='y', labelcolor='brown')

    # Add legends
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax3.legend(loc="lower right")

    plt.title("Forces Acting on Vehicle vs Speed & Gradient")
    plt.show()

def plot_power(motor_power_predicted, motor_power_measured, vehicle_velocity, gradient_global):
    ticks = np.arange(len(motor_power_predicted))

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot motor power (Predicted & Measured) on the primary y-axis
    ax1.plot(ticks, motor_power_predicted, linestyle='-', label="Predicted Motor Power", color='blue')
    ax1.plot(ticks, motor_power_measured, linestyle='-', label="Measured Motor Power", color='red')
    ax1.set_xlabel("Tick")
    ax1.set_ylabel("Power (Watts)", color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True)

    # Create secondary y-axis for vehicle velocity
    ax2 = ax1.twinx()
    ax2.plot(ticks, vehicle_velocity, linestyle='-', label="Vehicle Velocity", color='green')
    ax2.set_ylabel("Vehicle Velocity (km/h)", color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    ax3 = ax1.twinx()
    ax3.plot(ticks, gradient_global, linestyle='-', label="Gradient", color='orange')
    ax3.set_ylabel("Gradient", color='orange')
    ax3.tick_params(axis='y', labelcolor='orange')

    # Add legends for both plots
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    # Set title
    plt.title("Motor Power vs Vehicle Velocity")

    # Show the plot
    plt.show()



def plot_cornering_data(slip_distances, slip_angles, centripetal_lateral_force):
    ticks = np.arange(len(slip_angles))  # X-axis as ticks from 0 to len(speed) (same size for all arrays)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Left y-axis (Centripetal Lateral Force)
    ax1.set_xlabel("Ticks")
    ax1.set_ylabel("Centripetal Lateral Force (N)", color="r")
    ax1.plot(ticks, centripetal_lateral_force, label="Centripetal Lateral Force (N)", linestyle="-", color="r")
    ax1.tick_params(axis='y', labelcolor="r")

    # Third y-axis (Slip Distances)
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 0))  # Offset for clarity
    ax3.set_ylabel("Slip Distances (m)", color="g")
    ax3.plot(ticks, slip_distances, label="Slip Distances (m)", linestyle="-", color="g")
    ax3.tick_params(axis='y', labelcolor="g")

    # Fourth y-axis (Slip Angles)
    ax4 = ax1.twinx()
    ax4.spines["right"].set_position(("outward", 60))  # Further offset for clarity
    ax4.set_ylabel("Slip Angles (degrees)", color="c")
    ax4.plot(ticks, slip_angles, label="Slip Angles (degrees)", linestyle="-", color="c")
    ax4.tick_params(axis='y', labelcolor="c")

    # Legends
    ax1.legend(loc="upper left")  # Lateral force on left
    ax3.legend(loc="upper right")
    ax4.legend(loc="lower right")

    plt.title("Race Dynamics: Speed, Slip Distances, Slip Angles, and Lateral Forces")
    plt.grid(True)
    plt.show()

def plot_singe_value(value):
    ticks = np.arange(len(value))  # X-axis as ticks from 0 to len(speed) (same size for all arrays)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Left y-axis (Centripetal Lateral Force)
    ax1.set_xlabel("Ticks")
    ax1.set_ylabel("Centripetal Lateral Force (N)", color="r")
    ax1.plot(ticks, value, label="Centripetal Lateral Force (N)", linestyle="-", color="r")
    ax1.tick_params(axis='y', labelcolor="r")


    plt.title('singe value plot')
    plt.grid(True)
    plt.show()