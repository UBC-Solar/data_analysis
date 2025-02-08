import pandas as pd
import matplotlib.pyplot as plt

# coords_day3 = np.load(day3_path)
# coords_day1 = np.load(day1_path)

def load_data(path):
    data = pd.read_csv(path)
    values = data["Value"].to_numpy()
    return values


def plot_data(values, label='values'):
    # Plot the values
    plt.figure(figsize=(10, 6))
    plt.plot(values, label=label, linewidth=1.5)
    # plt.title("Motor Power")
    plt.xlabel("Time (s)")
    # plt.ylabel("Power (W)")
    plt.legend()
    plt.grid(True)
    plt.show()


motor_current = load_data("motor_current.csv")
plot_data(motor_current, "motor current")
motor_voltage = load_data("motor_voltage.csv")
plot_data(motor_voltage, "motor voltage")

vehicle_velocity = load_data("vehicle_velocity.csv")
plot_data(vehicle_velocity, "vehicle velocity")


motor_power = motor_current * motor_voltage
plot_data(motor_power, "motor power")