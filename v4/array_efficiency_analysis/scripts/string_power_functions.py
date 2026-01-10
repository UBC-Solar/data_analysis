from data_tools import TimeSeries, DBClient
from datetime import datetime

def get_string_power(start: datetime, end: datetime, array_string: str, current_name: str, volt_name: str, client: DBClient) -> TimeSeries:
    '''
    Finds the total power of a string over a period of time via P = IV
    '''
    print(f"{current_name}{array_string}")
    print(f"{volt_name}{array_string}")
    current:TimeSeries = client.query_time_series(start, end, f"{current_name}{array_string}")
    voltage:TimeSeries = client.query_time_series(start, end, f"{volt_name}{array_string}")
    current, voltage = TimeSeries.align(current, voltage)
    power = current.promote(current * voltage)
    return power

def get_total_power(start: datetime, end: datetime, array_strings: list[str], current_name: str, volt_name: str, client: DBClient = None) -> TimeSeries:
    '''
    Finds the total power of a set of strings over a period of time
    
    :param start: Description
    :type start: datetime
    :param end: Description
    :type end: datetime
    :param array_strings: List of array strings (Eg. [1, 2] or ["A", "B", "C"])
    :type array_strings: list[str]
    :param current_name: The name of the current sensors
    :type current_name: str
    :param volt_name: The name of the voltage sensors
    :type volt_name: str
    :return: A time series of the power
    :rtype: TimeSeries
    '''
    if client == None: client = DBClient(url="100.120.214.69:8086")

    total_power = None

    for array_string in array_strings:
        power_in_string:TimeSeries = get_string_power(start, end, array_string, current_name, volt_name, client)

        if total_power is None:
            total_power:TimeSeries = power_in_string
        else:
            total_power, power_in_string = TimeSeries.align(total_power, power_in_string)
            total_power = total_power.promote(power_in_string + total_power)

    return total_power
