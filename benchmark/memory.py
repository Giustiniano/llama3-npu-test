import datetime
import time
import wmi
import sys

import psutil

def monitor_system():
    filename = sys.argv[1]
    while True:
        psutil.cpu_percent()
        try:
            w = wmi.WMI(namespace=r"root\wmi")
        # convert from tenth of Kelvin to Celsius
            temperature_info = (w.MSAcpi_ThermalZoneTemperature()[0].CurrentTemperature * 10) - 273.15
        except Exception:
            temperature_info = "N/A"
        with open(filename, "a") as system_monitoring_file:
            if system_monitoring_file.tell() == 0:
                system_monitoring_file.write("TIMESTAMP,USED_MEMORY_GB,CPU_LOAD,CPU_TEMPERATURE")
            system_monitoring_file.write(f"{datetime.datetime.now().isoformat()},"
                                         f"{psutil.virtual_memory().used / 1024**3},"
                                         f"{psutil.cpu_percent(2)},{temperature_info}\n")
        time.sleep(2)


if __name__ == '__main__':
    monitor_system()