import os
import subprocess
from pathlib import Path
from xml.etree import ElementTree


class BatteryReport:
    @classmethod
    def create_battery_report(cls, report_path: Path):
        cls.report_path = report_path
        process = subprocess.Popen(f"powercfg /batteryreport /xml /output {report_path}")
        process.communicate()
        return BatteryReport(report_path)

    def delete_battery_report(self):
        os.unlink(self.__report_path)

    def __init__(self, report_path: Path):
        self.__root = ElementTree.parse(report_path).getroot()
        self.__latest_battery_info = None
        self.__report_path = report_path

    def get_latest_battery_info(self):
        return self.__root.findall(
            "./{http://schemas.microsoft.com/battery/2012}RecentUsage/{http://schemas.microsoft.com/battery/2012}UsageEntry[@EntryType=\"ReportGenerated\"]")[
            0].attrib if self.__latest_battery_info is None else self.__latest_battery_info

    def is_plugged_in(self) -> bool:
        return self.get_latest_battery_info()["Discharge"] == "0"

    def current_mWh(self):
        return int(self.get_latest_battery_info()["ChargeCapacity"])


if __name__ == '__main__':
    BatteryReport.create_battery_report(Path(".", "battery_report.xml"))
