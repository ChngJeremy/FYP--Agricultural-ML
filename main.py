#collect sensor data from sensor.py
import time


from sensor import collect_sensor_data #import collect_sensor_data from sensor.py


def main():
    while True:
        temperature, moisture_level, ph_level, plant_condition = collect_sensor_data()
        print("Temperature: {}".format(temperature))
        print("Moisture level: {}".format(moisture_level))
        print("pH level: {}".format(ph_level))
        print("Plant condition: {}".format(plant_condition))
        print()
        time.sleep(1)

if __name__ == "__main__":
    main()
