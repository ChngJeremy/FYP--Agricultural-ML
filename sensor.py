import cv2
import board
import adafruit_dht
import adafruit_mcp9808
import busio
import adafruit_ads1x15.ads1015 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

# Camera setup
camera = cv2.VideoCapture(0)

# DHT11 temperature and humidity sensor setup
dht_device = adafruit_dht.DHT11(board.D4)

# MCP9808 temperature sensor setup
i2c = busio.I2C(board.SCL, board.SDA)
mcp9808 = adafruit_mcp9808.MCP9808(i2c)

# ADS1015 ADC setup for CO2 sensor and moisture content sensor
ads = ADS.ADS1015(i2c)
adc_ch0 = AnalogIn(ads, ADS.P0)
adc_ch1 = AnalogIn(ads, ADS.P1)

def capture_image():
    _, frame = camera.read()
    return frame

def read_temperature_dht11():
    try:
        temperature = dht_device.temperature
        return temperature
    except RuntimeError as error:
        print(error)
        return None

def read_temperature_mcp9808():
    temperature = mcp9808.temperature
    return temperature

def read_humidity_dht11():
    try:
        humidity = dht_device.humidity
        return humidity
    except RuntimeError as error:
        print(error)
        return None

def read_co2_level():
    co2_voltage = adc_ch0.voltage
    # Conversion from voltage to CO2 level
    co2_level = convert_voltage_to_co2(co2_voltage)
    return co2_level

def convert_voltage_to_co2(voltage_reading):
    voltage_range = 5.0  # Maximum voltage range of the sensor
    co2_range = 5000  # Maximum CO2 range in ppm (parts per million)
    
    # Calculate the CO2 level based on the voltage reading and calibration parameters
    co2_level = (voltage_reading / voltage_range) * co2_range
    return co2_level

def read_moisture_content():
    moisture_voltage = adc_ch1.voltage
    # Conversion parameters specific to your moisture sensor
    voltage_min = 0.0  # Minimum voltage reading
    voltage_max = 3.3  # Maximum voltage reading
    moisture_min = 0  # Minimum moisture content
    moisture_max = 100  # Maximum moisture content
    
    # Calculate the moisture content based on the voltage reading and calibration parameters
    moisture_content = ((moisture_voltage - voltage_min) / (voltage_max - voltage_min)) * (moisture_max - moisture_min) + moisture_min
    
    return moisture_content

def collect_sensor_data():
    # Capture image
    image = capture_image()

    # Read sensor values
    temperature_dht11 = read_temperature_dht11()
    temperature_mcp9808 = read_temperature_mcp9808()
    humidity_dht11 = read_humidity_dht11()
    co2_level = read_co2_level()
    moisture_content = read_moisture_content()

    return image, temperature_dht11, temperature_mcp9808, humidity_dht11, co2_level, moisture_content
